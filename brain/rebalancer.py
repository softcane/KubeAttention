"""
Proactive Rebalancer for KubeAttention (Phase 4)

Monitors the cluster for pods running on sub-optimal nodes and provides
rebalancing recommendations via pod annotations.
"""

import asyncio
import time
from typing import Dict, List, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .models.base import BaseScorer
from .tensor_encoder import ClusterTensorEncoder, PodContext
from .metrics_schema import NodeMetricsSnapshot
from .config import REBALANCER, SCORE

class Rebalancer:
    """
    Background worker that audits pod placements using the model's intelligence.
    """
    
    def __init__(
        self,
        model: BaseScorer,
        encoder: ClusterTensorEncoder,
        telemetry_cache: Dict[str, NodeMetricsSnapshot],
    ):
        self.model = model
        self.encoder = encoder
        self.telemetry_cache = telemetry_cache
        
        # Initialize K8s client
        try:
            config.load_incluster_config()
        except Exception:
            try:
                config.load_kube_config()
            except Exception:
                print("Rebalancer: Failed to load K8s config, running in offline mode")
        
        self.v1 = client.CoreV1Api()

    async def start(self):
        """Start the background poll loop."""
        if not REBALANCER.ENABLED:
            print("Rebalancer: Disabled in config")
            return
            
        print(f"Rebalancer: Starting background loop (interval: {REBALANCER.INTERVAL_SECONDS}s)")
        
        while True:
            try:
                await self.run_once()
            except Exception as e:
                print(f"Rebalancer: Error during run: {e}")
            
            await asyncio.sleep(REBALANCER.INTERVAL_SECONDS)

    async def run_once(self):
        """Perform a single cluster-wide audit."""
        if not self.telemetry_cache:
            print("Rebalancer: No telemetry in cache yet, skipping scan")
            return

        print("Rebalancer: Auditing cluster placements...")
        
        # 1. Get all relevant pods
        try:
            pods = self.v1.list_pod_for_all_namespaces().items
        except Exception as e:
            print(f"Rebalancer: Failed to list pods: {e}")
            return

        rebalance_count = 0
        
        for pod in pods:
            if rebalance_count >= REBALANCER.MAX_PODS_PER_RUN:
                break
                
            if self._should_audit_pod(pod):
                rec = await self._audit_pod(pod)
                if rec:
                    self._apply_recommendation(pod, rec)
                    rebalance_count += 1

        if rebalance_count > 0:
            print(f"Rebalancer: Identified {rebalance_count} rebalancing opportunities")

    def _should_audit_pod(self, pod) -> bool:
        """Filter for pods that are worth auditing."""
        # Skip if in excluded namespace
        if pod.metadata.namespace in REBALANCER.EXCLUDE_NAMESPACES:
            return False
            
        # Skip if not running on a node
        if not pod.spec.node_name or pod.status.phase != "Running":
            return False
            
        # Skip if already being rebalanced
        if pod.metadata.annotations and "kubeattention.io/rebalance-target" in pod.metadata.annotations:
            # We already recommended a move recently
            return False
            
        return True

    async def _audit_pod(self, pod) -> Optional[Dict]:
        """Determine if a pod should be moved."""
        import numpy as np
        from .metrics_schema import FEATURE_NAMES
        
        namespace = pod.metadata.namespace
        name = pod.metadata.name
        current_node = pod.spec.node_name
        
        # 1. Collect resources
        cpu_milli = 0
        mem_bytes = 0
        for container in pod.spec.containers:
            # Simplified resource extraction
            req = container.resources.requests or {}
            # pod req parsing is complex in K8s, we use 1000m/1Gi fallback for audit
            cpu_val = req.get('cpu', '1000m')
            mem_val = req.get('memory', '1Gi')
            # (Note: real parser would go here)
            cpu_milli += 1000 # placeholder
            mem_bytes += 1024*1024*1024 # placeholder
            
        # 2. Build pod context feature vector (5 features)
        # [cpu_norm, mem_norm, workload_type_encoded, criticality_encoded, priority]
        cpu_norm = min(cpu_milli / 4000.0, 1.0)  # Normalize to [0,1]
        mem_norm = min(mem_bytes / (16 * 1024**3), 1.0)  # Normalize assuming 16GB max
        criticality = pod.metadata.annotations.get("kubeattention.io/criticality", "unknown") if pod.metadata.annotations else "unknown"
        crit_map = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25, "unknown": 0.5}
        crit_val = crit_map.get(criticality, 0.5)
        pod_features = np.array([cpu_norm, mem_norm, 0.5, crit_val, 0.5], dtype=np.float32)
        
        # 3. Extract node features from telemetry cache
        candidate_nodes = list(self.telemetry_cache.keys())
        if current_node not in candidate_nodes:
            return None # Can't score current node
        
        # Build node feature matrix from cached snapshots
        node_features_list = []
        for node_name in candidate_nodes:
            snapshot = self.telemetry_cache[node_name]
            # Extract features in FEATURE_NAMES order
            features = []
            for feat_name in FEATURE_NAMES:
                val = getattr(snapshot, feat_name, 0.0)
                if val is None:
                    val = 0.0
                features.append(float(val))
            node_features_list.append(features)
        
        node_features = np.array(node_features_list, dtype=np.float32)
        
        # 4. Run inference using new score_nodes interface
        results = self.model.score_nodes(node_features, pod_features, candidate_nodes)
        
        # 5. Find current vs best
        current_score = -1
        best_score = -1
        best_node = None
        best_reason = ""
        
        for res in results:
            node = res.node_name
            score = res.score
            if node == current_node:
                current_score = score
            if score > best_score:
                best_score = score
                best_node = node
                best_reason = res.reasoning

        # 6. Decision threshold
        # Rebalance if:
        # a) Current node is performing poorly (score < MIN_SOURCE_SCORE)
        # b) There is a significantly better alternative (delta > THRESHOLD)
        if current_score < REBALANCER.MIN_SOURCE_SCORE and (best_score - current_score) >= REBALANCER.SCORE_DELTA_THRESHOLD:
            return {
                "target_node": best_node,
                "current_score": current_score,
                "new_score": best_score,
                "reason": best_reason
            }
            
        return None


    def _apply_recommendation(self, pod, rec: Dict):
        """Annotate the pod with the recommendation."""
        try:
            annotations = pod.metadata.annotations or {}
            annotations.update({
                "kubeattention.io/rebalance-target": rec["target_node"],
                "kubeattention.io/rebalance-delta": f"{rec['new_score'] - rec['current_score']}",
                "kubeattention.io/rebalance-reason": rec["reason"],
                "kubeattention.io/rebalance-timestamp": str(int(time.time()))
            })
            
            body = {"metadata": {"annotations": annotations}}
            self.v1.patch_namespaced_pod(pod.metadata.name, pod.metadata.namespace, body)
            print(f"Rebalancer: RECOMMENDED MOVE {pod.metadata.namespace}/{pod.metadata.name}: "
                  f"{pod.spec.node_name} ({rec['current_score']}) -> {rec['target_node']} ({rec['new_score']})")
        except ApiException as e:
            print(f"Rebalancer: Error patching pod {pod.metadata.name}: {e}")
