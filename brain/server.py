"""
gRPC Server for KubeAttention Brain

Listens on Unix Domain Socket for Score requests from the Go scheduler plugin.
Implements the Brain service defined in scheduler.proto.
"""

import asyncio
import time
import os
import signal
from concurrent import futures
from typing import Optional

import grpc
from grpc import aio

# Import generated proto stubs - REQUIRED for production
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gen', 'python'))

try:
    import scheduler_pb2 as scheduler_pb2
    import scheduler_pb2_grpc as scheduler_pb2_grpc
except ImportError as e:
    raise RuntimeError(
        f"FATAL: Proto stubs not generated!\n"
        f"Run: python -m grpc_tools.protoc -I./proto --python_out=./gen/python "
        f"--grpc_python_out=./gen/python ./proto/scheduler.proto\n"
        f"Error: {e}"
    )

from .models import get_model
from .models.base import BaseScorer
from .tensor_encoder import ClusterTensorEncoder, ClusterTensor, PodContext
from .metrics_schema import NodeMetricsSnapshot
from .config import INFERENCE, TELEMETRY, MODEL_SELECTION
from .utils import create_neutral_result


# Default UDS path
DEFAULT_UDS_PATH = "/var/run/kubeattention/brain.sock"

# For local development/testing
DEV_UDS_PATH = "/tmp/kubeattention-brain.sock"


class BrainServicer:
    """
    Implements the Brain gRPC service.
    
    Handles Score, BatchScore, and HealthCheck RPCs.
    
    Safety Features:
    - Rejects requests with stale telemetry (configured via TELEMETRY.MAX_STALENESS_MS)
    - Returns neutral score on NaN/Inf input
    """
    
    def __init__(
        self,
        model: Optional[BaseScorer] = None,
        encoder: Optional[ClusterTensorEncoder] = None,
        model_version: str = "v0.2.0",
    ):
        self.model = model or get_model(MODEL_SELECTION.MODEL_TYPE)
        self.encoder = encoder or ClusterTensorEncoder()
        self.model_version = model_version
        self.last_latency_ms = 0
        self._request_count = 0
        
        # Cache for proactive rebalancing (Phase 4)
        # node_name -> NodeMetricsSnapshot
        self.last_telemetry_cache = {}
    
    async def Score(self, request, context):
        """Handle single node score request."""
        start_time = time.perf_counter()
        
        try:
            # SAFETY: Check for stale telemetry data
            if hasattr(request, 'node_telemetry') and request.node_telemetry.timestamp_unix_ms > 0:
                current_time_ms = int(time.time() * 1000)
                telemetry_age_ms = current_time_ms - request.node_telemetry.timestamp_unix_ms
                
                if telemetry_age_ms > TELEMETRY.MAX_STALENESS_MS:
                    context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                    context.set_details(f"Telemetry data is {telemetry_age_ms}ms old (max: {TELEMETRY.MAX_STALENESS_MS}ms)")
                    return scheduler_pb2.ScoreResponse(
                        score=INFERENCE.FALLBACK_SCORE,
                        reasoning=f"STALE DATA: Telemetry {telemetry_age_ms}ms old, using neutral score",
                        confidence=INFERENCE.FALLBACK_CONFIDENCE,
                    )
            
            # Extract node snapshot
            if request.node_telemetry.node_name:
                snapshot = NodeMetricsSnapshot.from_proto(request.node_telemetry)
            else:
                snapshot = NodeMetricsSnapshot(
                    node_name=request.node_name,
                    cpu_utilization=request.telemetry.get("cpu_utilization", 0.5),
                    memory_utilization=request.telemetry.get("memory_utilization", 0.5),
                )
            
            # Convert snapshot to numpy feature array
            from .metrics_schema import FEATURE_NAMES
            import numpy as np
            
            node_features = []
            for feat_name in FEATURE_NAMES:
                val = getattr(snapshot, feat_name, 0.0)
                node_features.append(float(val) if val is not None else 0.0)
            node_features = np.array([node_features], dtype=np.float32)
            
            # Build pod context features
            pod_cpu_norm = min(request.pod_requirements.cpu_milli / 4000.0, 1.0) if hasattr(request, 'pod_requirements') else 0.25
            pod_mem_norm = min(request.pod_requirements.memory_bytes / (16 * 1024**3), 1.0) if hasattr(request, 'pod_requirements') else 0.25
            pod_features = np.array([pod_cpu_norm, pod_mem_norm, 0.5, 0.5, 0.5], dtype=np.float32)
            
            # Run model inference using new score_nodes interface
            results = self.model.score_nodes(node_features, pod_features, [snapshot.node_name])
            result = results[0]
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.last_latency_ms = int(elapsed_ms)
            
            # Update cache for proactive rebalancing
            self.last_telemetry_cache[snapshot.node_name] = snapshot
            
            return scheduler_pb2.ScoreResponse(
                score=result.score,
                reasoning=result.reasoning,
                confidence=result.confidence,
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return scheduler_pb2.ScoreResponse(score=50, reasoning=f"Error: {e}")

    
    async def BatchScore(self, request, context):
        """Handle batch scoring for multiple nodes."""
        start_time = time.perf_counter()
        
        try:
            from .metrics_schema import FEATURE_NAMES
            import numpy as np
            
            # SAFETY: Check for stale telemetry (Issue 3 fix)
            current_time_ms = int(time.time() * 1000)
            stale_nodes = []
            for node in request.nodes:
                if node.timestamp_unix_ms > 0:
                    telemetry_age_ms = current_time_ms - node.timestamp_unix_ms
                    if telemetry_age_ms > TELEMETRY.MAX_STALENESS_MS:
                        stale_nodes.append(node.node_name)
            
            if stale_nodes:
                # Return neutral scores for all nodes if any are stale
                scores = [
                    scheduler_pb2.NodeScore(
                        node_name=node.node_name,
                        score=INFERENCE.FALLBACK_SCORE,
                        reasoning="STALE DATA: Telemetry too old, using neutral score",
                        confidence=INFERENCE.FALLBACK_CONFIDENCE,
                    )
                    for node in request.nodes
                ]
                return scheduler_pb2.BatchScoreResponse(scores=scores)
            
            # Convert all node telemetry to snapshots and feature arrays
            node_features_list = []
            node_names = []
            for node in request.nodes:
                snap = NodeMetricsSnapshot.from_proto(node)
                # Update cache for proactive rebalancing
                self.last_telemetry_cache[node.node_name] = snap
                node_names.append(node.node_name)
                
                # Extract features
                features = []
                for feat_name in FEATURE_NAMES:
                    val = getattr(snap, feat_name, 0.0)
                    features.append(float(val) if val is not None else 0.0)
                node_features_list.append(features)
            
            node_features = np.array(node_features_list, dtype=np.float32)
            
            # Build pod context features
            pod_cpu_norm = min(request.pod_requirements.cpu_milli / 4000.0, 1.0)
            pod_mem_norm = min(request.pod_requirements.memory_bytes / (16 * 1024**3), 1.0)
            pod_features = np.array([pod_cpu_norm, pod_mem_norm, 0.5, 0.5, 0.5], dtype=np.float32)
            
            # Run model inference using new score_nodes interface
            results = self.model.score_nodes(node_features, pod_features, node_names)
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.last_latency_ms = int(elapsed_ms)
            
            # Check for timeout
            if elapsed_ms > INFERENCE.MAX_LATENCY_MS:
                scores = [
                    scheduler_pb2.NodeScore(
                        node_name=node.node_name,
                        score=INFERENCE.FALLBACK_SCORE,
                        reasoning=f"TIMEOUT: Inference exceeded {INFERENCE.MAX_LATENCY_MS}ms, using neutral score",
                        confidence=INFERENCE.FALLBACK_CONFIDENCE,
                    )
                    for node in request.nodes
                ]
                return scheduler_pb2.BatchScoreResponse(scores=scores)
            
            scores = [
                scheduler_pb2.NodeScore(
                    node_name=r.node_name,
                    score=r.score,
                    reasoning=r.reasoning,
                    confidence=r.confidence,
                )
                for r in results
            ]
            
            return scheduler_pb2.BatchScoreResponse(scores=scores)
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return scheduler_pb2.BatchScoreResponse(scores=[])

    
    async def HealthCheck(self, request, context):
        """Health check for circuit breaker."""
        return scheduler_pb2.HealthCheckResponse(
            healthy=True,
            latency_ms=self.last_latency_ms,
            model_version=self.model_version,
        )


class BrainServer:
    """
    Async gRPC server that listens on Unix Domain Socket.
    """
    
    def __init__(
        self,
        uds_path: str = DEV_UDS_PATH,
        max_workers: int = 4,
        model: Optional[BaseScorer] = None,
    ):
        self.uds_path = uds_path
        self.max_workers = max_workers
        self.servicer = BrainServicer(model=model)
        self.server: Optional[aio.Server] = None
        
        # Initialize Rebalancer (Phase 4)
        from .rebalancer import Rebalancer
        self.rebalancer = Rebalancer(
            model=self.servicer.model,
            encoder=self.servicer.encoder,
            telemetry_cache=self.servicer.last_telemetry_cache
        )
    
    async def start(self):
        """Start the gRPC server on UDS and optionally TCP."""
        # ... existing socket setup ...
        
        # Start Rebalancer background task (Phase 4)
        asyncio.create_task(self.rebalancer.start())
        
        # Ensure socket directory exists
        socket_dir = os.path.dirname(self.uds_path)
        if socket_dir and not os.path.exists(socket_dir):
            os.makedirs(socket_dir, exist_ok=True)
        
        # Remove existing socket file
        if os.path.exists(self.uds_path):
            os.unlink(self.uds_path)
        
        # Create server
        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        
        # Register servicer - proto stubs are REQUIRED
        scheduler_pb2_grpc.add_BrainServicer_to_server(
            self.servicer, self.server
        )
        
        # Bind to UDS
        self.server.add_insecure_port(f"unix://{self.uds_path}")
        
        # Also bind to TCP port for Kubernetes health checks and external access
        tcp_port = os.environ.get("BRAIN_TCP_PORT", "50051")
        self.server.add_insecure_port(f"[::]:{tcp_port}")
        
        print(f"Brain server starting on unix://{self.uds_path} and TCP port {tcp_port}")
        await self.server.start()
        print(f"Brain server ready!")
    
    async def stop(self):
        """Stop the server gracefully."""
        if self.server:
            await self.server.stop(grace=5)
        if os.path.exists(self.uds_path):
            os.unlink(self.uds_path)
        print("Brain server stopped")

    
    async def wait_for_termination(self):
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()


async def serve(uds_path: str = DEV_UDS_PATH):
    """Main entry point to run the Brain server."""
    server = BrainServer(uds_path=uds_path)
    
    # Handle shutdown signals (must use get_running_loop inside async context)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))
    
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KubeAttention Brain Server")
    parser.add_argument(
        "--socket", 
        default=DEV_UDS_PATH,
        help="Unix socket path"
    )
    args = parser.parse_args()
    
    asyncio.run(serve(args.socket))
