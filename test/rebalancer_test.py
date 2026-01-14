"""
Verification test for KubeAttention Proactive Rebalancer.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch
import torch

from brain.rebalancer import Rebalancer
from brain.metrics_schema import NodeMetricsSnapshot
from brain.config import REBALANCER

class TestRebalancer(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Mock model and encoder
        self.model = MagicMock()
        self.encoder = MagicMock()
        self.telemetry_cache = {}
        
        # Patch K8s config loading
        with patch('kubernetes.config.load_incluster_config'), \
             patch('kubernetes.config.load_kube_config'):
            self.rebalancer = Rebalancer(self.model, self.encoder, self.telemetry_cache)
            # Mock the V1 API
            self.rebalancer.v1 = MagicMock()

    async def test_rebalance_trigger(self):
        """Test that a rebalance is triggered when a better node exists."""
        # 1. Setup mock pod
        mock_pod = MagicMock()
        mock_pod.metadata.name = "test-pod"
        mock_pod.metadata.namespace = "default"
        mock_pod.metadata.annotations = {}
        mock_pod.spec.node_name = "node-bad"
        mock_pod.status.phase = "Running"
        mock_pod.spec.containers = []
        
        self.rebalancer.v1.list_pod_for_all_namespaces.return_value.items = [mock_pod]
        
        # 2. Setup telemetry cache
        snap_bad = NodeMetricsSnapshot(node_name="node-bad", cpu_utilization=0.9)
        snap_good = NodeMetricsSnapshot(node_name="node-good", cpu_utilization=0.1)
        self.telemetry_cache["node-bad"] = snap_bad
        self.telemetry_cache["node-good"] = snap_good
        
        # 3. Setup model return values
        # score_batch returns list of results
        self.model.score_batch.return_value = [
            {"node_name": "node-bad", "score": 30, "confidence": 0.8, "reasoning": "High load"},
            {"node_name": "node-good", "score": 80, "confidence": 0.9, "reasoning": "Low load"}
        ]
        
        # 4. Run rebalancer
        await self.rebalancer.run_once()
        
        # 5. Verify patch was called with recommendation
        self.rebalancer.v1.patch_namespaced_pod.assert_called_once()
        args, kwargs = self.rebalancer.v1.patch_namespaced_pod.call_args
        
        name, namespace, body = args
        self.assertEqual(name, "test-pod")
        self.assertEqual(body["metadata"]["annotations"]["kubeattention.io/rebalance-target"], "node-good")
        self.assertEqual(body["metadata"]["annotations"]["kubeattention.io/rebalance-delta"], "50")

    async def test_no_rebalance_if_already_annotated(self):
        """Test that we don't re-annotate if recommendation already exists."""
        mock_pod = MagicMock()
        mock_pod.metadata.annotations = {"kubeattention.io/rebalance-target": "node-some"}
        mock_pod.spec.node_name = "node-bad"
        mock_pod.status.phase = "Running"
        
        self.rebalancer.v1.list_pod_for_all_namespaces.return_value.items = [mock_pod]
        
        await self.rebalancer.run_once()
        
        self.rebalancer.v1.patch_namespaced_pod.assert_not_called()

if __name__ == "__main__":
    unittest.main()
