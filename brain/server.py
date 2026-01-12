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

from .model import AttentionScorer, create_model
from .tensor_encoder import ClusterTensorEncoder, ClusterTensor, PodContext
from .metrics_schema import NodeMetricsSnapshot
from .config import INFERENCE, TELEMETRY
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
        model: Optional[AttentionScorer] = None,
        encoder: Optional[ClusterTensorEncoder] = None,
        model_version: str = "v0.1.0",
    ):
        self.model = model or create_model()
        self.encoder = encoder or ClusterTensorEncoder()
        self.model_version = model_version
        self.last_latency_ms = 0
        self._request_count = 0
    
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
            
            # Build ClusterTensor from request
            if request.node_telemetry.node_name:
                # Use structured telemetry
                snapshot = NodeMetricsSnapshot.from_proto(request.node_telemetry)
                pod = PodContext(
                    pod_name=request.pod_requirements.pod_name or request.pod_name,
                    pod_namespace=request.pod_requirements.pod_namespace or request.pod_namespace,
                    cpu_milli=request.pod_requirements.cpu_milli,
                    memory_bytes=request.pod_requirements.memory_bytes,
                    workload_type=request.pod_requirements.workload_type or "unknown",
                )
                cluster_tensor = self.encoder.encode_node_snapshots(
                    [[snapshot]], pod
                )
            else:
                # Legacy: simple telemetry map
                snapshot = NodeMetricsSnapshot(
                    node_name=request.node_name,
                    cpu_utilization=request.telemetry.get("cpu_utilization", 0.5),
                    memory_utilization=request.telemetry.get("memory_utilization", 0.5),
                )
                pod = PodContext(
                    pod_name=request.pod_name,
                    pod_namespace=request.pod_namespace,
                    cpu_milli=1000,
                    memory_bytes=1024 * 1024 * 512,
                )
                cluster_tensor = self.encoder.encode_node_snapshots(
                    [[snapshot]], pod
                )
            
            # Run model inference
            results = self.model.score_batch(cluster_tensor)
            result = results[0]
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.last_latency_ms = int(elapsed_ms)
            
            return scheduler_pb2.ScoreResponse(
                score=result["score"],
                reasoning=result["reasoning"],
                confidence=result["confidence"],
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return scheduler_pb2.ScoreResponse(score=50, reasoning=f"Error: {e}")
    
    async def BatchScore(self, request, context):
        """Handle batch scoring for multiple nodes."""
        start_time = time.perf_counter()
        
        try:
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
            
            # Convert all node telemetry to snapshots
            snapshots = [
                [NodeMetricsSnapshot.from_proto(node)]
                for node in request.nodes
            ]
            
            pod = PodContext(
                pod_name=request.pod_requirements.pod_name,
                pod_namespace=request.pod_requirements.pod_namespace,
                cpu_milli=request.pod_requirements.cpu_milli,
                memory_bytes=request.pod_requirements.memory_bytes,
                workload_type=request.pod_requirements.workload_type or "unknown",
            )
            
            cluster_tensor = self.encoder.encode_node_snapshots(snapshots, pod)
            
            # SAFETY: Configured timeout with fallback
            try:
                results = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(
                        None, self.model.score_batch, cluster_tensor
                    ),
                    timeout=INFERENCE.MAX_LATENCY_MS / 1000.0
                )
            except asyncio.TimeoutError:
                # Fallback: return neutral scores if inference takes > 45ms
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self.last_latency_ms = int(elapsed_ms)
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
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.last_latency_ms = int(elapsed_ms)
            
            scores = [
                scheduler_pb2.NodeScore(
                    node_name=r["node_name"],
                    score=r["score"],
                    reasoning=r["reasoning"],
                    confidence=r["confidence"],
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
        model: Optional[AttentionScorer] = None,
    ):
        self.uds_path = uds_path
        self.max_workers = max_workers
        self.servicer = BrainServicer(model=model)
        self.server: Optional[aio.Server] = None
    
    async def start(self):
        """Start the gRPC server on UDS."""
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
        
        print(f"🧠 Brain server starting on unix://{self.uds_path}")
        await self.server.start()
        print(f"✅ Brain server ready!")
    
    async def stop(self):
        """Stop the server gracefully."""
        if self.server:
            await self.server.stop(grace=5)
        if os.path.exists(self.uds_path):
            os.unlink(self.uds_path)
        print("🛑 Brain server stopped")
    
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
