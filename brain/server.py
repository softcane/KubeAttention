"""
gRPC Server for KubeAttention Brain

Listens on Unix Domain Socket for Score requests from the Go scheduler plugin.
Implements the Brain service defined in scheduler.proto.
"""

import asyncio
from concurrent import futures
import math
import os
import signal
import sys
import time
from typing import Optional

import grpc
from grpc import aio
import numpy as np

# Import generated proto stubs.
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

from .config import INFERENCE, MODEL_SELECTION, TELEMETRY
from .metrics_schema import (
    FEATURE_SCHEMA_VERSION,
    NODE_FEATURE_SCHEMA,
    NodeMetricsSnapshot,
    REQUIRED_PROTO_METRICS,
)
from .models import get_model
from .models.base import BaseScorer
from .tensor_encoder import ClusterTensorEncoder, PodContext
from .metrics_exporter import record_scoring_request, run_http_server, set_model_ready


# Default UDS path
DEFAULT_UDS_PATH = "/var/run/kubeattention/brain.sock"

# For local development/testing
DEV_UDS_PATH = "/tmp/kubeattention-brain.sock"


class RequestValidationError(ValueError):
    def __init__(self, code: grpc.StatusCode, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class BrainServicer:
    """Validated, fail-safe implementation of the Brain RPC boundary."""

    def __init__(
        self,
        model: Optional[BaseScorer] = None,
        encoder: Optional[ClusterTensorEncoder] = None,
        model_version: str = "v0.2.0",
        model_path: Optional[str] = None,
    ):
        self.model = model or get_model(MODEL_SELECTION.MODEL_TYPE)
        self.encoder = encoder or ClusterTensorEncoder()
        self.model_version = model_version
        self.model_load_error: Optional[str] = None
        self.last_latency_ms = 0
        self._request_count = 0
        self.last_telemetry_cache: dict[str, NodeMetricsSnapshot] = {}

        if model_path:
            if not os.path.isfile(model_path):
                self.model_load_error = f"model checkpoint not found: {model_path}"
            else:
                try:
                    self.model.load(model_path)
                except Exception as error:
                    self.model_load_error = f"incompatible model checkpoint: {error}"
        elif not self.model.ready:
            self.model_load_error = "no compatible model checkpoint loaded"
        set_model_ready(self.ready)

    @property
    def ready(self) -> bool:
        return self.model.ready and self.model_load_error is None

    @staticmethod
    def _criticality_name(value: int) -> str:
        names = ("unknown", "low", "medium", "high")
        if value < 0 or value >= len(names):
            raise RequestValidationError(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"pod criticality {value} is invalid",
            )
        return names[value]

    @staticmethod
    def _validate_pod(request) -> PodContext:
        if not request.HasField("pod_requirements"):
            raise RequestValidationError(
                grpc.StatusCode.INVALID_ARGUMENT, "pod_requirements is required"
            )
        pod = request.pod_requirements
        if not pod.pod_name or not pod.pod_namespace:
            raise RequestValidationError(
                grpc.StatusCode.INVALID_ARGUMENT,
                "pod name and namespace are required",
            )
        if pod.cpu_milli < 0 or pod.memory_bytes < 0:
            raise RequestValidationError(
                grpc.StatusCode.INVALID_ARGUMENT,
                "pod CPU and memory requirements must be non-negative",
            )
        return PodContext(
            pod_name=pod.pod_name,
            pod_namespace=pod.pod_namespace,
            cpu_milli=pod.cpu_milli,
            memory_bytes=pod.memory_bytes,
            priority=pod.priority,
            workload_type=pod.workload_type or "unknown",
            criticality=BrainServicer._criticality_name(pod.criticality),
            labels=dict(pod.labels),
        )

    @staticmethod
    def _validate_nodes(nodes) -> tuple[list[NodeMetricsSnapshot], Optional[str]]:
        if not nodes:
            raise RequestValidationError(
                grpc.StatusCode.INVALID_ARGUMENT, "at least one candidate node is required"
            )

        now_ms = int(time.time() * 1000)
        names: set[str] = set()
        snapshots: list[NodeMetricsSnapshot] = []
        degraded_reasons: list[str] = []
        known_metrics = {
            spec.proto_metric for spec in NODE_FEATURE_SCHEMA if spec.proto_metric is not None
        }
        for telemetry in nodes:
            if not telemetry.node_name:
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT, "candidate node name is required"
                )
            if telemetry.node_name in names:
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"duplicate candidate node {telemetry.node_name!r}",
                )
            names.add(telemetry.node_name)
            if telemetry.schema_version != FEATURE_SCHEMA_VERSION:
                raise RequestValidationError(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"node {telemetry.node_name!r} uses schema "
                    f"{telemetry.schema_version!r}, expected {FEATURE_SCHEMA_VERSION!r}",
                )
            if telemetry.timestamp_unix_ms <= 0:
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"node {telemetry.node_name!r} has no observation timestamp",
                )
            if telemetry.timestamp_unix_ms > now_ms:
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"node {telemetry.node_name!r} has a future observation timestamp",
                )
            if telemetry.observation_window_ms < 0:
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"node {telemetry.node_name!r} has a negative observation window",
                )

            available = list(telemetry.available_metrics)
            if len(available) != len(set(available)) or any(
                metric not in known_metrics for metric in available
            ):
                raise RequestValidationError(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"node {telemetry.node_name!r} has invalid metric availability",
                )

            snapshot = NodeMetricsSnapshot.from_proto(telemetry)
            for spec in NODE_FEATURE_SCHEMA:
                value = float(getattr(snapshot, spec.name))
                if not math.isfinite(value):
                    raise RequestValidationError(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"node {telemetry.node_name!r} field {spec.name} must be finite",
                    )
                if value < spec.minimum or value > spec.maximum:
                    raise RequestValidationError(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"node {telemetry.node_name!r} field {spec.name} is out of range",
                    )

            age_ms = now_ms - telemetry.timestamp_unix_ms
            if age_ms > TELEMETRY.MAX_STALENESS_MS:
                degraded_reasons.append(
                    f"node {telemetry.node_name!r} telemetry is {age_ms}ms old"
                )
            missing = REQUIRED_PROTO_METRICS - snapshot.available_metrics
            if missing:
                degraded_reasons.append(
                    f"node {telemetry.node_name!r} is missing required metrics "
                    f"{sorted(missing)}"
                )
            snapshots.append(snapshot)

        degradation = "; ".join(degraded_reasons) if degraded_reasons else None
        return snapshots, degradation

    @staticmethod
    def _neutral_scores(nodes, reason: str):
        return [
            scheduler_pb2.NodeScore(
                node_name=node.node_name,
                score=INFERENCE.FALLBACK_SCORE,
                reasoning=reason,
            )
            for node in nodes
        ]

    async def _infer(self, node_features, pod_features, node_names, context):
        timeout_seconds = INFERENCE.MAX_LATENCY_MS / 1000
        time_remaining = getattr(context, "time_remaining", None)
        if callable(time_remaining):
            remaining = time_remaining()
            if remaining is not None:
                timeout_seconds = min(timeout_seconds, max(0.0, remaining))
        if timeout_seconds <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(
            asyncio.to_thread(
                self.model.score_nodes, node_features, pod_features, node_names
            ),
            timeout=timeout_seconds,
        )

    async def Score(self, request, context):
        """Use the same validated batch path for a single candidate."""
        batch_request = scheduler_pb2.BatchScoreRequest(
            pod_requirements=request.pod_requirements,
            nodes=[request.node_telemetry],
        )
        response = await self.BatchScore(batch_request, context)
        if not response.scores:
            return scheduler_pb2.ScoreResponse()
        result = response.scores[0]
        return scheduler_pb2.ScoreResponse(
            score=result.score,
            reasoning=result.reasoning,
        )

    def _finish_batch(self, start_time, result: str, scores=None):
        response_scores = list(scores or [])
        latency_seconds = time.perf_counter() - start_time
        self.last_latency_ms = int(latency_seconds * 1000)
        record_scoring_request(result, latency_seconds, response_scores)
        return scheduler_pb2.BatchScoreResponse(scores=response_scores)

    async def BatchScore(self, request, context):
        start_time = time.perf_counter()
        self._request_count += 1
        try:
            pod = self._validate_pod(request)
            snapshots, degradation = self._validate_nodes(request.nodes)
        except RequestValidationError as error:
            context.set_code(error.code)
            context.set_details(error.detail)
            return self._finish_batch(start_time, "invalid_request")

        if degradation:
            return self._finish_batch(
                start_time,
                "degraded_telemetry",
                self._neutral_scores(
                    request.nodes, f"telemetry unavailable: {degradation}"
                ),
            )
        if not self.ready:
            return self._finish_batch(
                start_time,
                "model_unavailable",
                self._neutral_scores(
                    request.nodes, f"model unavailable: {self.model_load_error}"
                ),
            )

        node_features = np.asarray(
            [snapshot.to_feature_vector() for snapshot in snapshots],
            dtype=np.float32,
        )
        pod_features = np.asarray(pod.to_feature_vector(), dtype=np.float32)
        node_names = [snapshot.node_name for snapshot in snapshots]
        try:
            results = await self._infer(
                node_features, pod_features, node_names, context
            )
        except asyncio.TimeoutError:
            return self._finish_batch(
                start_time,
                "deadline_exceeded",
                self._neutral_scores(
                    request.nodes, "inference deadline exceeded; using neutral score"
                ),
            )
        except Exception as error:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"model inference failed: {error}")
            return self._finish_batch(start_time, "inference_error")

        if (
            len(results) != len(node_names)
            or [result.node_name for result in results] != node_names
        ):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("model returned an invalid candidate set")
            return self._finish_batch(start_time, "invalid_model_output")
        if any(result.score < 0 or result.score > 100 for result in results):
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("model returned an out-of-range score")
            return self._finish_batch(start_time, "invalid_model_output")

        for snapshot in snapshots:
            self.last_telemetry_cache[snapshot.node_name] = snapshot
        scores = [
            scheduler_pb2.NodeScore(
                node_name=result.node_name,
                score=result.score,
                reasoning=result.reasoning,
            )
            for result in results
        ]
        return self._finish_batch(start_time, "success", scores)

    async def HealthCheck(self, request, context):
        return scheduler_pb2.HealthCheckResponse(
            healthy=self.ready,
            latency_ms=self.last_latency_ms,
            model_version=self.model_version,
            model_schema_version=FEATURE_SCHEMA_VERSION if self.ready else "",
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
        model_path: Optional[str] = None,
    ):
        self.uds_path = uds_path
        self.max_workers = max_workers
        if model_path is None and model is None:
            model_path = os.environ.get("MODEL_PATH", "/models/best_model.pt")
        self.servicer = BrainServicer(model=model, model_path=model_path)
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
        metrics_port = os.environ.get("METRICS_PORT")
        if metrics_port:
            run_http_server(int(metrics_port))
    
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
