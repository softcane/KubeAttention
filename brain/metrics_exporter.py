"""Prometheus metrics populated only from observed Brain runtime state."""

from collections.abc import Iterable

from prometheus_client import Counter, Gauge, Histogram, start_http_server


SCORING_REQUESTS = Counter(
    "kubeattention_brain_scoring_requests_total",
    "Completed Brain scoring requests by result.",
    ("result",),
)
INFERENCE_LATENCY = Histogram(
    "kubeattention_brain_inference_latency_seconds",
    "Observed end-to-end Brain scoring latency.",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
CANDIDATE_SCORE = Gauge(
    "kubeattention_brain_candidate_score",
    "Most recently observed absolute score for a candidate node.",
    ("node",),
)
MODEL_READY = Gauge(
    "kubeattention_brain_model_ready",
    "Whether Brain has a loaded checkpoint compatible with the serving schema.",
)


def set_model_ready(ready: bool) -> None:
    MODEL_READY.set(1 if ready else 0)


def record_scoring_request(result: str, latency_seconds: float, scores: Iterable) -> None:
    SCORING_REQUESTS.labels(result=result).inc()
    INFERENCE_LATENCY.observe(latency_seconds)
    for score in scores:
        CANDIDATE_SCORE.labels(node=score.node_name).set(score.score)


def run_http_server(port: int) -> None:
    start_http_server(port)
