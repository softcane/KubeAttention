"""
KubeAttention Brain - Transformer-based Node Scoring

This package implements the AI "Brain" for KubeAttention scheduler:
- config: Centralized configuration constants
- metrics_schema: eBPF Tetragon metrics definitions
- tensor_encoder: ClusterTensor encoding from raw telemetry
- model: PyTorch Transformer for node scoring
- server: gRPC server over Unix Domain Socket
- utils: Shared utility functions
"""

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "INFERENCE",
    "TELEMETRY", 
    "SCORE",
    "NORMALIZATION",
    "MODEL",
    # Core classes
    "AttentionScorer",
    "create_model",
    "ClusterTensorEncoder",
    "ClusterTensor",
    "PodContext",
    "NodeMetricsSnapshot",
    # Server
    "BrainServer",
    "BrainServicer",
    # Utilities
    "create_neutral_result",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("INFERENCE", "TELEMETRY", "SCORE", "NORMALIZATION", "MODEL"):
        from . import config
        return getattr(config, name)
    elif name == "AttentionScorer":
        from .model import AttentionScorer
        return AttentionScorer
    elif name == "create_model":
        from .model import create_model
        return create_model
    elif name in ("ClusterTensorEncoder", "ClusterTensor", "PodContext"):
        from . import tensor_encoder
        return getattr(tensor_encoder, name)
    elif name == "NodeMetricsSnapshot":
        from .metrics_schema import NodeMetricsSnapshot
        return NodeMetricsSnapshot
    elif name in ("BrainServer", "BrainServicer"):
        from . import server
        return getattr(server, name)
    elif name == "create_neutral_result":
        from .utils import create_neutral_result
        return create_neutral_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
