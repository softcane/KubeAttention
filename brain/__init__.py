"""
KubeAttention Brain - ML-based Node Scoring

This package implements the AI "Brain" for KubeAttention scheduler:
- config: Centralized configuration constants
- metrics_schema: eBPF Tetragon metrics definitions
- tensor_encoder: ClusterTensor encoding from raw telemetry
- models: Scoring models (MLP, XGBoost)
- server: gRPC server over Unix Domain Socket
- utils: Shared utility functions
"""

__version__ = "0.2.0"

__all__ = [
    # Configuration
    "INFERENCE",
    "TELEMETRY", 
    "SCORE",
    "NORMALIZATION",
    "MODEL_SELECTION",
    # Core classes
    "get_model",
    "BaseScorer",
    "MLPScorer",
    "XGBoostScorer",
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
    if name in ("INFERENCE", "TELEMETRY", "SCORE", "NORMALIZATION", "MODEL_SELECTION"):
        from . import config
        return getattr(config, name)
    elif name == "get_model":
        from .models import get_model
        return get_model
    elif name == "BaseScorer":
        from .models.base import BaseScorer
        return BaseScorer
    elif name == "MLPScorer":
        from .models.mlp_scorer import MLPScorer
        return MLPScorer
    elif name == "XGBoostScorer":
        from .models.xgboost_scorer import XGBoostScorer
        return XGBoostScorer
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
