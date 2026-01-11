"""
KubeAttention Brain - Transformer-based Node Scoring

This package implements the AI "Brain" for KubeAttention scheduler:
- metrics_schema: eBPF Tetragon metrics definitions
- tensor_encoder: ClusterTensor encoding from raw telemetry
- model: PyTorch Transformer for node scoring
- server: gRPC server over Unix Domain Socket
"""

__version__ = "0.1.0"
__all__ = ["ClusterTensorEncoder", "AttentionScorer", "BrainServer"]
