"""
Cluster Tensor Encoder for KubeAttention

Transforms raw node telemetry and pod requirements into a structured
tensor representation suitable for Transformer input.

The ClusterTensor has shape: (num_nodes, seq_len, feature_dim)
where:
  - num_nodes: Number of candidate nodes
  - seq_len: Temporal window of metrics (for attention over time)
  - feature_dim: Number of eBPF metrics per timestep
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .metrics_schema import (
    NodeMetricsSnapshot,
    FEATURE_DIM,
    FEATURE_NAMES,
    TETRAGON_METRICS_SCHEMA,
)


@dataclass
class PodContext:
    """Pod requirements context for scoring."""
    pod_name: str
    pod_namespace: str
    cpu_milli: int
    memory_bytes: int
    workload_type: str = "unknown"
    labels: dict[str, str] = None
    
    # Feature names for dynamic dimension calculation
    FEATURE_NAMES: list[str] = None  # Set after class definition
    
    # Workload type categories (order matters for one-hot encoding)
    WORKLOAD_TYPES: list[str] = None  # Set after class definition
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
    
    def to_feature_vector(self) -> list[float]:
        """Convert pod requirements to normalized features."""
        from .config import NORMALIZATION
        
        # Normalize CPU
        cpu_norm = min(1.0, self.cpu_milli / NORMALIZATION.MAX_CPU_MILLI)
        
        # Normalize memory
        mem_norm = min(1.0, self.memory_bytes / NORMALIZATION.MAX_MEMORY_BYTES)
        
        # Workload type one-hot encoding
        workload_onehot = [1.0 if self.workload_type == wt else 0.0 for wt in PodContext.WORKLOAD_TYPES]
        
        return [cpu_norm, mem_norm] + workload_onehot


# Class-level constants (set after class definition to avoid dataclass issues)
PodContext.WORKLOAD_TYPES = ["cpu-bound", "memory-bound", "io-bound", "balanced", "unknown"]
PodContext.FEATURE_NAMES = ["cpu_normalized", "memory_normalized"] + [f"workload_{wt}" for wt in PodContext.WORKLOAD_TYPES]

# Derived constant for use in model.py
POD_CONTEXT_DIM = len(PodContext.FEATURE_NAMES)


@dataclass  
class ClusterTensor:
    """
    Structured tensor representation of cluster state for Transformer input.
    
    Attributes:
        node_features: (N, T, F) - Node telemetry over time
        node_names: List of node names (length N)
        pod_context: (P,) - Pod requirements vector
        attention_mask: (N,) - Mask for valid nodes (1=valid, 0=masked)
        timestamps: (T,) - Timestamps for temporal features
    """
    node_features: torch.Tensor      # (N, T, F)
    node_names: list[str]            # Length N
    pod_context: torch.Tensor        # (P,)
    attention_mask: torch.Tensor     # (N,)
    timestamps: torch.Tensor         # (T,)
    
    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]
    
    @property
    def seq_len(self) -> int:
        return self.node_features.shape[1]
    
    @property
    def feature_dim(self) -> int:
        return self.node_features.shape[2]
    
    def to(self, device: torch.device) -> "ClusterTensor":
        """Move all tensors to device."""
        return ClusterTensor(
            node_features=self.node_features.to(device),
            node_names=self.node_names,
            pod_context=self.pod_context.to(device),
            attention_mask=self.attention_mask.to(device),
            timestamps=self.timestamps.to(device),
        )


class ClusterTensorEncoder(nn.Module):
    """
    Encodes raw telemetry data into ClusterTensor format.
    
    This encoder handles:
    1. Normalization of metrics to [0, 1] range
    2. Temporal stacking for sequence modeling
    3. Pod context embedding
    4. Missing value imputation
    """
    
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        max_nodes: int = 100,
        max_seq_len: int = 10,
        pod_context_dim: int = 7,  # cpu, mem, 5 workload types
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_nodes = max_nodes
        self.max_seq_len = max_seq_len
        self.pod_context_dim = pod_context_dim
        
        # Learnable position encoding for temporal dimension
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, feature_dim) * 0.02
        )
        
        # Node position encoding (nodes don't have inherent order)
        self.node_pos_encoding = nn.Parameter(
            torch.randn(1, max_nodes, 1) * 0.02
        )
        
        # Running statistics for online normalization
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))
        self.register_buffer("num_samples", torch.tensor(0))
    
    def encode_node_snapshots(
        self,
        snapshots: list[list[NodeMetricsSnapshot]],  # (N, T) list of snapshots per node
        pod: PodContext,
        device: torch.device = torch.device("cpu"),
    ) -> ClusterTensor:
        """
        Encode node metric snapshots into a ClusterTensor.
        
        Args:
            snapshots: List of T snapshots per node (N nodes total)
            pod: Pod requirements context
            device: Target device for tensors
            
        Returns:
            ClusterTensor ready for model input
        """
        num_nodes = len(snapshots)
        seq_len = len(snapshots[0]) if snapshots else 1
        
        # Build node features tensor
        node_features = torch.zeros(num_nodes, seq_len, self.feature_dim)
        node_names = []
        timestamps = torch.zeros(seq_len)
        
        for i, node_snapshots in enumerate(snapshots):
            node_names.append(node_snapshots[0].node_name if node_snapshots else f"node-{i}")
            for t, snapshot in enumerate(node_snapshots):
                features = snapshot.to_feature_vector()
                node_features[i, t, :] = torch.tensor(features)
                if i == 0:
                    timestamps[t] = snapshot.timestamp_ms
        
        # Add temporal position encoding
        if seq_len <= self.max_seq_len:
            node_features = node_features + self.temporal_pos_encoding[:, :seq_len, :]
        
        # Build pod context vector
        pod_context = torch.tensor(pod.to_feature_vector(), dtype=torch.float32)
        
        # Attention mask (all valid for now)
        attention_mask = torch.ones(num_nodes)
        
        return ClusterTensor(
            node_features=node_features.to(device),
            node_names=node_names,
            pod_context=pod_context.to(device),
            attention_mask=attention_mask.to(device),
            timestamps=timestamps.to(device),
        )
    
    def encode_from_proto(
        self,
        nodes_telemetry: list,  # List of NodeTelemetry protos
        pod_requirements,       # PodRequirements proto
        device: torch.device = torch.device("cpu"),
    ) -> ClusterTensor:
        """
        Encode directly from protobuf messages.
        
        Args:
            nodes_telemetry: List of NodeTelemetry proto messages
            pod_requirements: PodRequirements proto message
            device: Target device
            
        Returns:
            ClusterTensor ready for model input
        """
        # Convert protos to internal format
        snapshots = [
            [NodeMetricsSnapshot.from_proto(node)]
            for node in nodes_telemetry
        ]
        
        pod = PodContext(
            pod_name=pod_requirements.pod_name,
            pod_namespace=pod_requirements.pod_namespace,
            cpu_milli=pod_requirements.cpu_milli,
            memory_bytes=pod_requirements.memory_bytes,
            workload_type=pod_requirements.workload_type or "unknown",
            labels=dict(pod_requirements.labels) if pod_requirements.labels else {},
        )
        
        return self.encode_node_snapshots(snapshots, pod, device)
    
    def forward(self, cluster_tensor: ClusterTensor) -> torch.Tensor:
        """
        Forward pass applies final normalization.
        
        Returns node_features ready for Transformer: (N, T, F)
        """
        return cluster_tensor.node_features
