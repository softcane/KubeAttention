"""
PyTorch Dataset for KubeAttention training.

Loads scheduling events from Parquet files and converts them to
tensors suitable for training the AttentionScorer model.
"""

import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Import feature definitions from single source of truth
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from brain.metrics_schema import FEATURE_NAMES, TETRAGON_METRICS_SCHEMA
from brain.tensor_encoder import PodContext


class SchedulingDataset(Dataset):
    """
    Dataset of scheduling decisions with outcomes.
    
    Each sample contains:
    - node_features: (N, T, F) telemetry for all candidate nodes
    - pod_context: (P,) pod requirements
    - label: index of the best node (based on outcome)
    - weight: importance weight based on outcome severity
    """
    
    # Feature names from metrics_schema.py (single source of truth)
    # DO NOT redefine here - import from metrics_schema to avoid mismatch!
    
    # Outcome to label mapping
    OUTCOME_LABELS = {
        "running": 1.0,      # Success
        "restarted": 0.5,    # Partial success
        "terminated": 0.3,   # Failure
        "oom_killed": 0.0,   # Critical failure
        "evicted": 0.0,      # Critical failure (noisy neighbor)
        "failed": 0.0,       # Failure
        "deleted": 0.5,      # Unknown (treat as neutral)
        "unknown": 0.5,      # Unknown
    }
    
    # Outcome weights for loss function
    OUTCOME_WEIGHTS = {
        "running": 1.0,
        "restarted": 1.5,
        "terminated": 2.0,
        "oom_killed": 3.0,   # Penalize heavily
        "evicted": 3.0,      # Penalize heavily
        "failed": 2.0,
        "deleted": 0.5,
        "unknown": 0.5,
    }
    
    def __init__(
        self,
        data_path: str,
        max_nodes: int = 100,
        temporal_window: int = 1,  # Number of time steps
    ):
        self.data_path = Path(data_path)
        self.max_nodes = max_nodes
        self.temporal_window = temporal_window
        
        # Load data
        self.events = self._load_data()
        
        print(f"📊 Loaded {len(self.events)} scheduling events from {data_path}")
    
    def _load_data(self) -> list:
        """Load events from JSONL or Parquet file."""
        events = []
        
        if self.data_path.suffix == ".parquet" and HAS_PARQUET:
            table = pq.read_table(self.data_path)
            df = table.to_pandas()
            for _, row in df.iterrows():
                events.append(row.to_dict())
        elif self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Filter events with valid outcomes
        events = [e for e in events if e.get("outcome") != "pending"]
        
        return events
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> dict:
        event = self.events[idx]
        
        # Extract node telemetry
        node_telemetry = event.get("node_telemetry", {})
        node_names = list(node_telemetry.keys())[:self.max_nodes]
        num_nodes = len(node_names)
        
        # Build node features tensor: (N, T, F)
        node_features = np.zeros((self.max_nodes, self.temporal_window, len(FEATURE_NAMES)))
        
        for i, node_name in enumerate(node_names):
            metrics = node_telemetry[node_name]
            for j, feature_name in enumerate(FEATURE_NAMES):
                value = metrics.get(feature_name, 0.0)
                # Normalize to [0, 1] range
                node_features[i, 0, j] = self._normalize_feature(feature_name, value)
        
        # Build pod context using the shared PodContext class (Issue 10 fix: unify encoding)
        pod = PodContext(
            pod_name=event.get("pod_name", ""),
            pod_namespace=event.get("pod_namespace", ""),
            cpu_milli=event.get("cpu_request_milli", 0),
            memory_bytes=event.get("memory_request_bytes", 0),
            workload_type=event.get("workload_type", "unknown"),
            labels=event.get("pod_labels", {})
        )
        pod_context = np.array(pod.to_feature_vector(), dtype=np.float32)
        
        # Create label: which node was chosen
        chosen_node = event.get("chosen_node", "")
        chosen_idx = node_names.index(chosen_node) if chosen_node in node_names else 0
        
        # Create binary labels for all nodes based on outcome
        outcome = event.get("outcome", "unknown")
        outcome_score = self.OUTCOME_LABELS.get(outcome, 0.5)
        
        # Target: the chosen node gets the outcome score
        labels = np.zeros(self.max_nodes, dtype=np.float32)
        if chosen_idx < self.max_nodes:
            labels[chosen_idx] = outcome_score
        
        # Attention mask: 1 for real nodes, 0 for padding
        attention_mask = np.zeros(self.max_nodes, dtype=np.float32)
        attention_mask[:num_nodes] = 1.0
        
        # Sample weight based on outcome
        weight = self.OUTCOME_WEIGHTS.get(outcome, 1.0)
        
        return {
            "node_features": torch.tensor(node_features, dtype=torch.float32),
            "pod_context": torch.tensor(pod_context, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "chosen_idx": torch.tensor(chosen_idx, dtype=torch.long),
            "num_nodes": torch.tensor(num_nodes, dtype=torch.long),
        }
    
    def _normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize a feature to [0, 1] range using TETRAGON_METRICS_SCHEMA."""
        # Use single source of truth for normalization ranges
        for spec in TETRAGON_METRICS_SCHEMA:
            if spec.name == feature_name:
                normalized = (value - spec.min_value) / (spec.max_value - spec.min_value + 1e-8)
                return max(0.0, min(1.0, normalized))
        # Fallback for unknown features
        return max(0.0, min(1.0, value))


def create_dataloader(
    data_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for training."""
    dataset = SchedulingDataset(data_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def generate_synthetic_data(output_path: str, num_samples: int = 10000):
    """
    Generate synthetic bootstrapping data for cold-start or CI/CD.
    
    WARNING: For production use, always train on real data collected from 
    your cluster via the Collector and Shadow Mode. Synthetic data is 
    a simplified model used for verification and bootstrapping only.
    """
    import random
    
    events = []
    
    node_names = [f"node-{i}" for i in range(10)]
    
    for i in range(num_samples):
        # Random pod requirements
        cpu_milli = random.randint(100, 4000)
        memory_bytes = random.randint(128 * 1024**2, 16 * 1024**3)
        
        # Generate telemetry for each node
        node_telemetry = {}
        best_node = None
        best_score = -1
        
        for node_name in node_names:
            # Random utilization
            cpu_util = random.uniform(0.1, 0.9)
            mem_util = random.uniform(0.1, 0.9)
            cache_miss = random.uniform(0.0, 0.5)
            
            # Simple scoring: lower utilization = better
            score = (1 - cpu_util) * 0.4 + (1 - mem_util) * 0.4 + (1 - cache_miss) * 0.2
            
            if score > best_score:
                best_score = score
                best_node = node_name
            
            node_telemetry[node_name] = {
                "node_name": node_name,
                "timestamp": "2026-01-11T00:00:00Z",
                "cpu_utilization": cpu_util,
                "memory_utilization": mem_util,
                "l3_cache_miss_rate": cache_miss,
                "l3_cache_occupancy_mb": random.uniform(0, 64),
                "memory_bandwidth_gbps": random.uniform(0, 50),
                "disk_io_wait_ms": random.uniform(0, 100),
                "disk_iops": random.uniform(0, 10000),
                "network_rx_packets_sec": random.uniform(0, 100000),
                "network_tx_packets_sec": random.uniform(0, 100000),
                "network_drop_rate": random.uniform(0, 0.01),
                "cpu_throttle_rate": random.uniform(0, 100),
            }
        
        # Simulate outcome based on placement quality
        # If placed on best node: high success rate
        # If placed on bad node: high failure rate
        chosen_node = random.choice(node_names)
        was_optimal = (chosen_node == best_node)
        
        if was_optimal:
            outcome = random.choices(
                ["running", "restarted", "oom_killed"],
                weights=[0.95, 0.04, 0.01]
            )[0]
        else:
            outcome = random.choices(
                ["running", "restarted", "oom_killed", "evicted"],
                weights=[0.1, 0.3, 0.3, 0.3]
            )[0]
        
        events.append({
            "timestamp": "2026-01-11T00:00:00Z",
            "event_id": f"synthetic-{i}",
            "pod_uid": f"pod-{i}",
            "pod_name": f"test-pod-{i}",
            "pod_namespace": "default",
            "pod_labels": {"app": "synthetic"},
            "cpu_request_milli": cpu_milli,
            "memory_request_bytes": memory_bytes,
            "candidate_nodes": node_names,
            "node_telemetry": node_telemetry,
            "chosen_node": chosen_node,
            "scheduler_name": "default-scheduler",
            "outcome": outcome,
            "outcome_timestamp": "2026-01-11T00:05:00Z",
            "p99_latency_ms": random.uniform(1, 100),
        })
    
    # Write to JSONL
    with open(output_path, 'w') as f:
        for event in events:
            f.write(json.dumps(event) + '\n')
    
    print(f"✅ Generated {num_samples} synthetic training samples to {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate synthetic data for testing
    generate_synthetic_data("training_data.jsonl", 10000)
    
    # Test dataset loading
    dataset = SchedulingDataset("training_data.jsonl")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Node features shape:", sample["node_features"].shape)
    print("Pod context shape:", sample["pod_context"].shape)
