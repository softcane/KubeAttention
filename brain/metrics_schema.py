"""
eBPF Tetragon Metrics Schema for KubeAttention

Defines the structure of metrics collected via eBPF/Tetragon probes.
These metrics form the input features for the Transformer model.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class MetricCategory(Enum):
    """Categories of eBPF metrics for feature grouping."""
    CPU = "cpu"
    MEMORY = "memory"
    CACHE = "cache"
    IO = "io"
    NETWORK = "network"


@dataclass
class TetragonMetricSpec:
    """Specification for a single Tetragon metric."""
    name: str
    category: MetricCategory
    unit: str
    min_value: float
    max_value: float
    description: str
    ebpf_probe: str  # The eBPF probe type (kprobe, tracepoint, etc.)


# Complete eBPF metrics schema matching scheduler.proto NodeTelemetry
TETRAGON_METRICS_SCHEMA: list[TetragonMetricSpec] = [
    # CPU Metrics
    TetragonMetricSpec(
        name="cpu_utilization",
        category=MetricCategory.CPU,
        unit="ratio",
        min_value=0.0,
        max_value=1.0,
        description="CPU utilization ratio across all cores",
        ebpf_probe="tracepoint:sched:sched_stat_runtime",
    ),
    TetragonMetricSpec(
        name="cpu_throttle_rate",
        category=MetricCategory.CPU,
        unit="events/sec",
        min_value=0.0,
        max_value=10000.0,
        description="CPU throttling events per second",
        ebpf_probe="tracepoint:cgroup:cgroup_throttle",
    ),
    
    # Memory Metrics
    TetragonMetricSpec(
        name="memory_utilization",
        category=MetricCategory.MEMORY,
        unit="ratio",
        min_value=0.0,
        max_value=1.0,
        description="Memory utilization ratio",
        ebpf_probe="kprobe:__alloc_pages",
    ),
    TetragonMetricSpec(
        name="memory_bandwidth_gbps",
        category=MetricCategory.MEMORY,
        unit="GB/s",
        min_value=0.0,
        max_value=200.0,
        description="Memory bandwidth utilization",
        ebpf_probe="perf:mem_load_retired.l3_miss",
    ),
    
    # L3 Cache Metrics (critical for noisy neighbor detection)
    TetragonMetricSpec(
        name="l3_cache_miss_rate",
        category=MetricCategory.CACHE,
        unit="ratio",
        min_value=0.0,
        max_value=1.0,
        description="L3 cache miss rate - key noisy neighbor indicator",
        ebpf_probe="perf:cache_misses",
    ),
    TetragonMetricSpec(
        name="l3_cache_occupancy_mb",
        category=MetricCategory.CACHE,
        unit="MB",
        min_value=0.0,
        max_value=256.0,
        description="L3 cache occupancy in megabytes",
        ebpf_probe="perf:llc_occupancy",
    ),
    
    # I/O Metrics
    TetragonMetricSpec(
        name="disk_io_wait_ms",
        category=MetricCategory.IO,
        unit="ms",
        min_value=0.0,
        max_value=1000.0,
        description="Average disk I/O wait time",
        ebpf_probe="tracepoint:block:block_rq_complete",
    ),
    TetragonMetricSpec(
        name="disk_iops",
        category=MetricCategory.IO,
        unit="ops/sec",
        min_value=0.0,
        max_value=1000000.0,
        description="Disk I/O operations per second",
        ebpf_probe="tracepoint:block:block_rq_issue",
    ),
    
    # Network Metrics
    TetragonMetricSpec(
        name="network_rx_packets_sec",
        category=MetricCategory.NETWORK,
        unit="packets/sec",
        min_value=0.0,
        max_value=10000000.0,
        description="Network receive packets per second",
        ebpf_probe="tracepoint:net:netif_receive_skb",
    ),
    TetragonMetricSpec(
        name="network_tx_packets_sec",
        category=MetricCategory.NETWORK,
        unit="packets/sec",
        min_value=0.0,
        max_value=10000000.0,
        description="Network transmit packets per second",
        ebpf_probe="tracepoint:net:net_dev_xmit",
    ),
    TetragonMetricSpec(
        name="network_drop_rate",
        category=MetricCategory.NETWORK,
        unit="ratio",
        min_value=0.0,
        max_value=1.0,
        description="Network packet drop rate",
        ebpf_probe="tracepoint:skb:kfree_skb",
    ),
    
    # Meta Metrics (Phase 2 & 4)
    TetragonMetricSpec(
        name="node_cost_index",
        category=MetricCategory.NETWORK,  # Reusing category or adding META
        unit="index",
        min_value=0.0,
        max_value=1.0,
        description="Relative cost index of the instance type",
        ebpf_probe="static:node_labels",
    ),
    TetragonMetricSpec(
        name="zone_diversity_score",
        category=MetricCategory.NETWORK,
        unit="index",
        min_value=0.0,
        max_value=1.0,
        description="Zone diversity score for resilience",
        ebpf_probe="static:topology",
    ),
    TetragonMetricSpec(
        name="spot_interruption_risk",
        category=MetricCategory.NETWORK,
        unit="index",
        min_value=0.0,
        max_value=1.0,
        description="Risk of spot interruption",
        ebpf_probe="static:node_labels",
    ),
    TetragonMetricSpec(
        name="is_spot_instance",
        category=MetricCategory.NETWORK,
        unit="bool",
        min_value=0.0,
        max_value=1.0,
        description="Boolean for Spot/Preemptible status",
        ebpf_probe="static:node_labels",
    ),
]


@dataclass
class NodeMetricsSnapshot:
    """A point-in-time snapshot of all metrics for a node."""
    node_name: str
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # CPU
    cpu_utilization: float = 0.0
    cpu_throttle_rate: float = 0.0
    
    # Memory
    memory_utilization: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    
    # Cache
    l3_cache_miss_rate: float = 0.0
    l3_cache_occupancy_mb: float = 0.0
    
    # I/O
    disk_io_wait_ms: float = 0.0
    disk_iops: float = 0.0
    
    # Network
    network_rx_packets_sec: float = 0.0
    network_tx_packets_sec: float = 0.0
    network_drop_rate: float = 0.0
    
    # Cost (Phase 2)
    node_cost_index: float = 0.0
    is_spot_instance: bool = False
    
    # Resilience (Phase 4)
    availability_zone: str = "unknown"
    zone_diversity_score: float = 0.5
    spot_interruption_risk: float = 0.0
    
    def to_feature_vector(self) -> list[float]:
        """Convert to normalized feature vector for model input."""
        features = []
        for spec in TETRAGON_METRICS_SCHEMA:
            if spec.name == "is_spot_instance":
                features.append(1.0 if self.is_spot_instance else 0.0)
                continue
                
            value = getattr(self, spec.name)
            if isinstance(value, str):
                # Strings (like zone) are handled via zone_diversity_score helper
                continue
                
            # Normalize to [0, 1] range
            normalized = (value - spec.min_value) / (spec.max_value - spec.min_value + 1e-8)
            features.append(max(0.0, min(1.0, normalized)))
        return features
    
    @classmethod
    def from_proto(cls, proto_telemetry) -> "NodeMetricsSnapshot":
        """Create from protobuf NodeTelemetry message."""
        return cls(
            node_name=proto_telemetry.node_name,
            timestamp_ms=proto_telemetry.timestamp_unix_ms,
            cpu_utilization=proto_telemetry.cpu_utilization,
            cpu_throttle_rate=proto_telemetry.cpu_throttle_rate,
            memory_utilization=proto_telemetry.memory_utilization,
            memory_bandwidth_gbps=proto_telemetry.memory_bandwidth_gbps,
            l3_cache_miss_rate=proto_telemetry.l3_cache_miss_rate,
            l3_cache_occupancy_mb=proto_telemetry.l3_cache_occupancy_mb,
            disk_io_wait_ms=proto_telemetry.disk_io_wait_ms,
            disk_iops=proto_telemetry.disk_iops,
            network_rx_packets_sec=proto_telemetry.network_rx_packets_sec,
            network_tx_packets_sec=proto_telemetry.network_tx_packets_sec,
            network_drop_rate=proto_telemetry.network_drop_rate,
            node_cost_index=getattr(proto_telemetry, "cost_per_hour", 0.0),
            is_spot_instance=getattr(proto_telemetry, "is_spot_instance", False),
            availability_zone=getattr(proto_telemetry, "availability_zone", "unknown"),
            spot_interruption_risk=getattr(proto_telemetry, "spot_interruption_risk", 0.0),
        )


# Feature dimension (number of metrics)
FEATURE_DIM = len(TETRAGON_METRICS_SCHEMA)

# Feature names for interpretability
FEATURE_NAMES = [spec.name for spec in TETRAGON_METRICS_SCHEMA]
