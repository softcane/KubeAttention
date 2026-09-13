"""Versioned node and pod feature contract shared by training and serving."""

from dataclasses import dataclass, field
import math
from typing import Mapping


FEATURE_SCHEMA_VERSION = "node-pod-v1"
WORKLOAD_TYPES = ("cpu-bound", "memory-bound", "io-bound", "balanced", "unknown")
CRITICALITY_LEVELS = ("unknown", "low", "medium", "high")
POD_FEATURE_NAMES = (
    "cpu_normalized",
    "memory_normalized",
    "priority_normalized",
    *(f"workload_{name}" for name in WORKLOAD_TYPES),
    *(f"criticality_{name}" for name in CRITICALITY_LEVELS),
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    minimum: float
    maximum: float
    proto_metric: int | None = None
    required_for_active_scoring: bool = False

    def normalize(self, value: float) -> float:
        return max(0.0, min(1.0, (value - self.minimum) / (self.maximum - self.minimum)))


NODE_FEATURE_SCHEMA = (
    FeatureSpec("cpu_utilization", 0.0, 1.0, 1, True),
    FeatureSpec("cpu_throttle_rate", 0.0, 10_000.0, 2),
    FeatureSpec("memory_utilization", 0.0, 1.0, 3, True),
    FeatureSpec("memory_bandwidth_gbps", 0.0, 200.0, 4),
    FeatureSpec("l3_cache_miss_rate", 0.0, 1.0, 5),
    FeatureSpec("l3_cache_occupancy_mb", 0.0, 256.0, 6),
    FeatureSpec("disk_io_wait_ms", 0.0, 1_000.0, 7),
    FeatureSpec("disk_iops", 0.0, 1_000_000.0, 8),
    FeatureSpec("network_rx_packets_sec", 0.0, 10_000_000.0, 9),
    FeatureSpec("network_tx_packets_sec", 0.0, 10_000_000.0, 10),
    FeatureSpec("network_drop_rate", 0.0, 1.0, 11),
    FeatureSpec("cost_per_hour", 0.0, 5.0),
    FeatureSpec("zone_diversity_score", 0.0, 1.0),
    FeatureSpec("spot_interruption_risk", 0.0, 1.0),
    FeatureSpec("is_spot_instance", 0.0, 1.0),
)
FEATURE_NAMES = tuple(spec.name for spec in NODE_FEATURE_SCHEMA)
FEATURE_DIM = len(FEATURE_NAMES)
MODEL_INPUT_NAMES = FEATURE_NAMES + POD_FEATURE_NAMES
MODEL_INPUT_DIM = len(MODEL_INPUT_NAMES)
REQUIRED_PROTO_METRICS = frozenset(
    spec.proto_metric for spec in NODE_FEATURE_SCHEMA if spec.required_for_active_scoring
)
REQUIRED_FEATURE_NAMES = frozenset(
    spec.name for spec in NODE_FEATURE_SCHEMA if spec.required_for_active_scoring
)


def normalize_node_record(record: Mapping[str, object]) -> list[float]:
    """Encode one training record with the serving normalization contract."""
    values = []
    for spec in NODE_FEATURE_SCHEMA:
        raw = record.get(spec.name, 0.0)
        if spec.name == "is_spot_instance":
            raw = 1.0 if bool(raw) else 0.0
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{spec.name} must be finite")
        values.append(spec.normalize(value))
    return values


@dataclass
class NodeMetricsSnapshot:
    """One node sample decoded according to FEATURE_SCHEMA_VERSION."""

    node_name: str
    timestamp_ms: int = 0
    available_metrics: frozenset[int] = field(default_factory=frozenset)
    telemetry_source: str = ""
    degradation_reason: str = ""
    observation_window_ms: int = 0
    schema_version: str = FEATURE_SCHEMA_VERSION

    cpu_utilization: float = 0.0
    cpu_throttle_rate: float = 0.0
    memory_utilization: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    l3_cache_miss_rate: float = 0.0
    l3_cache_occupancy_mb: float = 0.0
    disk_io_wait_ms: float = 0.0
    disk_iops: float = 0.0
    network_rx_packets_sec: float = 0.0
    network_tx_packets_sec: float = 0.0
    network_drop_rate: float = 0.0
    cost_per_hour: float = 0.0
    is_spot_instance: bool = False
    availability_zone: str = ""
    zone_diversity_score: float = 0.5
    spot_interruption_risk: float = 0.0

    def to_feature_vector(self) -> list[float]:
        return normalize_node_record(self.__dict__)

    def missing_required_metrics(self) -> frozenset[int]:
        return REQUIRED_PROTO_METRICS - self.available_metrics

    @classmethod
    def from_proto(cls, telemetry) -> "NodeMetricsSnapshot":
        return cls(
            node_name=telemetry.node_name,
            timestamp_ms=telemetry.timestamp_unix_ms,
            available_metrics=frozenset(telemetry.available_metrics),
            telemetry_source=telemetry.telemetry_source,
            degradation_reason=telemetry.degradation_reason,
            observation_window_ms=telemetry.observation_window_ms,
            schema_version=telemetry.schema_version,
            cpu_utilization=telemetry.cpu_utilization,
            cpu_throttle_rate=telemetry.cpu_throttle_rate,
            memory_utilization=telemetry.memory_utilization,
            memory_bandwidth_gbps=telemetry.memory_bandwidth_gbps,
            l3_cache_miss_rate=telemetry.l3_cache_miss_rate,
            l3_cache_occupancy_mb=telemetry.l3_cache_occupancy_mb,
            disk_io_wait_ms=telemetry.disk_io_wait_ms,
            disk_iops=telemetry.disk_iops,
            network_rx_packets_sec=telemetry.network_rx_packets_sec,
            network_tx_packets_sec=telemetry.network_tx_packets_sec,
            network_drop_rate=telemetry.network_drop_rate,
            cost_per_hour=telemetry.cost_per_hour,
            is_spot_instance=telemetry.is_spot_instance,
            availability_zone=telemetry.availability_zone,
            spot_interruption_risk=telemetry.spot_interruption_risk,
        )
