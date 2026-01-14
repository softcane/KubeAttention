"""
KubeAttention Configuration Constants

All tunable parameters in one place for easy auditing and modification.
This prevents magic numbers from being scattered throughout the codebase.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InferenceConfig:
    """Inference latency and safety constraints."""
    MAX_LATENCY_MS: int = 45
    FALLBACK_SCORE: int = 50
    FALLBACK_CONFIDENCE: float = 0.1


@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry staleness and data quality thresholds."""
    MAX_STALENESS_MS: int = 10_000  # 10 seconds


@dataclass(frozen=True)
class ScoreConfig:
    """Scoring output constraints."""
    MIN_SCORE: int = 0
    MAX_SCORE: int = 100
    MIN_CONFIDENCE: float = 0.0
    MAX_CONFIDENCE: float = 1.0


@dataclass(frozen=True)
class NormalizationConfig:
    """Feature normalization bounds for pod context."""
    MAX_CPU_MILLI: int = 8_000       # 8 cores
    MAX_MEMORY_BYTES: int = 32 * 1024**3  # 32GB


@dataclass(frozen=True)
class ModelConfig:
    """Transformer model hyperparameters."""
    D_MODEL: int = 128
    N_LAYERS: int = 3
    N_HEADS: int = 4
    DROPOUT: float = 0.1


@dataclass(frozen=True)
class CostConfig:
    """Cost-aware scoring parameters (Phase 2).
    
    NOTE: These are DEFAULT initialization values. The actual weights are
    LEARNABLE via nn.Parameter during training. These defaults serve as
    initialization and fallback.
    """
    # Default initialization for learnable weights
    WEIGHT_ALPHA_INIT: float = 0.3  # Reliability weight
    WEIGHT_BETA_INIT: float = 0.3   # Performance weight
    WEIGHT_GAMMA_INIT: float = 0.2  # Resilience weight
    WEIGHT_LAMBDA_INIT: float = 0.2 # Cost penalty weight
    
    # Cost normalization bounds
    MIN_COST_PER_HOUR: float = 0.01  # ~$0.01/hr (t3.nano)
    MAX_COST_PER_HOUR: float = 5.00  # ~$5/hr (large instances)

    # Relative cost index (normalized roughly [0, 1] for common AWS/GCP types)
    INSTANCE_COST_MAP: dict[str, float] = field(default_factory=lambda: {
        "t3.nano": 0.01,
        "t3.small": 0.05,
        "t3.medium": 0.1,
        "m5.large": 0.2,
        "m5.xlarge": 0.4,
        "c5.2xlarge": 0.6,
        "r5.4xlarge": 0.8,
        "p3.2xlarge": 1.0,
        "unknown": 0.1,
    })


@dataclass(frozen=True)
class ResilienceConfig:
    """Resilience scoring parameters (Phase 4).
    
    NOTE: These are DEFAULT initialization values for learnable weights.
    """
    # Default initialization for resilience sub-weights
    ZONE_WEIGHT_INIT: float = 0.5   # Zone diversity importance
    SPOT_WEIGHT_INIT: float = 0.5   # Spot interruption risk importance
    
    # Static risk profiles
    SPOT_INTERRUPTION_RISK: float = 0.2  # 20% baseline risk index


@dataclass(frozen=True)
class RebalancerConfig:
    """Proactive rebalancing parameters (Phase 4)."""
    ENABLED: bool = True
    INTERVAL_SECONDS: int = 60  # Scan the cluster every minute
    SCORE_DELTA_THRESHOLD: int = 20  # Rebalance if target node is 20 points better
    MIN_SOURCE_SCORE: int = 40  # Only rebalance if source node is performing poorly
    MAX_PODS_PER_RUN: int = 5   # Throttling to avoid cluster churn
    
    # Namespaces to skip
    EXCLUDE_NAMESPACES: list[str] = field(default_factory=lambda: [
        "kube-system",
        "kube-public",
        "kube-node-lease",
        "kubeattention-system",
    ])


@dataclass(frozen=True)
class ClusterConfig:
    """Cluster topology configuration (centralized source of truth)."""
    # Availability zones - extend this list for your cloud provider
    AVAILABILITY_ZONES: list[str] = field(default_factory=lambda: [
        "us-east-1a",
        "us-east-1b",
        "us-east-1c",
        "us-east-1d",
        "us-east-1e",
        "us-east-1f",
    ])
    
    # Instance types for cost mapping (extend as needed)
    INSTANCE_TYPES: list[str] = field(default_factory=lambda: [
        "t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large",
        "m5.large", "m5.xlarge", "m5.2xlarge", "m5.4xlarge",
        "c5.large", "c5.xlarge", "c5.2xlarge",
        "r5.large", "r5.xlarge",
        "p3.2xlarge", "p3.8xlarge",
    ])


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and training data configuration."""
    # Dynamic sizing - no artificial cap on nodes
    MAX_NODES: int = 1000  # Reasonable upper bound for padding, not a hard cap
    
    # Temporal window for time series features
    TEMPORAL_WINDOW: int = 10  # Number of historical timesteps
    
    # Synthetic data generation defaults
    DEFAULT_NUM_NODES: int = 10
    DEFAULT_NUM_SAMPLES: int = 10000


@dataclass(frozen=True)
class ModelSelectionConfig:
    """Model selection configuration."""
    # Model type: "mlp" or "xgboost"
    # MLP: Lower latency (0.05ms), smaller size (16KB)
    # XGBoost: Faster training (0.1s), same accuracy
    MODEL_TYPE: str = "mlp"  # Default to MLP for lowest latency


# Singleton instances for easy access
INFERENCE = InferenceConfig()
TELEMETRY = TelemetryConfig()
SCORE = ScoreConfig()
NORMALIZATION = NormalizationConfig()
MODEL = ModelConfig()
COST = CostConfig()
RESILIENCE = ResilienceConfig()
REBALANCER = RebalancerConfig()
CLUSTER = ClusterConfig()
DATASET = DatasetConfig()
MODEL_SELECTION = ModelSelectionConfig()
