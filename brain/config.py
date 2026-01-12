"""
KubeAttention Configuration Constants

All tunable parameters in one place for easy auditing and modification.
This prevents magic numbers from being scattered throughout the codebase.
"""

from dataclasses import dataclass


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


# Singleton instances for easy access
INFERENCE = InferenceConfig()
TELEMETRY = TelemetryConfig()
SCORE = ScoreConfig()
NORMALIZATION = NormalizationConfig()
MODEL = ModelConfig()
