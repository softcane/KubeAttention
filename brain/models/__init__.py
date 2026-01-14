"""
Model Registry for KubeAttention.

Provides a unified interface to switch between different scorer implementations.
"""

from typing import Optional
from .base import BaseScorer


# Registry of available models
_MODEL_REGISTRY = {}


def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str = "mlp", **kwargs) -> BaseScorer:
    """
    Get a scorer model by name.
    
    Args:
        name: Model name ("mlp" or "xgboost")
        **kwargs: Model-specific configuration
        
    Returns:
        Initialized scorer model
        
    Example:
        >>> scorer = get_model("mlp")
        >>> scorer = get_model("xgboost", n_estimators=200)
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    
    return _MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List all available models."""
    return list(_MODEL_REGISTRY.keys())


# Import models to register them
from .mlp_scorer import MLPScorer
from .xgboost_scorer import XGBoostScorer
