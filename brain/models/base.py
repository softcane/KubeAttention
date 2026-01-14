"""
Base Scorer Interface for KubeAttention.

All scoring models (MLP, XGBoost, etc.) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ScoringResult:
    """Result from a scoring operation."""
    node_name: str
    score: int  # 0-100
    confidence: float  # 0-1
    reasoning: str


class BaseScorer(ABC):
    """
    Abstract base class for all node scoring models.
    
    Implementations:
    - MLPScorer: Lightweight 2-layer neural network
    - XGBoostScorer: Gradient boosted trees for ranking
    """
    
    @abstractmethod
    def score_nodes(
        self,
        node_features: np.ndarray,  # (N, F) - N nodes, F features
        pod_features: np.ndarray,   # (P,) - Pod context features
        node_names: List[str],
    ) -> List[ScoringResult]:
        """
        Score all candidate nodes for a pod placement decision.
        
        Args:
            node_features: Feature matrix for N candidate nodes
            pod_features: Feature vector for the pod being scheduled
            node_names: Names of the candidate nodes
            
        Returns:
            List of ScoringResult, one per node
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X: np.ndarray,      # (N_samples, F) - flattened features
        y: np.ndarray,      # (N_samples,) - labels/scores
        weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on labeled data.
        
        Args:
            X: Feature matrix
            y: Target labels (1.0 = good placement, 0.0 = bad placement)
            weights: Optional sample weights
            
        Returns:
            Training metrics dict
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for logging."""
        pass
    
    @property
    @abstractmethod
    def num_parameters(self) -> int:
        """Number of trainable parameters (0 for tree models)."""
        pass


def generate_reasoning(node_name: str, score: int, features: np.ndarray) -> str:
    """Generate human-readable reasoning for a score based on actual features."""
    # Feature order (from FEATURE_NAMES): cpu_util, cpu_throttle, mem_util, mem_bw, l3_miss, ...
    cpu_util = features[0] if len(features) > 0 else 0.5
    mem_util = features[2] if len(features) > 2 else 0.5
    l3_miss = features[4] if len(features) > 4 else 0.1
    
    reasons = []
    
    # CPU analysis
    if cpu_util > 0.8:
        reasons.append("high CPU pressure")
    elif cpu_util < 0.3:
        reasons.append("low CPU load")
    
    # Memory analysis
    if mem_util > 0.85:
        reasons.append("memory constrained")
    elif mem_util < 0.4:
        reasons.append("ample memory")
    
    # Cache contention
    if l3_miss > 0.5:
        reasons.append("cache contention detected")
    
    if not reasons:
        if score >= 70:
            reasons.append("good overall capacity")
        elif score >= 50:
            reasons.append("moderate load")
        else:
            reasons.append("elevated resource pressure")
    
    return f"Node {node_name}: {', '.join(reasons)} (CPU={cpu_util:.0%}, Mem={mem_util:.0%})"

