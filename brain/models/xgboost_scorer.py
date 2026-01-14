"""
XGBoost Scorer for KubeAttention.

Gradient boosted trees for node ranking.
<1ms inference, excellent for tabular data.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseScorer, ScoringResult, generate_reasoning
from . import register_model

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None


@register_model("xgboost")
class XGBoostScorer(BaseScorer):
    """
    XGBoost-based scorer using gradient boosted trees.
    
    Characteristics:
        - <1ms inference
        - No GPU required
        - Excellent for tabular data
        - ~100KB model file
        
    Note: Requires `pip install xgboost`
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: str = "reg:squarederror",
    ):
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Run: pip install xgboost"
            )
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        
        self.model: Optional[xgb.XGBRegressor] = None
        self._is_trained = False
        
    @property
    def name(self) -> str:
        return "XGBoost"
    
    @property
    def num_parameters(self) -> int:
        # Tree models don't have traditional parameters
        # Return number of trees * average nodes as approximation
        if self.model is None:
            return 0
        return self.n_estimators * (2 ** self.max_depth)
    
    def score_nodes(
        self,
        node_features: np.ndarray,
        pod_features: np.ndarray,
        node_names: List[str],
    ) -> List[ScoringResult]:
        """Score all candidate nodes."""
        if not self._is_trained:
            # Return neutral scores if not trained
            return [
                ScoringResult(
                    node_name=name,
                    score=50,
                    confidence=0.5,
                    reasoning=f"Model not trained, using neutral score for {name}",
                )
                for name in node_names
            ]
        
        # Concatenate node features with pod context (broadcast pod to all nodes)
        N = node_features.shape[0]
        if pod_features is not None and len(pod_features) > 0:
            if len(pod_features.shape) == 1:
                pod_broadcast = np.tile(pod_features, (N, 1))
            else:
                pod_broadcast = pod_features
            X_combined = np.hstack([node_features, pod_broadcast])
        else:
            X_combined = node_features
        
        # Predict raw scores (can be any range)
        raw_scores = self.model.predict(X_combined)
        
        # Min-max normalization to [0, 100]
        # This handles cases where XGBoost outputs negative or >1 values
        score_min = raw_scores.min()
        score_max = raw_scores.max()
        if score_max - score_min > 1e-8:
            scores = (raw_scores - score_min) / (score_max - score_min) * 100
        else:
            # All same score - return neutral
            scores = np.full_like(raw_scores, 50.0)
        
        # Confidence based on score spread (higher spread = more confident in ranking)
        score_std = np.std(scores)
        base_confidence = min(score_std / 30.0, 0.9)  # Confidence increases with score variance
        confidences = np.full_like(scores, base_confidence + 0.1)
        
        results = []
        for i, node_name in enumerate(node_names):
            score = int(np.clip(scores[i], 0, 100))
            conf = float(np.clip(confidences[i], 0.3, 0.95))
            results.append(ScoringResult(
                node_name=node_name,
                score=score,
                confidence=conf,
                reasoning=generate_reasoning(node_name, score, node_features[i]),
            ))
        
        return results


    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        eval_set: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """Train the XGBoost model on labeled data."""
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective=self.objective,
            random_state=42,
            n_jobs=-1,
        )
        
        fit_params = {}
        if weights is not None:
            fit_params["sample_weight"] = weights
        if eval_set is not None:
            fit_params["eval_set"] = [eval_set]
            fit_params["verbose"] = False
        
        self.model.fit(X, y, **fit_params)
        self._is_trained = True
        
        # Compute training metrics
        train_pred = self.model.predict(X)
        mse = float(np.mean((train_pred - y) ** 2))
        
        return {
            "train_mse": mse,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_samples": len(X),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_model(path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self._is_trained = True
