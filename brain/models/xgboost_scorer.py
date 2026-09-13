"""
XGBoost Scorer for KubeAttention.

Gradient boosted trees for node ranking.
<1ms inference, excellent for tabular data.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseScorer, ScoringResult, generate_reasoning
from . import register_model
from brain.metrics_schema import FEATURE_SCHEMA_VERSION, MODEL_INPUT_DIM, MODEL_INPUT_NAMES

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

    @property
    def ready(self) -> bool:
        return self._is_trained

    def predict_quality(self, features: np.ndarray) -> np.ndarray:
        if not self._is_trained or self.model is None:
            raise ValueError("model is not trained")
        if features.ndim != 2 or features.shape[1] != MODEL_INPUT_DIM:
            raise ValueError(f"features must have shape (N, {MODEL_INPUT_DIM})")
        return np.clip(
            np.asarray(self.model.predict(features), dtype=np.float64),
            0.0,
            1.0,
        )
    
    def score_nodes(
        self,
        node_features: np.ndarray,
        pod_features: np.ndarray,
        node_names: List[str],
    ) -> List[ScoringResult]:
        """Return absolute node-quality scores using the loaded model."""
        if not self._is_trained:
            return [
                ScoringResult(name, 50, f"Model not trained, using neutral score for {name}")
                for name in node_names
            ]
        node_count = node_features.shape[0]
        if pod_features is None or pod_features.ndim != 1:
            raise ValueError("pod_features must be a one-dimensional feature vector")
        pod_broadcast = np.broadcast_to(pod_features, (node_count, pod_features.shape[0]))
        combined = np.hstack([node_features, pod_broadcast])
        if combined.shape[1] != MODEL_INPUT_DIM:
            raise ValueError(
                f"model input has {combined.shape[1]} features, expected {MODEL_INPUT_DIM}"
            )
        scores = self.predict_quality(combined) * 100.0
        return [
            ScoringResult(
                node_name=name,
                score=int(scores[index]),
                reasoning=generate_reasoning(name, int(scores[index]), node_features[index]),
            )
            for index, name in enumerate(node_names)
        ]


    
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
        if not self._is_trained or self.model is None:
            raise ValueError("Model not trained yet")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_model(path)
        with open(f"{path}.metadata.json", "w") as metadata_file:
            json.dump({
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "model_input_names": MODEL_INPUT_NAMES,
            }, metadata_file)

    def load(self, path: str) -> None:
        with open(f"{path}.metadata.json") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema is incompatible")
        if tuple(metadata.get("model_input_names", ())) != MODEL_INPUT_NAMES:
            raise ValueError("checkpoint feature order is incompatible")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        self._is_trained = True
