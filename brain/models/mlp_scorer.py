"""
2-Layer MLP Scorer for KubeAttention.

A lightweight neural network for node scoring.
~3,000 parameters, ~1ms inference.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseScorer, ScoringResult, generate_reasoning
from . import register_model

from brain.metrics_schema import (
    FEATURE_SCHEMA_VERSION,
    MODEL_INPUT_DIM,
    MODEL_INPUT_NAMES,
)

DEFAULT_INPUT_DIM = MODEL_INPUT_DIM


class MLPNetwork(nn.Module):
    """Two-layer regressor for absolute node quality."""

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.score_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, scale_to_100: bool = True) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        scores = torch.sigmoid(self.score_head(h))
        if scale_to_100:
            scores = scores * 100
        return scores.squeeze(-1)



@register_model("mlp")
class MLPScorer(BaseScorer):
    """
    Lightweight 2-layer MLP for node scoring.
    
    Architecture:
        Input(F) → Dense(64, ReLU) → Dense(32, ReLU) → Score(1)
        
    Characteristics:
        - ~3,000 parameters
        - ~1ms inference
        - ~15KB model file
    """
    
    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        if input_dim != MODEL_INPUT_DIM:
            raise ValueError(f"input_dim {input_dim} does not match schema dimension {MODEL_INPUT_DIM}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.model = MLPNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = None
        self._is_trained = False
        
    @property
    def name(self) -> str:
        return "MLP (2-layer)"
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def ready(self) -> bool:
        return self._is_trained

    def predict_quality(self, features: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("model is not trained")
        if features.ndim != 2 or features.shape[1] != MODEL_INPUT_DIM:
            raise ValueError(f"features must have shape (N, {MODEL_INPUT_DIM})")
        self.model.eval()
        inputs = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(inputs, scale_to_100=False).cpu().numpy()
    
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
        self.model.eval()
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
                score=int(np.clip(scores[index], 0, 100)),
                reasoning=generate_reasoning(name, int(scores[index]), node_features[index]),
            )
            for index, name in enumerate(node_names)
        ]

    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Train the MLP on labeled data."""
        self.model.train()
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        if weights is not None:
            w_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            w_tensor = torch.ones_like(y_tensor)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y, batch_w in loader:
                self.optimizer.zero_grad()
                
                scores = self.model(batch_X, scale_to_100=False)
                
                # Loss: scores are [0,1], labels are [0,1] - direct comparison
                loss = (batch_w * (scores - batch_y) ** 2).mean()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(loader))

        self._is_trained = True
        
        return {
            "final_loss": losses[-1],
            "epochs": epochs,
            "num_samples": len(X),
        }

    
    def save(self, path: str) -> None:
        if not self._is_trained:
            raise ValueError("Model not trained yet")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_input_names": MODEL_INPUT_NAMES,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
            raise ValueError("checkpoint feature schema is missing or incompatible")
        if tuple(checkpoint.get("model_input_names", ())) != MODEL_INPUT_NAMES:
            raise ValueError("checkpoint feature order is incompatible")
        if checkpoint.get("input_dim") != MODEL_INPUT_DIM:
            raise ValueError("checkpoint input dimension is incompatible")
        self.input_dim = MODEL_INPUT_DIM
        self.hidden_dim = checkpoint["hidden_dim"]
        self.model = MLPNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._is_trained = True
