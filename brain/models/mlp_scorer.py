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

# Dynamic feature count from schema
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from brain.metrics_schema import FEATURE_NAMES

# Input dimension: node features + pod context (5 features: cpu_norm, mem_norm, workload_type, criticality, priority)
NODE_FEATURE_DIM = len(FEATURE_NAMES)
POD_CONTEXT_DIM = 11
DEFAULT_INPUT_DIM = NODE_FEATURE_DIM + POD_CONTEXT_DIM


class MLPNetwork(nn.Module):
    """Simple 2-layer MLP for scoring."""
    
    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.score_head = nn.Linear(hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor, scale_to_100: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (N, F) input features
            scale_to_100: If True, output scores in [0, 100]. If False, output in [0, 1].
            
        Returns:
            scores: (N,) node scores in [0, 100] or [0, 1]
            confidences: (N,) confidence in [0, 1]
        """
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        # Raw sigmoid output in [0, 1]
        raw_scores = torch.sigmoid(self.score_head(h))
        
        # Scale to [0, 100] only if requested (inference mode)
        if scale_to_100:
            scores = raw_scores * 100
        else:
            scores = raw_scores
            
        confidences = torch.sigmoid(self.confidence_head(h))  # [0, 1]
        
        return scores.squeeze(-1), confidences.squeeze(-1)



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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        
        self.model = MLPNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = None
        
    @property
    def name(self) -> str:
        return "MLP (2-layer)"
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
    
    def score_nodes(
        self,
        node_features: np.ndarray,
        pod_features: np.ndarray,
        node_names: List[str],
    ) -> List[ScoringResult]:
        """Score all candidate nodes."""
        self.model.eval()
        
        # Concatenate node features with pod context (broadcast pod to all nodes)
        N = node_features.shape[0]
        
        # Ensure pod_features is 2D: (N, P)
        if pod_features is not None and len(pod_features) > 0:
            if len(pod_features.shape) == 1:
                pod_broadcast = np.tile(pod_features, (N, 1))
            else:
                pod_broadcast = pod_features
            # Concatenate: [node_features | pod_features]
            X_combined = np.hstack([node_features, pod_broadcast])
        else:
            X_combined = node_features
        
        X = torch.tensor(X_combined, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            scores, confidences = self.model(X)
        
        results = []
        for i, node_name in enumerate(node_names):
            score = int(scores[i].item())
            conf = float(confidences[i].item())
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
                
                # Use scale_to_100=False to get raw [0,1] output for training
                scores, _ = self.model(batch_X, scale_to_100=False)
                
                # Loss: scores are [0,1], labels are [0,1] - direct comparison
                loss = (batch_w * (scores - batch_y) ** 2).mean()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(loader))
        
        return {
            "final_loss": losses[-1],
            "epochs": epochs,
            "num_samples": len(X),
        }

    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint.get("input_dim", self.input_dim)
        self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
        self.model = MLPNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
