"""
Transformer Model for KubeAttention Node Scoring

A compact Transformer that processes ClusterTensor input to produce
node suitability scores. Designed with:
- Multi-head self-attention over nodes
- Cross-attention with pod requirements
- Int8 quantization support for 2GB memory limit
- <50ms inference latency target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .config import SCORE, INFERENCE, MODEL
from .metrics_schema import FEATURE_DIM
from .tensor_encoder import ClusterTensor, POD_CONTEXT_DIM


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1), :]


class NodeAttentionBlock(nn.Module):
    """
    Self-attention block for inter-node relationships.
    
    Allows nodes to "attend" to each other's resource states,
    enabling the model to detect noisy neighbor patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self, 
        x: torch.Tensor,  # (B, N, D)
        mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> torch.Tensor:
        B, N, D = x.shape
        
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use PyTorch 2.0+ scaled_dot_product_attention (FlashAttention/SDPA)
        # This is 2-4x faster than manual attention and memory-efficient
        attn_mask = None
        if mask is not None:
            # Convert (B, N) mask to (B, 1, 1, N) for broadcasting
            attn_mask = mask.unsqueeze(1).unsqueeze(2).bool()
            # Invert: True means attend, False means mask out
            attn_mask = ~attn_mask
        
        # FlashAttention-compatible path (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # Node attention is not causal
        )
        
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        x = residual + self.dropout(out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


class PodCrossAttention(nn.Module):
    """
    Cross-attention from nodes to pod requirements.
    
    Allows each node representation to attend to the incoming
    pod's resource requirements for context-aware scoring.
    """
    
    def __init__(self, d_model: int, pod_dim: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(pod_dim, d_model)
        self.v_proj = nn.Linear(pod_dim, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        nodes: torch.Tensor,     # (B, N, D)
        pod_ctx: torch.Tensor,   # (B, P)
    ) -> torch.Tensor:
        B, N, D = nodes.shape
        
        residual = nodes
        nodes = self.norm(nodes)
        
        # Pod context as single "token"
        pod_ctx = pod_ctx.unsqueeze(1)  # (B, 1, P)
        
        q = self.q_proj(nodes).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(pod_ctx).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(pod_ctx).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return residual + out


class AttentionScorer(nn.Module):
    """
    Main Transformer model for node scoring.
    
    Architecture:
    1. Feature projection (F -> D)
    2. Temporal pooling (aggregate T timesteps)
    3. N-layer node self-attention
    4. Cross-attention with pod context
    5. Score head (D -> 1, scaled to 0-100)
    
    Designed for:
    - <50ms inference latency
    - <2GB memory footprint (with int8 quantization)
    - Interpretable attention weights
    """
    
    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        d_model: int = MODEL.D_MODEL,
        n_layers: int = MODEL.N_LAYERS,
        n_heads: int = MODEL.N_HEADS,
        pod_context_dim: int = POD_CONTEXT_DIM,
        dropout: float = MODEL.DROPOUT,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Temporal aggregation (simple mean pooling + learned weight)
        self.temporal_weight = nn.Parameter(torch.ones(1))
        
        # Node self-attention layers
        self.node_attention_layers = nn.ModuleList([
            NodeAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Pod cross-attention
        self.pod_cross_attn = PodCrossAttention(d_model, pod_context_dim, n_heads)
        
        # Score prediction head
        self.score_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        cluster_tensor: ClusterTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for node scoring.
        
        Args:
            cluster_tensor: Encoded cluster state
            
        Returns:
            scores: (N,) scores in range [0, 100]
            confidences: (N,) confidence values in [0, 1]
        """
        # Get dimensions
        node_features = cluster_tensor.node_features  # (N, T, F)
        N, T, F = node_features.shape
        
        # SAFETY: Check for NaN/Inf inputs (Issue 6 fix)
        if torch.isnan(node_features).any() or torch.isinf(node_features).any():
            # Return neutral scores for all nodes if input contains NaN/Inf
            neutral_scores = torch.full((N,), 50.0, device=node_features.device)
            neutral_confidences = torch.full((N,), 0.1, device=node_features.device)
            return neutral_scores, neutral_confidences
        
        if torch.isnan(cluster_tensor.pod_context).any() or torch.isinf(cluster_tensor.pod_context).any():
            neutral_scores = torch.full((N,), 50.0, device=node_features.device)
            neutral_confidences = torch.full((N,), 0.1, device=node_features.device)
            return neutral_scores, neutral_confidences
        
        # Add batch dimension for nn.Module compatibility
        node_features = node_features.unsqueeze(0)  # (1, N, T, F)
        pod_ctx = cluster_tensor.pod_context.unsqueeze(0)  # (1, P)
        mask = cluster_tensor.attention_mask.unsqueeze(0)  # (1, N)
        
        # Project features
        x = self.feature_proj(node_features)  # (1, N, T, D)
        
        # Temporal aggregation: mean pool over time
        x = x.mean(dim=2)  # (1, N, D)
        
        # Node self-attention layers
        for layer in self.node_attention_layers:
            x = layer(x, mask)
        
        # Cross-attention with pod context
        x = self.pod_cross_attn(x, pod_ctx)  # (1, N, D)
        
        # Score prediction
        scores = self.score_head(x).squeeze(-1).squeeze(0)  # (N,)
        
        # Scale to configured range with explicit clamping
        scores = torch.sigmoid(scores) * SCORE.MAX_SCORE
        scores = torch.clamp(scores, SCORE.MIN_SCORE, SCORE.MAX_SCORE)
        
        # Confidence estimation
        confidences = self.confidence_head(x).squeeze(-1).squeeze(0)  # (N,)
        confidences = torch.clamp(confidences, SCORE.MIN_CONFIDENCE, SCORE.MAX_CONFIDENCE)
        
        return scores, confidences
    
    def score_batch(
        self,
        cluster_tensor: ClusterTensor,
    ) -> list[dict]:
        """
        Score all nodes and return structured results.
        
        Returns:
            List of dicts with node_name, score, confidence, reasoning
        """
        scores, confidences = self.forward(cluster_tensor)
        
        results = []
        for i, node_name in enumerate(cluster_tensor.node_names):
            score = int(scores[i].item())
            confidence = float(confidences[i].item())
            
            # Generate simple reasoning based on score
            if score >= 80:
                reasoning = f"Node {node_name} shows excellent resource availability"
            elif score >= 60:
                reasoning = f"Node {node_name} has good capacity with minor contention"
            elif score >= 40:
                reasoning = f"Node {node_name} shows moderate resource pressure"
            else:
                reasoning = f"Node {node_name} has significant contention - avoid if possible"
            
            results.append({
                "node_name": node_name,
                "score": score,
                "confidence": confidence,
                "reasoning": reasoning,
            })
        
        return results
    
    @torch.no_grad()
    def quantize_int8(self) -> "AttentionScorer":
        """
        Quantize model to int8 for reduced memory footprint.
        
        Note: Requires PyTorch 2.0+ for best results.
        """
        # Dynamic quantization for linear layers
        quantized = torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized


def create_model(
    pretrained_path: Optional[str] = None,
    quantized: bool = False,
    device: torch.device = torch.device("cpu"),
) -> AttentionScorer:
    """
    Factory function to create and optionally load a pretrained model.
    
    Args:
        pretrained_path: Path to saved model weights (checkpoint file)
        quantized: Whether to apply int8 quantization
        device: Target device
        
    Returns:
        Initialized AttentionScorer model
    """
    import os
    
    model = AttentionScorer()
    
    # Check for default checkpoint path
    if pretrained_path is None:
        # Look for trained model in standard locations
        candidates = [
            "checkpoints/best_model.pt",
            "checkpoints/final_model.pt",
            os.path.expanduser("~/.kubeattention/model.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                pretrained_path = candidate
                print(f"📂 Found trained model at {pretrained_path}")
                break
    
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        # Handle both checkpoint format and raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Loaded trained model from {pretrained_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights from {pretrained_path}")
        else:
            print(f"⚠️ Unknown checkpoint format at {pretrained_path}, using random weights")
    else:
        print("⚠️ No pretrained model found - using random weights!")
    
    if quantized:
        model = model.quantize_int8()
    
    model = model.to(device)
    model.eval()
    
    return model

