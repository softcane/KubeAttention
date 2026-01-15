"""
Training script for KubeAttention MLP/XGBoost models.

Trains the scorer model on scheduling events with outcomes.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.models import get_model, list_models
from brain.training.dataset import SchedulingDataset, create_dataloader
from brain.config import MODEL_SELECTION


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data
    train_data_path: str = "training_data.jsonl"
    val_data_path: Optional[str] = None
    
    # Model selection: "mlp" or "xgboost"
    model_type: str = MODEL_SELECTION.MODEL_TYPE
    
    # MLP-specific
    hidden_dim: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 32
    
    # XGBoost-specific
    n_estimators: int = 100
    max_depth: int = 6
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"


def prepare_training_data(data_path: str) -> tuple:
    """Load and prepare training data from JSONL."""
    from brain.metrics_schema import FEATURE_NAMES
    
    dataset = SchedulingDataset(data_path)
    
    X_list = []
    y_list = []
    w_list = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        num_nodes = sample["num_nodes"].item()
        
        # Node features: use last timestep
        node_feats = sample["node_features"][:num_nodes, -1, :].numpy()  # (N, F)
        
        # Pod context features: broadcast to all nodes
        pod_ctx = sample["pod_context"].numpy()  # (P,)
        pod_broadcast = np.tile(pod_ctx, (num_nodes, 1))  # (N, P)
        
        # Concatenate node features with pod context for full input
        combined_feats = np.hstack([node_feats, pod_broadcast])  # (N, F+P)
        
        labels = sample["labels"][:num_nodes].numpy()
        weight = sample["weight"].item()
        
        X_list.append(combined_feats)
        y_list.append(labels)
        w_list.extend([weight] * num_nodes)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    w = np.array(w_list)
    
    return X, y, w



def train_model(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    model_type: str = "mlp",
    **kwargs,
) -> str:
    """
    High-level function to train a model.
    
    Args:
        train_data_path: Path to training JSONL data
        val_data_path: Optional path to validation data
        output_dir: Output directory for checkpoints
        model_type: "mlp" or "xgboost"
        **kwargs: Model-specific parameters
        
    Returns:
        Path to the saved model checkpoint
    """
    print(f"Training {model_type.upper()} model...")
    print(f"  Data: {train_data_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    print("Loading training data...")
    X_train, y_train, w_train = prepare_training_data(train_data_path)
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {X_train.shape[1]}")
    
    X_val, y_val = None, None
    if val_data_path:
        X_val, y_val, _ = prepare_training_data(val_data_path)
        print(f"  Validation samples: {len(X_val):,}")
    
    # Initialize model with correct input dimension
    input_dim = X_train.shape[1]
    if model_type == "mlp":
        model = get_model(model_type, input_dim=input_dim, hidden_dim=kwargs.get("hidden_dim", 64))
    else:
        model = get_model(model_type, n_estimators=kwargs.get("n_estimators", 100), max_depth=kwargs.get("max_depth", 6))
    print(f"  Model: {model.name}")
    print(f"  Parameters: {model.num_parameters:,}")
    print(f"  Input dim: {input_dim}")

    
    # Train
    start_time = time.perf_counter()
    
    if model_type == "mlp":
        metrics = model.train(
            X_train, y_train,
            weights=w_train,
            epochs=kwargs.get("epochs", 50),
            lr=kwargs.get("learning_rate", 1e-3),
            batch_size=kwargs.get("batch_size", 32),
        )
    else:
        eval_set = (X_val, y_val) if X_val is not None else None
        metrics = model.train(X_train, y_train, weights=w_train, eval_set=eval_set)
    
    elapsed = time.perf_counter() - start_time
    print(f"Training complete in {elapsed:.1f}s")
    
    # Save model
    model_ext = ".pt" if model_type == "mlp" else ".json"
    model_path = os.path.join(output_dir, f"best_model{model_ext}")
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training results
    results = {
        "model_type": model_type,
        "num_samples": len(X_train),
        "elapsed_seconds": elapsed,
        **metrics,
    }
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train KubeAttention model")
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--val-data", help="Path to validation data")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--model", default="mlp", choices=["mlp", "xgboost"],
                        help="Model type (default: mlp)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (MLP)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (MLP)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (MLP)")
    parser.add_argument("--n-estimators", type=int, default=100, help="Trees (XGBoost)")
    parser.add_argument("--max-depth", type=int, default=6, help="Max depth (XGBoost)")
    
    args = parser.parse_args()
    
    best_model_path = train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        model_type=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    
    print(f"\nBest model saved to: {best_model_path}")
