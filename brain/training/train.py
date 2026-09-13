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

from brain.config import MODEL_SELECTION
from brain.metrics_schema import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    REQUIRED_FEATURE_NAMES,
)
from brain.models import get_model, list_models
from brain.training.dataset import SchedulingDataset


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


def prepare_training_data(data_path: str, require_measured: bool = False) -> tuple:
    """Load model inputs and optionally require measured cluster outcomes."""
    dataset = SchedulingDataset(data_path)
    if not dataset.events:
        raise ValueError(f"no completed scheduling events in {data_path}")
    if require_measured:
        invalid_evidence = []
        incomplete_telemetry = []
        for event in dataset.events:
            event_id = event.get("event_id", event.get("pod_name", "unknown"))
            if event.get("evidence_source") != "measured":
                invalid_evidence.append(event_id)
                continue
            candidate_nodes = event.get("candidate_nodes") or []
            telemetry = event.get("node_telemetry") or {}
            for node_name in candidate_nodes:
                sample = telemetry.get(node_name) or {}
                available = set(sample.get("available_metrics") or [])
                if not REQUIRED_FEATURE_NAMES.issubset(available):
                    incomplete_telemetry.append(event_id)
                    break
        if invalid_evidence:
            raise ValueError(
                f"validation data contains {len(invalid_evidence)} events without "
                "evidence_source='measured'"
            )
        if incomplete_telemetry:
            raise ValueError(
                f"validation data contains {len(incomplete_telemetry)} events without "
                "complete required interference measurements"
            )

    X_list = []
    y_list = []
    weights = []
    for index in range(len(dataset)):
        sample = dataset[index]
        node_count = sample["num_nodes"].item()
        if node_count == 0:
            continue
        node_features = sample["node_features"][:node_count, -1, :].numpy()
        pod_features = sample["pod_context"].numpy()
        combined = np.hstack([
            node_features,
            np.broadcast_to(pod_features, (node_count, pod_features.shape[0])),
        ])
        X_list.append(combined)
        y_list.append(sample["labels"][:node_count].numpy())
        weights.extend([sample["weight"].item()] * node_count)
    if not X_list:
        raise ValueError(f"no candidate node telemetry in {data_path}")
    return np.vstack(X_list), np.concatenate(y_list), np.asarray(weights)


def evaluate_promotion(model, X_val: np.ndarray, y_val: np.ndarray, minimum_improvement: float) -> dict:
    """Compare held-out model quality with the non-ML resource-pressure rule."""
    predictions = model.predict_quality(X_val)
    cpu = X_val[:, 0]
    memory = X_val[:, 2]
    cache_miss = X_val[:, 4]
    baseline = (1.0 - cpu) * 0.4 + (1.0 - memory) * 0.4 + (1.0 - cache_miss) * 0.2
    model_mse = float(np.mean((predictions - y_val) ** 2))
    baseline_mse = float(np.mean((baseline - y_val) ** 2))
    threshold = baseline_mse * (1.0 - minimum_improvement)
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "held_out_model_mse": model_mse,
        "held_out_baseline_mse": baseline_mse,
        "minimum_relative_improvement": minimum_improvement,
        "promoted": model_mse < threshold,
    }



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
    
    if not val_data_path:
        raise ValueError("a held-out measured validation data set is required")
    X_val, y_val, _ = prepare_training_data(val_data_path, require_measured=True)
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
        metrics = model.train(
            X_train, y_train, weights=w_train, eval_set=(X_val, y_val)
        )
    
    elapsed = time.perf_counter() - start_time
    print(f"Training complete in {elapsed:.1f}s")
    
    promotion = evaluate_promotion(
        model,
        X_val,
        y_val,
        minimum_improvement=kwargs.get("minimum_improvement", 0.01),
    )
    results = {
        "model_type": model_type,
        "num_samples": len(X_train),
        "elapsed_seconds": elapsed,
        **metrics,
        **promotion,
    }
    results_path = os.path.join(output_dir, "training_results.json")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file, indent=2)
    if not promotion["promoted"]:
        raise RuntimeError(
            "model promotion rejected: held-out model MSE did not beat the baseline"
        )

    model_ext = ".pt" if model_type == "mlp" else ".json"
    model_path = os.path.join(output_dir, f"best_model{model_ext}")
    model.save(model_path)
    print(f"Promoted model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train KubeAttention model")
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--val-data", required=True, help="Measured held-out JSONL data")
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
