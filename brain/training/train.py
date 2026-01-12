"""
Training script for KubeAttention AttentionScorer model.

Trains the Transformer model on scheduling events with outcomes,
using cross-entropy loss to predict which node leads to the best outcome.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain.model import AttentionScorer, create_model
from brain.training.dataset import SchedulingDataset, create_dataloader
from brain.config import MODEL


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Data
    train_data_path: str = "training_data.jsonl"
    val_data_path: Optional[str] = None
    
    # Model (imported from centralized config)
    d_model: int = MODEL.D_MODEL
    n_layers: int = MODEL.N_LAYERS
    n_heads: int = MODEL.N_HEADS
    dropout: float = MODEL.DROPOUT
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """
    Trainer for the AttentionScorer model.
    
    Uses weighted cross-entropy loss where:
    - OOM/eviction outcomes have higher weight (penalize bad placements more)
    - Success outcomes train the model to prefer those nodes
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize model
        self.model = AttentionScorer(
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        ).to(self.device)
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 Model initialized with {num_params:,} parameters")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
        )
        
        # NOTE: We use MSE loss directly in train_epoch/validate, not a stored loss_fn
        # This is because we're training for regression (score prediction) not classification
        # The per-sample loss is computed as: F.mse_loss(scores / 100.0, target)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            node_features = batch["node_features"].to(self.device)
            pod_context = batch["pod_context"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            weights = batch["weight"].to(self.device)
            
            # Forward pass - need to construct ClusterTensor
            # For now, use a simplified forward
            batch_size = node_features.shape[0]
            
            # Reshape for model: (B, N, T, F) -> process each sample
            losses = []
            for i in range(batch_size):
                # Create ClusterTensor-like structure
                from brain.tensor_encoder import ClusterTensor
                
                num_nodes = batch["num_nodes"][i].item()
                seq_len = node_features.shape[2]  # T dimension
                
                cluster_tensor = ClusterTensor(
                    node_features=node_features[i, :num_nodes],  # (N, T, F)
                    pod_context=pod_context[i],  # (P,)
                    attention_mask=attention_mask[i, :num_nodes],  # (N,)
                    node_names=[f"node-{j}" for j in range(num_nodes)],
                    timestamps=torch.arange(seq_len, dtype=torch.float32),  # (T,)
                )
                
                scores, confidences = self.model(cluster_tensor)
                
                # Compute loss for this sample
                # Target: node that was chosen and led to good outcome
                target = labels[i, :num_nodes]
                
                # Use MSE loss: predict the outcome score for each node
                sample_loss = F.mse_loss(scores / 100.0, target)
                losses.append(sample_loss * weights[i])
            
            # Average loss across batch
            loss = torch.stack(losses).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            node_features = batch["node_features"].to(self.device)
            pod_context = batch["pod_context"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            batch_size = node_features.shape[0]
            
            losses = []
            for i in range(batch_size):
                from brain.tensor_encoder import ClusterTensor
                
                num_nodes = batch["num_nodes"][i].item()
                seq_len = node_features.shape[2]
                
                cluster_tensor = ClusterTensor(
                    node_features=node_features[i, :num_nodes],
                    pod_context=pod_context[i],
                    attention_mask=attention_mask[i, :num_nodes],
                    node_names=[f"node-{j}" for j in range(num_nodes)],
                    timestamps=torch.arange(seq_len, dtype=torch.float32),
                )
                
                scores, _ = self.model(cluster_tensor)
                target = labels[i, :num_nodes]
                sample_loss = F.mse_loss(scores / 100.0, target)
                losses.append(sample_loss)
            
            loss = torch.stack(losses).mean()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Full training loop."""
        print(f"🚀 Starting training for {self.config.num_epochs} epochs")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # Track best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
            
            # Update learning rate
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Log progress
            lr = self.optimizer.param_groups[0]['lr']
            log_msg = f"Epoch {epoch+1}/{self.config.num_epochs} | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | LR: {lr:.2e}"
            print(log_msg)
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_every_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Training complete in {elapsed/60:.1f} minutes")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "elapsed_seconds": elapsed,
        }
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print(f"📂 Loaded checkpoint from {path} (epoch {self.current_epoch})")


def train_model(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    output_dir: str = "checkpoints",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> str:
    """
    High-level function to train a model.
    
    Returns path to the best model checkpoint.
    """
    config = TrainingConfig(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        checkpoint_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_data_path,
        batch_size=batch_size,
        shuffle=True,
    )
    
    val_dataloader = None
    if val_data_path:
        val_dataloader = create_dataloader(
            val_data_path,
            batch_size=batch_size,
            shuffle=False,
        )
    
    # Train
    trainer = Trainer(config)
    results = trainer.train(train_dataloader, val_dataloader)
    
    # Save training results
    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "train_losses": results["train_losses"],
            "val_losses": results["val_losses"],
            "best_val_loss": results["best_val_loss"],
            "elapsed_seconds": results["elapsed_seconds"],
        }, f, indent=2)
    
    return str(Path(output_dir) / "best_model.pt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train KubeAttention model")
    parser.add_argument("--train-data", required=True, help="Path to training data")
    parser.add_argument("--val-data", help="Path to validation data")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    best_model_path = train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    print(f"\n🎉 Best model saved to: {best_model_path}")
