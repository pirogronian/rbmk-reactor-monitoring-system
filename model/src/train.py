"""
Training module for RBMK anomaly detection model.

Training Strategy:
1. For each batch, forward pass through autoencoder
2. Calculate reconstruction error (MSE loss)
3. Backpropagate and update weights
4. Monitor training/validation loss to detect overfitting
5. Save best model based on validation loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import json
import time
import pickle 

class AnomalyTrainer:
    """
    Handles the complete training pipeline for the autoencoder model.

    Features:
    - Training with loss computation
    - Validation to detect overfitting
    - Early stopping to avoid wasting time
    - Model checkpointing (save best model)
    - Training history logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler, 
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
    ):
        """
        Args:
            model: The LSTM/GRU autoencoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            lr: Learning rate (typical: 1e-3 to 1e-4)
                - Higher: faster learning but may be unstable
                - Lower: slower but more stable
            weight_decay: L2 regularization (typical: 1e-5 to 1e-4)
                         Prevents overfitting by penalizing large weights
            device: 'cpu' or 'cuda' (GPU)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler 
        self.device = device

        # Loss function: Mean Squared Error
        # Penalizes reconstruction errors equally across all features
        self.criterion = nn.MSELoss()

        # Adam optimizer: adaptive learning rate, good for deep learning
        # Combines benefits of momentum and RMSprop
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler: reduce learning rate when validation loss plateaus
        # Helps fine-tune the model after initial learning phase
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # Minimize validation loss
            factor=0.5,  # Multiply LR by 0.5
            patience=5,  # Wait 5 epochs before reducing
            verbose=True,
        )

        # Training history for visualization
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "epoch": [],
            "learning_rate": [],
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch (one pass through all training data).

        Process for each batch:
        1. Forward pass: get reconstructed sequence
        2. Calculate MSE loss between input and reconstruction
        3. Backward pass: compute gradients
        4. Optimizer step: update weights

        Returns:
            Average training loss for this epoch
        """
        self.model.train()  # Set to training mode (enables dropout)
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (sequences, _) in enumerate(self.train_loader):
            # Move batch to device (GPU if available)
            sequences = sequences.to(self.device)

            # ===== FORWARD PASS =====
            # Get reconstruction from autoencoder
            reconstructed = self.model(sequences)

            # ===== LOSS COMPUTATION =====
            # MSE penalizes large reconstruction errors
            loss = self.criterion(reconstructed, sequences)

            # ===== BACKWARD PASS =====
            # Clear old gradients
            self.optimizer.zero_grad()

            # Compute gradients of loss w.r.t. all parameters
            loss.backward()

            # ===== GRADIENT CLIPPING =====
            # Prevent exploding gradients (common in RNNs)
            # Caps gradient norm at 1.0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # ===== WEIGHT UPDATE =====
            # Update weights: θ = θ - lr * ∇L
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """
        Evaluate model on validation set (without training).

        Process for each batch:
        1. Forward pass (no gradients needed)
        2. Calculate MSE loss
        3. Accumulate

        Returns:
            Average validation loss
        """
        self.model.eval()  # Set to evaluation mode (disables dropout)
        total_loss = 0.0
        num_batches = 0

        # disable gradient computation (saves memory and speeds up)
        with torch.no_grad():
            for sequences, _ in self.val_loader:
                sequences = sequences.to(self.device)

                # Forward pass only
                reconstructed = self.model(sequences)
                loss = self.criterion(reconstructed, sequences)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def fit(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict:
        """
        Complete training loop with early stopping.

        Strategy:
        1. Train for up to num_epochs
        2. After each epoch, validate and check if val loss improved
        3. If no improvement for early_stopping_patience epochs, stop
        4. Save best model based on val loss

        Args:
            num_epochs: Maximum epochs to train (typical: 50-200)
            early_stopping_patience: Epochs to wait before stopping (typical: 5-15)
            checkpoint_dir: Directory to save best model

        Returns:
            Training history dict
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)

        best_val_loss = float("inf")
        best_model_path = checkpoint_path / "best_model.pth"
        patience_counter = 0

        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Max epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*70}\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # ===== TRAINING PHASE =====
            train_loss = self.train_epoch()

            # ===== VALIDATION PHASE =====
            val_loss = self.validate()

            # ===== LOGGING =====
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # Print epoch summary
            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Time: {elapsed:.2f}s"
            )

            # ===== EARLY STOPPING =====
            if val_loss < best_val_loss:
                # Validation improved! Save the model
                best_val_loss = val_loss
                patience_counter = 0

                # Save checkpoint
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                    },
                    best_model_path,
                )
                
                # Save scaler separately
                scaler_path = checkpoint_path / "scaler.pkl"
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.6f})")

            else:
                # Validation did not improve
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(
                        f"\n⚠️  Early stopping triggered after {epoch + 1} epochs "
                        f"(patience: {early_stopping_patience})"
                    )
                    break

            # ===== LEARNING RATE SCHEDULING =====
            # Reduce learning rate if validation loss plateaus
            self.scheduler.step(val_loss)

            print()

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best val loss: {best_val_loss:.6f} (epoch {np.argmin(self.history['val_loss']) + 1})")
        print(f"Best model saved to: {best_model_path}")
        print(f"{'='*70}\n")

        # Save training history to JSON
        history_path = checkpoint_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")

        return self.history

    def load_best_model(self, checkpoint_path: str = "checkpoints/best_model.pth"):
        """Load the best saved model for inference"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Model loaded from {checkpoint_path}")
        return self.model
