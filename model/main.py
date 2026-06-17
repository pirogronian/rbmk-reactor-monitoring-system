"""
Main entry point for RBMK reactor anomaly detection system.

Workflow:
1. Load and preprocess data
2. Create model (LSTM or GRU autoencoder)
3. Train on normal behavior data
4. Evaluate on test set and compute anomaly threshold
5. Detect anomalies in new data
"""
import sys
from pathlib import Path
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import Tuple

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataset import RBMKDataLoader
from model import AnomalyDetector
from train import AnomalyTrainer


def setup_device(use_cpu: bool = False) -> str:
    """
    Determine whether to use GPU (CUDA) or CPU.

    CUDA GPU:
    - Much faster training (10-100x speedup for deep learning)
    - Required for large models
    - Only available if PyTorch is built with CUDA and GPU exists

    CPU:
    - Always available
    - Slower but useful for debugging

    Args:
        use_cpu: Force CPU-only execution (useful if CUDA has compatibility issues)
    """
    if use_cpu:
        device = "cpu"
        print("⚠️  CPU mode forced (--use-cpu flag set)")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print("⚠️  GPU not available, using CPU (training will be slow)")

    return device


def plot_training_history(history: dict, save_path: str = "checkpoints/training_plot.png"):
    """
    Visualize training progress over epochs.

    Shows:
    - Training loss (should decrease smoothly)
    - Validation loss (should decrease, may fluctuate)
    - Gap between them indicates overfitting
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    ax.plot(epochs, train_loss, label="Training Loss", marker="o", linewidth=2)
    ax.plot(epochs, val_loss, label="Validation Loss", marker="s", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Autoencoder Training Progress", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Log scale to see small differences

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"✓ Training plot saved to {save_path}")
    plt.close()


def compute_anomaly_threshold(
    model: torch.nn.Module,
    val_loader,
    device: str,
    percentile: float = 95.0,
) -> float:
    """
    Compute anomaly detection threshold from validation set.

    Strategy:
    1. Calculate reconstruction errors for all validation samples
    2. Use percentile-based threshold (e.g., 95th percentile)
    3. Samples with error > threshold are flagged as anomalies

    Args:
        model: Trained autoencoder
        val_loader: Validation data loader
        device: CPU or CUDA
        percentile: Threshold percentile (95 = top 5% are anomalies)
                   Adjust based on false alarm tolerance

    Returns:
        Threshold value
    """
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for sequences, _ in val_loader:
            sequences = sequences.to(device)
            reconstructed = model(sequences)

            # Calculate per-sequence reconstruction error
            mse = torch.mean((sequences - reconstructed) ** 2, dim=[1, 2])
            anomaly_scores.extend(mse.cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)
    threshold = np.percentile(anomaly_scores, percentile)

    print(f"\n{'='*70}")
    print("ANOMALY DETECTION THRESHOLD")
    print(f"{'='*70}")
    print(f"Percentile: {percentile}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Mean score: {anomaly_scores.mean():.6f}")
    print(f"Min score: {anomaly_scores.min():.6f}")
    print(f"Max score: {anomaly_scores.max():.6f}")
    print(f"{'='*70}\n")

    return threshold


def detect_anomalies(
    model: torch.nn.Module,
    test_loader,
    threshold: float,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies in test set using trained model.

    For each sequence:
    1. Compute reconstruction error
    2. Compare with threshold
    3. Flag if error > threshold

    Args:
        model: Trained autoencoder
        test_loader: Test data loader
        threshold: Anomaly threshold
        device: CPU or CUDA

    Returns:
        anomaly_scores: Array of reconstruction errors
        predictions: Binary array (0=normal, 1=anomaly)
    """
    model.eval()
    anomaly_scores = []

    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(device)
            reconstructed = model(sequences)

            # Per-sequence reconstruction error
            mse = torch.mean((sequences - reconstructed) ** 2, dim=[1, 2])
            anomaly_scores.extend(mse.cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)
    predictions = (anomaly_scores > threshold).astype(int)

    # Statistics
    num_anomalies = predictions.sum()
    num_total = len(predictions)
    anomaly_rate = num_anomalies / num_total * 100

    print(f"\n{'='*70}")
    print("ANOMALY DETECTION RESULTS")
    print(f"{'='*70}")
    print(f"Total sequences: {num_total}")
    print(f"Anomalies detected: {num_anomalies} ({anomaly_rate:.1f}%)")
    print(f"Normal sequences: {num_total - num_anomalies} ({100-anomaly_rate:.1f}%)")
    print(f"Score range: [{anomaly_scores.min():.6f}, {anomaly_scores.max():.6f}]")
    print(f"Threshold: {threshold:.6f}")
    print(f"{'='*70}\n")

    return anomaly_scores, predictions


def plot_anomaly_scores(
    scores: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    save_path: str = "checkpoints/anomaly_scores.png",
):
    """
    Visualize anomaly scores with threshold line.

    Shows distribution of reconstruction errors and anomaly threshold.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Time series of anomaly scores
    ax = axes[0]
    ax.plot(scores, label="Reconstruction Error", linewidth=1, alpha=0.7)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.6f})")
    ax.scatter(
        np.where(predictions == 1)[0],
        scores[predictions == 1],
        color="red",
        s=20,
        label="Detected Anomalies",
    )
    ax.set_xlabel("Sequence Index", fontsize=11)
    ax.set_ylabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_title("Anomaly Scores Over Time", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Histogram of anomaly scores
    ax = axes[1]
    ax.hist(scores, bins=50, alpha=0.7, color="blue", label="Reconstruction Errors")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution of Reconstruction Errors", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"✓ Anomaly scores plot saved to {save_path}")
    plt.close()


def main(args):
    """
    Main pipeline: load data → train model → detect anomalies
    """
    print("\n" + "=" * 70)
    print("RBMK REACTOR ANOMALY DETECTION SYSTEM")
    print("=" * 70 + "\n")

    # ===== SETUP =====
    device = setup_device(use_cpu=args.use_cpu)

    # ===== LOAD DATA =====
    print("\n📊 Loading data...")
    data_loader = RBMKDataLoader(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 for Windows, can use more on Linux/Mac
    )

    # Get number of features from first batch
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[2]
    print(f"✓ Input features: {input_size}")

    # ===== CREATE MODEL =====
    print(f"\n🧠 Creating {args.model_type.upper()} autoencoder...")
    model = AnomalyDetector(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_type=args.model_type,
    )

    # Print model architecture
    print(f"Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # ===== TRAIN MODEL =====
    if args.mode == "train":
        print(f"\n🚀 Training...")
        trainer = AnomalyTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=data_loader.scaler,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
        )

        history = trainer.fit(
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir="checkpoints",
        )

        # Plot training progress
        plot_training_history(history)

        # Load best model
        model = trainer.load_best_model("checkpoints/best_model.pth")

    elif args.mode == "inference":
        # Load pre-trained model
        print(f"\n📂 Loading pre-trained model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Model loaded")

    # ===== COMPUTE ANOMALY THRESHOLD =====
    print("\n📈 Computing anomaly threshold...")
    threshold = compute_anomaly_threshold(
        model=model,
        val_loader=val_loader,
        device=device,
        percentile=args.threshold_percentile,
    )

    # ===== DETECT ANOMALIES =====
    print("\n🔍 Detecting anomalies in test set...")
    anomaly_scores, predictions = detect_anomalies(
        model=model,
        test_loader=test_loader,
        threshold=threshold,
        device=device,
    )

    # ===== VISUALIZE RESULTS =====
    plot_anomaly_scores(anomaly_scores, predictions, threshold)

    print("\n✅ Pipeline complete!")
    print(f"Results saved to 'checkpoints/' directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RBMK Reactor Anomaly Detection using LSTM/GRU Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py --mode train --model-type lstm --num-epochs 50

  # Use GPU for faster training
  python main.py --mode train --model-type gru --num-epochs 100

  # Load a pre-trained model and detect anomalies
  python main.py --mode inference --checkpoint checkpoints/best_model.pth
        """,
    )

    # ===== DEVICE ARGUMENTS =====
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force CPU-only execution (useful if CUDA has compatibility issues)",
    )

    # ===== DATA ARGUMENTS =====
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Influx_RBML_data.parquet",
        help="Path to parquet data file",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=50,
        help="Sequence length for time windows (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    # ===== MODEL ARGUMENTS =====
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lstm", "gru"],
        default="lstm",
        help="Model type (default: lstm)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM/GRU hidden size (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM/GRU layers (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )

    # ===== TRAINING ARGUMENTS =====
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        default="train",
        help="Mode: train new model or load pre-trained (default: train)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to checkpoint for inference mode",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="L2 regularization weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    # ===== ANOMALY DETECTION ARGUMENTS =====
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile for anomaly threshold (default: 95.0)",
    )

    args = parser.parse_args()
    main(args)
