"""
Utility functions for saving and loading the scaler.

Usage in training script:
    from src.dataset import RBMKDataLoader
    from utils import save_scaler
    
    dataloader = RBMKDataLoader()
    train_loader, val_loader, test_loader = dataloader.get_loaders()
    
    # After training...
    save_scaler(dataloader.scaler, "checkpoints/scaler.pkl")
"""

import joblib
from pathlib import Path


def save_scaler(scaler, filepath: str) -> None:
    """Save the fitted scaler to disk for later inference."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"✓ Scaler saved to {filepath}")


def load_scaler(filepath: str):
    """Load a previously saved scaler."""
    return joblib.load(filepath)
