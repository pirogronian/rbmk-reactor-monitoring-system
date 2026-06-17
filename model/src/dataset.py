"""
Dataset module for RBMK reactor time-series data.

This module handles:
1. Loading parquet data
2. Preprocessing and normalization
3. Creating sliding windows for time-series sequences
4. Splitting into train/val/test sets
5. Creating PyTorch DataLoaders for efficient batch processing
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Optional
import pyarrow


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series sequences.

    Creates sliding windows of time-series data for LSTM input.
    Each sample is a sequence of shape (sequence_length, num_features)
    """

    def __init__(self, data: np.ndarray, sequence_length: int):
        """
        Args:
            data: Preprocessed time-series data of shape (num_samples, num_features)
            sequence_length: Number of timesteps in each window (e.g., 50)
        """
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        """Total number of windows we can create from the data"""
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single time-series window.

        Returns:
            input_seq: Current window [sequence_length, features]
            target_seq: Same window (for autoencoder reconstruction)
        """
        window = self.data[idx : idx + self.sequence_length]
        # Convert to float32 for PyTorch compatibility
        window_tensor = torch.tensor(window, dtype=torch.float32)
        return window_tensor, window_tensor


class RBMKDataLoader:
    """
    Handles loading, preprocessing, and splitting RBMK reactor data.

    Workflow:
    1. Load parquet file
    2. Select numeric features (reactor parameters)
    3. Normalize using StandardScaler
    4. Create sliding windows
    5. Split into train/val/test
    6. Create PyTorch DataLoaders
    """

    def __init__(
        self,
        data_path: str,
        sequence_length: int = 50,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Args:
            data_path: Path to parquet file (e.g., 'Influx_RBML_data.parquet')
            sequence_length: Number of timesteps per window (typical: 20-100)
            train_ratio: Proportion for training (default 70%)
            val_ratio: Proportion for validation (default 15%)
            test_ratio: Proportion for testing (default 15%)
        """
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Storage for scalers (needed for inverse transformation later)
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self) -> np.ndarray:
        """
        Load and preprocess the parquet file.

        Steps:
        1. Read parquet file
        2. Select only numeric columns (reactor parameters)
        3. Remove any rows with missing values
        4. Normalize to mean=0, std=1 using StandardScaler

        Returns:
            Normalized data array of shape (num_samples, num_features)
        """
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path, engine='pyarrow')

        # Select only numeric columns (ignore timestamps, strings, etc.)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = numeric_cols

        print(f"Selected {len(numeric_cols)} numeric features: {numeric_cols[:5]}...")

        # Extract numeric data
        data = df[numeric_cols].values

        # Remove rows with NaN values
        mask = ~np.isnan(data).any(axis=1)
        data = data[mask]
        print(f"Data shape after cleaning: {data.shape}")

        return data

    def split_train_val_test(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Uses sequential split (not random) to preserve temporal structure:
        - First 70% → Training
        - Next 15% → Validation
        - Last 15% → Testing

        This prevents time-series leakage where future data influences past predictions.

        Returns:
            train_data, val_data, test_data arrays
        """
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        print(f"Data split:")
        print(f"  Train: {len(train_data)} samples ({self.train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_data)} samples ({self.val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_data)} samples ({self.test_ratio*100:.0f}%)")

        return train_data, val_data, test_data

    def create_dataloaders(
        self, batch_size: int = 32, num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create the complete pipeline: load → preprocess → split → create DataLoaders.

        Args:
            batch_size: Number of sequences per batch (typical: 16-64)
            num_workers: Number of parallel workers for data loading (0 for single-threaded)

        Returns:
            train_loader, val_loader, test_loader
        """
        # Load and normalize data
        data = self.load_data()


        train_data, val_data, test_data = self.split_train_val_test(data)

        train_data_scaled = self.scaler.fit_transform(train_data)

        # Dane walidacyjne i testowe tylko przekształcamy (transform), nie ucząc na nich skalera!
        val_data_scaled = self.scaler.transform(val_data)
        test_data_scaled = self.scaler.transform(test_data)

        #  POPRAWNA WERSJA:
        train_dataset = TimeSeriesDataset(train_data_scaled, self.sequence_length) # Przekazane SKALOWANE dane!
        val_dataset = TimeSeriesDataset(val_data_scaled, self.sequence_length)
        test_dataset = TimeSeriesDataset(test_data_scaled, self.sequence_length)

        print(f"\nDataset windows created:")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Val sequences:   {len(val_dataset)}")
        print(f"  Test sequences:  {len(test_dataset)}")

        # Create DataLoaders for efficient batching and shuffling
        # Shuffle training data to improve learning, but keep val/test in order
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # Keeps data on GPU memory for faster transfer
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for consistent evaluation
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def inverse_transform(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.

        Useful for interpreting anomaly scores in the original units.

        Args:
            normalized_data: Data in normalized space

        Returns:
            Data in original scale
        """
        return self.scaler.inverse_transform(normalized_data)