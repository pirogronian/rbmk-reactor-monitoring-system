"""
LSTM/GRU Autoencoder model for anomaly detection in RBMK reactor data.

Architecture:
- ENCODER: LSTM/GRU layers that compress the input sequence into latent features
- DECODER: LSTM/GRU layers that reconstruct the original sequence

Anomaly Detection Strategy:
- Normal behavior → Low reconstruction error
- Abnormal behavior → High reconstruction error
- Threshold-based classification: error > threshold = anomaly
"""
import torch
import torch.nn as nn
from typing import Tuple, List


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection.

    How it works:
    1. ENCODER reads the input sequence and compresses it into a hidden state
    2. DECODER uses this hidden state to reconstruct the original sequence
    3. Reconstruction error indicates if the sequence is normal or anomalous
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: Number of features (e.g., 10 reactor parameters)
            hidden_size: Number of LSTM hidden units (typical: 32-128)
                        Higher = more capacity but slower training
            num_layers: Number of stacked LSTM layers (typical: 1-3)
                       Deeper = better feature extraction but may overfit
            dropout: Regularization to prevent overfitting (typical: 0.1-0.3)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ============ ENCODER ============
        # Takes input sequence and compresses it
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,  # Input shape: (batch, seq_len, features)
        )

        # ============ DECODER ============
        # Takes the encoder's final state and reconstructs the sequence
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Final linear layer to map hidden states back to input features
        self.decoder_output = nn.Linear(hidden_size, input_size)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
               Example: (32, 50, 10) = 32 sequences, 50 timesteps, 10 features

        Returns:
            reconstructed: Reconstructed sequence (same shape as input)
            encoder_hidden: Hidden state from encoder (bottleneck representation)
            encoder_cell: Cell state from encoder
        """
        # ===== ENCODING PHASE =====
        # LSTM reads the entire sequence and produces a hidden state (bottleneck)
        # encoder_output shape: (batch, seq_len, hidden_size) - we don't use this
        # encoder_hidden: (num_layers, batch, hidden_size) - the compressed representation
        # encoder_cell: (num_layers, batch, hidden_size) - LSTM cell state

        encoder_output, (encoder_hidden, encoder_cell) = self.encoder_lstm(x)

        # ===== DECODING PHASE =====
        # Use encoder's final hidden state as initialization for decoder
        # This forces the decoder to reconstruct using only the compressed info
        # We repeat the hidden state for sequence_length timesteps

        seq_len = x.size(1)  # Original sequence length (e.g., 50)

        # Create decoder input: all zeros with shape (batch, seq_len, hidden_size)
        # The decoder learns to reconstruct from the hidden state alone
        decoder_input = torch.zeros(x.size(0), seq_len, self.hidden_size, device=x.device)

        # Decode: feed zeros and the encoder state to generate reconstruction
        decoder_output, _ = self.decoder_lstm(decoder_input, (encoder_hidden, encoder_cell))

        # Reshape decoder output to match input dimensions
        # decoder_output: (batch, seq_len, hidden_size) → (batch, seq_len, input_size)
        reconstructed = self.decoder_output(decoder_output)

        return reconstructed, encoder_hidden, encoder_cell


class GRUAutoencoder(nn.Module):
    """
    GRU-based Autoencoder for time-series anomaly detection.

    GRU vs LSTM:
    - GRU: Simpler, fewer parameters, faster training
    - LSTM: More complex, more parameters, often better performance
    Choose GRU for faster iteration, LSTM for better accuracy.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """Same parameters as LSTMAutoencoder"""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ============ ENCODER ============
        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # ============ DECODER ============
        self.decoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection layer
        self.decoder_output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (same logic as LSTM, but GRU has only hidden state, no cell state)

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            reconstructed: Reconstructed sequence
            encoder_hidden: Compressed representation
        """
        # Encode
        encoder_output, encoder_hidden = self.encoder_gru(x)

        # Decode
        seq_len = x.size(1)
        decoder_input = torch.zeros(x.size(0), seq_len, self.hidden_size, device=x.device)
        decoder_output, _ = self.decoder_gru(decoder_input, encoder_hidden)

        # Reconstruct
        reconstructed = self.decoder_output(decoder_output)

        return reconstructed, encoder_hidden


class AnomalyDetector(nn.Module):
    """
    Wrapper around autoencoder that provides convenient anomaly scoring.

    Usage:
        model = AnomalyDetector(input_size=10, model_type='lstm')
        reconstructed = model(x)
        anomaly_scores = model.compute_anomaly_score(x)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_type: str = "lstm",
    ):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM/GRU hidden size
            num_layers: Number of layers
            dropout: Dropout rate
            model_type: 'lstm' or 'gru'
        """
        super().__init__()

        if model_type.lower() == "lstm":
            self.autoencoder = LSTMAutoencoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif model_type.lower() == "gru":
            self.autoencoder = GRUAutoencoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"model_type must be 'lstm' or 'gru', got {model_type}")

        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through autoencoder and return reconstruction"""
        if self.model_type.lower() == "lstm":
            reconstructed, _, _ = self.autoencoder(x)
        else:
            reconstructed, _ = self.autoencoder(x)
        return reconstructed

    @staticmethod
    def compute_anomaly_score(
        original: torch.Tensor, reconstructed: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute reconstruction error as anomaly score.

        Strategy:
        - For each sequence, calculate MSE between original and reconstructed
        - High MSE → likely anomaly
        - Low MSE → likely normal behavior

        Args:
            original: Original sequences (batch, seq_len, features)
            reconstructed: Reconstructed sequences (same shape)
            reduction: 'mean' (scalar) or 'none' (per-sequence scores)

        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Mean Squared Error: average squared difference
        mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2])  # (batch_size,)

        if reduction == "mean":
            return mse.mean()
        else:
            return mse
