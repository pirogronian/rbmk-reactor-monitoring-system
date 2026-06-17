"""
Configuration file for RBMK anomaly detection system.

Edit this file to set default parameters, then run:
    python main_with_config.py

Or pass parameters directly:
    python main.py --mode train --hidden-size 128 --num-epochs 100
"""

# ============ DATA CONFIGURATION ============
DATA_CONFIG = {
    # Path to input parquet file (relative to model/ directory)
    "data_path": "../Influx_RBML_data.parquet",

    # Number of timesteps per sequence (typical: 20-100)
    # Higher = more context but slower training
    "sequence_length": 50,

    # Batch size for training (typical: 16-128)
    # Higher = faster but more memory
    "batch_size": 32,
}

# ============ MODEL CONFIGURATION ============
MODEL_CONFIG = {
    # Model type: 'lstm' (more accurate) or 'gru' (faster)
    "model_type": "lstm",

    # Hidden size: number of LSTM/GRU units (typical: 32-256)
    # Higher = more capacity but slower and risks overfitting
    "hidden_size": 64,

    # Number of stacked layers (typical: 1-3)
    # Deeper networks can learn complex patterns but are harder to train
    "num_layers": 2,

    # Dropout rate for regularization (typical: 0.1-0.5)
    # Prevents overfitting by randomly disabling neurons
    "dropout": 0.2,
}

# ============ TRAINING CONFIGURATION ============
TRAINING_CONFIG = {
    # Training mode: 'train' (new model) or 'inference' (load existing)
    "mode": "train",

    # Path to checkpoint for inference mode
    "checkpoint": "checkpoints/best_model.pth",

    # Maximum number of training epochs
    # Early stopping will likely end before this
    "num_epochs": 100,

    # Learning rate: step size for weight updates (typical: 1e-4 to 1e-2)
    # Too high: unstable learning, loss oscillates
    # Too low: slow learning, may not converge
    "learning_rate": 1e-3,

    # L2 regularization strength (typical: 1e-5 to 1e-3)
    # Penalizes large weights, prevents overfitting
    "weight_decay": 1e-5,

    # Early stopping patience: epochs to wait before stopping
    # if validation loss doesn't improve (typical: 5-20)
    "early_stopping_patience": 10,
}

# ============ ANOMALY DETECTION CONFIGURATION ============
ANOMALY_CONFIG = {
    # Percentile for threshold (typical: 85-99)
    # 95 = top 5% are anomalies
    # Lower = more sensitive (more false alarms)
    # Higher = less sensitive (may miss anomalies)
    "threshold_percentile": 95.0,
}

# ============ PRESET CONFIGURATIONS ============

# For quick experimentation
QUICK_SETUP = {
    **DATA_CONFIG,
    **{
        **MODEL_CONFIG,
        "model_type": "gru",  # Faster
        "hidden_size": 32,
        "num_layers": 1,
    },
    **{
        **TRAINING_CONFIG,
        "num_epochs": 50,
    },
    **ANOMALY_CONFIG,
}

# For best accuracy (takes longer)
BEST_ACCURACY = {
    **DATA_CONFIG,
    **{
        **MODEL_CONFIG,
        "model_type": "lstm",
        "hidden_size": 256,
        "num_layers": 3,
        "dropout": 0.3,
    },
    **{
        **TRAINING_CONFIG,
        "num_epochs": 300,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "early_stopping_patience": 15,
    },
    **{
        **ANOMALY_CONFIG,
        "threshold_percentile": 90.0,
    },
}

# For GPU with lots of memory (maximize performance)
GPU_OPTIMIZED = {
    **DATA_CONFIG,
    **{
        **MODEL_CONFIG,
        "model_type": "lstm",
        "hidden_size": 512,
        "num_layers": 4,
        "dropout": 0.2,
    },
    **{
        **TRAINING_CONFIG,
        "num_epochs": 200,
        "batch_size": 128,  # Larger batch on GPU
        "learning_rate": 1e-3,
    },
    **ANOMALY_CONFIG,
}

# For CPU with limited memory
CPU_MINIMAL = {
    **DATA_CONFIG,
    **{
        "data_path": "../Influx_RBML_data.parquet",
        "sequence_length": 30,  # Shorter sequences
        "batch_size": 8,  # Smaller batch
    },
    **{
        **MODEL_CONFIG,
        "model_type": "gru",
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.1,
    },
    **{
        **TRAINING_CONFIG,
        "num_epochs": 30,
        "learning_rate": 1e-4,
    },
    **ANOMALY_CONFIG,
}

# ============ USAGE EXAMPLES ============

if __name__ == "__main__":
    """
    USAGE:

    1. Edit the configuration above

    2. Run with configuration:
       python -c "from config import *; print(BEST_ACCURACY)"

    3. Or use in your code:
       from config import TRAINING_CONFIG, MODEL_CONFIG
       model = create_model(**MODEL_CONFIG)
       trainer = train(**TRAINING_CONFIG)

    4. Select preset:
       SELECTED_CONFIG = QUICK_SETUP  # or BEST_ACCURACY, GPU_OPTIMIZED, etc.
    """

    print("Default Configuration:")
    print(f"  Model: {MODEL_CONFIG['model_type'].upper()}")
    print(f"  Hidden size: {MODEL_CONFIG['hidden_size']}")
    print(f"  Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print()
    print("Available presets:")
    print("  - QUICK_SETUP: Fast iteration (GRU, small)")
    print("  - BEST_ACCURACY: Best performance (large LSTM)")
    print("  - GPU_OPTIMIZED: Maximum GPU utilization")
    print("  - CPU_MINIMAL: Limited memory optimization")
