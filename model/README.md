# RBMK Reactor Anomaly Detection System

A PyTorch-based LSTM/GRU autoencoder for detecting anomalies in simulated RBMK reactor monitoring data.

## 📋 Overview

**What it does:**
- Learns normal reactor behavior patterns from historical data
- Detects deviations (anomalies) that might indicate issues
- Uses reconstruction-based approach: large reconstruction errors = anomaly

**How it works:**
1. Autoencoder compresses time-series sequences into latent features
2. Decoder reconstructs the original sequence
3. Normal behavior → low reconstruction error
4. Abnormal behavior → high reconstruction error

## 🚀 Quick Start

### Prerequisites

The Docker devcontainer already includes PyTorch and essential packages. If running locally:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn matplotlib numpy
```

### 1. Explore Your Data

First, understand the structure of your parquet file:

```bash
cd model
python explore_data.py
```

This will show:
- Number of samples and features
- Data types and missing values
- Statistical properties
- Column names (you'll need these!)

### 2. Train a Model

Train on your RBMK reactor data:

```bash
cd model
python main.py --mode train --model-type lstm --num-epochs 100
```

**Key arguments:**
- `--model-type`: `lstm` (more accurate, slower) or `gru` (faster, slightly less accurate)
- `--num-epochs`: More epochs = better but slower (start with 50-100)
- `--hidden-size`: Model capacity (64-256, higher = more powerful but slower)
- `--learning-rate`: Step size for gradient descent (1e-4 to 1e-2)
- `--batch-size`: Sequences per batch (16-64, higher = faster but more memory)

**Example with custom parameters:**

```bash
python main.py --mode train \
  --model-type lstm \
  --num-epochs 150 \
  --hidden-size 128 \
  --learning-rate 1e-3 \
  --batch-size 32 \
  --sequence-length 50
```

### 3. Detect Anomalies

Load a trained model and find anomalies:

```bash
python main.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --threshold-percentile 95.0
```

**What this does:**
- Loads the best trained model
- Computes reconstruction errors on validation set
- Uses 95th percentile as threshold (adjust if needed)
- Detects anomalies in test set
- Generates visualization

## 📁 Project Structure

```
model/
├── main.py                 # Entry point - orchestrates pipeline
├── explore_data.py         # Data exploration script
├── src/
│   ├── dataset.py         # Data loading & preprocessing
│   ├── model.py           # LSTM/GRU autoencoder architectures
│   ├── train.py           # Training loop & early stopping
│   └── __init__.py        # Package initialization
├── checkpoints/           # Saved models & training history
│   ├── best_model.pth     # Best model weights
│   ├── training_history.json
│   ├── training_plot.png
│   └── anomaly_scores.png
├── data/                  # Input/output data
└── .devcontainer/         # Docker configuration
```

## 🔧 Code Components Explained

### 1. **dataset.py** - Data Preparation

**RBMKDataLoader:**
- Loads parquet file
- Selects numeric features (reactor parameters)
- Normalizes using StandardScaler (mean=0, std=1)
- Creates sliding windows (e.g., 50 timesteps per window)
- Splits into train (70%) / val (15%) / test (15%)
- Creates PyTorch DataLoaders for efficient batch processing

**Why normalization?**
- LSTM learns faster with normalized input
- Prevents features with large ranges from dominating

**Why sliding windows?**
- LSTM needs fixed-length sequences
- Overlapping windows create more training samples

### 2. **model.py** - Neural Network Architecture

**LSTMAutoencoder:**
```
Input sequence (50 timesteps, 10 features)
         ↓
    ENCODER LSTM (2 layers, 64 hidden units)
         ↓
    Hidden state (compressed representation)
         ↓
    DECODER LSTM (2 layers, 64 hidden units)
         ↓
    Linear layer
         ↓
Reconstructed sequence (same shape as input)
```

**Why LSTM?**
- Remembers long-range dependencies
- Learns complex temporal patterns
- Good for time-series data

**GRU alternative:**
- Simpler than LSTM (fewer parameters)
- Faster training
- Often similar performance

### 3. **train.py** - Training Loop

**AnomalyTrainer:**
- **Forward pass**: Input → Autoencoder → Reconstruction
- **Loss**: MSE between input and reconstruction
- **Backward pass**: Compute gradients
- **Optimization**: Adam optimizer updates weights
- **Early stopping**: Stop if validation loss doesn't improve

**Key techniques:**
- **Gradient clipping**: Prevents exploding gradients (common in RNNs)
- **Learning rate scheduler**: Reduces LR when progress plateaus
- **Model checkpointing**: Saves best model based on validation loss

### 4. **main.py** - Orchestration

**Pipeline:**
1. Load and preprocess data
2. Create LSTM/GRU model
3. Train with early stopping
4. Compute anomaly threshold (95th percentile of validation errors)
5. Detect anomalies in test set
6. Generate visualizations

## 📊 Interpreting Results

### Training Plot (`training_plot.png`)
- **Training loss**: Should decrease smoothly
- **Validation loss**: Should decrease, may fluctuate
- **Gap between them**: Indicates overfitting (stop earlier if too wide)

### Anomaly Scores (`anomaly_scores.png`)
- **Top plot**: Time series of reconstruction errors
- **Red line**: Anomaly threshold
- **Red dots**: Detected anomalies
- **Bottom plot**: Histogram showing error distribution

### What the threshold means:
- Sequences with error > threshold = anomalies
- 95th percentile threshold = top 5% flagged as anomalies
- Adjust percentile based on acceptable false alarm rate

## 🎛️ Hyperparameter Tuning

### Model Capacity
| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| `hidden_size` | Model complexity | Start with 64, increase if underfitting |
| `num_layers` | Depth | 2-3 is usually optimal |
| `dropout` | Regularization | 0.1-0.3, increase if overfitting |

### Training
| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| `learning_rate` | Step size | 1e-3 is good starting point |
| `batch_size` | Samples per update | 32 is usually fine |
| `num_epochs` | Training duration | Use early stopping (typical: 50-200) |

### Anomaly Detection
| Parameter | Effect | Recommendation |
|-----------|--------|-----------------|
| `threshold_percentile` | Sensitivity | 90 = more alerts, 99 = fewer alerts |
| `sequence_length` | Context window | 20-100, higher = slower but more context |

## 🔍 Troubleshooting

**Model not learning (loss stays high):**
- Increase learning rate
- Increase hidden_size
- Check data normalization

**Overfitting (val loss increases):**
- Increase dropout
- Decrease hidden_size
- Decrease learning_rate
- Collect more training data

**Out of memory:**
- Decrease batch_size
- Decrease hidden_size
- Decrease sequence_length

**Too many false alarms:**
- Increase threshold_percentile (e.g., 98)
- Check that you're using training data without anomalies

## 📈 Expected Performance

On typical reactor data:
- Training loss: ~1e-3 to 1e-2
- Validation loss: ~1e-3 to 1e-2
- Training time: 5-30 minutes depending on hardware

If significantly different, check data preprocessing.

## 🚀 Running in Docker

The `.devcontainer` folder is configured with PyTorch:

```bash
# Open in VS Code with Dev Containers extension
# Or manually:
docker build -f .devcontainer/Dockerfile -t rbmk-ml .
docker run -it -v $(pwd):/workspace rbmk-ml bash
cd model && python main.py --mode train
```

## 📝 Next Steps

1. **Customize for your data**: Edit column names in `dataset.py` if needed
2. **Tune hyperparameters**: Experiment with hidden_size, learning_rate, etc.
3. **Validate on known issues**: Test with reactor data containing known anomalies
4. **Deploy**: Use trained model for real-time anomaly detection
5. **Monitor**: Track false alarm rate and retrain periodically

## 📚 Key Concepts

**Autoencoder**: Neural network that learns to compress and reconstruct data
- Unsupervised learning (no labels needed)
- Anomalies = sequences it can't reconstruct well

**LSTM (Long Short-Term Memory)**:
- Remembers important information over long sequences
- Forgets irrelevant information automatically
- Great for complex temporal patterns

**Reconstruction-based anomaly detection**:
- Train on normal behavior only
- Abnormal sequences won't compress/reconstruct well
- Simple, interpretable, and effective

**Threshold selection**:
- Percentile-based: straightforward and robust
- Adjust based on tolerance for false alarms vs missed anomalies

## 📞 Support

For issues or questions, check the code comments in each module for detailed explanations.
