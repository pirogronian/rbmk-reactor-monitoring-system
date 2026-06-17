# Code Structure & Explanation

This document explains every part of the RBMK anomaly detection system.

## 🏗️ System Architecture

```
DATA FLOW:
Raw Parquet File
    ↓
[dataset.py] - Load, normalize, create sliding windows
    ↓
PyTorch DataLoaders (batch processing)
    ↓
[model.py] - LSTM/GRU autoencoder architecture
    ↓
[train.py] - Training loop with validation & early stopping
    ↓
[main.py] - Orchestration, threshold selection, anomaly detection
    ↓
Results: trained model + anomaly predictions + visualizations
```

---

## 📦 Module Deep-Dives

### 1️⃣ dataset.py - Data Loading & Preprocessing

#### **TimeSeriesDataset class**
```python
def __init__(self, data: np.ndarray, sequence_length: int):
    self.data = data                    # All data points
    self.sequence_length = sequence_length  # Window size (e.g., 50)

def __getitem__(self, idx: int):
    # Create sliding window starting at index idx
    window = self.data[idx : idx + sequence_length]
    # Return (input, target) - both are the same for reconstruction
    return window, window
```

**Why sliding windows?**
- LSTM needs fixed-length sequences
- 1000 data points + 50-length window = 951 training samples
- Overlapping windows maximize data usage

**Example:**
```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Sequence length: 3

Window 0: [1, 2, 3]
Window 1: [2, 3, 4]
Window 2: [3, 4, 5]
... and so on
```

#### **RBMKDataLoader class - The Full Pipeline**

1. **Load data**: `pd.read_parquet(filepath)`
   - Reads binary parquet format (efficient storage)

2. **Select numeric columns**: Filters out timestamps, strings, etc.
   - Example: 50 columns → 10 numeric sensor readings

3. **Remove NaN values**: Cleans bad data points
   - Keeps only rows with all values present

4. **Normalize (StandardScaler)**:
   ```
   normalized = (original - mean) / std
   Result: mean ≈ 0, std ≈ 1
   
   Why? LSTM learns much better with normalized input
   ```

5. **Train/Val/Test split** (sequential, not random):
   ```
   First 70% → Training (learn normal behavior)
   Next 15% → Validation (tune hyperparameters, detect overfitting)
   Last 15% → Test (evaluate on unseen data)
   
   Sequential split preserves temporal order
   (random split would cause time leakage)
   ```

6. **Create DataLoaders** for efficient batching:
   ```
   Train: shuffle=True (randomize order to improve learning)
   Val/Test: shuffle=False (keep original order for analysis)
   ```

---

### 2️⃣ model.py - Neural Network Architecture

#### **LSTMAutoencoder - How It Works**

```
INPUT: [batch_size, sequence_length, num_features]
       [32, 50, 10]  ← 32 sequences, 50 timesteps, 10 sensors

         ↓ ENCODER LSTM ↓
Reads the entire 50-timestep window
Compresses to hidden state: [num_layers, batch, hidden_size]
                            [2, 32, 64]  ← Bottleneck!

         ↓ DECODER LSTM ↓
Given just the hidden state, tries to RECONSTRUCT the input
Generates: [batch, sequence_length, hidden_size]
           [32, 50, 64]

         ↓ Linear Layer ↓
Projects back to original feature space: [32, 50, 10]

OUTPUT: [batch, sequence_length, input_size] - Reconstructed sequence
```

#### **Key Architecture Decisions**

**Why LSTM?**
- **Regular RNN**: Struggles with long sequences (vanishing gradient problem)
- **LSTM**: Has memory cells that preserve information over 50+ timesteps
- **GRU**: Simpler LSTM variant, faster training, similar performance

**Dropout (0.2)**:
- Randomly zeros 20% of neurons during training
- Forces network to learn redundant features
- Prevents overfitting without needing more data

**Num_layers=2**:
- Layer 1: Learns low-level temporal patterns
- Layer 2: Learns high-level patterns from Layer 1's outputs
- 3+ layers usually unnecessary for this task

---

### 3️⃣ train.py - Training Loop

#### **Training Process (one epoch)**

```python
for batch in train_loader:
    # 1. FORWARD PASS
    reconstructed = model(batch)
    
    # 2. COMPUTE LOSS
    loss = MSE(original, reconstructed)
    # Example: original[0,0,0] = 1.5, reconstructed[0,0,0] = 1.3
    # Contribution to loss: (1.5 - 1.3)^2 = 0.04
    
    # 3. BACKWARD PASS
    loss.backward()  # Compute ∂loss/∂weights
    
    # 4. GRADIENT CLIPPING
    clip_grad_norm_(parameters, max_norm=1.0)  # Prevent exploding gradients
    
    # 5. WEIGHT UPDATE
    optimizer.step()  # θ = θ - lr * ∇loss
```

#### **Why MSE Loss?**

Mean Squared Error for autoencoder:
```
MSE = mean((original - reconstructed)^2)

Penalizes large errors quadratically:
- Error 0.1 → contributes 0.01
- Error 1.0 → contributes 1.00
- Error 10.0 → contributes 100.00

Result: Model heavily focuses on large reconstruction errors
        = Good for detecting anomalies (which have large errors)
```

#### **Early Stopping Logic**

```
Epoch 1:  val_loss = 0.050  ← Save model (best so far)
Epoch 2:  val_loss = 0.048  ← Save model (improved!)
Epoch 3:  val_loss = 0.049  ← No improvement, patience_counter++
Epoch 4:  val_loss = 0.050  ← No improvement, patience_counter++
...
Epoch 10: val_loss = 0.052  ← patience_counter reaches limit → STOP

Prevents training too long (saves time, reduces overfitting)
```

#### **Learning Rate Scheduler**

```python
ReduceLROnPlateau(
    mode='min',      # Minimize validation loss
    factor=0.5,      # Multiply LR by 0.5
    patience=5,      # Wait 5 epochs
)

Epoch 1-5:   LR = 1e-3  (fast learning)
             val_loss plateaus

Epoch 6:     val_loss still high → LR *= 0.5 = 5e-4
Epoch 7-11:  LR = 5e-4  (finer-grain learning)
             val_loss plateaus again

Epoch 12:    LR *= 0.5 = 2.5e-4
```

---

### 4️⃣ model.py - Anomaly Detector Wrapper

```python
model = AnomalyDetector(input_size=10, model_type='lstm')

# Training: model minimizes reconstruction error on NORMAL data only
# This learns what "normal" looks like

# Inference: given new sequence,
# compute reconstruction error
anomaly_score = MSE(original, model(original))

# Classification:
if anomaly_score > threshold:
    print("ANOMALY DETECTED!")
else:
    print("Normal behavior")
```

**Why unsupervised?**
- No need for labeled data (anomalies, normal)
- Just train on normal data
- Anything different = anomaly

---

### 5️⃣ main.py - Orchestration

#### **Training Mode**

```python
python main.py --mode train --model-type lstm --num-epochs 100

1. Load data with RBMKDataLoader
   └─ Creates train/val/test loaders

2. Initialize LSTM model
   └─ Instantiate with input_size determined from data

3. Create AnomalyTrainer
   └─ Sets up optimizer, loss function, scheduler

4. Train for up to 100 epochs
   └─ Saves best model when val_loss improves
   └─ Stops early if no improvement after 10 epochs

5. Load best model
   └─ Restore weights from best epoch

6. Compute anomaly threshold
   └─ Use 95th percentile of validation reconstruction errors
   └─ Means top 5% are flagged as anomalies

7. Detect anomalies on test set
   └─ Calculate reconstruction error for each test sequence
   └─ Flag if > threshold

8. Visualize
   └─ training_plot.png: loss over epochs
   └─ anomaly_scores.png: errors and threshold
```

#### **Threshold Selection**

```python
# On validation set, collect all reconstruction errors
errors = [0.001, 0.002, 0.003, ..., 0.5, 0.6]  # 1000 errors

# Calculate 95th percentile
threshold = np.percentile(errors, 95)  # e.g., 0.08

# Now on test set:
for sequence in test_set:
    error = MSE(sequence, model(sequence))
    if error > 0.08:
        print("ANOMALY")  # Top 5% are flagged
    else:
        print("Normal")
```

**Why percentile?**
- Robust to outliers
- Easy to adjust sensitivity (95 vs 90 vs 99)
- No assumptions about error distribution

---

### 6️⃣ Inference Mode

```python
python main.py --mode inference --checkpoint checkpoints/best_model.pth

1. Load pre-trained model
   └─ Restore weights from checkpoint

2. Skip training, jump to threshold computation
   └─ Use validation set from earlier training

3. Detect anomalies
   └─ Run on test set without modifying weights
```

---

## 🎯 Data Flow Example

**Input: 10 reactor parameters, 1000 timesteps**

```
Step 1 - Load & Normalize:
[1000, 10] array → StandardScaler → [1000, 10] normalized

Step 2 - Create windows (length=50):
[1000, 10] → [951, 50, 10]  (951 sequences)

Step 3 - Split 70/15/15:
Train: [665, 50, 10]  ← Learn normal behavior
Val:   [143, 50, 10]  ← Tune hyperparameters
Test:  [143, 50, 10]  ← Evaluate

Step 4 - Batch (batch_size=32):
Train batches: 665 sequences → 21 batches of 32

Step 5 - Forward pass (one batch):
[32, 50, 10] → LSTM Encoder → [2, 32, 64] bottleneck
            → LSTM Decoder → [32, 50, 64]
            → Linear layer → [32, 50, 10] reconstructed

Step 6 - Loss:
MSE([32,50,10] original, [32,50,10] reconstructed) = scalar

Step 7 - Backward pass & update

Repeat for all 21 batches = 1 epoch
Repeat for all epochs (early stop when no improvement)
```

---

## 🔍 Debugging Guide

**Problem: Model loss very high, not decreasing**

```
Symptoms:
- train_loss stays > 0.1
- val_loss oscillates wildly

Causes:
1. Learning rate too high → weights swing wildly
2. Learning rate too low → no learning
3. Data not normalized → large values confuse optimizer
4. Network too small → can't learn complex patterns

Solutions:
1. Try LR in [1e-4, 1e-3, 1e-2] range, see which helps
2. Check: is train_loss decreasing AT ALL? 
   - If yes, increase num_epochs
   - If no, learning rate is likely wrong
3. Print normalized data: min/max should be ~[-3, 3]
4. Increase hidden_size from 64 → 128
```

**Problem: Overfitting (train_loss low, val_loss high)**

```
Symptoms:
train_loss: 0.001
val_loss:   0.1

Causes:
- Model too powerful for the data
- Training too long

Solutions:
1. Increase dropout: 0.2 → 0.3 or 0.4
2. Decrease hidden_size: 128 → 64
3. Decrease num_layers: 3 → 2
4. Decrease learning_rate (slower training = more time to generalize)
5. Add L2 regularization: weight_decay 1e-5 → 1e-4
```

---

## 📊 Expected Metrics

**Normal training:**
```
Epoch 1:   Train: 0.050, Val: 0.048
Epoch 10:  Train: 0.010, Val: 0.012  ← Both decreasing
Epoch 50:  Train: 0.005, Val: 0.006  ← Converged

Early stop at epoch 65 (no improvement for 10 epochs)
Best val loss: 0.005
```

**Anomaly detection:**
```
Reconstruction errors on validation (normal data):
Mean: 0.006
Max:  0.015
95th percentile: 0.010  ← Threshold

Test set results:
Total sequences: 143
Anomalies detected: 7 (4.9%)
Normal: 136 (95.1%)
```

---

## 🚀 Performance Tips

**GPU vs CPU:**
- GPU (CUDA): 10-100x faster for training
- Docker has GPU support with CUDA 12.6

**Reduce training time:**
1. Use GRU instead of LSTM (fewer parameters)
2. Reduce hidden_size (64 → 32)
3. Reduce sequence_length (50 → 30)
4. Reduce num_layers (2 → 1)

**Improve accuracy:**
1. Use LSTM instead of GRU
2. Increase hidden_size (64 → 128)
3. Increase num_layers (2 → 3)
4. More training data
5. Fine-tune learning_rate
