# Quick Reference - Getting Started

## 🚀 First Run (Recommended Steps)

### 1. Explore Your Data
```bash
cd model
python explore_data.py
```
**Output:** 
- Data shape, columns, types
- Missing values check
- Statistical summary

**Next:** Note down the numeric column names for reference

---

### 2. Train a Model
```bash
python main.py --mode train \
  --model-type lstm \
  --num-epochs 100 \
  --hidden-size 64 \
  --batch-size 32
```

**What happens:**
- Loads data (takes 30 seconds)
- Creates LSTM model
- Trains for up to 100 epochs with early stopping
- Saves best model to `checkpoints/best_model.pth`
- Generates visualizations

**Expected duration:** 5-30 minutes (depends on GPU/CPU)

**Output files:**
```
checkpoints/
├── best_model.pth              # Your trained model
├── training_history.json        # Loss over epochs
├── training_plot.png            # Loss visualization
└── anomaly_scores.png           # Detection results
```

---

### 3. Interpret Results

#### training_plot.png
- **Both losses decreasing?** → Good, model is learning
- **Validation loss increasing?** → Overfitting (increase dropout)
- **Losses flat?** → Learning rate too low or data too simple

#### anomaly_scores.png
- **Top plot:** Reconstruction errors with threshold line
- **Red dots:** Detected anomalies
- **Bottom plot:** Histogram of error distribution

---

### 4. Adjust Parameters Based on Results

**If underfitting (both losses high):**
```bash
python main.py --mode train \
  --hidden-size 128 \           # ← Increase
  --num-layers 3 \              # ← Increase
  --learning-rate 1e-3 \        # ← Try different value
  --num-epochs 150              # ← Train longer
```

**If overfitting (val_loss > train_loss):**
```bash
python main.py --mode train \
  --hidden-size 32 \            # ← Decrease
  --dropout 0.4 \               # ← Increase
  --learning-rate 5e-4 \        # ← Decrease
  --num-epochs 100
```

**For faster iteration:**
```bash
python main.py --mode train \
  --model-type gru \            # Faster than LSTM
  --hidden-size 32 \            # Smaller
  --num-epochs 50               # Fewer epochs
```

---

## 📋 Command Reference

### Basic Training
```bash
# Start here - LSTM with defaults
python main.py --mode train --num-epochs 100

# Faster alternative - GRU
python main.py --mode train --model-type gru --num-epochs 100
```

### Advanced Training
```bash
python main.py --mode train \
  --model-type lstm \
  --hidden-size 128 \
  --num-layers 3 \
  --dropout 0.3 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --sequence-length 50 \
  --num-epochs 200 \
  --early-stopping-patience 15
```

### Inference (on trained model)
```bash
# Use the trained model to detect anomalies
python main.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --threshold-percentile 95.0

# More sensitive (catch more anomalies)
python main.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --threshold-percentile 90.0

# Less sensitive (fewer false alarms)
python main.py --mode inference \
  --checkpoint checkpoints/best_model.pth \
  --threshold-percentile 99.0
```

### Explore Data
```bash
python explore_data.py
```

---

## 🎯 Common Workflows

### "I want to train and detect anomalies"
```bash
# 1. Train
python main.py --mode train --num-epochs 100

# 2. Detect (automatically uses trained model)
# Just run inference with same checkpoint
python main.py --mode inference
```

### "I want to experiment quickly"
```bash
# Use GRU for faster training
python main.py --mode train --model-type gru --num-epochs 50
```

### "I want the best possible accuracy"
```bash
# Use LSTM with larger model
python main.py --mode train \
  --model-type lstm \
  --hidden-size 256 \
  --num-layers 3 \
  --num-epochs 300 \
  --batch-size 16 \
  --learning-rate 5e-4
``` 
python main.py --mode train \
  --use-cpu \
  --model-type lstm \
  --hidden-size 256 \
  --num-layers 3 \
  --num-epochs 300 \
  --batch-size 16 \
  --learning-rate 5e-4

### "I have a GPU and want maximum speed"
```bash
# Large batch, fast iterations
python main.py --mode train \
  --batch-size 128 \
  --num-epochs 50 \
  --hidden-size 64
```

### "My model is overfitting"
```bash
python main.py --mode train \
  --hidden-size 32 \
  --dropout 0.5 \
  --learning-rate 1e-4 \
  --num-epochs 100
```

---

## 📊 Hyperparameter Guide

| Parameter | Try These Values | Effect |
|-----------|------------------|--------|
| `hidden-size` | 32, 64, 128, 256 | Larger = more capacity but slower |
| `num-layers` | 1, 2, 3 | Deeper = better features but harder to train |
| `dropout` | 0.1, 0.2, 0.3, 0.4 | Higher = prevent overfitting |
| `learning-rate` | 1e-4, 5e-4, 1e-3, 5e-3 | Higher = faster but unstable |
| `batch-size` | 16, 32, 64, 128 | Larger = faster but more memory |
| `sequence-length` | 20, 50, 100 | Longer = more context but slower |
| `num-epochs` | 50, 100, 200 | Early stopping will end early |
| `threshold-percentile` | 90, 95, 99 | Lower = more anomalies detected |

---

## 🔧 Docker Usage

### Open in VS Code with Dev Containers
1. Install "Dev Containers" extension
2. Click button in lower left
3. Select "Reopen in Container"
4. Run commands as normal

### Manual Docker
```bash
# Build
docker build -f .devcontainer/devcontainer.json -t rbmk-ml .

# Run
docker run -it -v $(pwd):/workspace rbmk-ml bash
cd model && python main.py --mode train
```

---

## ⚡ Performance Tips

**GPU available but not using it?**
- Check: `nvidia-smi` command
- Docker needs `--gpus all` flag

**Training too slow?**
1. Use GRU instead of LSTM
2. Reduce `hidden-size` to 32
3. Reduce `sequence-length` to 30
4. Reduce `batch-size` to 16
5. Increase `num-epochs` but let early stopping handle it

**Out of memory?**
1. Reduce `batch-size` (32 → 16 → 8)
2. Reduce `hidden-size` (128 → 64)
3. Reduce `sequence-length` (50 → 30)

**Model not learning?**
1. Print first few batches to check data
2. Try learning rate: 1e-4, 1e-3, 1e-2 range
3. Check normalized data is in [-3, 3] range
4. Increase `hidden-size` to 128

---

## 📈 Monitoring Training

Open a new terminal and run:
```bash
# Watch training_history.json update in real-time
watch -n 5 tail checkpoints/training_history.json

# Or with Python (better formatting)
python << 'EOF'
import json
import time
while True:
    try:
        with open('checkpoints/training_history.json') as f:
            h = json.load(f)
        print(f"Epoch: {h['epoch'][-1]}")
        print(f"Train: {h['train_loss'][-1]:.6f}")
        print(f"Val:   {h['val_loss'][-1]:.6f}")
        print(f"LR:    {h['learning_rate'][-1]:.2e}")
    except:
        pass
    time.sleep(5)
EOF
```

---

## ✅ Checklist Before Running

- [ ] Parquet file exists in parent directory
- [ ] Docker container running (or venv activated)
- [ ] Enough disk space for checkpoints (~100 MB)
- [ ] ~30 min free time (for first training run)

---

## 🆘 Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
- Install: `pip install torch`
- Or use Docker container

**"FileNotFoundError: ../Influx_RBML_data.parquet"**
- Check file path and spelling
- File should be in parent directory of model/

**"CUDA out of memory"**
- GPU memory full
- Reduce batch_size or hidden_size
- Or use CPU (slower but works)

**Model stopped early (early stopping triggered)**
- This is normal! Early stopping prevents overfitting
- Check if val_loss is still decreasing before it stopped
- If stopped too early, increase early_stopping_patience

**Training loss NaN or Inf**
- Exploding gradients (gradient clipping should help)
- Learning rate too high
- Try lower learning_rate: 1e-4 or 1e-5

---