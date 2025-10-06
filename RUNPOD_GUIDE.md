# RunPod GPU Training Guide

Complete guide for running dog breed classification experiments on RunPod GPU instances.

---

## Quick Start

```bash
# 1. Clone/upload project to RunPod
# 2. Run bootstrap script
bash bootstrap_runpod.sh

# 3. Download and preprocess data
PYTHONPATH=src python3 -m dbc.ingest
PYTHONPATH=src python3 -m dbc.preprocess

# 4. Run all three experiments
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml

# 5. Compare results
PYTHONPATH=src python3 -m dbc.experiments compare
```

---

## Step-by-Step Setup

### 1. Create RunPod Instance

**Recommended Template:**
- **GPU**: RTX 3090, RTX 4090, or A4000 (1 GPU sufficient)
- **Template**: PyTorch 2.1 or TensorFlow 2.15 (both include CUDA)
- **Disk Space**: 50GB minimum (for dataset + models)
- **Region**: Any with low queue time

**Cost Estimate:**
- RTX 3090: ~$0.30-0.50/hour
- Total runtime: ~1-2 hours for all 3 experiments
- **Total cost**: ~$0.60-1.00

### 2. Upload Project Files

**Option A: Git Clone (Recommended)**

```bash
# SSH into RunPod instance
cd /workspace

# Clone your repository
git clone https://github.com/yourusername/dog_breed_classification.git
cd dog_breed_classification
```

**Option B: Direct Upload**

Use RunPod's file upload feature to upload:
```
dog_breed_classification/
├── src/
├── configs/
├── bootstrap_runpod.sh
└── (other files)
```

### 3. Run Bootstrap Script

```bash
# Make script executable (if not already)
chmod +x bootstrap_runpod.sh

# Run bootstrap
bash bootstrap_runpod.sh
```

**What the bootstrap script does:**
- ✓ Verifies GPU availability
- ✓ Checks Python version (3.10+)
- ✓ Installs TensorFlow with GPU support
- ✓ Installs Pillow (image processing)
- ✓ Installs pandas, numpy, scikit-learn, matplotlib
- ✓ Installs PyYAML, scipy
- ✓ Verifies TensorFlow can access GPU

**Expected output:**
```
==========================================
Dog Breed Classification - RunPod Setup
==========================================
✓ GPU detected:
NVIDIA GeForce RTX 3090, 24576 MiB
✓ Python version: 3.10.12

Installing Python dependencies...
==================================
...

✓ TensorFlow can access GPU
  - /device:GPU:0

✓ All dependencies installed successfully!
```

---

## Running Experiments

### Download Dataset

```bash
PYTHONPATH=src python3 -m dbc.ingest
```

**Runtime:** ~5-10 minutes
**Downloads:** ~750MB
**Output:** `data/raw/Images/`, `data/raw/breed_mapping.csv`

### Preprocess Dataset

```bash
PYTHONPATH=src python3 -m dbc.preprocess
```

**Runtime:** ~2-3 minutes
**Output:**
- `artifacts/train_metadata.csv` (16,508 images)
- `artifacts/val_metadata.csv` (4,072 images)
- `artifacts/dataset_stats.json`

### Run Model Experiments

#### Experiment 1: ResNet50

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

**Runtime:** ~20-25 minutes
**Expected Accuracy:** 75-82%

#### Experiment 2: VGG16

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
```

**Runtime:** ~20-25 minutes
**Expected Accuracy:** 72-79%

#### Experiment 3: EfficientNetB0

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
```

**Runtime:** ~15-20 minutes
**Expected Accuracy:** 78-85% (best)

---

## Monitoring Training

### Watch GPU Usage

```bash
# In a separate terminal/screen session
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization: Should be 80-100% during training
- Memory usage: ~8-12GB (depending on model)

### Monitor Training Progress

Training output shows:
```
Epoch 1/20
1/516 ━━━━━━━━━━━━━━━━━━ 2s 158ms/step - accuracy: 0.0312 - loss: 4.8523
...
Epoch 1: val_accuracy improved from -inf to 0.6234, saving model to experiments/exp_.../checkpoints/best_model.keras
```

**Key metrics to watch:**
- `loss`: Should decrease over epochs
- `accuracy`: Should increase (target: >75%)
- `val_accuracy`: Validation accuracy (main metric)
- `val_top5_accuracy`: Top-5 accuracy (should be >90%)

---

## After Training

### Compare Results

```bash
PYTHONPATH=src python3 -m dbc.experiments compare
```

**Output:**
```
Experiment Comparison:

experiment_id          name                          val_accuracy  val_top5_accuracy
exp_20251006_140530    ResNet50 Transfer Learning    0.7856        0.9412
exp_20251006_153045    VGG16 Transfer Learning       0.7523        0.9287
exp_20251006_171122    EfficientNetB0 Transfer...    0.8234        0.9589

🏆 Best Model: EfficientNetB0 Transfer Learning
   Validation Accuracy: 0.8234
   Top-5 Accuracy: 0.9589
```

### List Experiments

```bash
PYTHONPATH=src python3 -m dbc.experiments list
```

### Generate Report

```bash
PYTHONPATH=src python3 -m dbc.experiments report
```

Creates: `experiments/experiments_report.md`

---

## Download Results

### Option 1: Download via RunPod Web UI

Navigate to:
```
/workspace/dog_breed_classification/experiments/
```

Download:
- `exp_*/checkpoints/best_model.keras` (best models)
- `experiments_report.md` (comparison report)

### Option 2: Download via SCP

```bash
# From your local machine
scp -r runpod_user@runpod_ip:/workspace/dog_breed_classification/experiments ./
```

### Option 3: Zip and Download

```bash
# On RunPod
cd /workspace/dog_breed_classification
zip -r experiments.zip experiments/
# Download experiments.zip via web UI
```

---

## File Structure After Training

```
dog_breed_classification/
├── data/
│   └── raw/
│       ├── Images/              # 20,580 dog images
│       └── breed_mapping.csv
├── artifacts/
│   ├── train_metadata.csv
│   ├── val_metadata.csv
│   └── dataset_stats.json
├── experiments/
│   ├── experiments_registry.json
│   ├── experiments_report.md
│   ├── exp_YYYYMMDD_HHMMSS/    # ResNet50
│   │   ├── checkpoints/
│   │   │   ├── best_model.keras     # ← Download this
│   │   │   └── final_model.keras
│   │   ├── config.json
│   │   ├── final_metrics.json
│   │   └── training_history.json
│   ├── exp_YYYYMMDD_HHMMSS/    # VGG16
│   └── exp_YYYYMMDD_HHMMSS/    # EfficientNetB0
└── src/dbc/
```

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check TensorFlow GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Fix:** Ensure you selected a GPU instance, not CPU-only.

### Out of Memory (OOM)

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Fix:** Reduce batch size in configs:

```yaml
# Edit configs/exp_*.yaml
train:
  batch_size: 16  # Reduce from 32 to 16 or 8
```

### Slow Training

**Expected speeds:**
- ResNet50: ~150-200ms/step
- VGG16: ~150-200ms/step
- EfficientNetB0: ~100-150ms/step

If slower:
- Check GPU utilization: `nvidia-smi`
- Ensure you're on a GPU instance
- Check if other processes are using GPU

### Download Pre-trained Weights Fails

**Error:** `Unable to download ResNet50 weights`

**Fix:** Manually download weights:

```bash
# ResNet50
curl -L -o ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16
curl -L -o ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# EfficientNetB0
curl -L -o ~/.keras/models/efficientnetb0_notop.h5 \
  "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
```

### Python Path Issues

Always use `PYTHONPATH=src` before running modules:

```bash
# Wrong
python3 -m dbc.train_cnn configs/exp_resnet50.yaml

# Correct
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

Or export it once:
```bash
export PYTHONPATH=src
python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

---

## Cost Optimization

### Minimize Costs

1. **Pre-download data locally** → Upload to RunPod (saves 5-10 min)
2. **Use spot instances** if available (cheaper but can be interrupted)
3. **Run experiments sequentially** (can't parallelize on 1 GPU anyway)
4. **Download results immediately** after training
5. **Terminate instance** as soon as done

### Estimated Costs

| GPU Type | $/hour | Total Time | Total Cost |
|----------|--------|------------|------------|
| RTX 3090 | $0.40 | 1.5 hours | ~$0.60 |
| RTX 4090 | $0.50 | 1.0 hours | ~$0.50 |
| A4000 | $0.30 | 2.0 hours | ~$0.60 |

**Breakdown:**
- Data download + preprocess: 10-15 min
- ResNet50 training: 20-25 min
- VGG16 training: 20-25 min
- EfficientNetB0 training: 15-20 min
- **Total: 65-85 minutes**

---

## Advanced Usage

### Run Experiments in Background

Use `screen` or `tmux` to keep training running if SSH disconnects:

```bash
# Start screen session
screen -S training

# Run experiments
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml

# Detach: Ctrl+A, then D
# Reattach: screen -r training
```

### Run All Three Experiments Automatically

Create a script:

```bash
# run_all_experiments.sh
#!/bin/bash
export PYTHONPATH=src

echo "Running ResNet50..."
python3 -m dbc.train_cnn configs/exp_resnet50.yaml

echo "Running VGG16..."
python3 -m dbc.train_cnn configs/exp_vgg16.yaml

echo "Running EfficientNetB0..."
python3 -m dbc.train_cnn configs/exp_efficientnet.yaml

echo "Generating comparison report..."
python3 -m dbc.experiments compare
python3 -m dbc.experiments report

echo "All experiments complete!"
```

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### Monitor Training Remotely

Use `tensorboard` (if installed):

```bash
# In training script, add TensorBoard callback
# Then run:
tensorboard --logdir experiments/ --port 6006

# Access via RunPod's port forwarding
```

---

## Summary

✅ **Bootstrap script created**: `bootstrap_runpod.sh`
✅ **Installs all dependencies**: TensorFlow, Pillow, pandas, etc.
✅ **Verifies GPU**: Checks TensorFlow can access GPU
✅ **Ready for training**: Run 3 experiments in ~1-2 hours
✅ **Automatic tracking**: All results saved and comparable

**Total cost: ~$0.50-1.00** for complete model comparison on GPU!

---

## Next Steps

After running experiments on RunPod:

1. **Download best model** (likely EfficientNetB0)
2. **Fine-tune best model** with unfrozen layers (if accuracy < 85%)
3. **Deploy model** to production (Flask API, mobile app, etc.)
4. **Test on new images** to verify performance

See [PIPELINE.md](PIPELINE.md) for next steps after model selection.
