# RunPod Training Checklist

Quick reference for running experiments on RunPod GPU.

---

## Pre-Flight Checklist

Before uploading to RunPod, verify locally:

- [x] Data downloaded: `data/raw/Images/` exists
- [x] Data preprocessed: `artifacts/train_metadata.csv` exists
- [x] Configs ready: `configs/exp_*.yaml` files exist
- [x] Bootstrap script: `bootstrap_runpod.sh` is executable

---

## RunPod Setup (First Time)

### 1. Create Instance

- **GPU**: RTX 3090/4090 or A4000
- **Template**: PyTorch 2.1 or TensorFlow 2.15
- **Disk**: 50GB minimum
- **Cost**: ~$0.40-0.50/hour

### 2. Upload Project

```bash
# Option A: Git clone (if you have a repo)
git clone https://github.com/yourusername/dog_breed_classification.git
cd dog_breed_classification

# Option B: Upload via RunPod file browser
# Upload entire project folder
```

### 3. Run Bootstrap

```bash
bash bootstrap_runpod.sh
```

**Verify output:**
- ✓ GPU detected
- ✓ TensorFlow version: 2.15.0+
- ✓ TensorFlow can access GPU
- ✓ All dependencies installed

---

## Data Preparation

### Option A: Download on RunPod (Recommended)

```bash
PYTHONPATH=src python3 -m dbc.ingest       # ~10 min
PYTHONPATH=src python3 -m dbc.preprocess   # ~3 min
```

### Option B: Upload Pre-processed Data

If you already ran preprocessing locally:

1. Zip locally: `zip -r data.zip data/ artifacts/`
2. Upload `data.zip` to RunPod
3. Extract: `unzip data.zip`

**Saves:** ~13 minutes

---

## Running Experiments

### Quick Start: All Three Models

```bash
./run_all_experiments.sh
```

**Runtime:** ~60-80 minutes
**Output:** All three models + comparison report

### Manual: One Model at a Time

```bash
export PYTHONPATH=src

# ResNet50 (~20-25 min)
python3 -m dbc.train_cnn configs/exp_resnet50.yaml

# VGG16 (~20-25 min)
python3 -m dbc.train_cnn configs/exp_vgg16.yaml

# EfficientNetB0 (~15-20 min)
python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
```

### Monitor Progress

```bash
# In separate terminal/screen
watch -n 1 nvidia-smi
```

**GPU should show:**
- Utilization: 80-100%
- Memory: 8-12GB
- Temperature: 60-80°C

---

## After Training

### 1. Compare Results

```bash
PYTHONPATH=src python3 -m dbc.experiments compare
```

**Expected output:**
```
🏆 Best Model: EfficientNetB0 Transfer Learning
   Validation Accuracy: 0.8234
```

### 2. Download Best Model

**Via Web UI:**
1. Navigate to `experiments/`
2. Find best experiment (e.g., `exp_20251006_171122/`)
3. Download `checkpoints/best_model.keras`

**Via Command Line:**
```bash
# Zip results
zip -r results.zip experiments/

# Download via SCP (from local machine)
scp runpod_user@runpod_ip:/workspace/dog_breed_classification/results.zip ./
```

### 3. Terminate Instance

**Important:** Stop instance to avoid charges!

1. Download all needed files
2. Click "Stop Pod" in RunPod dashboard
3. Verify instance stopped

---

## Expected Results

### Training Metrics

| Model | Accuracy | Top-5 | Params | Time |
|-------|----------|-------|--------|------|
| ResNet50 | 75-82% | 93-96% | 24.8M | 20-25 min |
| VGG16 | 72-79% | 91-94% | 16.8M | 20-25 min |
| EfficientNetB0 | **78-85%** | **94-97%** | 5.8M | 15-20 min |

### File Outputs

```
experiments/
├── experiments_registry.json
├── experiments_report.md          ← Comparison report
├── exp_YYYYMMDD_HHMMSS/          ← ResNet50
│   ├── checkpoints/
│   │   └── best_model.keras      ← Download this
│   ├── config.json
│   ├── final_metrics.json
│   └── training_history.json
├── exp_YYYYMMDD_HHMMSS/          ← VGG16
└── exp_YYYYMMDD_HHMMSS/          ← EfficientNetB0
```

---

## Troubleshooting

### Problem: GPU not detected

**Solution:**
```bash
nvidia-smi  # Should show GPU info
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If no GPU, ensure GPU instance selected (not CPU).

### Problem: Out of Memory

**Solution:** Reduce batch size in configs:
```yaml
train:
  batch_size: 16  # or 8
```

### Problem: Can't download pre-trained weights

**Solution:** Manual download:
```bash
mkdir -p ~/.keras/models
cd ~/.keras/models

# ResNet50
curl -L -O "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16
curl -L -O "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# EfficientNetB0
curl -L -O "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
```

### Problem: Python module not found

**Solution:**
```bash
# Always use PYTHONPATH
export PYTHONPATH=src
python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

---

## Cost Summary

### Breakdown (RTX 3090 @ $0.40/hr)

| Task | Time | Cost |
|------|------|------|
| Data download | 10 min | $0.07 |
| Preprocessing | 3 min | $0.02 |
| ResNet50 | 22 min | $0.15 |
| VGG16 | 22 min | $0.15 |
| EfficientNetB0 | 18 min | $0.12 |
| **Total** | **75 min** | **$0.50** |

### Cost Optimization Tips

1. ✓ Pre-download data locally, upload to RunPod (saves $0.07)
2. ✓ Use spot instances if available (30-50% cheaper)
3. ✓ Run all experiments in one session
4. ✓ Download results immediately
5. ✓ Terminate instance as soon as done

**Minimum cost:** ~$0.40-0.60 for all three models

---

## Success Criteria

Training successful if:

- ✓ All 3 experiments completed without errors
- ✓ Validation accuracy: 70-85%
- ✓ Top-5 accuracy: >90%
- ✓ Training loss decreased over epochs
- ✓ No OOM errors
- ✓ Best model saved to `experiments/*/checkpoints/best_model.keras`

---

## Quick Commands Reference

```bash
# Setup
bash bootstrap_runpod.sh

# Data prep
PYTHONPATH=src python3 -m dbc.ingest
PYTHONPATH=src python3 -m dbc.preprocess

# Run all experiments
./run_all_experiments.sh

# Or individual
export PYTHONPATH=src
python3 -m dbc.train_cnn configs/exp_resnet50.yaml
python3 -m dbc.train_cnn configs/exp_vgg16.yaml
python3 -m dbc.train_cnn configs/exp_efficientnet.yaml

# Compare
python3 -m dbc.experiments compare
python3 -m dbc.experiments list
python3 -m dbc.experiments report

# Download
zip -r results.zip experiments/
```

---

## Timeline

| Minute | Task |
|--------|------|
| 0-5 | Create RunPod instance, upload project |
| 5-10 | Run bootstrap script |
| 10-20 | Download data |
| 20-25 | Preprocess data |
| 25-45 | Train ResNet50 |
| 45-65 | Train VGG16 |
| 65-80 | Train EfficientNetB0 |
| 80-85 | Compare results, download |
| **85** | **Terminate instance ✓** |

**Total: ~1.5 hours, ~$0.60**

---

## Next Steps After RunPod

1. **Identify best model** from comparison
2. **Fine-tune** if accuracy < 85%:
   - Create new config with `trainable_base: true`
   - Use lower learning rate: `0.0001`
   - Train for 10-15 more epochs
3. **Deploy model** to production
4. **Test on real-world images**

See [PIPELINE.md](PIPELINE.md) for deployment guide.
