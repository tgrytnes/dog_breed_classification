# Dog Breed Classification - Complete Pipeline Guide

This guide covers the entire ML pipeline from data download to model training and comparison.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Step-by-Step Guide](#step-by-step-guide)
3. [Experiment Tracking](#experiment-tracking)
4. [Model Comparison](#model-comparison)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Cloud GPU Training](#cloud-gpu-training)

---

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ingest    │────▶│ Preprocess   │────▶│    Train     │────▶│  Evaluate   │
│   (data)    │     │  (clean)     │     │    (CNN)     │     │  (compare)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
      │                    │                     │                     │
      ▼                    ▼                     ▼                     ▼
  data/raw/          artifacts/           experiments/          experiments/
  Images/            train_metadata       exp_*/                 report.md
  Annotation/        val_metadata         checkpoints/
  breed_mapping                           metrics/
```

### Pipeline Stages

1. **Ingest**: Download Stanford Dogs Dataset (~750MB)
2. **Preprocess**: Validate, clean, and split images (train/val 80/20)
3. **Train**: Train CNN models with experiment tracking
4. **Compare**: Compare experiments and select best model

---

## Step-by-Step Guide

### 1. Download Dataset

```bash
PYTHONPATH=src python3 -m dbc.ingest
```

**What it does:**
- Downloads images, annotations, and lists from Stanford
- Extracts to `data/raw/`
- Creates `breed_mapping.csv` with 120 breeds

**Output:**
```
data/raw/
├── Images/              # 20,580 dog images
├── Annotation/          # Bounding box annotations
├── breed_mapping.csv    # Class ID → breed name mapping
└── [train/test]_list.mat
```

**Time:** ~5-10 minutes (depending on connection)

---

### 2. Preprocess Dataset

```bash
PYTHONPATH=src python3 -m dbc.preprocess
```

**What it does:**
- Scans all images and validates them (checks for corruption)
- Removes invalid/corrupt images
- Filters by size constraints (min 50px, max 10MB)
- Creates stratified train/val split (80/20)
- Computes dataset statistics

**Output:**
```
artifacts/
├── dataset_scan.csv     # Full scan results (all images)
├── dataset_stats.json   # Dataset statistics
├── train_metadata.csv   # Training set (16,508 images)
└── val_metadata.csv     # Validation set (4,072 images)
```

**Expected Results:**
- Total images: ~20,565 (after cleaning)
- Train: 16,508 images (80%)
- Val: 4,072 images (20%)
- All 120 breeds represented in both splits

**Time:** ~2-3 minutes

---

### 3. Verify with Smoke Test

```bash
PYTHONPATH=src python3 -m dbc.smoke
```

**What it does:**
- Tests data loading (breed mapping, directories)
- Tests image loading (sample 10 random images)
- Tests preprocessing artifacts
- Tests batch loading (4 images → 224×224)
- Checks breed distribution

**Time:** <30 seconds

---

### 4. Train Models

#### Option A: Quick Smoke Test (CPU, 2 minutes)

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_smoke_test.yaml
```

- 2 epochs, 80 training images, 40 validation images
- Verifies entire pipeline works before GPU training

#### Option B: CPU Test (5-10 minutes)

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_cpu_test.yaml
```

- 5 epochs, full dataset, batch size 16
- Suitable for testing on Mac Mini without GPU

#### Option C: Full Training (GPU recommended, 20-30 minutes)

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_baseline.yaml
```

- 20 epochs, full dataset, batch size 32
- Requires GPU (Google Colab, cloud instance)

**Training Output:**
```
experiments/
└── exp_YYYYMMDD_HHMMSS/
    ├── config.json              # Experiment configuration
    ├── training_history.json    # Loss/accuracy per epoch
    ├── final_metrics.json       # Final validation metrics
    ├── training_log.csv         # CSV log (for plotting)
    └── checkpoints/
        ├── best_model.keras     # Best model (by val_accuracy)
        └── final_model.keras    # Final epoch model
```

---

## Experiment Tracking

### Automatic Tracking

Every training run is automatically tracked with:

- **Unique ID**: `exp_YYYYMMDD_HHMMSS`
- **Configuration**: All hyperparameters saved
- **Metrics**: Loss, accuracy, top-5 accuracy per epoch
- **Artifacts**: Model checkpoints, training logs

### Experiment Metadata

Configure in YAML:

```yaml
# configs/my_experiment.yaml
experiment_name: "ResNet50 Baseline"
experiment_description: "Transfer learning with frozen ResNet50 base, 20 epochs"
experiment_tags: ["baseline", "resnet50", "transfer"]

paths:
  experiments: "experiments"  # Experiment storage directory

train:
  model_type: "transfer"
  base_model: "resnet50"
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
```

---

## Model Comparison

### List All Experiments

```bash
PYTHONPATH=src python3 -m dbc.experiments list
```

**Output:**
```
Found 3 experiment(s):

ID                        Name                           Status       Tags
------------------------------------------------------------------------------------------
exp_20251006_140530       ResNet50 Baseline              completed    baseline, resnet50
exp_20251006_153045       VGG16 Transfer                 completed    vgg16, transfer
exp_20251006_171122       EfficientNetB0                 completed    efficientnet
```

### Filter Experiments

```bash
# By status
PYTHONPATH=src python3 -m dbc.experiments list --status completed

# By tags
PYTHONPATH=src python3 -m dbc.experiments list --tags resnet50,baseline
```

### Compare Experiments

```bash
PYTHONPATH=src python3 -m dbc.experiments compare
```

**Output:**
```
Experiment Comparison:

experiment_id          name                model_type  base_model   val_accuracy  val_top5_accuracy
exp_20251006_140530    ResNet50 Baseline   transfer    resnet50     0.8234        0.9567
exp_20251006_153045    VGG16 Transfer      transfer    vgg16        0.8112        0.9501
exp_20251006_171122    EfficientNetB0      transfer    efficientnet 0.8456        0.9623

🏆 Best Model: EfficientNetB0
   Validation Accuracy: 0.8456
   Top-5 Accuracy: 0.9623
```

### Compare Specific Experiments

```bash
PYTHONPATH=src python3 -m dbc.experiments compare --ids exp_20251006_140530,exp_20251006_153045
```

### Show Experiment Details

```bash
PYTHONPATH=src python3 -m dbc.experiments show exp_20251006_140530
```

**Output:**
```
============================================================
Experiment: ResNet50 Baseline
============================================================
ID: exp_20251006_140530
Status: completed
Created: 2025-10-06T14:05:30
Tags: baseline, resnet50, transfer
Description: Transfer learning with frozen ResNet50 base, 20 epochs

Configuration:
  Model: transfer
  Base Model: resnet50
  Batch Size: 32
  Learning Rate: 0.001
  Epochs: 20

Final Results:
  Val Loss: 0.6543
  Val Accuracy: 0.8234
  Val Top-5 Accuracy: 0.9567
  Epochs Trained: 20
  Best Model: experiments/exp_20251006_140530/checkpoints/best_model.keras
```

### Generate Report

```bash
PYTHONPATH=src python3 -m dbc.experiments report
```

Creates `experiments/experiments_report.md` with:
- Summary of all experiments
- Comparison table
- Top 5 models by accuracy
- Detailed results for each experiment

---

## Hyperparameter Tuning

### Create Experiment Configs

Create multiple config files for different hyperparameters:

```yaml
# configs/exp_resnet50_lr001.yaml
experiment_name: "ResNet50 LR=0.001"
experiment_tags: ["resnet50", "lr_sweep"]
train:
  base_model: "resnet50"
  learning_rate: 0.001
  batch_size: 32
  epochs: 20

# configs/exp_resnet50_lr0001.yaml
experiment_name: "ResNet50 LR=0.0001"
experiment_tags: ["resnet50", "lr_sweep"]
train:
  base_model: "resnet50"
  learning_rate: 0.0001
  batch_size: 32
  epochs: 20
```

### Run Multiple Experiments

```bash
# Run experiments sequentially
for config in configs/exp_*.yaml; do
    PYTHONPATH=src python3 -m dbc.train_cnn "$config"
done
```

### Compare Results

```bash
# Compare all experiments with "lr_sweep" tag
PYTHONPATH=src python3 -m dbc.experiments list --tags lr_sweep
PYTHONPATH=src python3 -m dbc.experiments compare
```

### Hyperparameter Grid

Common hyperparameters to tune:

| Parameter | Options | Notes |
|-----------|---------|-------|
| `base_model` | resnet50, vgg16, efficientnetb0 | Architecture |
| `learning_rate` | 0.001, 0.0001, 0.00001 | Adam optimizer |
| `batch_size` | 16, 32, 64 | Memory permitting |
| `dropout` | 0.3, 0.5, 0.7 | Regularization |
| `trainable_base` | false, true | Fine-tune or freeze |
| `epochs` | 10, 20, 30 | With early stopping |

---

## Cloud GPU Training

### Google Colab Setup

1. **Upload files to Google Drive:**
   ```
   dog_breed_classification/
   ├── src/dbc/
   ├── configs/
   └── [artifacts/]  (optional, can preprocess locally)
   ```

2. **Colab notebook:**
   ```python
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')

   # Navigate to project
   %cd /content/drive/MyDrive/dog_breed_classification

   # Install dependencies (if needed)
   !pip install tensorflow pandas pyyaml pillow

   # Download data (or upload preprocessed artifacts)
   !PYTHONPATH=src python3 -m dbc.ingest
   !PYTHONPATH=src python3 -m dbc.preprocess

   # Train model
   !PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_baseline.yaml

   # Compare experiments
   !PYTHONPATH=src python3 -m dbc.experiments compare
   ```

3. **Download results:**
   - Best model: `experiments/exp_*/checkpoints/best_model.keras`
   - Metrics: `experiments/exp_*/final_metrics.json`
   - Report: `experiments/experiments_report.md`

### AWS/GCP Setup

```bash
# SSH into GPU instance
ssh gpu-instance

# Clone/upload project
scp -r dog_breed_classification gpu-instance:~/

# SSH and run
cd dog_breed_classification
PYTHONPATH=src python3 -m dbc.ingest
PYTHONPATH=src python3 -m dbc.preprocess
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_baseline.yaml

# Download results
scp -r gpu-instance:~/dog_breed_classification/experiments ./
```

---

## Expected Results

### Baseline Performance

Based on Stanford Dogs Dataset benchmarks:

| Model | Expected Val Accuracy | Top-5 Accuracy | Training Time (GPU) |
|-------|----------------------|----------------|---------------------|
| ResNet50 (frozen) | 75-82% | 93-96% | ~20 mins |
| VGG16 (frozen) | 72-79% | 91-94% | ~25 mins |
| EfficientNetB0 | 78-85% | 94-97% | ~15 mins |
| ResNet50 (fine-tuned) | 80-87% | 95-98% | ~40 mins |

### Dataset Challenges

This is a **difficult** dataset:
- 120 fine-grained classes (similar breeds)
- Variable image quality
- Multiple objects in images
- Complex backgrounds
- Class imbalance (62 breeds have <130 samples)

Achieving >80% accuracy is considered good performance.

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size: `batch_size: 16` or `8`
- Use smaller model: `efficientnetb0` instead of `resnet50`
- Use mixed precision training (advanced)

### Training Too Slow (CPU)

- Use `configs/cnn_cpu_test.yaml` for quick testing
- Move to GPU for full training (Google Colab free tier)

### Poor Validation Accuracy

- Check for data leakage (train/val split)
- Increase epochs or reduce learning rate
- Try data augmentation adjustments
- Consider fine-tuning base model: `trainable_base: true`

### Experiment Not Tracked

- Ensure `experiments` directory in config:
  ```yaml
  paths:
    experiments: "experiments"
  ```
- Check experiment registry: `experiments/experiments_registry.json`

---

## Next Steps

1. **Run baseline experiments:**
   - ResNet50, VGG16, EfficientNetB0
   - Compare results

2. **Hyperparameter tuning:**
   - Learning rate sweep
   - Batch size optimization
   - Dropout regularization

3. **Fine-tuning:**
   - Unfreeze base layers: `trainable_base: true`
   - Lower learning rate: `0.0001`

4. **Advanced techniques:**
   - Class weighting for imbalanced breeds
   - Test-time augmentation
   - Ensemble models

5. **Production deployment:**
   - Convert best model to TFLite
   - Create inference API
   - Build web/mobile app
