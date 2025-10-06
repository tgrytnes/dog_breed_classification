# Configuration Files Guide

This guide explains all available configuration files for training CNN models.

## Config File Overview

| Config File | Purpose | Model | Epochs | Batch Size | Runtime | Use Case |
|-------------|---------|-------|--------|------------|---------|----------|
| `cnn_smoke_test.yaml` | Quick verification | ResNet50 | 2 | 8 | 2-3 min | Test pipeline on CPU |
| `cnn_cpu_test.yaml` | CPU testing | ResNet50 | 5 | 16 | 5-10 min | Test on Mac without GPU |
| `cnn_baseline.yaml` | Legacy baseline | ResNet50 | 20 | 32 | 20-25 min | Original baseline config |
| `exp_resnet50.yaml` | **ResNet50 experiment** | ResNet50 | 20 | 32 | 20-25 min | **GPU comparison** |
| `exp_vgg16.yaml` | **VGG16 experiment** | VGG16 | 20 | 32 | 20-25 min | **GPU comparison** |
| `exp_efficientnet.yaml` | **EfficientNet experiment** | EfficientNetB0 | 20 | 32 | 15-20 min | **GPU comparison** |
| `exp_baseline.yaml` | Data preprocessing | N/A | N/A | N/A | N/A | Used by ingest/preprocess |

---

## Testing Configs (Before Cloud GPU)

### 1. cnn_smoke_test.yaml
**Purpose:** Verify entire pipeline works (data loading → training → saving)

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_smoke_test.yaml
```

**Settings:**
- Model: ResNet50 (frozen base)
- Epochs: 2 (minimal)
- Batch size: 8 (small)
- Training data: 10 batches = 80 images
- Validation data: 5 batches = 40 images

**Runtime:** 2-3 minutes on CPU (Mac Mini)

**Use when:** Before deploying to cloud GPU, verify everything works

---

### 2. cnn_cpu_test.yaml
**Purpose:** Test full dataset on CPU (slower but complete)

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_cpu_test.yaml
```

**Settings:**
- Model: ResNet50 (frozen base)
- Epochs: 5
- Batch size: 16 (reduced for CPU memory)
- Training data: Full dataset (16,508 images)
- Validation data: Full dataset (4,072 images)

**Runtime:** 5-10 minutes on CPU

**Use when:** Testing on Mac Mini without GPU access

---

## Cloud GPU Experiment Configs (Main Comparison)

These are the **three configs you should run on cloud GPU** to compare models.

### 3. exp_resnet50.yaml ⭐
**Purpose:** ResNet50 baseline experiment

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

**Settings:**
- Model: ResNet50 (50 layers, 23.6M frozen params + 1.2M trainable)
- Architecture: Deep residual network with skip connections
- Expected accuracy: **75-82%**
- Expected top-5: **93-96%**

**Strengths:**
- Industry standard baseline
- Good balance of accuracy and speed
- Proven architecture for image classification

**Weaknesses:**
- Larger model size (94MB)
- May overfit on limited data

---

### 4. exp_vgg16.yaml
**Purpose:** VGG16 comparison experiment

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
```

**Settings:**
- Model: VGG16 (16 layers, 14.7M frozen params + 2.4M trainable)
- Architecture: Simple stacked convolutional layers
- Expected accuracy: **72-79%**
- Expected top-5: **91-94%**

**Strengths:**
- Simple, interpretable architecture
- Well-studied model

**Weaknesses:**
- Lower accuracy than modern architectures
- More trainable parameters (slower training)
- Typically the weakest of the three

---

### 5. exp_efficientnet.yaml ⭐⭐⭐ (Recommended)
**Purpose:** EfficientNetB0 experiment - **likely best performance**

```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
```

**Settings:**
- Model: EfficientNetB0 (modern, efficient architecture)
- Parameters: 4.0M frozen + 1.3M trainable
- Expected accuracy: **78-85%** (highest)
- Expected top-5: **94-97%** (highest)

**Strengths:**
- Best expected performance
- Most parameter-efficient (smallest model: 20MB)
- Fastest training time (~15-20 min)
- State-of-the-art architecture (2019)

**Weaknesses:**
- None for this task (expected best model)

---

## Recommended Cloud GPU Workflow

### Step 1: Run All Three Experiments

```bash
# On Google Colab or cloud GPU instance
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
```

**Total runtime:** ~60-80 minutes

---

### Step 2: Compare Results

```bash
PYTHONPATH=src python3 -m dbc.experiments compare
```

Example output:
```
Experiment Comparison:

experiment_id          name                          val_accuracy  val_top5_accuracy
exp_20251006_140530    ResNet50 Transfer Learning    0.7856        0.9412
exp_20251006_153045    VGG16 Transfer Learning       0.7523        0.9287
exp_20251006_171122    EfficientNetB0 Transfer...    0.8234        0.9589  ← Best

🏆 Best Model: EfficientNetB0 Transfer Learning
```

---

### Step 3: Select Best Model

Based on comparison (likely EfficientNetB0), use that model for:
- Further fine-tuning
- Deployment to production
- Test set evaluation

---

## Common Settings Across All Experiment Configs

All three experiment configs use **identical settings** for fair comparison:

```yaml
train:
  batch_size: 32              # Same batch size
  epochs: 20                  # Same number of epochs
  learning_rate: 0.001        # Same learning rate
  optimizer: "adam"           # Same optimizer
  dropout: 0.5                # Same regularization
  trainable_base: false       # All freeze base model
  early_stopping: true        # All use early stopping
  patience: 5                 # Same patience
  image_size: [224, 224]      # Same image size
```

**Only difference:** `base_model` (resnet50 / vgg16 / efficientnetb0)

This ensures a **fair apples-to-apples comparison**.

---

## Modifying Configs for Hyperparameter Tuning

After finding the best model (likely EfficientNetB0), you can create variants:

### Example: Fine-tuning EfficientNetB0

Create `configs/exp_efficientnet_finetune.yaml`:

```yaml
experiment_name: "EfficientNetB0 Fine-tuned"
experiment_description: "Fine-tune EfficientNetB0 with unfrozen base layers"
experiment_tags: ["efficientnet", "fine-tuned"]

train:
  base_model: "efficientnetb0"
  trainable_base: true        # ← Unfreeze base
  learning_rate: 0.0001       # ← Lower LR for fine-tuning
  epochs: 15                  # ← Fewer epochs
  batch_size: 16              # ← Smaller batch (more memory needed)
```

### Example: Learning Rate Sweep

```yaml
# exp_efficientnet_lr0001.yaml
experiment_name: "EfficientNetB0 LR=0.0001"
train:
  learning_rate: 0.0001

# exp_efficientnet_lr00001.yaml
experiment_name: "EfficientNetB0 LR=0.00001"
train:
  learning_rate: 0.00001
```

---

## Config File Structure

All experiment configs follow this structure:

```yaml
# Experiment metadata (for tracking)
experiment_name: "Model Name"
experiment_description: "Description of experiment"
experiment_tags: ["tag1", "tag2"]

# Paths
paths:
  data: "data/raw"
  artifacts: "artifacts"
  experiments: "experiments"

# Training settings
train:
  # Model
  model_type: "transfer"  # or "custom"
  base_model: "resnet50"  # or "vgg16", "efficientnetb0"
  trainable_base: false

  # Training
  batch_size: 32
  epochs: 20
  learning_rate: 0.001
  optimizer: "adam"

  # Regularization
  dropout: 0.5

  # Early stopping
  early_stopping: true
  patience: 5

  # Data
  image_size: [224, 224]
  num_classes: 120
```

---

## Quick Reference - Which Config to Use?

| Situation | Config | Command |
|-----------|--------|---------|
| "Does the pipeline work?" | `cnn_smoke_test.yaml` | `PYTHONPATH=src python3 -m dbc.train_cnn configs/cnn_smoke_test.yaml` |
| "Test on my Mac" | `cnn_cpu_test.yaml` | Same as above |
| "Compare all models" | All three `exp_*.yaml` | See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) |
| "Best single model" | `exp_efficientnet.yaml` | `PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml` |

---

## Summary

✅ **Three experiment configs ready for cloud GPU comparison:**
- [configs/exp_resnet50.yaml](configs/exp_resnet50.yaml) - ResNet50 baseline
- [configs/exp_vgg16.yaml](configs/exp_vgg16.yaml) - VGG16 comparison
- [configs/exp_efficientnet.yaml](configs/exp_efficientnet.yaml) - **EfficientNetB0 (expected best)**

✅ **Identical training settings for fair comparison**

✅ **Expected winner: EfficientNetB0** (78-85% accuracy)

✅ **Total runtime: ~60-80 minutes** on Google Colab GPU

See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) for complete Google Colab instructions!
