# Running Model Comparison Experiments on Cloud GPU

This guide shows how to run all three baseline models (ResNet50, VGG16, EfficientNetB0) on Google Colab and compare results.

## Quick Start - Google Colab

### 1. Upload Project to Google Drive

Upload the entire `dog_breed_classification` folder to your Google Drive:
```
Google Drive/
└── dog_breed_classification/
    ├── src/
    ├── configs/
    ├── data/raw/          # Optional: upload if already downloaded
    └── artifacts/         # Optional: upload if already preprocessed
```

### 2. Open Google Colab

Create a new notebook or use this template:

```python
# ==============================================================================
# DOG BREED CLASSIFICATION - MODEL COMPARISON
# Run all three baseline models: ResNet50, VGG16, EfficientNetB0
# ==============================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
import os
os.chdir('/content/drive/MyDrive/dog_breed_classification')

# Verify GPU is available
!nvidia-smi

# Install dependencies (if needed)
# Most should already be available in Colab
# !pip install tensorflow pandas pyyaml pillow

# ==============================================================================
# STEP 1: Data Setup (if not already done)
# ==============================================================================

# Option A: Download data directly on Colab (recommended)
!PYTHONPATH=src python3 -m dbc.ingest

# Option B: Skip if you already uploaded preprocessed data to Drive

# ==============================================================================
# STEP 2: Preprocess Data (if not already done)
# ==============================================================================

!PYTHONPATH=src python3 -m dbc.preprocess

# Verify preprocessing
!ls -lh artifacts/*.csv

# ==============================================================================
# STEP 3: Run Experiments
# ==============================================================================

print("\n" + "="*80)
print("EXPERIMENT 1/3: ResNet50 Transfer Learning")
print("="*80)
!PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml

print("\n" + "="*80)
print("EXPERIMENT 2/3: VGG16 Transfer Learning")
print("="*80)
!PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml

print("\n" + "="*80)
print("EXPERIMENT 3/3: EfficientNetB0 Transfer Learning")
print("="*80)
!PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml

# ==============================================================================
# STEP 4: Compare Results
# ==============================================================================

print("\n" + "="*80)
print("EXPERIMENT COMPARISON")
print("="*80)

# List all experiments
!PYTHONPATH=src python3 -m dbc.experiments list

# Compare experiments
!PYTHONPATH=src python3 -m dbc.experiments compare

# Generate report
!PYTHONPATH=src python3 -m dbc.experiments report

# View report
with open('experiments/experiments_report.md', 'r') as f:
    print(f.read())

# ==============================================================================
# STEP 5: Download Best Model
# ==============================================================================

# The comparison above will show the best model
# Download the best model checkpoint to your local machine

# Example: Download ResNet50 model (replace with best experiment ID)
from google.colab import files
# files.download('experiments/exp_YYYYMMDD_HHMMSS/checkpoints/best_model.keras')

print("\n✅ All experiments complete! Check the comparison above to see the best model.")
```

### 3. Run the Notebook

Click **Runtime → Run all** to execute all three experiments sequentially.

**Expected Runtime:**
- Data download: ~5-10 minutes (if not pre-uploaded)
- Preprocessing: ~2-3 minutes (if not pre-done)
- ResNet50 training: ~20-25 minutes
- VGG16 training: ~20-25 minutes
- EfficientNetB0 training: ~15-20 minutes
- **Total: ~60-80 minutes**

---

## Individual Experiment Commands

If you prefer to run experiments separately:

### ResNet50
```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_resnet50.yaml
```

### VGG16
```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_vgg16.yaml
```

### EfficientNetB0
```bash
PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml
```

---

## Experiment Configurations

All three experiments use identical training settings for fair comparison:

| Setting | Value |
|---------|-------|
| Epochs | 20 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Dropout | 0.5 |
| Base Model | Frozen (trainable=false) |
| Early Stopping | Enabled (patience=5) |
| Image Size | 224×224 |

**Only difference:** Base model architecture (ResNet50 vs VGG16 vs EfficientNetB0)

---

## Expected Results

Based on Stanford Dogs Dataset benchmarks:

| Model | Params (Total) | Params (Trainable) | Expected Val Accuracy | Top-5 Accuracy |
|-------|----------------|--------------------|-----------------------|----------------|
| **ResNet50** | 24.8M | 1.2M | 75-82% | 93-96% |
| **VGG16** | 16.8M | 2.4M | 72-79% | 91-94% |
| **EfficientNetB0** | 5.8M | 1.3M | 78-85% | 94-97% |

**EfficientNetB0 is expected to perform best** (modern architecture, better parameter efficiency).

---

## After Training - Compare Results

### List All Experiments
```bash
PYTHONPATH=src python3 -m dbc.experiments list
```

Output:
```
Found 3 experiment(s):

ID                        Name                           Status       Tags
------------------------------------------------------------------------------------------
exp_20251006_140530       ResNet50 Transfer Learning     completed    resnet50, transfer
exp_20251006_153045       VGG16 Transfer Learning        completed    vgg16, transfer
exp_20251006_171122       EfficientNetB0 Transfer...     completed    efficientnet, transfer
```

### Compare Experiments
```bash
PYTHONPATH=src python3 -m dbc.experiments compare
```

Output:
```
Experiment Comparison:

experiment_id          name                          val_accuracy  val_top5_accuracy  epochs_trained
exp_20251006_140530    ResNet50 Transfer Learning    0.7856        0.9412            18
exp_20251006_153045    VGG16 Transfer Learning       0.7523        0.9287            20
exp_20251006_171122    EfficientNetB0 Transfer...    0.8234        0.9589            17

🏆 Best Model: EfficientNetB0 Transfer Learning
   Validation Accuracy: 0.8234
   Top-5 Accuracy: 0.9589
```

### View Specific Experiment
```bash
PYTHONPATH=src python3 -m dbc.experiments show exp_20251006_171122
```

### Generate Report
```bash
PYTHONPATH=src python3 -m dbc.experiments report
```

Creates `experiments/experiments_report.md` with detailed comparison.

---

## Downloading Results from Colab

After training completes, download:

1. **Best Models:**
   ```python
   from google.colab import files

   # Download best EfficientNetB0 model (replace with your best experiment ID)
   files.download('experiments/exp_YYYYMMDD_HHMMSS/checkpoints/best_model.keras')
   ```

2. **Experiment Report:**
   ```python
   files.download('experiments/experiments_report.md')
   ```

3. **All Experiments (zip entire directory):**
   ```python
   !zip -r experiments.zip experiments/
   files.download('experiments.zip')
   ```

---

## Troubleshooting

### Out of Memory (OOM) on Colab

Reduce batch size in all configs:
```yaml
train:
  batch_size: 16  # or even 8
```

### Training Too Slow

Verify GPU is being used:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If no GPU, enable it: **Runtime → Change runtime type → GPU**

### Download Pre-trained Weights Fails

Manually download on Colab:
```bash
# ResNet50
!curl -L -o ~/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  "https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

# VGG16
!curl -L -o ~/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 \
  "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# EfficientNetB0
!curl -L -o ~/.keras/models/efficientnetb0_notop.h5 \
  "https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5"
```

---

## Next Steps After Comparison

1. **Select Best Model:** Based on comparison results (likely EfficientNetB0)

2. **Fine-tune Best Model:**
   - Unfreeze base layers: `trainable_base: true`
   - Lower learning rate: `learning_rate: 0.0001`
   - Train for 10-15 more epochs

3. **Hyperparameter Tuning:**
   - Experiment with dropout (0.3, 0.5, 0.7)
   - Try different learning rates (0.001, 0.0001, 0.00001)
   - Adjust batch size

4. **Deploy Best Model:**
   - Convert to TFLite for mobile
   - Create Flask/FastAPI inference service
   - Build web app for breed prediction

---

## File Structure After Training

```
dog_breed_classification/
├── experiments/
│   ├── experiments_registry.json           # Master registry
│   ├── experiments_report.md               # Comparison report
│   ├── exp_20251006_140530/               # ResNet50
│   │   ├── checkpoints/
│   │   │   ├── best_model.keras           # Best ResNet50 model
│   │   │   └── final_model.keras
│   │   ├── config.json
│   │   ├── final_metrics.json
│   │   ├── training_history.json
│   │   └── training_log.csv
│   ├── exp_20251006_153045/               # VGG16
│   │   └── [same structure]
│   └── exp_20251006_171122/               # EfficientNetB0
│       └── [same structure]
└── artifacts/
    ├── train_metadata.csv
    └── val_metadata.csv
```

---

## Summary

✅ All three models (ResNet50, VGG16, EfficientNetB0) are configured
✅ Identical training settings for fair comparison
✅ Automatic experiment tracking and comparison
✅ Expected best: EfficientNetB0 (78-85% accuracy)
✅ Total runtime: ~60-80 minutes on Colab GPU

Run the Colab notebook above to execute all experiments and find your best model!
