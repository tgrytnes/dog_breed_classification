# ResNet50 Training Guide: Dog Breed Classification

## Overview

This document chronicles the ResNet50 training experiments for the Stanford Dogs Dataset (120 breeds). It covers both the **preprocessing failure from yesterday** and **today's successful fine-tuning** with selective layer unfreezing.

---

## Key Learnings

1. **Preprocessing Order Matters**: Augmentation must occur BEFORE model-specific preprocessing
2. **Preprocessed Data Creates Distribution Mismatch**: Using pre-saved preprocessed images breaks augmentation
3. **Selective Unfreezing Works Better**: Unfreezing last 50 layers (out of 175) achieved 77.68% accuracy
4. **On-the-Fly Loading is Essential**: Load raw images → augment → resize → preprocess in correct order

---

## Yesterday's Failure: Preprocessed Data Issue

### What We Did Wrong

We created a preprocessed dataset using ResNet50-specific preprocessing (Caffe mode):

```bash
PYTHONPATH=src python3 -m dbc.preprocess_images --model-name resnet50
```

This saved images with:
- ResNet50 Caffe mode preprocessing applied (BGR channel order, mean subtraction)
- Images already normalized and transformed

### Why It Failed

When we loaded these preprocessed images during training:

1. **Augmentation was applied to already-preprocessed images**
   - Rotation/flipping was applied to BGR-ordered, mean-subtracted pixel values
   - This created an invalid distribution that didn't match what ResNet50 expects

2. **Distribution Mismatch**
   - ResNet50 expects: Raw RGB → Augment → Convert to BGR → Subtract ImageNet mean
   - What we gave it: (BGR - mean) → Augment → No further preprocessing
   - The augmentation transformed the carefully calibrated preprocessed values

3. **Model Couldn't Learn**
   - Fine-tuning plateaued around 66-68% accuracy
   - Model was trying to learn from a corrupted feature space

### The Root Cause

**Correct preprocessing order:**
```
Raw Image [0-255 RGB]
  → Augmentation (rotation, flip, etc.)
  → Resize to 224×224
  → Caffe preprocessing (RGB→BGR, subtract mean)
```

**What we did (WRONG):**
```
Raw Image [0-255 RGB]
  → Resize to 224×224
  → Caffe preprocessing (RGB→BGR, subtract mean)
  → Save to disk
  → Load preprocessed image
  → Augmentation (operates on BGR-ordered, mean-subtracted values) ❌
```

---

## Today's Success: On-the-Fly Loading

### The Fix

Switched from preprocessed data to **on-the-fly loading** with correct preprocessing order:

**Configuration:** [configs/cnn_selective_finetune_v2.yaml](../configs/cnn_selective_finetune_v2.yaml)

```yaml
train:
  model_type: "transfer"
  base_model: "resnet50"

  # Load from frozen baseline checkpoint
  load_checkpoint: "experiments/exp_001/checkpoints/best_model.h5"

  # Selective unfreezing - unfreeze last 50 layers (out of 175 total)
  unfreeze_last_n: 50

  batch_size: 64
  epochs: 50
  learning_rate: 0.0001  # 10x lower than frozen baseline
  optimizer: "adam"
  dropout: 0.5

  image_size: [224, 224]
  num_classes: 120
```

### Data Loader Implementation

The [src/dbc/data_loader.py](../src/dbc/data_loader.py) implements correct preprocessing order:

```python
# ResNet50 configuration
'resnet50': {
    'preprocess_mode': 'caffe',  # BGR, ImageNet mean subtraction
    'image_size': (224, 224),
    'description': 'ResNet50: Caffe mode (BGR, mean=[103.939, 116.779, 123.68]), 224x224'
}

# In __getitem__ method:
def __getitem__(self, idx):
    # 1. Load raw image [0-255 RGB]
    img = Image.open(img_path).convert('RGB')

    # 2. Apply augmentation (if training)
    if self.augment:
        img = self.augmentation_pipeline(img)

    # 3. Resize
    img = img.resize(self.image_size, Image.Resampling.LANCZOS)

    # 4. Convert to array and apply model-specific preprocessing
    img_array = np.array(img, dtype=np.float32)

    # 5. ResNet50: Apply Caffe mode preprocessing
    if self.model_preprocess_mode == 'caffe':
        img_array = keras_preprocess_input(img_array, mode='caffe')
        # This converts RGB→BGR and subtracts ImageNet mean
```

---

## Training Results

### Phase 1: Frozen Baseline (Yesterday)

**Configuration:** [configs/cnn_baseline.yaml](../configs/cnn_baseline.yaml)

| Metric | Value |
|--------|-------|
| **Val Accuracy** | **54.69%** |
| Val Top-5 Accuracy | 79.68% |
| Val Loss | 1.7503 |
| Training Time | ~20 epochs |
| GPU | NVIDIA A100 |

**Analysis:**
- Reasonable baseline for frozen ResNet50
- Top-5 accuracy shows model learned meaningful features
- Ready for fine-tuning

### Phase 2: Selective Fine-Tuning (Today)

**Configuration:** [configs/cnn_selective_finetune_v2.yaml](../configs/cnn_selective_finetune_v2.yaml)

| Metric | Value |
|--------|-------|
| **Val Accuracy** | **77.68%** |
| Val Top-5 Accuracy | 94.43% |
| Val Loss | 0.8744 |
| Training Time | ~50 epochs |
| Improvement over Frozen | **+22.99%** |

**Key Settings:**
- Loaded frozen baseline checkpoint
- Unfroze last 50 layers (out of 175 total = 28.6%)
- Learning rate: 0.0001 (10x lower than frozen baseline)
- Batch size: 64
- On-the-fly data loading with correct preprocessing order

**Analysis:**
- Massive improvement from fixing preprocessing order
- Selective unfreezing (50 layers) prevented overfitting
- Top-5 accuracy of 94.43% shows excellent feature learning
- Model successfully learned dog breed distinctions

---

## Training Strategy: Selective Layer Unfreezing

### Why Selective Unfreezing?

Instead of unfreezing all 175 layers at once, we used **selective unfreezing**:

```python
# In train_cnn.py
unfreeze_last_n = config['train'].get('unfreeze_last_n', 0)

if unfreeze_last_n > 0:
    total_layers = len(base_model.layers)
    freeze_until = total_layers - unfreeze_last_n

    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= freeze_until)
```

**Benefits:**
1. **Prevents Catastrophic Forgetting**: Lower layers preserve ImageNet features
2. **Faster Training**: Fewer parameters to update
3. **Better Generalization**: Reduces overfitting risk
4. **Computational Efficiency**: Lower memory usage

**ResNet50 Architecture:**
- Total layers: 175
- Unfrozen: 50 layers (last 28.6%)
- Frozen: 125 layers (first 71.4%)

---

## Comparison: Preprocessed vs On-the-Fly

| Approach | Val Accuracy | Issue |
|----------|--------------|-------|
| **Preprocessed Data** (Yesterday) | ~66-68% (plateau) | Augmentation applied to preprocessed images |
| **On-the-Fly Loading** (Today) | **77.68%** | Correct preprocessing order ✓ |

**Time Trade-off:**
- Preprocessed: Faster epoch time (~30s/epoch) but wrong results
- On-the-fly: Slower epoch time (~60s/epoch) but correct results

**Conclusion:** Correctness > Speed. Always use on-the-fly loading for transfer learning.

---

## Experiment Registry

Both experiments are tracked in [experiments/experiments_registry.json](../experiments/experiments_registry.json):

```json
{
  "exp_001": {
    "name": "ResNet50 Baseline",
    "model": "resnet50",
    "best_val_accuracy": 0.5469,
    "status": "completed"
  },
  "exp_004": {
    "name": "ResNet50 Fine-Tuning v2 - Selective (50)",
    "model": "resnet50",
    "best_val_accuracy": 0.7768,
    "status": "completed"
  }
}
```

---

## Recommendations for Future Training

### ✅ Do This

1. **Always use on-the-fly loading** for transfer learning
2. **Apply augmentation to raw images** before preprocessing
3. **Use selective unfreezing** (25-30% of layers) for fine-tuning
4. **Lower learning rate** by 10x when fine-tuning
5. **Monitor first few epochs** to catch preprocessing issues early

### ❌ Avoid This

1. **Never preprocess and save images** when using augmentation
2. **Don't unfreeze all layers** at once (risk of catastrophic forgetting)
3. **Don't use same LR** for frozen vs fine-tuning phases
4. **Don't skip model-specific preprocessing** (ResNet50 needs Caffe mode)

---

## Technical Details

### ResNet50 Preprocessing Requirements

ResNet50 uses **Caffe mode** preprocessing:

```python
# Keras preprocessing
keras.applications.resnet50.preprocess_input(x, mode='caffe')
```

This performs:
1. **RGB → BGR** channel conversion
2. **Mean subtraction**: Subtract ImageNet mean per channel
   - Blue: 103.939
   - Green: 116.779
   - Red: 123.68
3. **No scaling**: Keeps values in range ~[-128, 128]

### Why Caffe Mode?

ResNet50 was originally trained in Caffe framework, which:
- Uses BGR channel order (OpenCV convention)
- Applies per-channel mean subtraction
- Does NOT divide by standard deviation

Using a different preprocessing mode (e.g., 'torch' or 'tf') would cause the model to fail.

---

## Conclusion

The ResNet50 experiments demonstrate a critical lesson in transfer learning:

> **Preprocessing order is not optional—it's fundamental to model performance.**

By switching from preprocessed data to on-the-fly loading, we:
- Fixed the distribution mismatch caused by augmenting preprocessed images
- Improved validation accuracy from ~68% (plateau) to **77.68%**
- Achieved 94.43% top-5 accuracy with selective layer unfreezing

**Final Result:** ResNet50 with selective fine-tuning achieved **77.68% accuracy** on 120 dog breeds, a strong baseline for comparison with EfficientNetB4 (93.12%).

---

## Files Referenced

- **Configs:**
  - [configs/cnn_baseline.yaml](../configs/cnn_baseline.yaml) - Frozen baseline
  - [configs/cnn_selective_finetune_v2.yaml](../configs/cnn_selective_finetune_v2.yaml) - Fine-tuning

- **Code:**
  - [src/dbc/data_loader.py](../src/dbc/data_loader.py) - Data loading and preprocessing
  - [src/dbc/train_cnn.py](../src/dbc/train_cnn.py) - Training script with selective unfreezing

- **Documentation:**
  - [docs/efficientnet_training_guide.md](./efficientnet_training_guide.md) - EfficientNetB4 guide
  - [experiments/experiments_registry.json](../experiments/experiments_registry.json) - Experiment tracking
