# Model-Specific Image Preprocessing Guide

## Overview

Different pre-trained models expect images to be preprocessed in specific ways. Using the correct preprocessing for each model is **critical** for achieving good performance.

## Implemented Models

### VGG16
- **Image Size**: 224×224
- **Channel Order**: BGR (reversed from RGB)
- **Preprocessing**: Caffe-style
  - Mean subtraction: `[103.939, 116.779, 123.68]` (in BGR order)
  - No scaling - raw pixel values [0-255] before mean subtraction
  - Result range: approximately `[-123, +151]`
- **Usage**:
  ```python
  from tensorflow.keras.applications.vgg16 import preprocess_input
  preprocessed = preprocess_input(image_array)
  ```

### ResNet50
- **Image Size**: 224×224
- **Channel Order**: BGR (reversed from RGB)
- **Preprocessing**: Caffe-style (same as VGG16)
  - Mean subtraction: `[103.939, 116.779, 123.68]` (in BGR order)
  - No scaling - raw pixel values [0-255] before mean subtraction
  - Result range: approximately `[-123, +151]`
- **Usage**:
  ```python
  from tensorflow.keras.applications.resnet50 import preprocess_input
  preprocessed = preprocess_input(image_array)
  ```

### EfficientNetB0
- **Image Size**: 224×224
- **Channel Order**: RGB (original order)
- **Preprocessing**: Torch-style
  - Scale to [0, 1]: `pixel / 255.0`
  - Normalize: `(pixel - mean) / std`
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
  - Result range: approximately `[-2.1, +2.6]`
- **Usage**:
  ```python
  from tensorflow.keras.applications.efficientnet import preprocess_input
  preprocessed = preprocess_input(image_array)
  ```

## Using Model-Specific Preprocessing

### In Your Training Code

```python
from dbc.data_loader import create_data_loaders

# For VGG16
train_gen, val_gen = create_data_loaders(
    train_metadata_path="artifacts/train_metadata.csv",
    val_metadata_path="artifacts/val_metadata.csv",
    data_root="data/raw",
    batch_size=32,
    model_name='vgg16'  # Automatically applies VGG16 preprocessing
)

# For ResNet50
train_gen, val_gen = create_data_loaders(
    ...,
    model_name='resnet50'  # Automatically applies ResNet50 preprocessing
)

# For EfficientNetB0
train_gen, val_gen = create_data_loaders(
    ...,
    model_name='efficientnetb0'  # Automatically applies EfficientNet preprocessing
)
```

### Expected Value Ranges

After preprocessing, you should see these approximate ranges:

| Model | Range | Mean (approx) | Notes |
|-------|-------|---------------|-------|
| VGG16 | [-124, +151] | Around -30 per channel | Caffe-style, BGR |
| ResNet50 | [-124, +151] | Around -30 per channel | Caffe-style, BGR |
| EfficientNetB0 | [-2.1, +2.6] | Around 0 per channel | Torch-style, RGB |

## Why This Matters

Using incorrect preprocessing can lead to:
- **Poor accuracy**: The model sees input distribution different from training
- **Slow convergence**: The model must learn to adapt to wrong inputs
- **Training instability**: Gradient explosions or vanishing gradients

## Testing Your Preprocessing

Run the test script to verify preprocessing is working:

```bash
python3 test_model_preprocessing.py
```

This will:
1. Load sample images with each model's preprocessing
2. Display the value ranges and statistics
3. Verify that preprocessing matches expectations

## Implementation Details

The preprocessing is implemented in `src/dbc/data_loader.py`:

1. **get_model_preprocessing_config()**: Returns preprocessing configuration for each model
2. **DogBreedDataset**: Applies model-specific preprocessing if `model_name` is provided
3. **create_data_loaders()**: Creates data loaders with the correct preprocessing

### Order of Operations

1. Load image from disk (PIL)
2. Convert to RGB if needed
3. Resize to target size (224×224)
4. Convert to numpy array (float32)
5. Apply augmentation (if training)
6. **Apply model-specific preprocessing** (if model_name specified)
7. Return preprocessed image

## Common Issues

### Issue: EfficientNet shows raw [0-255] values
**Cause**: Preprocessing function not being called
**Fix**: Ensure `model_name='efficientnetb0'` is passed to `create_data_loaders()`

### Issue: VGG16/ResNet50 accuracy is poor
**Cause**: Using ImageNet normalization instead of Caffe preprocessing
**Fix**: Use `model_name='vgg16'` instead of `normalize='imagenet'`

### Issue: Different results between training and inference
**Cause**: Preprocessing mismatch between training and inference
**Fix**: Use the same `model_name` parameter for both training and inference data loaders

## References

- [Keras Applications Preprocessing](https://keras.io/api/applications/#usage-examples-for-image-classification-models)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
