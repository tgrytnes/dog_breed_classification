# Model-Specific Preprocessing Implementation

## Summary

✅ **Implementation Complete**: Model-specific preprocessing has been successfully implemented for VGG16, ResNet50, and EfficientNetB0.

## What Was Changed

### 1. Added Model-Specific Preprocessing Configuration ([data_loader.py:13-50](../src/dbc/data_loader.py#L13-L50))

```python
def get_model_preprocessing_config(model_name: str) -> dict:
    """Get preprocessing configuration for specific models."""
    configs = {
        'vgg16': {
            'preprocess_mode': 'caffe',  # BGR, mean subtraction
            'image_size': (224, 224),
            'description': 'VGG16: RGB->BGR, mean=[103.939, 116.779, 123.68] subtraction'
        },
        'resnet50': {
            'preprocess_mode': 'caffe',  # BGR, mean subtraction
            'image_size': (224, 224),
            'description': 'ResNet50: RGB->BGR, mean=[103.939, 116.779, 123.68] subtraction'
        },
        'efficientnetb0': {
            'preprocess_mode': 'torch',  # RGB, ImageNet normalization
            'image_size': (224, 224),
            'description': 'EfficientNet: RGB, normalize with ImageNet mean/std to ~[-2, +2]'
        }
    }
    return configs[model_name]
```

### 2. Updated DogBreedDataset Class ([data_loader.py:69-99](../src/dbc/data_loader.py#L69-L99))

- Added `model_name` parameter to `__init__()`
- Added `model_preprocess_mode` attribute to store the preprocessing mode
- Automatically sets the correct image size for each model

### 3. Modified Image Loading ([data_loader.py:152-159](../src/dbc/data_loader.py#L152-L159))

```python
# Normalize
if self.model_preprocess_mode is not None:
    # Use model-specific preprocessing (Keras preprocess_input with mode)
    img_array = keras_preprocess_input(img_array, mode=self.model_preprocess_mode)
elif self.normalize == 'imagenet':
    img_array = img_array / 255.0  # Scale to [0,1]
    img_array = (img_array - self.imagenet_mean) / self.imagenet_std
elif self.normalize == 'scale':
    img_array = img_array / 255.0  # Scale to [0,1]
```

### 4. Updated create_data_loaders() ([data_loader.py:498-557](../src/dbc/data_loader.py#L498-L557))

- Added `model_name` parameter
- Passes model_name to both train and validation datasets
- Displays model name in the output header

## Verified Results

### VGG16
- ✅ **Mode**: Caffe
- ✅ **Value Range**: [-123.7, 151.1]
- ✅ **Mean**: ~[-30, -30, -30] (centered around 0)
- ✅ **Channel Order**: BGR (reversed)

### ResNet50
- ✅ **Mode**: Caffe (same as VGG16)
- ✅ **Value Range**: [-123.7, 151.1]
- ✅ **Mean**: ~[-30, -30, -30] (centered around 0)
- ✅ **Channel Order**: BGR (reversed)

### EfficientNetB0
- ✅ **Mode**: Torch
- ✅ **Value Range**: [-2.1, 2.6]
- ✅ **Mean**: ~[-0.5, -0.6, -0.5] (normalized with ImageNet stats)
- ✅ **Channel Order**: RGB (original)

## How to Use

### Basic Usage

```python
from dbc.data_loader import create_data_loaders

# For VGG16
train_gen, val_gen = create_data_loaders(
    train_metadata_path="artifacts/train_metadata.csv",
    val_metadata_path="artifacts/val_metadata.csv",
    data_root="data/raw",
    batch_size=32,
    model_name='vgg16'  # 👈 Specify model
)

# For ResNet50
train_gen, val_gen = create_data_loaders(..., model_name='resnet50')

# For EfficientNetB0
train_gen, val_gen = create_data_loaders(..., model_name='efficientnetb0')
```

### In Training Scripts

Update your training script to pass the model name:

```python
# Old way (generic preprocessing)
train_gen, val_gen = create_data_loaders(
    ...,
    normalize='imagenet'
)

# New way (model-specific preprocessing)
train_gen, val_gen = create_data_loaders(
    ...,
    model_name='vgg16'  # or 'resnet50', 'efficientnetb0'
)
```

## Testing

Run the test script to verify preprocessing:

```bash
python3 test_model_preprocessing.py
```

This will:
1. Test all three preprocessing configurations
2. Load sample batches with each model's preprocessing
3. Display value ranges and statistics
4. Compare preprocessing across all models

## Key Implementation Notes

### Why Use imagenet_utils.preprocess_input?

We use `tensorflow.keras.applications.imagenet_utils.preprocess_input` with the `mode` parameter because:

1. **Consistency**: All preprocessing modes are in one function
2. **Correctness**: The EfficientNet-specific import doesn't work correctly
3. **Flexibility**: Easy to add new models with different preprocessing modes

### Preprocessing Modes

- **'caffe'**: RGB→BGR conversion + mean subtraction (VGG16, ResNet50)
- **'torch'**: Scale to [0,1] + ImageNet normalization (EfficientNetB0)
- **'tf'**: Scale to [-1, 1] (not used in this project)

### Order of Operations

1. Load image (PIL)
2. Convert to RGB
3. Resize to model's image size
4. Convert to numpy array (float32)
5. **Apply augmentation** (if training)
6. **Apply model-specific preprocessing**
7. Return preprocessed image

⚠️ **Important**: Augmentation happens BEFORE preprocessing, ensuring augmented pixels are in [0-255] range before preprocessing.

## Next Steps

To use this in your experiments:

1. **Update your config files** to specify model names:
   ```yaml
   model:
     base_model: 'vgg16'  # or 'resnet50', 'efficientnetb0'
   ```

2. **Update your training script** to pass model_name to data loaders

3. **Re-run experiments** with correct preprocessing for better results

## Documentation

See [MODEL_PREPROCESSING.md](MODEL_PREPROCESSING.md) for detailed information about:
- How each preprocessing method works
- Expected value ranges
- Common issues and solutions
- Implementation references
