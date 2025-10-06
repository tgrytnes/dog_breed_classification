# Dog Breed Classification - Setup Guide

This guide walks you through setting up the dog breed classification project.

## Project Structure

```
dog_breed_classification/
├── src/dbc/              # Source code (renamed from yourproj)
│   ├── config.py         # Configuration loader
│   ├── ingest.py         # Dataset download and ingestion
│   ├── preprocess.py     # Image validation, cleaning, and splitting
│   ├── smoke.py          # Smoke tests
│   ├── train.py          # Model training
│   ├── eval.py           # Model evaluation
│   └── utils.py          # Utility functions
├── data/raw/             # Raw dataset (downloaded)
├── artifacts/            # Processed data and results
├── configs/              # Configuration files
└── notebooks/            # Jupyter notebooks
```

## Step-by-Step Setup

### 1. Download the Dataset

Download the Stanford Dogs Dataset (~750MB):

```bash
python -m dbc.ingest
```

This will:
- Download images, annotations, and file lists from Stanford
- Extract to `data/raw/`
- Create `breed_mapping.csv` with 120 dog breeds
- Create `train_list.csv` and `test_list.csv` if available

**Expected output:**
```
Downloading Stanford Dogs Dataset...
✓ Download complete!
  Images: data/raw/Images
  Annotations: data/raw/Annotation
  Lists: data/raw/lists
✓ Created breed mapping with 120 breeds
```

### 2. Preprocess the Dataset

Scan, validate, clean, and split the images:

```bash
python -m dbc.preprocess
```

This will:
- Scan all images and validate them (check for corruption, size, format)
- Remove invalid/corrupt images
- Filter by size constraints (min 50px, max 10MB)
- Compute dataset statistics
- Create stratified train/validation split (80/20)
- Save metadata to `artifacts/`

**Expected output:**
```
SCANNING DATASET
Found 20,580 images across 120 breeds

CLEANING DATASET
  Initial images: 20,580
  Removed invalid/corrupt: 12
  Removed size violations: 3
  Final clean images: 20,565

DATASET STATISTICS
  Total images: 20,565
  Total breeds: 120
  Avg images/breed: 171.4
  Avg dimensions: 400x300

CREATING TRAIN/VAL SPLIT
  Train: 16,452 images (80.0%)
  Val: 4,113 images (20.0%)
```

**Artifacts created:**
- `artifacts/dataset_scan.csv` - Full scan results
- `artifacts/dataset_stats.json` - Dataset statistics
- `artifacts/train_metadata.csv` - Training set metadata
- `artifacts/val_metadata.csv` - Validation set metadata

### 3. Run Smoke Tests

Verify everything is working:

```bash
python -m dbc.smoke
```

This tests:
- ✓ Data loading (breed mapping, image directories)
- ✓ Image loading (sample 10 random images)
- ✓ Preprocessing artifacts (train/val splits)
- ✓ Batch loading (load and resize 4 images to 224x224)
- ✓ Breed distribution (check for class imbalance)

**Expected output:**
```
DOG BREED CLASSIFICATION - SMOKE TEST
============================================================

Testing data loading...
  ✓ Loaded 120 breeds from breed mapping
  ✓ Found 120 breed directories in Images/

Testing image loading with 10 random samples...
  ✓ n02085620_10074.jpg: 500x375, mode=RGB
  ...
  Loaded 10/10 images successfully

Testing batch loading (4 images)...
  ✓ Batch shape: (4, 224, 224, 3)
  ✓ Labels shape: (4,)
  ✓ Pixel value range: [0, 255]

SMOKE TEST SUMMARY
============================================================
  ✓ PASS: Data Loading
  ✓ PASS: Image Loading
  ✓ PASS: Preprocessing Artifacts
  ✓ PASS: Batch Loading
  ✓ PASS: Breed Distribution

  Total: 5/5 tests passed
  🎉 All smoke tests passed!
```

## Quick Start (All Steps)

Run all setup steps in sequence:

```bash
# 1. Download dataset
python -m dbc.ingest

# 2. Preprocess
python -m dbc.preprocess

# 3. Verify with smoke test
python -m dbc.smoke
```

## Dataset Information

### Stanford Dogs Dataset

- **Source**: http://vision.stanford.edu/aditya86/ImageNetDogs/
- **Classes**: 120 dog breeds
- **Images**: ~20,000 total
- **Size**: ~750MB compressed
- **Format**: JPEG images
- **License**: Research/academic use

### Dataset Challenges

This dataset presents several real-world challenges:

1. **Multiple Objects**: Many images contain humans, other animals, or various background objects alongside the dogs, which can complicate the classification task
2. **Variable Composition**: Dogs may occupy only a small portion of the image frame
3. **Background Clutter**: Complex backgrounds with furniture, outdoor scenery, or other distractions
4. **Occlusion**: Dogs may be partially hidden or cropped in some images

These challenges make the dataset more realistic but also more difficult, requiring the model to learn discriminative breed features despite visual noise.

### Breed Examples

The dataset includes breeds like:
- Chihuahua
- Japanese_spaniel
- Maltese_dog
- Pekinese
- Siberian_husky
- Alaskan_malamute
- Golden_retriever
- Labrador_retriever
- ... (120 total)

## Troubleshooting

### Download fails
If the download fails, you can manually download from:
- Images: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
- Annotations: http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

Extract to `data/raw/` and run `python -m dbc.ingest` again.

### Corrupt images
The preprocessing step automatically detects and removes corrupt images. Check `artifacts/dataset_scan.csv` for details.

### Memory issues
If you run out of memory during preprocessing, the validation runs per-image and doesn't load everything at once. Reduce batch size in smoke test if needed.

### 4. Test Data Pipeline (Optional)

Test the complete data loading and augmentation pipeline:

```bash
python -m dbc.data_loader
```

This creates data loaders with:
- Image resizing to 224×224
- ImageNet normalization
- Data augmentation (rotation, flip, zoom, brightness, contrast)
- Batch generation (32 samples per batch)

**Expected output:**
```
CREATING DATA LOADERS
============================================================

Train dataset:
  Images: 16,508
  Classes: 120
  Image size: (224, 224)
  Normalization: imagenet
  Augmentation: True

Validation dataset:
  Images: 4,072
  Classes: 120
  Augmentation: False

Testing batch loading...
Batch shape: (4, 224, 224, 3), Labels shape: (4,)
Image value range: [-2.118, 2.405]

✓ Data loader test passed!
```

## Data Pipeline Features

### Image Preprocessing
- **Resize**: All images resized to 224×224 (standard for ImageNet models)
- **RGB Conversion**: Automatic conversion of RGBA/grayscale to RGB
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Data Augmentation (Training Only)
- **Random Rotation**: ±15 degrees
- **Horizontal Flip**: 50% probability
- **Random Zoom**: 90-110% scale
- **Brightness**: ±20% adjustment
- **Contrast**: ±20% adjustment

**Why augment?** Multiplies effective training data from 16,508 → millions of variations, preventing overfitting and improving generalization.

## Next Steps

After successful setup:

1. **Explore the data** - Open `notebooks/main.ipynb` for EDA and data pipeline visualization
2. **Train a model** - Implement CNN architectures (ResNet50, VGG16, EfficientNet, custom CNN)
3. **Evaluate** - Use `src/dbc/eval.py` for model evaluation

## Configuration

Edit `configs/exp_baseline.yaml` to customize:

```yaml
seed: 42
paths:
  raw_data: data/raw
  artifacts: artifacts
```

You can create additional config files for different experiments.
