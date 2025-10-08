"""
Preprocess images and save to numpy arrays for fast loading during training.
This eliminates the I/O bottleneck by preprocessing once and loading directly into memory.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import argparse
from tensorflow.keras.applications.imagenet_utils import preprocess_input as keras_preprocess_input


def preprocess_and_save_images(
    metadata_path: Path,
    data_root: Path,
    output_path: Path,
    image_size: tuple = (224, 224),
    normalize: str = 'imagenet',
    model_name: str = None
):
    """
    Preprocess all images and save as a single numpy array for fast loading.

    Args:
        metadata_path: Path to metadata CSV
        data_root: Root directory containing images
        output_path: Path to save preprocessed .npz file
        image_size: Target image size
        normalize: Normalization method
        model_name: Model name for model-specific preprocessing (resnet50, efficientnetb0, etc)
    """
    print(f"\nPreprocessing images from: {metadata_path}")
    print(f"Saving to: {output_path}")
    if model_name:
        print(f"Using model-specific preprocessing for: {model_name}")

    # Get model-specific preprocessing mode
    from .data_loader import get_model_preprocessing_config
    preprocess_mode = None
    if model_name:
        config = get_model_preprocessing_config(model_name)
        preprocess_mode = config['preprocess_mode']
        image_size = config['image_size']
        print(f"  Preprocessing mode: {config['description']}")

    # Load metadata
    df = pd.read_csv(metadata_path)
    n_images = len(df)

    # ImageNet normalization stats (only used if not using model-specific preprocessing)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    # Pre-allocate arrays
    images = np.zeros((n_images, image_size[1], image_size[0], 3), dtype=np.float32)
    labels = np.zeros(n_images, dtype=np.int32)

    print(f"Processing {n_images} images...")

    # Process each image
    for idx in range(n_images):
        if (idx + 1) % 1000 == 0 or (idx + 1) == n_images:
            print(f"  Processed {idx + 1}/{n_images} images...")
        row = df.iloc[idx]

        # Load image
        img_path = data_root / row['file_path']
        img = Image.open(img_path)

        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize(image_size, Image.BILINEAR)

        # Convert to array
        img_array = np.array(img, dtype=np.float32)

        # Apply model-specific preprocessing or standard normalization
        if preprocess_mode:
            # Use Keras model-specific preprocessing (caffe, tf, or torch mode)
            img_array = keras_preprocess_input(img_array, mode=preprocess_mode)
        elif normalize == 'imagenet':
            img_array = img_array / 255.0
            img_array = (img_array - imagenet_mean) / imagenet_std
        elif normalize == 'scale':
            img_array = img_array / 255.0

        images[idx] = img_array
        labels[idx] = row['class_id'] - 1  # Convert to 0-119

    # Save as separate .npy files (supports memory mapping for instant loading!)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to disk...")

    # Save images and labels separately
    images_path = output_path.parent / f"{output_path.stem}_images.npy"
    labels_path = output_path.parent / f"{output_path.stem}_labels.npy"

    np.save(images_path, images)
    np.save(labels_path, labels)

    images_size_mb = images_path.stat().st_size / (1024 * 1024)
    labels_size_mb = labels_path.stat().st_size / (1024 * 1024)
    total_size_mb = images_size_mb + labels_size_mb

    print(f"\n✓ Saved preprocessed data:")
    print(f"  Images: {images_path} ({images_size_mb:.1f} MB)")
    print(f"  Labels: {labels_path} ({labels_size_mb:.1f} MB)")
    print(f"  Total: {total_size_mb:.1f} MB")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Note: Saved as .npy files for instant memory-mapped loading")


def main():
    parser = argparse.ArgumentParser(description='Preprocess images for fast training')
    parser.add_argument('--data-root', type=str, default='data/raw',
                        help='Root directory containing images')
    parser.add_argument('--artifacts', type=str, default='artifacts',
                        help='Directory containing metadata CSVs')
    parser.add_argument('--output-dir', type=str, default='artifacts/preprocessed',
                        help='Directory to save preprocessed images')
    parser.add_argument('--image-size', type=int, nargs=2, default=[224, 224],
                        help='Image size (width height)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name for model-specific preprocessing (resnet50, efficientnetb0, efficientnetb4)')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    artifacts_dir = Path(args.artifacts)
    output_dir = Path(args.output_dir)

    # Preprocess training data
    print("="*60)
    print("PREPROCESSING TRAINING DATA")
    print("="*60)
    preprocess_and_save_images(
        metadata_path=artifacts_dir / 'train_metadata.csv',
        data_root=data_root,
        output_path=output_dir / 'train_data.npz',
        image_size=tuple(args.image_size),
        model_name=args.model_name
    )

    # Preprocess validation data
    print("\n" + "="*60)
    print("PREPROCESSING VALIDATION DATA")
    print("="*60)
    preprocess_and_save_images(
        metadata_path=artifacts_dir / 'val_metadata.csv',
        data_root=data_root,
        output_path=output_dir / 'val_data.npz',
        image_size=tuple(args.image_size),
        model_name=args.model_name
    )

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("\nTo use preprocessed data, train with:")
    print("  PYTHONPATH=src python3 -m dbc.train_cnn configs/exp_efficientnet.yaml --use-preprocessed")


if __name__ == '__main__':
    main()
