from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
import shutil
from .config import load_config
from .utils import ensure_dir


def validate_image(image_path: Path) -> dict:
    """
    Validate an image file and return statistics.

    Returns:
        dict with validation results: {
            'valid': bool,
            'width': int,
            'height': int,
            'format': str,
            'mode': str,
            'size_kb': float,
            'error': str (if invalid)
        }
    """
    result = {
        'valid': False,
        'width': None,
        'height': None,
        'format': None,
        'mode': None,
        'size_kb': None,
        'error': None
    }

    try:
        # Check if file exists
        if not image_path.exists():
            result['error'] = 'File not found'
            return result

        # Get file size
        result['size_kb'] = image_path.stat().st_size / 1024

        # Try to open and validate image
        with Image.open(image_path) as img:
            result['width'], result['height'] = img.size
            result['format'] = img.format
            result['mode'] = img.mode

            # Check for corrupt images by loading pixel data
            img.verify()

        # Re-open to actually load the image (verify() closes the file)
        with Image.open(image_path) as img:
            img.load()

        result['valid'] = True

    except Exception as e:
        result['error'] = str(e)

    return result


def scan_dataset(images_dir: Path, breeds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan all images in the dataset and validate them.

    Args:
        images_dir: Path to Images directory
        breeds_df: DataFrame with breed mapping

    Returns:
        DataFrame with columns: file_path, breed_name, class_id, valid, width, height, etc.
    """
    print("Scanning dataset images...")

    all_images = []

    for _, breed_row in breeds_df.iterrows():
        breed_dir = images_dir / breed_row['breed_dir']

        if not breed_dir.exists():
            print(f"Warning: Breed directory not found: {breed_dir}")
            continue

        # Find all image files
        image_files = list(breed_dir.glob('*.jpg')) + \
                     list(breed_dir.glob('*.jpeg')) + \
                     list(breed_dir.glob('*.JPEG')) + \
                     list(breed_dir.glob('*.JPG'))

        for img_file in image_files:
            # Validate image
            validation = validate_image(img_file)

            # Create relative path from Images directory
            rel_path = img_file.relative_to(images_dir.parent)

            all_images.append({
                'file_path': str(rel_path),
                'absolute_path': str(img_file),
                'breed_name': breed_row['breed_name'],
                'breed_dir': breed_row['breed_dir'],
                'class_id': breed_row['class_id'],
                'filename': img_file.name,
                'valid': validation['valid'],
                'width': validation['width'],
                'height': validation['height'],
                'format': validation['format'],
                'mode': validation['mode'],
                'size_kb': validation['size_kb'],
                'error': validation['error']
            })

    df = pd.DataFrame(all_images)
    print(f"Found {len(df)} images across {breeds_df['breed_name'].nunique()} breeds")

    return df


def clean_dataset(
    images_df: pd.DataFrame,
    min_size: int = 50,
    max_size_kb: int = 10000,
    remove_grayscale: bool = False,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Clean dataset by removing invalid, corrupt, or problematic images.

    Args:
        images_df: DataFrame from scan_dataset
        min_size: Minimum width/height in pixels
        max_size_kb: Maximum file size in KB
        remove_grayscale: If True, remove non-RGB images (L, RGBA, etc.)
        remove_duplicates: If True, remove duplicate filenames

    Returns:
        Cleaned DataFrame with removed_reason column
    """
    initial_count = len(images_df)
    df = images_df.copy()

    # Add a column to track removal reasons
    df['removed_reason'] = None

    # 1. Remove invalid/corrupt images
    invalid_mask = df['valid'] == False
    df.loc[invalid_mask, 'removed_reason'] = df.loc[invalid_mask, 'error']
    df = df[df['valid'] == True].copy()
    invalid_count = initial_count - len(df)

    # 2. Remove duplicates (same filename)
    duplicate_count = 0
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=['filename'], keep='first')
        duplicate_count = before - len(df)
        if duplicate_count > 0:
            print(f"  Note: Removed {duplicate_count} duplicate filenames")

    # 3. Filter by size constraints
    size_mask = (
        (df['width'] < min_size) |
        (df['height'] < min_size) |
        (df['size_kb'] > max_size_kb)
    )
    size_filtered = df[~size_mask].copy()
    size_count = len(df) - len(size_filtered)

    # 4. Optionally remove grayscale/non-RGB images
    grayscale_count = 0
    if remove_grayscale:
        before = len(size_filtered)
        size_filtered = size_filtered[size_filtered['mode'] == 'RGB'].copy()
        grayscale_count = before - len(size_filtered)

    # 5. Check for extreme aspect ratios (potential issues)
    size_filtered['aspect_ratio'] = size_filtered['width'] / size_filtered['height']
    extreme_ratio = (
        (size_filtered['aspect_ratio'] < 0.2) |
        (size_filtered['aspect_ratio'] > 5.0)
    )
    extreme_count = extreme_ratio.sum()
    if extreme_count > 0:
        print(f"  Warning: {extreme_count} images with extreme aspect ratios (< 0.2 or > 5.0)")
        print(f"           These are kept but may need special handling")

    print("\nCleaning summary:")
    print(f"  Initial images: {initial_count}")
    print(f"  Removed invalid/corrupt: {invalid_count}")
    print(f"  Removed duplicates: {duplicate_count}")
    print(f"  Removed size violations: {size_count}")
    print(f"  Removed grayscale/non-RGB: {grayscale_count}")
    print(f"  Final clean images: {len(size_filtered)}")

    if invalid_count > 0:
        print(f"\nInvalid images found:")
        invalid_df = images_df[images_df['valid'] == False]
        for _, row in invalid_df.head(10).iterrows():
            print(f"  {row['file_path']}: {row['error']}")

    return size_filtered


def compute_dataset_statistics(images_df: pd.DataFrame) -> dict:
    """
    Compute statistics about the dataset.

    Returns:
        dict with statistics
    """
    stats = {
        'total_images': len(images_df),
        'total_breeds': images_df['breed_name'].nunique(),
        'images_per_breed': images_df.groupby('breed_name').size().to_dict(),
        'avg_images_per_breed': images_df.groupby('breed_name').size().mean(),
        'min_images_per_breed': images_df.groupby('breed_name').size().min(),
        'max_images_per_breed': images_df.groupby('breed_name').size().max(),
        'avg_width': images_df['width'].mean(),
        'avg_height': images_df['height'].mean(),
        'avg_size_kb': images_df['size_kb'].mean(),
        'image_modes': images_df['mode'].value_counts().to_dict(),
        'image_formats': images_df['format'].value_counts().to_dict()
    }

    return stats


def create_train_val_split(
    images_df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation split.

    Args:
        images_df: DataFrame with image information
        val_size: Fraction of data for validation
        seed: Random seed

    Returns:
        (train_df, val_df)
    """
    np.random.seed(seed)

    train_dfs = []
    val_dfs = []

    # Split per breed to ensure stratification
    for breed_name, group in images_df.groupby('breed_name'):
        n_val = max(1, int(len(group) * val_size))

        # Shuffle and split
        shuffled = group.sample(frac=1, random_state=seed)
        val_dfs.append(shuffled.iloc[:n_val])
        train_dfs.append(shuffled.iloc[n_val:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)

    print(f"\nTrain/Val split:")
    print(f"  Train: {len(train_df)} images ({len(train_df)/len(images_df)*100:.1f}%)")
    print(f"  Val: {len(val_df)} images ({len(val_df)/len(images_df)*100:.1f}%)")

    return train_df, val_df


def main(config_path: str = "configs/exp_baseline.yaml"):
    """Preprocess the dataset: scan, validate, clean, and split."""
    cfg = load_config(config_path)

    raw_data = Path(cfg['paths']['raw_data'])
    artifacts = Path(cfg['paths']['artifacts'])
    ensure_dir(artifacts)

    # Load breed mapping
    breeds_file = raw_data / "breed_mapping.csv"
    if not breeds_file.exists():
        print(f"Error: Breed mapping not found at {breeds_file}")
        print("Please run ingest.py first to download the dataset.")
        return

    breeds_df = pd.read_csv(breeds_file)
    images_dir = raw_data / "Images"

    # Scan dataset
    print("\n" + "="*60)
    print("SCANNING DATASET")
    print("="*60)
    images_df = scan_dataset(images_dir, breeds_df)

    # Save raw scan results
    scan_file = artifacts / "dataset_scan.csv"
    images_df.to_csv(scan_file, index=False)
    print(f"\n✓ Saved scan results to {scan_file}")

    # Clean dataset
    print("\n" + "="*60)
    print("CLEANING DATASET")
    print("="*60)
    clean_df = clean_dataset(
        images_df,
        min_size=50,
        max_size_kb=10000,
        remove_grayscale=False,  # Keep RGBA/grayscale, will convert during training
        remove_duplicates=True   # Remove duplicate filenames
    )

    # Compute statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    stats = compute_dataset_statistics(clean_df)
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total breeds: {stats['total_breeds']}")
    print(f"  Avg images/breed: {stats['avg_images_per_breed']:.1f}")
    print(f"  Min images/breed: {stats['min_images_per_breed']}")
    print(f"  Max images/breed: {stats['max_images_per_breed']}")
    print(f"  Avg dimensions: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")
    print(f"  Avg file size: {stats['avg_size_kb']:.1f} KB")
    print(f"  Image modes: {stats['image_modes']}")

    # Save statistics
    import json
    stats_file = artifacts / "dataset_stats.json"
    # Convert numpy types to Python types for JSON serialization
    stats_clean = {k: (int(v) if isinstance(v, np.integer) else
                      float(v) if isinstance(v, np.floating) else v)
                  for k, v in stats.items() if k not in ['images_per_breed']}
    with open(stats_file, 'w') as f:
        json.dump(stats_clean, f, indent=2)
    print(f"\n✓ Saved statistics to {stats_file}")

    # Create train/val split
    print("\n" + "="*60)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*60)
    train_df, val_df = create_train_val_split(clean_df, val_size=0.2, seed=cfg.get('seed', 42))

    # Save splits
    train_file = artifacts / "train_metadata.csv"
    val_file = artifacts / "val_metadata.csv"
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    print(f"\n✓ Saved train metadata to {train_file}")
    print(f"✓ Saved val metadata to {val_file}")

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp_baseline.yaml")
