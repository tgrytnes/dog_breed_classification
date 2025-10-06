from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import random
from .config import load_config
from .utils import ensure_dir


def test_data_loading():
    """Test that data files can be loaded."""
    print("Testing data loading...")

    # Check for raw data directory
    raw_data = Path("data/raw")
    assert raw_data.exists(), f"Raw data directory not found: {raw_data}"

    # Check for breed mapping
    breeds_file = raw_data / "breed_mapping.csv"
    if breeds_file.exists():
        breeds_df = pd.read_csv(breeds_file)
        assert len(breeds_df) > 0, "Breed mapping is empty"
        assert 'breed_name' in breeds_df.columns, "breed_name column missing"
        assert 'class_id' in breeds_df.columns, "class_id column missing"
        print(f"  ✓ Loaded {len(breeds_df)} breeds from breed mapping")
    else:
        print(f"  ⚠ Breed mapping not found at {breeds_file}")
        print("  Run: python -m dbc.ingest first")
        return False

    # Check for Images directory
    images_dir = raw_data / "Images"
    if images_dir.exists():
        breed_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
        print(f"  ✓ Found {len(breed_dirs)} breed directories in Images/")
    else:
        print(f"  ⚠ Images directory not found at {images_dir}")
        return False

    return True


def test_image_loading(n_samples: int = 10):
    """Test loading and processing a sample of images."""
    print(f"\nTesting image loading with {n_samples} random samples...")

    raw_data = Path("data/raw")
    images_dir = raw_data / "Images"

    # Find all image files
    all_images = list(images_dir.glob("*/*.jpg"))[:100]  # Limit search

    if len(all_images) == 0:
        print("  ⚠ No images found")
        return False

    # Sample random images
    sample_images = random.sample(all_images, min(n_samples, len(all_images)))

    success_count = 0
    for img_path in sample_images:
        try:
            # Try to open and load image
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                img.load()  # Actually load pixel data

            print(f"  ✓ {img_path.name}: {width}x{height}, mode={mode}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ {img_path.name}: {e}")

    print(f"\n  Loaded {success_count}/{len(sample_images)} images successfully")
    return success_count == len(sample_images)


def test_preprocessing_artifacts():
    """Test that preprocessing artifacts exist."""
    print("\nTesting preprocessing artifacts...")

    artifacts = Path("artifacts")

    # Check for train/val metadata
    train_file = artifacts / "train_metadata.csv"
    val_file = artifacts / "val_metadata.csv"

    if train_file.exists() and val_file.exists():
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)

        print(f"  ✓ Train metadata: {len(train_df)} images")
        print(f"  ✓ Val metadata: {len(val_df)} images")

        # Check for required columns
        required_cols = ['file_path', 'breed_name', 'class_id']
        for col in required_cols:
            assert col in train_df.columns, f"Missing column in train: {col}"
            assert col in val_df.columns, f"Missing column in val: {col}"

        print(f"  ✓ All required columns present")

        # Check no overlap
        train_files = set(train_df['file_path'])
        val_files = set(val_df['file_path'])
        overlap = train_files & val_files

        assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} files"
        print(f"  ✓ No train/val overlap")

        return True
    else:
        print(f"  ⚠ Preprocessing artifacts not found")
        print("  Run: python -m dbc.preprocess first")
        return False


def test_image_batch_loading(batch_size: int = 4):
    """Test loading a batch of images for model input."""
    print(f"\nTesting batch loading ({batch_size} images)...")

    artifacts = Path("artifacts")
    train_file = artifacts / "train_metadata.csv"

    if not train_file.exists():
        print("  ⚠ Train metadata not found")
        return False

    train_df = pd.read_csv(train_file)

    # Sample a batch
    batch = train_df.sample(n=min(batch_size, len(train_df)))

    raw_data = Path("data/raw")
    images = []
    labels = []

    target_size = (224, 224)  # Standard size for transfer learning

    for _, row in batch.iterrows():
        img_path = raw_data / row['file_path']

        try:
            # Load and resize image
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize
                img_resized = img.resize(target_size)

                # Convert to array
                img_array = np.array(img_resized)

                images.append(img_array)
                labels.append(row['class_id'])

                print(f"  ✓ Loaded {img_path.name}: shape={img_array.shape}, breed={row['breed_name']}")

        except Exception as e:
            print(f"  ✗ Failed to load {img_path}: {e}")
            return False

    # Stack into batch
    batch_images = np.stack(images)
    batch_labels = np.array(labels)

    print(f"\n  ✓ Batch shape: {batch_images.shape}")
    print(f"  ✓ Labels shape: {batch_labels.shape}")
    print(f"  ✓ Pixel value range: [{batch_images.min()}, {batch_images.max()}]")

    assert batch_images.shape == (len(batch), target_size[0], target_size[1], 3)
    assert batch_labels.shape == (len(batch),)

    return True


def test_breed_distribution():
    """Test that dataset has reasonable breed distribution."""
    print("\nTesting breed distribution...")

    artifacts = Path("artifacts")
    train_file = artifacts / "train_metadata.csv"

    if not train_file.exists():
        print("  ⚠ Train metadata not found")
        return False

    train_df = pd.read_csv(train_file)

    # Count images per breed
    breed_counts = train_df['breed_name'].value_counts()

    print(f"  Total breeds: {len(breed_counts)}")
    print(f"  Images per breed - Min: {breed_counts.min()}, Max: {breed_counts.max()}, Mean: {breed_counts.mean():.1f}")

    # Check for very imbalanced classes
    if breed_counts.min() < 5:
        print(f"  ⚠ Warning: Some breeds have very few images (< 5)")
        print(f"  Breeds with < 5 images: {(breed_counts < 5).sum()}")

    print(f"\n  Top 5 breeds by count:")
    for breed, count in breed_counts.head(5).items():
        print(f"    {breed}: {count} images")

    return True


def main(config_path: str = "configs/exp_baseline.yaml"):
    """Run all smoke tests."""
    print("="*60)
    print("DOG BREED CLASSIFICATION - SMOKE TEST")
    print("="*60)

    cfg = load_config(config_path)

    tests = [
        ("Data Loading", test_data_loading),
        ("Image Loading", lambda: test_image_loading(n_samples=10)),
        ("Preprocessing Artifacts", test_preprocessing_artifacts),
        ("Batch Loading", lambda: test_image_batch_loading(batch_size=4)),
        ("Breed Distribution", test_breed_distribution),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n  ✗ {test_name} failed with error: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All smoke tests passed!")

        # Save smoke test artifact
        artifacts = Path(cfg['paths']['artifacts'])
        ensure_dir(artifacts)
        smoke_file = artifacts / "smoke_test_passed.txt"
        with open(smoke_file, 'w') as f:
            f.write("Smoke tests passed\n")
            for test_name, result in results.items():
                f.write(f"{test_name}: {'PASS' if result else 'FAIL'}\n")
        print(f"  Saved smoke test results to {smoke_file}")
    else:
        print("\n  ⚠ Some tests failed. Please investigate.")

    print("="*60)

    return passed == total


if __name__ == "__main__":
    import sys
    success = main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp_baseline.yaml")
    sys.exit(0 if success else 1)
