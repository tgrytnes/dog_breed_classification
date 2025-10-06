from __future__ import annotations
import pandas as pd
from pathlib import Path
import urllib.request
import zipfile
import tarfile
import shutil
from typing import Optional
from .utils import ensure_dir


def download_stanford_dogs(data_dir: Path, force_download: bool = False) -> dict:
    """
    Download Stanford Dogs Dataset.

    The Stanford Dogs Dataset contains images of 120 breeds of dogs from around the world.
    This dataset has been built using images and annotation from ImageNet for the task
    of fine-grained image categorization.

    Dataset info:
    - 120 dog breed classes
    - ~20,000 images total
    - Images from ImageNet

    Args:
        data_dir: Directory to download and extract data to
        force_download: If True, re-download even if files exist

    Returns:
        dict with paths to images and annotations
    """
    ensure_dir(data_dir)

    # URLs for Stanford Dogs Dataset
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"

    images_dir = data_dir / "Images"
    annotations_dir = data_dir / "Annotation"
    lists_dir = data_dir / "lists"

    # Check if already downloaded
    if not force_download and images_dir.exists() and annotations_dir.exists():
        print(f"Dataset already exists at {data_dir}")
        return {
            "images_dir": images_dir,
            "annotations_dir": annotations_dir,
            "lists_dir": lists_dir
        }

    # Download and extract images
    print("Downloading Stanford Dogs Dataset...")
    print("This may take several minutes (images are ~750MB)...")

    images_tar = data_dir / "images.tar"
    if not images_tar.exists() or force_download:
        print(f"Downloading images from {images_url}...")
        urllib.request.urlretrieve(images_url, images_tar)
        print(f"Downloaded images to {images_tar}")

    # Extract images
    if not images_dir.exists() or force_download:
        print(f"Extracting images to {data_dir}...")
        with tarfile.open(images_tar, 'r') as tar:
            tar.extractall(data_dir)
        print(f"Extracted images to {images_dir}")

    # Download and extract annotations
    annotations_tar = data_dir / "annotation.tar"
    if not annotations_tar.exists() or force_download:
        print(f"Downloading annotations from {annotations_url}...")
        urllib.request.urlretrieve(annotations_url, annotations_tar)
        print(f"Downloaded annotations to {annotations_tar}")

    if not annotations_dir.exists() or force_download:
        print(f"Extracting annotations to {data_dir}...")
        with tarfile.open(annotations_tar, 'r') as tar:
            tar.extractall(data_dir)
        print(f"Extracted annotations to {annotations_dir}")

    # Download and extract file lists
    lists_tar = data_dir / "lists.tar"
    if not lists_tar.exists() or force_download:
        print(f"Downloading file lists from {lists_url}...")
        urllib.request.urlretrieve(lists_url, lists_tar)
        print(f"Downloaded file lists to {lists_tar}")

    if not lists_dir.exists() or force_download:
        print(f"Extracting file lists to {data_dir}...")
        with tarfile.open(lists_tar, 'r') as tar:
            tar.extractall(data_dir)
        print(f"Extracted file lists to {lists_dir}")

    print("\n✓ Download complete!")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Lists: {lists_dir}")

    return {
        "images_dir": images_dir,
        "annotations_dir": annotations_dir,
        "lists_dir": lists_dir
    }


def load_train_list(lists_dir: Path) -> pd.DataFrame:
    """Load training file list."""
    train_list = lists_dir / "train_list.mat"
    if not train_list.exists():
        # Fallback to text file if exists
        train_txt = lists_dir / "train_list.txt"
        if train_txt.exists():
            files = []
            with open(train_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        files.append({'file_path': parts[0], 'class_id': int(parts[1])})
            return pd.DataFrame(files)
        return pd.DataFrame(columns=['file_path', 'class_id'])

    # Load .mat file
    try:
        from scipy.io import loadmat
        data = loadmat(train_list)
        # Extract file list from mat structure
        file_list = data.get('file_list', data.get('train_list', []))
        labels = data.get('labels', data.get('train_labels', []))

        files = []
        for i, (fname, label) in enumerate(zip(file_list, labels)):
            if hasattr(fname, 'item'):
                fname = fname.item()
            if isinstance(fname, bytes):
                fname = fname.decode('utf-8')
            files.append({'file_path': fname, 'class_id': int(label)})

        return pd.DataFrame(files)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return pd.DataFrame(columns=['file_path', 'class_id'])


def load_test_list(lists_dir: Path) -> pd.DataFrame:
    """Load test file list."""
    test_list = lists_dir / "test_list.mat"
    if not test_list.exists():
        # Fallback to text file
        test_txt = lists_dir / "test_list.txt"
        if test_txt.exists():
            files = []
            with open(test_txt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        files.append({'file_path': parts[0], 'class_id': int(parts[1])})
            return pd.DataFrame(files)
        return pd.DataFrame(columns=['file_path', 'class_id'])

    try:
        from scipy.io import loadmat
        data = loadmat(test_list)
        file_list = data.get('file_list', data.get('test_list', []))
        labels = data.get('labels', data.get('test_labels', []))

        files = []
        for fname, label in zip(file_list, labels):
            if hasattr(fname, 'item'):
                fname = fname.item()
            if isinstance(fname, bytes):
                fname = fname.decode('utf-8')
            files.append({'file_path': fname, 'class_id': int(label)})

        return pd.DataFrame(files)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return pd.DataFrame(columns=['file_path', 'class_id'])


def create_breed_mapping(images_dir: Path) -> pd.DataFrame:
    """
    Create mapping from class_id to breed name by scanning image directories.

    Returns:
        DataFrame with columns: class_id, breed_name
    """
    breed_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])

    breeds = []
    for class_id, breed_dir in enumerate(breed_dirs, start=1):
        breed_name = breed_dir.name
        # Clean up breed name (e.g., "n02085620-Chihuahua" -> "Chihuahua")
        if '-' in breed_name:
            breed_name = breed_name.split('-', 1)[1]

        breeds.append({
            'class_id': class_id,
            'breed_name': breed_name,
            'breed_dir': breed_dir.name
        })

    return pd.DataFrame(breeds)


def main(config_path: str = "configs/exp_baseline.yaml"):
    """Download and prepare dataset."""
    from .config import load_config

    cfg = load_config(config_path)
    data_dir = Path(cfg['paths']['raw_data'])

    # Download dataset
    paths = download_stanford_dogs(data_dir, force_download=False)

    # Create breed mapping
    breeds_df = create_breed_mapping(paths['images_dir'])
    breeds_file = data_dir / "breed_mapping.csv"
    breeds_df.to_csv(breeds_file, index=False)
    print(f"\n✓ Created breed mapping with {len(breeds_df)} breeds: {breeds_file}")

    # Try to load train/test splits if they exist
    if paths['lists_dir'].exists():
        train_df = load_train_list(paths['lists_dir'])
        test_df = load_test_list(paths['lists_dir'])

        if not train_df.empty:
            train_file = data_dir / "train_list.csv"
            train_df.to_csv(train_file, index=False)
            print(f"✓ Saved train list with {len(train_df)} images: {train_file}")

        if not test_df.empty:
            test_file = data_dir / "test_list.csv"
            test_df.to_csv(test_file, index=False)
            print(f"✓ Saved test list with {len(test_df)} images: {test_file}")

    print("\nDataset summary:")
    print(f"  Total breeds: {len(breeds_df)}")
    print(f"  Images directory: {paths['images_dir']}")

    return paths


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/exp_baseline.yaml")
