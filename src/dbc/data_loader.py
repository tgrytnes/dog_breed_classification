from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, List
import random
from tensorflow import keras


class DogBreedDataset:
    """
    Dataset class for dog breed images with preprocessing and augmentation.

    Handles:
    - Image loading and resizing
    - RGB conversion
    - Normalization (ImageNet stats or [0,1] scaling)
    - Data augmentation (rotation, flip, zoom, brightness)
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        data_root: Path,
        image_size: Tuple[int, int] = (224, 224),
        normalize: str = 'imagenet',
        augment: bool = False,
        augment_params: Optional[dict] = None
    ):
        """
        Initialize dataset.

        Args:
            metadata_df: DataFrame with columns: file_path, breed_name, class_id
            data_root: Root directory containing the images
            image_size: Target size for images (width, height)
            normalize: 'imagenet' for ImageNet stats, 'scale' for [0,1], None for no normalization
            augment: Whether to apply data augmentation
            augment_params: Dict with augmentation parameters
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment

        # Default augmentation parameters
        self.augment_params = augment_params or {
            'rotation_range': 15,      # ±15 degrees
            'horizontal_flip': True,   # 50% probability
            'zoom_range': (0.9, 1.1),  # 90-110%
            'brightness_range': (0.8, 1.2),  # ±20%
            'contrast_range': (0.8, 1.2)     # ±20%
        }

        # ImageNet normalization statistics
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])

        print(f"Dataset initialized:")
        print(f"  Images: {len(self.metadata_df)}")
        print(f"  Classes: {self.metadata_df['breed_name'].nunique()}")
        print(f"  Image size: {self.image_size}")
        print(f"  Normalization: {self.normalize}")
        print(f"  Augmentation: {self.augment}")

    def __len__(self) -> int:
        return len(self.metadata_df)

    def load_image(self, idx: int) -> Tuple[np.ndarray, int, str]:
        """
        Load and preprocess a single image.

        Returns:
            (image_array, class_id, breed_name)
        """
        row = self.metadata_df.iloc[idx]

        # Load image
        img_path = self.data_root / row['file_path']
        img = Image.open(img_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize
        img = img.resize(self.image_size, Image.BILINEAR)

        # Convert to array
        img_array = np.array(img, dtype=np.float32)

        # Apply augmentation if enabled
        if self.augment:
            img_array = self._augment_image(img_array)

        # Normalize
        if self.normalize == 'imagenet':
            img_array = img_array / 255.0  # Scale to [0,1]
            img_array = (img_array - self.imagenet_mean) / self.imagenet_std
        elif self.normalize == 'scale':
            img_array = img_array / 255.0  # Scale to [0,1]

        # Convert class_id from 1-120 to 0-119 for Keras
        return img_array, row['class_id'] - 1, row['breed_name']

    def _augment_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.

        Args:
            img_array: Image as numpy array (H, W, 3)

        Returns:
            Augmented image array
        """
        img = Image.fromarray(img_array.astype(np.uint8))

        # Random rotation
        if 'rotation_range' in self.augment_params:
            angle = random.uniform(
                -self.augment_params['rotation_range'],
                self.augment_params['rotation_range']
            )
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

        # Random horizontal flip
        if self.augment_params.get('horizontal_flip', False):
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random zoom (resize then crop/pad)
        if 'zoom_range' in self.augment_params:
            zoom_factor = random.uniform(*self.augment_params['zoom_range'])
            new_size = tuple(int(dim * zoom_factor) for dim in img.size)
            img = img.resize(new_size, Image.BILINEAR)

            # Crop or pad to original size
            if zoom_factor > 1:  # Zoomed in, need to crop
                left = (new_size[0] - self.image_size[0]) // 2
                top = (new_size[1] - self.image_size[1]) // 2
                img = img.crop((left, top, left + self.image_size[0], top + self.image_size[1]))
            else:  # Zoomed out, need to pad
                new_img = Image.new('RGB', self.image_size, (128, 128, 128))
                paste_x = (self.image_size[0] - new_size[0]) // 2
                paste_y = (self.image_size[1] - new_size[1]) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img

        img_array = np.array(img, dtype=np.float32)

        # Random brightness
        if 'brightness_range' in self.augment_params:
            brightness_factor = random.uniform(*self.augment_params['brightness_range'])
            img_array = np.clip(img_array * brightness_factor, 0, 255)

        # Random contrast
        if 'contrast_range' in self.augment_params:
            contrast_factor = random.uniform(*self.augment_params['contrast_range'])
            mean = img_array.mean()
            img_array = np.clip((img_array - mean) * contrast_factor + mean, 0, 255)

        return img_array

    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a batch of images.

        Args:
            indices: List of dataset indices

        Returns:
            (images, labels) where images is (N, H, W, 3) and labels is (N,)
        """
        images = []
        labels = []

        for idx in indices:
            img, label, _ = self.load_image(idx)
            images.append(img)
            labels.append(label)

        return np.array(images), np.array(labels)


class DataGenerator(keras.utils.Sequence):
    """
    Generator for creating batches of training/validation data.
    Inherits from keras.utils.Sequence for proper Keras integration.
    """

    def __init__(
        self,
        dataset: DogBreedDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize data generator.

        Args:
            dataset: DogBreedDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.indices = np.arange(len(dataset))
        self.n_batches = int(np.ceil(len(dataset) / batch_size))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index):
        """
        Get batch at given index (required by keras.utils.Sequence).

        Args:
            index: Batch index

        Returns:
            Tuple of (images, labels) for the batch
        """
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]

        images, labels = self.dataset.get_batch(batch_indices)

        return images, labels

    def __iter__(self):
        """Iterate over batches (for compatibility)."""
        if self.shuffle:
            np.random.shuffle(self.indices)

        for i in range(self.n_batches):
            yield self[i]

    def on_epoch_end(self):
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_data_loaders(
    train_metadata_path: Path,
    val_metadata_path: Path,
    data_root: Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    normalize: str = 'imagenet',
    augment_train: bool = True,
    seed: int = 42
) -> Tuple[DataGenerator, DataGenerator]:
    """
    Create train and validation data loaders.

    Args:
        train_metadata_path: Path to train metadata CSV
        val_metadata_path: Path to validation metadata CSV
        data_root: Root directory containing images
        batch_size: Batch size
        image_size: Target image size (width, height)
        normalize: Normalization method ('imagenet', 'scale', or None)
        augment_train: Whether to augment training data
        seed: Random seed

    Returns:
        (train_generator, val_generator)
    """
    # Load metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)

    print("\n" + "="*60)
    print("CREATING DATA LOADERS")
    print("="*60)

    # Create datasets
    print("\nTrain dataset:")
    train_dataset = DogBreedDataset(
        train_df,
        data_root,
        image_size=image_size,
        normalize=normalize,
        augment=augment_train
    )

    print("\nValidation dataset:")
    val_dataset = DogBreedDataset(
        val_df,
        data_root,
        image_size=image_size,
        normalize=normalize,
        augment=False  # No augmentation for validation
    )

    # Create generators
    train_gen = DataGenerator(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_gen = DataGenerator(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_gen)} ({batch_size} samples/batch)")
    print(f"  Val batches: {len(val_gen)} ({batch_size} samples/batch)")
    print("="*60)

    return train_gen, val_gen


if __name__ == "__main__":
    # Test data loader
    from .config import load_config

    cfg = load_config("configs/exp_baseline.yaml")

    train_gen, val_gen = create_data_loaders(
        train_metadata_path=Path("artifacts/train_metadata.csv"),
        val_metadata_path=Path("artifacts/val_metadata.csv"),
        data_root=Path(cfg['paths']['raw_data']),
        batch_size=4,
        augment_train=True
    )

    # Test loading a batch
    print("\nTesting batch loading...")
    for images, labels in train_gen:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"Unique labels: {np.unique(labels)}")
        break  # Just test first batch

    print("\n✓ Data loader test passed!")
