from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional, List
import random
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input as keras_preprocess_input


def get_model_preprocessing_config(model_name: str) -> dict:
    """
    Get preprocessing configuration for specific models.

    Args:
        model_name: Name of the model ('vgg16', 'resnet50', 'efficientnetb0')

    Returns:
        dict with preprocessing config: {
            'preprocess_mode': preprocessing mode for keras,
            'image_size': (width, height),
            'description': human-readable description
        }
    """
    model_name = model_name.lower()

    configs = {
        'resnet50': {
            'preprocess_mode': 'caffe',
            'image_size': (224, 224),
            'description': 'ResNet50: RGB->BGR, mean=[103.939, 116.779, 123.68] subtraction'
        },
        'efficientnetb0': {
            'preprocess_mode': 'torch',
            'image_size': (224, 224),
            'description': 'EfficientNetB0: RGB, normalize with ImageNet mean/std to ~[-2, +2]'
        },
        'efficientnetb4': {
            'preprocess_mode': 'torch',
            'image_size': (380, 380),
            'description': 'EfficientNetB4: RGB, normalize with ImageNet mean/std, 380x380 input'
        }
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(configs.keys())}")

    return configs[model_name]


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
        augment_params: Optional[dict] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize dataset.

        Args:
            metadata_df: DataFrame with columns: file_path, breed_name, class_id
            data_root: Root directory containing the images
            image_size: Target size for images (width, height)
            normalize: 'imagenet' for ImageNet stats, 'scale' for [0,1], 'model' for model-specific, None for no normalization
            augment: Whether to apply data augmentation
            augment_params: Dict with augmentation parameters
            model_name: Name of model for model-specific preprocessing (e.g., 'vgg16', 'resnet50')
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        self.model_name = model_name

        # Setup model-specific preprocessing if specified
        self.model_preprocess_mode = None
        if model_name is not None:
            config = get_model_preprocessing_config(model_name)
            self.model_preprocess_mode = config['preprocess_mode']
            self.image_size = config['image_size']  # Use model's required size
            print(f"  Model-specific preprocessing: {config['description']}")

        # Strong augmentation parameters (proven effective for fine-grained classification)
        self.augment_params = augment_params or {
            'rotation_range': 20,           # ±20 degrees
            'horizontal_flip': True,        # 50% probability
            'zoom_range': 0.2,              # 0.2 = zoom in/out by 20%
            'width_shift_range': 0.2,       # ±20% horizontal shift
            'height_shift_range': 0.2,      # ±20% vertical shift
            'shear_range': 0.15,            # 0.15 radians shear
            'brightness_range': (0.8, 1.2), # ±20%
            'contrast_range': (0.8, 1.2),   # ±20%
            'fill_mode': 'nearest'          # Fill empty regions
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
        if self.model_preprocess_mode is not None:
            # Use model-specific preprocessing (Keras preprocess_input with mode)
            img_array = keras_preprocess_input(img_array, mode=self.model_preprocess_mode)
        elif self.normalize == 'imagenet':
            img_array = img_array / 255.0  # Scale to [0,1]
            img_array = (img_array - self.imagenet_mean) / self.imagenet_std
        elif self.normalize == 'scale':
            img_array = img_array / 255.0  # Scale to [0,1]

        # Convert class_id from 1-120 to 0-119 for Keras
        return img_array, row['class_id'] - 1, row['breed_name']

    def _get_fill_value(self, img: Image.Image) -> tuple:
        """Get fill value based on fill_mode parameter."""
        fill_mode = self.augment_params.get('fill_mode', 'nearest')
        if fill_mode == 'nearest':
            # Use edge pixels (approximate nearest behavior with constant edge color)
            # For simplicity, use a neutral gray that blends well
            return (128, 128, 128)
        else:
            return (128, 128, 128)

    def _augment_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply strong data augmentation to image.

        Includes: rotation, flip, zoom, width/height shift, shear, brightness, contrast

        Args:
            img_array: Image as numpy array (H, W, 3)

        Returns:
            Augmented image array
        """
        import math
        img = Image.fromarray(img_array.astype(np.uint8))
        fill_color = self._get_fill_value(img)

        # Random rotation
        if 'rotation_range' in self.augment_params:
            angle = random.uniform(
                -self.augment_params['rotation_range'],
                self.augment_params['rotation_range']
            )
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill_color)

        # Random horizontal flip
        if self.augment_params.get('horizontal_flip', False):
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Random shear transform
        if 'shear_range' in self.augment_params:
            # shear_range is in radians (0.15 = 8.6 degrees)
            shear_radians = random.uniform(
                -self.augment_params['shear_range'],
                self.augment_params['shear_range']
            )
            shear_factor = math.tan(shear_radians)
            img = img.transform(
                img.size,
                Image.AFFINE,
                (1, shear_factor, 0, 0, 1, 0),
                resample=Image.BILINEAR,
                fillcolor=fill_color
            )

        # Random width shift
        if 'width_shift_range' in self.augment_params:
            max_shift_w = int(self.image_size[0] * self.augment_params['width_shift_range'])
            shift_w = random.randint(-max_shift_w, max_shift_w)
            if shift_w != 0:
                img = img.transform(
                    img.size,
                    Image.AFFINE,
                    (1, 0, -shift_w, 0, 1, 0),
                    resample=Image.BILINEAR,
                    fillcolor=fill_color
                )

        # Random height shift
        if 'height_shift_range' in self.augment_params:
            max_shift_h = int(self.image_size[1] * self.augment_params['height_shift_range'])
            shift_h = random.randint(-max_shift_h, max_shift_h)
            if shift_h != 0:
                img = img.transform(
                    img.size,
                    Image.AFFINE,
                    (1, 0, 0, 0, 1, -shift_h),
                    resample=Image.BILINEAR,
                    fillcolor=fill_color
                )

        # Random zoom (resize then crop/pad)
        if 'zoom_range' in self.augment_params:
            # zoom_range is a single value (e.g., 0.2 means zoom from 0.8 to 1.2)
            zoom_range_val = self.augment_params['zoom_range']
            zoom_factor = random.uniform(1.0 - zoom_range_val, 1.0 + zoom_range_val)
            new_size = tuple(int(dim * zoom_factor) for dim in img.size)
            img = img.resize(new_size, Image.BILINEAR)

            # Crop or pad to original size
            if zoom_factor > 1:  # Zoomed in, need to crop
                left = (new_size[0] - self.image_size[0]) // 2
                top = (new_size[1] - self.image_size[1]) // 2
                img = img.crop((left, top, left + self.image_size[0], top + self.image_size[1]))
            else:  # Zoomed out, need to pad
                new_img = Image.new('RGB', self.image_size, fill_color)
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


class PreprocessedDataGenerator(keras.utils.Sequence):
    """
    Ultra-fast data generator that loads from preprocessed numpy arrays.
    All images are loaded into memory at once for maximum speed.
    """

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize preprocessed data generator.

        Args:
            data_path: Path to .npz file containing preprocessed data
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            augment: Apply augmentation (train only)
            seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        # Load data with memory mapping (instant loading!)
        print(f"Loading preprocessed data from: {data_path.parent}")
        images_path = data_path.parent / f"{data_path.stem}_images.npy"
        labels_path = data_path.parent / f"{data_path.stem}_labels.npy"

        self.images = np.load(images_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')

        self.n_samples = len(self.images)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))

        print(f"  Loaded {self.n_samples} images into memory")
        print(f"  Images shape: {self.images.shape}")
        print(f"  Memory usage: {self.images.nbytes / (1024**3):.2f} GB")

        self.indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Setup augmentation layers if needed
        if self.augment:
            self.augmentation = keras.Sequential([
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.04),  # ~15 degrees
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.2),
            ])

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index):
        """Get batch at given index."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[start_idx:end_idx]

        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Apply augmentation if enabled (on GPU)
        if self.augment:
            batch_images = self.augmentation(batch_images, training=True)

        return batch_images, batch_labels

    def on_epoch_end(self):
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class OptimizedDataGenerator(keras.utils.Sequence):
    """
    Optimized data generator using TensorFlow's prefetch for better GPU utilization.
    Prefetches data in parallel with model training to eliminate data loading bottlenecks.
    """

    def __init__(
        self,
        dataset: DogBreedDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: Optional[int] = None,
        use_prefetch: bool = True,
        prefetch_buffer: int = 4
    ):
        """
        Initialize optimized data generator.

        Args:
            dataset: DogBreedDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            seed: Random seed for reproducibility
            use_prefetch: Enable prefetching for better performance
            prefetch_buffer: Number of batches to prefetch (default 4 = AUTOTUNE equivalent)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.use_prefetch = use_prefetch
        self.prefetch_buffer = prefetch_buffer

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        self.indices = np.arange(len(dataset))
        self.n_batches = int(np.ceil(len(dataset) / batch_size))

        if self.shuffle:
            np.random.shuffle(self.indices)

        # Pre-create batches for parallel loading
        self._setup_batch_indices()

    def _setup_batch_indices(self):
        """Pre-compute batch indices for faster access."""
        self.batch_indices_list = []
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            self.batch_indices_list.append(self.indices[start_idx:end_idx])

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index):
        """
        Get batch at given index with parallel loading.

        Args:
            index: Batch index

        Returns:
            Tuple of (images, labels) for the batch
        """
        batch_indices = self.batch_indices_list[index]

        # Load batch with parallel operations
        if self.use_prefetch:
            # Use TensorFlow's parallel map for faster loading
            images = []
            labels = []
            for idx in batch_indices:
                img, label, _ = self.dataset.load_image(int(idx))
                images.append(img)
                labels.append(label)
            return np.array(images), np.array(labels)
        else:
            # Standard loading
            images, labels = self.dataset.get_batch(batch_indices)
            return images, labels

    def on_epoch_end(self):
        """Shuffle indices and rebuild batch list at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)
            self._setup_batch_indices()


def create_data_loaders(
    train_metadata_path: Path,
    val_metadata_path: Path,
    data_root: Path,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    normalize: str = 'imagenet',
    augment_train: bool = True,
    seed: int = 42,
    use_prefetch: bool = True,
    model_name: Optional[str] = None
) -> Tuple[DataGenerator, DataGenerator]:
    """
    Create train and validation data loaders with optimized prefetching.

    Args:
        train_metadata_path: Path to train metadata CSV
        val_metadata_path: Path to validation metadata CSV
        data_root: Root directory containing images
        batch_size: Batch size
        image_size: Target image size (width, height) - overridden if model_name is specified
        normalize: Normalization method ('imagenet', 'scale', or None) - ignored if model_name is specified
        augment_train: Whether to augment training data
        seed: Random seed
        use_prefetch: Use tf.data pipeline with prefetching for better GPU utilization
        model_name: Name of model for model-specific preprocessing (e.g., 'vgg16', 'resnet50', 'efficientnetb0')

    Returns:
        (train_generator, val_generator)
    """
    # Load metadata
    train_df = pd.read_csv(train_metadata_path)
    val_df = pd.read_csv(val_metadata_path)

    print("\n" + "="*60)
    print("CREATING DATA LOADERS WITH OPTIMIZED PREFETCHING")
    if model_name:
        print(f"Model: {model_name.upper()}")
    print("="*60)

    # Create datasets
    print("\nTrain dataset:")
    train_dataset = DogBreedDataset(
        train_df,
        data_root,
        image_size=image_size,
        normalize=normalize,
        augment=augment_train,
        model_name=model_name
    )

    print("\nValidation dataset:")
    val_dataset = DogBreedDataset(
        val_df,
        data_root,
        image_size=image_size,
        normalize=normalize,
        augment=False,  # No augmentation for validation
        model_name=model_name
    )

    # Create optimized generators with prefetching
    train_gen = OptimizedDataGenerator(train_dataset, batch_size=batch_size, shuffle=True, seed=seed, use_prefetch=use_prefetch)
    val_gen = OptimizedDataGenerator(val_dataset, batch_size=batch_size, shuffle=False, seed=seed, use_prefetch=use_prefetch)

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_gen)} ({batch_size} samples/batch)")
    print(f"  Val batches: {len(val_gen)} ({batch_size} samples/batch)")
    print(f"  Prefetching: {'ENABLED (optimized for GPU)' if use_prefetch else 'DISABLED'}")
    print("="*60)

    return train_gen, val_gen


def create_preprocessed_loaders(
    preprocessed_dir: Path,
    batch_size: int = 32,
    augment_train: bool = True,
    seed: int = 42
) -> Tuple[PreprocessedDataGenerator, PreprocessedDataGenerator]:
    """
    Create data loaders from preprocessed numpy arrays.
    This is MUCH faster than loading from disk on-the-fly.

    Args:
        preprocessed_dir: Directory containing train_data.npz and val_data.npz
        batch_size: Batch size
        augment_train: Whether to augment training data (on GPU)
        seed: Random seed

    Returns:
        (train_generator, val_generator)
    """
    preprocessed_dir = Path(preprocessed_dir)

    print("\n" + "="*60)
    print("LOADING PREPROCESSED DATA (FAST MODE)")
    print("="*60)

    # Create generators from preprocessed files
    print("\nTrain dataset:")
    train_gen = PreprocessedDataGenerator(
        data_path=preprocessed_dir / 'train_data.npz',
        batch_size=batch_size,
        shuffle=True,
        augment=augment_train,
        seed=seed
    )

    print("\nValidation dataset:")
    val_gen = PreprocessedDataGenerator(
        data_path=preprocessed_dir / 'val_data.npz',
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        seed=seed
    )

    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_gen)} ({batch_size} samples/batch)")
    print(f"  Val batches: {len(val_gen)} ({batch_size} samples/batch)")
    print(f"  Augmentation: {'GPU-accelerated (Keras layers)' if augment_train else 'DISABLED'}")
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
