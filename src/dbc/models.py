"""
CNN model architectures for dog breed classification.

This module provides transfer learning models using pre-trained ImageNet weights.
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50,
    VGG16,
    EfficientNetB0
)


def build_transfer_learning_model(
    base_model_name: str = 'resnet50',
    num_classes: int = 120,
    input_shape: tuple = (224, 224, 3),
    trainable_base: bool = False,
    dropout_rate: float = 0.5,
    unfreeze_last_n_layers: int = None
) -> Model:
    """
    Build a transfer learning model with a pre-trained base.

    Args:
        base_model_name: Name of base model ('resnet50', 'vgg16', 'efficientnetb0')
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        trainable_base: Whether to make base layers trainable (False = feature extraction)
        dropout_rate: Dropout rate before final classification layer
        unfreeze_last_n_layers: If set, freeze all layers except last N (overrides trainable_base)
                               Recommended: 4-8 for VGG16, 10-20 for ResNet50, 20-40 for EfficientNet

    Returns:
        Compiled Keras model
    """
    # Load pre-trained base model
    base_models = {
        'resnet50': ResNet50,
        'vgg16': VGG16,
        'efficientnetb0': EfficientNetB0
    }

    if base_model_name.lower() not in base_models:
        raise ValueError(f"Unknown base model: {base_model_name}. "
                        f"Available: {list(base_models.keys())}")

    base_model_class = base_models[base_model_name.lower()]

    # Create base model (without top classification layer)
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Apply freezing strategy
    if unfreeze_last_n_layers is not None:
        # Selective unfreezing: freeze all except last N layers
        base_model.trainable = True
        total_layers = len(base_model.layers)

        # Freeze early layers
        for layer in base_model.layers[:-unfreeze_last_n_layers]:
            layer.trainable = False

        # Unfreeze last N layers
        for layer in base_model.layers[-unfreeze_last_n_layers:]:
            layer.trainable = True

        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"  Selective unfreezing: {trainable_count}/{total_layers} layers trainable (last {unfreeze_last_n_layers})")
    else:
        # All-or-nothing freezing
        base_model.trainable = trainable_base
        if trainable_base:
            print(f"  All {len(base_model.layers)} base layers trainable")
        else:
            print(f"  All {len(base_model.layers)} base layers frozen")

    # Build full model
    inputs = keras.Input(shape=input_shape)

    # Base model
    x = base_model(inputs, training=False)

    # Global pooling to convert feature maps to single vector
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers for classification
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def build_custom_cnn(
    num_classes: int = 120,
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.3
) -> Model:
    """
    Build a custom CNN from scratch (baseline for comparison).

    Args:
        num_classes: Number of output classes
        input_shape: Input image shape (height, width, channels)
        dropout_rate: Dropout rate

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)  # Higher dropout at end
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate + 0.2)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.001,
    optimizer: str = 'adam'
) -> Model:
    """
    Compile a model with appropriate loss and metrics.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')

    Returns:
        Compiled model
    """
    # Create optimizer
    optimizers = {
        'adam': keras.optimizers.Adam(learning_rate=learning_rate),
        'sgd': keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=learning_rate)
    }

    if optimizer.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer}. "
                        f"Available: {list(optimizers.keys())}")

    opt = optimizers[optimizer.lower()]

    # Compile with categorical crossentropy for multi-class classification
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',  # Use sparse for integer labels
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )

    return model
