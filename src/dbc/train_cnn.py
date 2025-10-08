"""
Training script for CNN dog breed classification models.
"""

from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tensorflow import keras

from .config import load_config
from .data_loader import create_data_loaders
from .models import build_transfer_learning_model, build_custom_cnn, compile_model
from .utils import ensure_dir, save_json
from .experiment_tracker import get_tracker


def train_model(config_path: str = "configs/cnn_baseline.yaml"):
    """
    Train a CNN model for dog breed classification.

    Args:
        config_path: Path to YAML config file
    """
    print("="*60)
    print("DOG BREED CLASSIFICATION - CNN TRAINING")
    print("="*60)

    # Load config
    cfg = load_config(config_path)

    # Initialize experiment tracker
    tracker = get_tracker(experiments_dir=cfg['paths'].get('experiments', 'experiments'))

    # Create experiment
    experiment_name = cfg.get('experiment_name', f"exp_{Path(config_path).stem}")
    experiment_desc = cfg.get('experiment_description', '')
    experiment_tags = cfg.get('experiment_tags', [])

    # Add automatic tags based on config
    if 'train' in cfg:
        if cfg['train'].get('model_type') == 'transfer':
            experiment_tags.append(cfg['train'].get('base_model', 'unknown'))
        experiment_tags.append(cfg['train'].get('model_type', 'unknown'))

    experiment_id = tracker.create_experiment(
        name=experiment_name,
        description=experiment_desc,
        tags=experiment_tags
    )

    # Log configuration
    tracker.log_config(experiment_id, cfg)

    # Extract paths
    train_metadata = Path(cfg['paths']['artifacts']) / "train_metadata.csv"
    val_metadata = Path(cfg['paths']['artifacts']) / "val_metadata.csv"
    data_root = Path(cfg['paths']['data'])
    artifacts_dir = Path(cfg['paths']['artifacts'])

    # Use experiment-specific directory for checkpoints
    exp_dir = Path(cfg['paths'].get('experiments', 'experiments')) / experiment_id
    checkpoint_dir = exp_dir / "checkpoints"

    # Ensure directories exist
    ensure_dir(artifacts_dir)
    ensure_dir(checkpoint_dir)

    # Extract training config
    train_cfg = cfg['train']
    batch_size = train_cfg.get('batch_size', 32)
    image_size = tuple(train_cfg.get('image_size', [224, 224]))
    epochs = train_cfg.get('epochs', 20)
    model_type = train_cfg.get('model_type', 'transfer')  # 'transfer' or 'custom'
    base_model = train_cfg.get('base_model', 'resnet50')
    learning_rate = train_cfg.get('learning_rate', 0.001)

    print(f"\nConfiguration:")
    print(f"  Model type: {model_type}")
    if model_type == 'transfer':
        print(f"  Base model: {base_model}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Create data loaders
    print("\nCreating data loaders...")

    # Check if preprocessed data exists (.npy format for memory mapping)
    preprocessed_dir = Path("artifacts/preprocessed")
    train_images_path = preprocessed_dir / "train_data_images.npy"
    val_images_path = preprocessed_dir / "val_data_images.npy"
    use_preprocessed = train_images_path.exists() and val_images_path.exists()

    if use_preprocessed:
        print(f"\n✓ Found preprocessed data, using instant memory-mapped loading!")
        from .data_loader import create_preprocessed_loaders
        train_gen, val_gen = create_preprocessed_loaders(
            preprocessed_dir=preprocessed_dir,
            batch_size=batch_size,
            augment_train=True,
            seed=42
        )
    else:
        print(f"\nUsing on-the-fly image loading with model-specific preprocessing")
        train_gen, val_gen = create_data_loaders(
            train_metadata_path=train_metadata,
            val_metadata_path=val_metadata,
            data_root=data_root,
            batch_size=batch_size,
            model_name=base_model,  # Critical: model-specific preprocessing
            augment_train=True,
            seed=42
        )

    # Build model
    print(f"\nBuilding {model_type} model...")
    num_classes = 120  # Stanford Dogs has 120 breeds

    # Check if we should load from checkpoint
    load_checkpoint = train_cfg.get('load_checkpoint', None)

    if load_checkpoint:
        print(f"\n✓ Loading model from checkpoint: {load_checkpoint}")
        model = keras.models.load_model(load_checkpoint)

        # Update trainable status if specified
        if model_type == 'transfer':
            # Find the base model layer (ResNet50, VGG16, etc.)
            base_layer = None
            for layer in model.layers:
                if layer.name in ['resnet50', 'vgg16', 'efficientnetb0', 'efficientnetb4']:
                    base_layer = layer
                    break

            if base_layer and 'unfreeze_last_n' in train_cfg:
                # Selective unfreezing: freeze first N layers, unfreeze last M layers
                unfreeze_last_n = train_cfg.get('unfreeze_last_n')
                total_layers = len(base_layer.layers)
                freeze_until = total_layers - unfreeze_last_n

                print(f"  Selective unfreezing: {base_layer.name} has {total_layers} layers")
                print(f"  Freezing first {freeze_until} layers, unfreezing last {unfreeze_last_n} layers")

                # Freeze all layers first
                base_layer.trainable = True
                batchnorm_count = 0
                for i, layer in enumerate(base_layer.layers):
                    if i >= freeze_until:
                        # Unfreeze this layer, but keep BatchNormalization layers frozen
                        # This is critical for fine-tuning - BatchNorm layers must stay frozen
                        if 'BatchNormalization' in layer.__class__.__name__:
                            layer.trainable = False
                            batchnorm_count += 1
                        else:
                            layer.trainable = True
                    else:
                        layer.trainable = False

                if batchnorm_count > 0:
                    print(f"  Kept {batchnorm_count} BatchNormalization layers frozen (critical for fine-tuning)")

            elif base_layer and 'trainable_base' in train_cfg:
                # Full freeze/unfreeze
                trainable = train_cfg.get('trainable_base', False)
                base_layer.trainable = trainable
                print(f"  Set base model ({base_layer.name}) trainable={trainable}")

        print(f"  Loaded model with {model.count_params():,} parameters")
        trainable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"  Trainable parameters: {trainable_params:,}")

    else:
        # Build new model
        if model_type == 'transfer':
            model = build_transfer_learning_model(
                base_model_name=base_model,
                num_classes=num_classes,
                input_shape=(*image_size, 3),
                trainable_base=train_cfg.get('trainable_base', False),
                dropout_rate=train_cfg.get('dropout', 0.5)
            )
        elif model_type == 'custom':
            model = build_custom_cnn(
                num_classes=num_classes,
                input_shape=(*image_size, 3),
                dropout_rate=train_cfg.get('dropout', 0.3)
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'transfer' or 'custom'")

    # Compile model (always recompile with new learning rate)
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    model = compile_model(
        model,
        learning_rate=learning_rate,
        optimizer=train_cfg.get('optimizer', 'adam'),
        label_smoothing=label_smoothing
    )

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Calculate steps per epoch
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)

    # Allow limiting batches for smoke testing
    if 'limit_train_batches' in train_cfg:
        steps_per_epoch = min(steps_per_epoch, train_cfg['limit_train_batches'])
        print(f"  ⚠️  LIMITING training to {steps_per_epoch} batches per epoch (smoke test mode)")

    if 'limit_val_batches' in train_cfg:
        validation_steps = min(validation_steps, train_cfg['limit_val_batches'])
        print(f"  ⚠️  LIMITING validation to {validation_steps} batches (smoke test mode)")

    print(f"\nTraining configuration:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Validation steps: {validation_steps}")

    # Setup callbacks
    callbacks = []

    # Model checkpoint - save best model
    checkpoint_path = checkpoint_dir / "best_model.h5"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint_cb)

    # Early stopping
    if train_cfg.get('early_stopping', True):
        early_stop_cb = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=train_cfg.get('patience', 5),
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callbacks.append(early_stop_cb)

    # Learning rate schedule: Cosine Annealing with Warmup or ReduceLROnPlateau
    lr_schedule_type = train_cfg.get('lr_schedule', 'reduce_on_plateau')

    if lr_schedule_type == 'cosine_warmup':
        # Cosine Annealing with Warmup (research-based, smoother convergence)
        warmup_epochs = train_cfg.get('warmup_epochs', int(epochs * 0.1))
        min_lr = train_cfg.get('min_learning_rate', learning_rate * 0.1)

        def cosine_annealing_with_warmup(epoch, lr):
            """Cosine annealing learning rate schedule with warmup."""
            if epoch < warmup_epochs:
                # Warmup phase: linearly increase LR
                return learning_rate * (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing phase
                import math
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        lr_scheduler_cb = keras.callbacks.LearningRateScheduler(cosine_annealing_with_warmup, verbose=1)
        callbacks.append(lr_scheduler_cb)
        print(f"  ✓ Cosine annealing with warmup: {warmup_epochs} warmup epochs, min_lr={min_lr}")
    else:
        # Default: Reduce learning rate on plateau
        reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_cb)

    # CSV logger
    csv_path = exp_dir / "training_log.csv"
    csv_logger_cb = keras.callbacks.CSVLogger(str(csv_path))
    callbacks.append(csv_logger_cb)

    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    print("="*60)

    # Adjust workers based on data source
    if use_preprocessed:
        # Preprocessed data is already in memory, no need for workers
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # On-the-fly loading benefits from multiprocessing
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
            workers=8,
            use_multiprocessing=True,
            max_queue_size=32
        )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_loss, val_acc, val_top5 = model.evaluate(val_gen, verbose=1)

    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    print(f"  Top-5 Accuracy: {val_top5:.4f}")

    # Save training history and metrics
    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    metrics = {
        'experiment_id': experiment_id,
        'model_type': model_type,
        'base_model': base_model if model_type == 'transfer' else None,
        'val_loss': float(val_loss),
        'val_accuracy': float(val_acc),
        'val_top5_accuracy': float(val_top5),
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'image_size': list(image_size),
        'timestamp': timestamp,
        'config_path': config_path,
        'checkpoint_path': str(checkpoint_path)
    }

    # Add training history
    history_data = {
        'train_loss': [float(x) for x in history.history['loss']],
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }

    # Log to experiment tracker
    tracker.log_training_history(experiment_id, history_data)
    tracker.log_final_results(experiment_id, metrics)

    # Save metrics to experiment directory
    metrics_path = exp_dir / "final_metrics.json"
    save_json(metrics, metrics_path)
    print(f"\nSaved metrics to {metrics_path}")

    # Also save to artifacts for backward compatibility
    artifacts_metrics = artifacts_dir / "cnn_metrics.json"
    save_json(metrics, artifacts_metrics)
    artifacts_history = artifacts_dir / "training_history.json"
    save_json(history_data, artifacts_history)

    # Save final model
    final_model_path = checkpoint_dir / "final_model.h5"
    model.save(final_model_path)
    print(f"Saved final model to {final_model_path}")

    print("\n" + "="*60)
    print("✓ Training pipeline complete!")
    print("="*60)


def main(config_path: str = "configs/cnn_baseline.yaml"):
    """Main entry point for training."""
    train_model(config_path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/cnn_baseline.yaml")
