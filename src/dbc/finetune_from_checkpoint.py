"""
Fine-tune a frozen model from its checkpoint.

This script loads a trained frozen model and fine-tunes it with unfrozen layers.
This prevents catastrophic forgetting by starting from already-learned features.
"""

import argparse
from pathlib import Path
import yaml
from tensorflow import keras
from .data_loader import create_preprocessed_loaders, create_data_loaders
import json
from datetime import datetime


def finetune_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    learning_rate: float = 0.00001,
    epochs: int = 20,
    batch_size: int = 64
):
    """
    Fine-tune a model from a frozen checkpoint.

    Args:
        checkpoint_path: Path to frozen model checkpoint (.h5 file)
        config_path: Path to config file for experiment setup
        learning_rate: Learning rate for fine-tuning (very low)
        epochs: Number of fine-tuning epochs
        batch_size: Batch size for fine-tuning
    """

    print("="*60)
    print("FINE-TUNING FROM CHECKPOINT")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load the frozen model
    print(f"\nLoading frozen model from: {checkpoint_path}")
    model = keras.models.load_model(checkpoint_path)

    print("\nModel loaded successfully!")
    print(f"Total parameters: {model.count_params():,}")

    # Find the base model layer and unfreeze it
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is likely the base model
            base_model = layer
            break

    if base_model:
        print(f"\nUnfreezing base model: {base_model.name}")
        base_model.trainable = True
        print(f"Trainable parameters: {sum([keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    else:
        print("\nWarning: Could not find base model to unfreeze. Unfreezing all layers.")
        for layer in model.layers:
            layer.trainable = True

    # Recompile with lower learning rate
    print(f"\nRecompiling model with learning rate: {learning_rate}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )

    # Create data loaders
    print("\nCreating data loaders...")
    preprocessed_dir = Path("artifacts/preprocessed")

    if (preprocessed_dir / "train_data_images.npy").exists():
        train_gen, val_gen = create_preprocessed_loaders(
            preprocessed_dir=preprocessed_dir,
            batch_size=batch_size,
            augment_train=True,
            seed=42
        )
    else:
        train_gen, val_gen = create_data_loaders(
            train_metadata_path=Path(config['paths']['artifacts']) / 'train_metadata.csv',
            val_metadata_path=Path(config['paths']['artifacts']) / 'val_metadata.csv',
            data_root=Path(config['paths']['data']),
            batch_size=batch_size,
            image_size=tuple(config['train']['image_size']),
            normalize='imagenet',
            augment_train=True,
            seed=42
        )

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp_{timestamp}"
    exp_dir = Path(config['paths']['experiments']) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    finetune_config = config.copy()
    finetune_config['experiment_name'] = config.get('experiment_name', '') + " - Fine-tuned"
    finetune_config['train']['learning_rate'] = learning_rate
    finetune_config['train']['batch_size'] = batch_size
    finetune_config['train']['epochs'] = epochs
    finetune_config['train']['trainable_base'] = True
    finetune_config['parent_checkpoint'] = checkpoint_path

    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(finetune_config, f, indent=2)

    print(f"\nCreated experiment: {exp_id}")

    # Setup callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(exp_dir / 'checkpoints' / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=str(exp_dir / 'training_log.csv')
        )
    ]

    # Create checkpoint directory
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)

    # Train
    print("\n" + "="*60)
    print(f"Starting fine-tuning for {epochs} epochs...")
    print("="*60)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final results
    final_results = {
        'experiment_id': exp_id,
        'model_type': 'transfer',
        'base_model': config['train']['base_model'],
        'val_loss': float(history.history['val_loss'][-1]),
        'val_accuracy': float(history.history['val_accuracy'][-1]),
        'val_top5_accuracy': float(history.history['val_top5_accuracy'][-1]),
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'image_size': config['train']['image_size'],
        'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        'config_path': config_path,
        'checkpoint_path': str(exp_dir / 'checkpoints' / 'best_model.h5'),
        'parent_checkpoint': checkpoint_path,
        'completed_at': datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    }

    with open(exp_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Final validation accuracy: {final_results['val_accuracy']:.4f}")
    print(f"Final top-5 accuracy: {final_results['val_top5_accuracy']:.4f}")
    print(f"Results saved to: {exp_dir}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='Fine-tune from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to frozen model checkpoint')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    finetune_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
