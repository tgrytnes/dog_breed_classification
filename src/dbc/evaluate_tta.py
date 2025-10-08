"""
Evaluate model performance using Test-Time Augmentation (TTA).
TTA applies multiple augmentations to each test image and averages predictions for better accuracy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
import argparse
import json

from dbc.data_loader import DogBreedDataset, predict_with_tta


def evaluate_model_with_tta(
    model_path: str,
    val_metadata_path: str,
    data_root: str,
    model_name: str = 'efficientnetb0',
    n_augmentations: int = 5,
    batch_size: int = 32
):
    """
    Evaluate a trained model using Test-Time Augmentation.

    Args:
        model_path: Path to trained model (.h5 file)
        val_metadata_path: Path to validation metadata CSV
        data_root: Root directory containing images
        model_name: Model architecture name for proper preprocessing
        n_augmentations: Number of augmented versions per image (default 5)
        batch_size: Batch size for processing (not used with TTA, kept for compatibility)
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL WITH TEST-TIME AUGMENTATION (TTA)")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Augmentations per image: {n_augmentations}")
    print("="*70 + "\n")

    # Load model
    print("Loading model...")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded: {model_path}")

    # Load validation data (without augmentation)
    print("\nLoading validation dataset...")
    val_df = pd.read_csv(val_metadata_path)
    val_dataset = DogBreedDataset(
        val_df,
        data_root,
        normalize='imagenet',
        augment=False,
        model_name=model_name
    )
    print(f"✓ Loaded {len(val_dataset)} validation images")

    # Evaluate with TTA
    print(f"\nEvaluating with TTA ({n_augmentations} augmentations per image)...")
    print("This will take longer than standard evaluation but provides better accuracy.\n")

    all_predictions = []
    all_labels = []
    top1_correct = 0
    top5_correct = 0

    for idx in range(len(val_dataset)):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(val_dataset)} images processed")

        # Load original image (without preprocessing)
        img, label, _ = val_dataset.load_image(idx)

        # Predict with TTA (handles preprocessing internally)
        pred_probs = predict_with_tta(
            model,
            img,
            n_augmentations=n_augmentations,
            normalize_fn=None  # Already normalized in load_image
        )

        # Get predictions
        top5_preds = np.argsort(pred_probs)[-5:][::-1]
        top1_pred = top5_preds[0]

        # Update metrics
        if top1_pred == label:
            top1_correct += 1
        if label in top5_preds:
            top5_correct += 1

        all_predictions.append(top1_pred)
        all_labels.append(label)

    # Calculate metrics
    n_samples = len(val_dataset)
    top1_accuracy = top1_correct / n_samples
    top5_accuracy = top5_correct / n_samples

    # Print results
    print("\n" + "="*70)
    print("RESULTS WITH TTA")
    print("="*70)
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    print(f"Total samples: {n_samples}")
    print(f"Augmentations per image: {n_augmentations}")
    print("="*70)

    # Save results
    results = {
        'model_path': model_path,
        'n_augmentations': n_augmentations,
        'top1_accuracy': float(top1_accuracy),
        'top5_accuracy': float(top5_accuracy),
        'n_samples': n_samples
    }

    output_path = Path(model_path).parent / 'tta_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")

    return results


def compare_with_without_tta(
    model_path: str,
    val_metadata_path: str,
    data_root: str,
    model_name: str = 'efficientnetb0',
    n_augmentations: int = 5
):
    """
    Compare model performance with and without TTA.
    """
    print("\n" + "="*70)
    print("COMPARING STANDARD EVALUATION vs TTA")
    print("="*70)

    # Load model
    model = keras.models.load_model(model_path)

    # Create dataset
    val_df = pd.read_csv(val_metadata_path)
    val_dataset = DogBreedDataset(
        val_df,
        data_root,
        normalize='imagenet',
        augment=False,
        model_name=model_name
    )

    # Standard evaluation (no TTA)
    print("\n1. Standard evaluation (no augmentation)...")
    top1_correct_std = 0
    top5_correct_std = 0

    for idx in range(len(val_dataset)):
        if idx % 100 == 0:
            print(f"  Standard eval: {idx}/{len(val_dataset)}")
        img, label, _ = val_dataset.load_image(idx)
        pred_probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        top5_preds = np.argsort(pred_probs)[-5:][::-1]

        if top5_preds[0] == label:
            top1_correct_std += 1
        if label in top5_preds:
            top5_correct_std += 1

    std_top1 = top1_correct_std / len(val_dataset)
    std_top5 = top5_correct_std / len(val_dataset)

    # TTA evaluation
    print(f"\n2. TTA evaluation ({n_augmentations} augmentations)...")
    top1_correct_tta = 0
    top5_correct_tta = 0

    for idx in range(len(val_dataset)):
        if idx % 100 == 0:
            print(f"  TTA eval: {idx}/{len(val_dataset)}")
        img, label, _ = val_dataset.load_image(idx)
        pred_probs = predict_with_tta(model, img, n_augmentations=n_augmentations)
        top5_preds = np.argsort(pred_probs)[-5:][::-1]

        if top5_preds[0] == label:
            top1_correct_tta += 1
        if label in top5_preds:
            top5_correct_tta += 1

    tta_top1 = top1_correct_tta / len(val_dataset)
    tta_top5 = top5_correct_tta / len(val_dataset)

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"Standard Top-1: {std_top1*100:.2f}%")
    print(f"TTA Top-1:      {tta_top1*100:.2f}% (Δ {(tta_top1-std_top1)*100:+.2f}%)")
    print(f"\nStandard Top-5: {std_top5*100:.2f}%")
    print(f"TTA Top-5:      {tta_top5*100:.2f}% (Δ {(tta_top5-std_top5)*100:+.2f}%)")
    print("="*70)

    results = {
        'standard': {'top1': float(std_top1), 'top5': float(std_top5)},
        'tta': {'top1': float(tta_top1), 'top5': float(tta_top5)},
        'improvement': {
            'top1': float(tta_top1 - std_top1),
            'top5': float(tta_top5 - std_top5)
        }
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with TTA')
    parser.add_argument('model_path', type=str, help='Path to trained model (.h5)')
    parser.add_argument('--model-name', type=str, default='efficientnetb0',
                        help='Model architecture name (efficientnetb0, resnet50, etc.)')
    parser.add_argument('--val-metadata', type=str, default='artifacts/val_metadata.csv',
                        help='Path to validation metadata')
    parser.add_argument('--data-root', type=str, default='data/raw',
                        help='Root directory for images')
    parser.add_argument('--n-aug', type=int, default=5,
                        help='Number of augmentations per image (default 5)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with and without TTA')

    args = parser.parse_args()

    if args.compare:
        compare_with_without_tta(
            args.model_path,
            args.val_metadata,
            args.data_root,
            args.model_name,
            args.n_aug
        )
    else:
        evaluate_model_with_tta(
            args.model_path,
            args.val_metadata,
            args.data_root,
            args.model_name,
            args.n_aug
        )
