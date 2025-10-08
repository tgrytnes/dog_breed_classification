"""
Evaluate ensemble of multiple models by averaging their predictions.
Ensemble typically provides 1-3% improvement over single models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
import argparse
import json
from typing import List, Tuple

from dbc.data_loader import DogBreedDataset


def load_models(model_paths: List[str]) -> List[keras.Model]:
    """Load multiple trained models."""
    models = []
    for path in model_paths:
        print(f"Loading: {path}")
        model = keras.models.load_model(path)
        models.append(model)
    return models


def predict_ensemble(
    models: List[keras.Model],
    image: np.ndarray,
    weights: List[float] = None
) -> np.ndarray:
    """
    Make ensemble prediction by averaging predictions from multiple models.

    Args:
        models: List of trained Keras models
        image: Input image (already preprocessed for each model)
        weights: Optional weights for each model (default: equal weights)

    Returns:
        Averaged prediction probabilities (num_classes,)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # Get predictions from all models
    predictions = []
    for model in models:
        pred = model.predict(np.expand_dims(image, 0), verbose=0)[0]
        predictions.append(pred)

    # Weighted average
    ensemble_pred = np.zeros_like(predictions[0])
    for pred, weight in zip(predictions, weights):
        ensemble_pred += pred * weight

    return ensemble_pred


def evaluate_ensemble(
    model_configs: List[dict],
    val_metadata_path: str,
    data_root: str
) -> dict:
    """
    Evaluate ensemble of models on validation set.

    Args:
        model_configs: List of dicts with keys: 'path', 'model_name', 'weight'
        val_metadata_path: Path to validation metadata CSV
        data_root: Root directory containing images

    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ENSEMBLE")
    print("="*70)
    print(f"Number of models: {len(model_configs)}")
    for i, cfg in enumerate(model_configs, 1):
        print(f"  {i}. {cfg['model_name']} (weight: {cfg.get('weight', 1.0/len(model_configs)):.2f})")
        print(f"     {cfg['path']}")
    print("="*70 + "\n")

    # Load all models
    print("Loading models...")
    models_data = []
    for cfg in model_configs:
        model = keras.models.load_model(cfg['path'])
        models_data.append({
            'model': model,
            'model_name': cfg['model_name'],
            'weight': cfg.get('weight', 1.0 / len(model_configs))
        })
    print(f"✓ Loaded {len(models_data)} models\n")

    # Create datasets for each model (different preprocessing)
    print("Creating datasets...")
    val_df = pd.read_csv(val_metadata_path)
    datasets = []
    for cfg in model_configs:
        dataset = DogBreedDataset(
            val_df,
            data_root,
            normalize='imagenet',
            augment=False,
            model_name=cfg['model_name']
        )
        datasets.append(dataset)
    n_samples = len(datasets[0])
    print(f"✓ Created datasets with {n_samples} samples\n")

    # Evaluate ensemble
    print("Evaluating ensemble...")
    top1_correct = 0
    top5_correct = 0

    # Also track individual model performance
    individual_top1 = [0] * len(models_data)
    individual_top5 = [0] * len(models_data)

    for idx in range(n_samples):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{n_samples} images")

        # Get true label (same across all datasets)
        _, label, _ = datasets[0].load_image(idx)

        # Collect predictions from each model
        all_predictions = []
        for i, (model_data, dataset) in enumerate(zip(models_data, datasets)):
            img, _, _ = dataset.load_image(idx)
            pred = model_data['model'].predict(np.expand_dims(img, 0), verbose=0)[0]
            all_predictions.append(pred)

            # Track individual performance
            top5_preds = np.argsort(pred)[-5:][::-1]
            if top5_preds[0] == label:
                individual_top1[i] += 1
            if label in top5_preds:
                individual_top5[i] += 1

        # Ensemble prediction (weighted average)
        weights = [m['weight'] for m in models_data]
        ensemble_pred = np.zeros_like(all_predictions[0])
        for pred, weight in zip(all_predictions, weights):
            ensemble_pred += pred * weight

        # Get top-5 predictions
        top5_preds = np.argsort(ensemble_pred)[-5:][::-1]

        # Update metrics
        if top5_preds[0] == label:
            top1_correct += 1
        if label in top5_preds:
            top5_correct += 1

    # Calculate metrics
    ensemble_top1 = top1_correct / n_samples
    ensemble_top5 = top5_correct / n_samples

    # Print results
    print("\n" + "="*70)
    print("INDIVIDUAL MODEL RESULTS")
    print("="*70)
    for i, model_data in enumerate(models_data):
        top1 = individual_top1[i] / n_samples
        top5 = individual_top5[i] / n_samples
        print(f"{model_data['model_name']}:")
        print(f"  Top-1: {top1*100:.2f}%")
        print(f"  Top-5: {top5*100:.2f}%")

    print("\n" + "="*70)
    print("ENSEMBLE RESULTS")
    print("="*70)
    print(f"Top-1 Accuracy: {ensemble_top1*100:.2f}%")
    print(f"Top-5 Accuracy: {ensemble_top5*100:.2f}%")
    print(f"Total samples: {n_samples}")

    # Show improvement over best individual model
    best_individual_top1 = max(individual_top1) / n_samples
    improvement = ensemble_top1 - best_individual_top1
    print(f"\nImprovement over best individual: {improvement*100:+.2f}%")
    print("="*70)

    # Save results
    results = {
        'ensemble': {
            'top1_accuracy': float(ensemble_top1),
            'top5_accuracy': float(ensemble_top5)
        },
        'individual_models': [
            {
                'model_name': m['model_name'],
                'weight': m['weight'],
                'top1_accuracy': float(individual_top1[i] / n_samples),
                'top5_accuracy': float(individual_top5[i] / n_samples)
            }
            for i, m in enumerate(models_data)
        ],
        'improvement_over_best': float(improvement),
        'n_samples': n_samples
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ensemble of models')
    parser.add_argument('--models', nargs='+', required=True,
                        help='List of model paths')
    parser.add_argument('--model-names', nargs='+', required=True,
                        help='List of model architecture names (same order as --models)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Optional weights for each model (default: equal weights)')
    parser.add_argument('--val-metadata', type=str, default='artifacts/val_metadata.csv',
                        help='Path to validation metadata')
    parser.add_argument('--data-root', type=str, default='data/raw',
                        help='Root directory for images')
    parser.add_argument('--output', type=str, default='ensemble_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    if len(args.models) != len(args.model_names):
        raise ValueError("Number of models must match number of model names")

    if args.weights and len(args.weights) != len(args.models):
        raise ValueError("Number of weights must match number of models")

    # Create model configs
    model_configs = []
    for i, (path, name) in enumerate(zip(args.models, args.model_names)):
        cfg = {'path': path, 'model_name': name}
        if args.weights:
            cfg['weight'] = args.weights[i]
        model_configs.append(cfg)

    # Evaluate
    results = evaluate_ensemble(
        model_configs,
        args.val_metadata,
        args.data_root
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {args.output}")
