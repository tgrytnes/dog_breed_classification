# MLOps Project Template

A **streamlined, reusable template** for machine learning projects that prioritizes simplicity and ease of use.

This template provides a **config-driven pipeline** with minimal abstractions, making it easy to understand, modify, and extend for your specific ML project needs.

## Key Features

- **Simple, inline model registry** - All models defined in a single dictionary in `train.py`
- **No complex abstractions** - Direct sklearn/xgboost usage without wrapper layers
- **Comprehensive metrics tracking** - ROC-AUC, PR-AUC, predictions, feature importance
- **Experiment history** - Automatic tracking via `results.json` and `results.csv`
- **Flexible data splitting** - Chronological or random splits based on your config
- **Smart feature scaling** - Automatic scaling for logistic regression, raw features for tree models

---

## How to Use This Template

⚡ **Quick Start:**
1. Click **"Use this template"** on GitHub to create your project repository
2. Initialize your project with the setup script (see below)
3. Update `configs/exp_baseline.yaml` for your task
4. Run the pipeline: `bash scripts/run_train.sh configs/exp_baseline.yaml`

**Do not develop directly in this template repository** - keep it as your clean base template.

---

## Initialize a New Project

Use the one-time init script to rename the template package and metadata:

```bash
# Dry run to preview changes
python scripts/init_project.py \
  --package my_project \
  --dist-name my-project \
  --title "My Project" \
  --kernel-name my-project-venv \
  --dry-run

# Apply changes
python scripts/init_project.py \
  --package my_project \
  --dist-name my-project \
  --title "My Project" \
  --kernel-name my-project-venv
```

This script:
- Renames `src/yourproj` to `src/<package>`
- Updates imports and metadata (pyproject.toml, kernel name)
- Adjusts titles in docs and notebook imports

---

## Quickstart

```bash
# 1. Set up environment (creates venv, installs dependencies)
source bootstrap_env.sh

# 2. Run feature engineering (customize src/yourproj/features.py)
python src/yourproj/features.py

# 3. Train models
bash scripts/run_train.sh configs/exp_baseline.yaml

# 4. View results
cat artifacts/results.csv
```

---

## Models Tested

This project tested multiple CNN architectures for dog breed classification:

### Baseline Training (Frozen Base)
| Architecture | Top-1 Accuracy | Top-5 Accuracy | Image Size |
|-------------|----------------|----------------|------------|
| ResNet50 | 54.69% | 89.54% | 224×224 |
| EfficientNetB0 | 83.96% | 97.62% | 224×224 |
| EfficientNetB4 | 93.71% | 99.63% | 380×380 |
| EfficientNetB5 | 94.55% | 99.71% | 456×456 |
| **EfficientNetV2-S** | **95.04%** | **99.53%** | **384×384** |

### Fine-Tuning (20 Layers Unfrozen)
| Architecture | Top-1 Accuracy | Top-5 Accuracy | Improvement |
|-------------|----------------|----------------|-------------|
| ResNet50 | 77.68% | 95.24% | +22.99% |
| EfficientNetB0 | 84.43% | 97.86% | +0.47% |
| EfficientNetB4 | 93.86% | 99.63% | +0.15% |
| **EfficientNetV2-S** | **95.14%** | **99.51%** | **+0.10%** |

### Advanced Techniques
| Technique | Top-1 Accuracy | Top-5 Accuracy | Notes |
|-----------|----------------|----------------|-------|
| Test-Time Augmentation (5 aug) | 95.33% | 99.53% | +0.19%, 5× slower |
| 3-Model Ensemble | 95.04% | 99.68% | Worse than single best model |

**Best Model:** EfficientNetV2-S Fine-tuned (95.14% top-1, 99.51% top-5)

See [EXPERIMENT_RESULTS_2025_10_08.md](docs/EXPERIMENT_RESULTS_2025_10_08.md) for full details.

---

## Configuration

All experiments are defined via YAML configs in `configs/`:

```yaml
seed: 42
paths:
  raw_data: data/raw
  artifacts: artifacts
task: your_project_name
train:
  target: y                    # Target column name
  test_size: 0.25              # Train/test split ratio
  date_col: null               # Optional: column for chronological split
  model: "logreg_baseline"     # Model to train
  model_args: {}               # Model-specific parameters
```

---

## Repository Structure

```
.
├── artifacts/                  # Model outputs, metrics, predictions
│   ├── results.csv             # All experiment results
│   ├── results.json            # Detailed experiment history
│   ├── features.csv            # Engineered features
│   ├── metrics.json            # Latest run metrics
│   ├── predictions.json        # Test set predictions
│   ├── feature_importance.json # Feature importance (tree models)
│   └── checkpoints/            # Saved model files
├── configs/                    # Experiment configurations (YAML)
│   └── exp_baseline.yaml       # Baseline experiment config
├── data/
│   └── raw/                    # Raw input data
├── notebooks/                  # Jupyter notebooks for analysis
├── src/yourproj/               # Source code package
│   ├── train.py                # Main training script
│   ├── features.py             # Feature engineering
│   ├── preprocess.py           # Data preprocessing
│   ├── ingest.py               # Data loading
│   ├── eval.py                 # Model evaluation
│   ├── smoke.py                # Smoke tests
│   └── utils.py                # Utility functions
├── scripts/
│   ├── run_train.sh            # Training pipeline entrypoint
│   └── init_project.py         # Project initialization
├── pyproject.toml              # Package configuration
├── bootstrap_env.sh            # Environment setup script
└── README.md                   # This file
```

---

## Metrics & Outputs

Each training run automatically saves:

1. **metrics.json** - Comprehensive metrics:
   - ROC-AUC, PR-AUC, accuracy
   - Train/test sizes and target rates
   - Model configuration
   - Timestamp

2. **predictions.json** - Test set predictions for visualization

3. **feature_importance.json** - Feature importance (tree models only)

4. **results.csv** - Flat file for quick experiment comparison

5. **checkpoints/model.pkl** - Trained model for later use

---

## Next Steps

After initializing your project:

1. **Update `src/yourproj/ingest.py`** - Load your raw data
2. **Update `src/yourproj/preprocess.py`** - Clean and prepare data
3. **Update `src/yourproj/features.py`** - Engineer features
4. **Create experiment configs** - Add configs for different model/hyperparameter combinations
5. **Run experiments** - Execute training with different configs
6. **Analyze results** - Use notebooks to compare `artifacts/results.csv`

---

## Design Principles

This template follows these principles from the simplified Hotel Cancellation Risk project:

- **Simplicity over abstraction** - Direct code is easier to understand and modify
- **Self-contained training** - Everything needed in one `train.py` file
- **Comprehensive tracking** - Save everything for reproducibility
- **Flexible defaults** - Smart defaults with easy override options
- **Clear data flow** - Explicit data loading, processing, and training steps

---

## License

This project template is provided as-is for ML project development.
