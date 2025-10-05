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

## Available Models

The template includes common sklearn/xgboost models with sensible defaults:

| Model Name | Algorithm | Use Case |
|------------|-----------|----------|
| `logreg_baseline` | Logistic Regression | Baseline, interpretable |
| `logreg_l1` | Logistic Regression (L1) | Feature selection |
| `logreg_l2` | Logistic Regression (L2) | Regularization |
| `random_forest` | Random Forest | Robust, non-linear |
| `xgboost` | XGBoost | High performance |

### Model Configuration

Configure models in your `configs/exp_*.yaml`:

```yaml
train:
  model: "random_forest"
  model_args:
    n_estimators: 300
    max_depth: 10
```

**Adding New Models:**
Simply add to the `MODELS` dictionary in [train.py](src/yourproj/train.py):

```python
MODELS = {
    'my_custom_model': lambda **kw: MyModelClass(**kw),
    ...
}
```

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
