from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
from .utils import ensure_dir, save_json
import numpy as np
import yaml

def load_config(path: str):
    """Simple YAML config loader."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Simple model registry - easy to follow and extend
MODELS = {
    'logreg_baseline': lambda **kw: LogisticRegression(
        class_weight='balanced', random_state=42, max_iter=1000, **kw
    ),
    'logreg_l1': lambda **kw: LogisticRegression(
        penalty=kw.get('penalty', 'l1'), C=kw.get('C', 0.1),
        class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000,
        **{k: v for k, v in kw.items() if k not in ['C', 'penalty']}
    ),
    'logreg_l2': lambda **kw: LogisticRegression(
        penalty=kw.get('penalty', 'l2'), C=kw.get('C', 1.0),
        class_weight='balanced', solver='liblinear', random_state=42, max_iter=1000,
        **{k: v for k, v in kw.items() if k not in ['C', 'penalty']}
    ),
    'random_forest': lambda **kw: RandomForestClassifier(
        class_weight='balanced', random_state=42,
        n_estimators=kw.get('n_estimators', 100),
        max_depth=kw.get('max_depth', None),
        **{k: v for k, v in kw.items() if k not in ['n_estimators', 'max_depth']}
    ),
    'xgboost': lambda **kw: _build_xgboost(**kw) if XGBOOST_AVAILABLE else None
}

def _build_xgboost(**kw):
    """Build XGBoost classifier with proper parameter handling."""
    from xgboost import XGBClassifier as XGB

    # Calculate scale_pos_weight based on class imbalance if needed
    scale_pos_weight = kw.get('scale_pos_weight', 1.0)

    return XGB(
        learning_rate=kw.get('learning_rate', 0.1),
        max_depth=kw.get('max_depth', 5),
        n_estimators=kw.get('n_estimators', 100),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        **{k: v for k, v in kw.items() if k not in [
            'learning_rate', 'max_depth', 'n_estimators', 'scale_pos_weight'
        ]}
    )

def arrays_from_dataframe(df, feature_cols, target_col):
    """Convert DataFrame to numpy arrays for training."""
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].astype(int).values.astype(np.int32)
    return X, y

def chrono_split(df: pd.DataFrame, date_col: str = None, test_size=0.25):
    """Split data chronologically by date column, or randomly if no date column."""
    if date_col is None or date_col not in df.columns:
        # Fallback to random split if no date column
        return train_test_split(df, test_size=test_size, random_state=42, stratify=df.get('y'))

    df = df.sort_values(date_col)
    n = len(df)
    n_test = max(1, int(round(n*test_size)))
    return df.iloc[:-n_test], df.iloc[-n_test:]

def main(config_path: str = "configs/exp_baseline.yaml"):
    cfg = load_config(config_path)

    # Load processed features
    feats_path = Path(cfg['paths']['artifacts']) / "features.csv"
    print(f"Loading features from {feats_path}")
    feats = pd.read_csv(feats_path)

    # Ensure target column exists
    target_col = cfg['train'].get('target', 'y')
    if target_col not in feats.columns:
        raise ValueError(f"Target column '{target_col}' not found in features")

    # Remove rows with missing target
    feats = feats.dropna(subset=[target_col])
    print(f"Dataset shape: {feats.shape}, Target rate: {feats[target_col].mean():.3f}")

    # Get feature columns (exclude target)
    if 'features' in cfg['train']:
        Xcols = cfg['train']['features']
    else:
        # Use all columns except target as features
        Xcols = [col for col in feats.columns if col != target_col]

    print(f"Using {len(Xcols)} features for training")

    # Split data chronologically if possible, otherwise randomly
    test_size = cfg['train'].get('test_size', 0.25)
    train, test = chrono_split(feats, cfg['train'].get('date_col'), test_size)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # Convert to arrays
    Xtr, ytr = arrays_from_dataframe(train, Xcols, target_col)
    Xte, yte = arrays_from_dataframe(test, Xcols, target_col)

    # Scale features for logistic regression (tree models don't need scaling)
    model_name = cfg['train'].get('model', 'logreg_baseline')
    if model_name.startswith('logreg'):
        print("Scaling features for logistic regression...")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
    elif model_name in ['random_forest', 'xgboost']:
        print(f"Using raw features for {model_name} (no scaling needed)...")

    # Simple model creation and training
    model_args = dict(cfg['train'].get('model_args', {}))

    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODELS.keys())}")

    print(f"Training {model_name} model...")
    model = MODELS[model_name](**model_args)
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    # Calculate comprehensive metrics
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, prob)) if len(set(yte)) > 1 else None,
        "n_test": int(len(yte)),
        "n_train": int(len(ytr)),
        "features": Xcols,
        "target_rate_test": float(yte.mean()),
        "target_rate_train": float(ytr.mean()),
    }

    # Add PR-AUC (important for imbalanced data)
    if len(set(yte)) > 1:
        precision, recall, _ = precision_recall_curve(yte, prob)
        metrics["pr_auc"] = float(auc(recall, precision))

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    metrics["model"] = model_name
    metrics["model_args"] = model_args
    metrics["config_path"] = config_path
    metrics["timestamp_utc"] = timestamp

    # Save metrics
    out = Path(cfg['paths']['artifacts']) / "metrics.json"
    ensure_dir(out.parent)
    save_json(metrics, out)

    # Track run history for easy comparison across experiments
    summary = {k: v for k, v in metrics.items() if k != "features"}
    summary["num_features"] = len(Xcols)
    summary["features"] = Xcols
    summary["train_rows"] = len(train)
    summary["test_rows"] = len(test)

    results_path = Path(cfg['paths']['artifacts']) / "results.json"
    if results_path.exists():
        try:
            history = json.loads(results_path.read_text())
            if not isinstance(history, list):
                history = [history]
        except json.JSONDecodeError:
            history = []
    else:
        history = []
    history.append(summary)
    save_json(history, results_path)

    # Persist a flat CSV for quick inspection / plotting
    csv_summary = summary.copy()
    csv_summary["model_args"] = json.dumps(model_args, sort_keys=True)
    csv_summary["features"] = "|".join(Xcols)

    results_csv = Path(cfg['paths']['artifacts']) / "results.csv"
    df_summary = pd.DataFrame([csv_summary])
    if results_csv.exists():
        try:
            existing = pd.read_csv(results_csv)
            df_summary = pd.concat([existing, df_summary], ignore_index=True)
        except Exception:
            pass
    df_summary.to_csv(results_csv, index=False)

    # Save predictions and probabilities for visualization
    pred_data = {
        'y_true': yte.tolist(),
        'y_pred': pred.tolist(),
        'y_prob': prob.tolist(),
        'model': model_name,
        'timestamp': timestamp
    }
    pred_path = Path(cfg['paths']['artifacts']) / "predictions.json"
    save_json(pred_data, pred_path)

    # Save feature importance for tree-based models
    if hasattr(model, 'feature_importances_'):
        importance_data = {
            'feature_names': Xcols,
            'importance_values': model.feature_importances_.tolist(),
            'model': model_name,
            'timestamp': timestamp
        }
        importance_path = Path(cfg['paths']['artifacts']) / "feature_importance.json"
        save_json(importance_data, importance_path)
        print(f"Saved feature importance to {importance_path}")

    # Save model checkpoint
    ckpt_dir = Path(cfg['paths']['artifacts']) / "checkpoints"
    ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / "model.pkl"
    try:
        import pickle
        with open(ckpt_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model to {ckpt_path}")
    except Exception as e:
        print(f"Warning: failed to save model checkpoint: {e}")

    print(f"Saved metrics to {out}")
    print(f"Saved predictions to {pred_path}")
    print(f"Model performance: ROC-AUC={metrics.get('roc_auc', 'N/A'):.3f}, PR-AUC={metrics.get('pr_auc', 'N/A'):.3f}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else "configs/exp_baseline.yaml")
