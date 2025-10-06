"""
Experiment tracking for model comparison and hyperparameter tuning.

This module provides tools to:
- Track multiple training runs
- Compare model performance
- Store hyperparameters and results
- Generate comparison reports
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from .utils import save_json, ensure_dir


class ExperimentTracker:
    """
    Track multiple experiments for model comparison.

    Each experiment consists of:
    - Unique experiment ID
    - Configuration (hyperparameters, model architecture)
    - Training metrics (loss, accuracy curves)
    - Validation results
    - Model artifacts (checkpoints, saved models)
    """

    def __init__(self, experiments_dir: Path):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiment logs
        """
        self.experiments_dir = Path(experiments_dir)
        ensure_dir(self.experiments_dir)

        self.registry_path = self.experiments_dir / "experiments_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load experiment registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'experiments': []}

    def _save_registry(self):
        """Save experiment registry to disk."""
        save_json(self.registry, self.registry_path)

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            description: Optional description
            tags: Optional tags for filtering (e.g., ['resnet50', 'baseline'])

        Returns:
            experiment_id: Unique ID for this experiment
        """
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        experiment_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        experiment = {
            'experiment_id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'created_at': timestamp,
            'status': 'created'
        }

        self.registry['experiments'].append(experiment)
        self._save_registry()

        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        ensure_dir(exp_dir)
        ensure_dir(exp_dir / "checkpoints")

        print(f"Created experiment: {experiment_id}")
        return experiment_id

    def log_config(self, experiment_id: str, config: Dict[str, Any]):
        """
        Log experiment configuration.

        Args:
            experiment_id: Experiment ID
            config: Configuration dictionary (hyperparameters, model settings)
        """
        exp_dir = self.experiments_dir / experiment_id
        config_path = exp_dir / "config.json"
        save_json(config, config_path)

        # Update registry
        for exp in self.registry['experiments']:
            if exp['experiment_id'] == experiment_id:
                exp['config'] = config
                exp['status'] = 'configured'
                break
        self._save_registry()

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Log metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            metrics: Dictionary of metrics (e.g., {'val_loss': 0.5, 'val_accuracy': 0.85})
            step: Optional step/epoch number
        """
        exp_dir = self.experiments_dir / experiment_id
        metrics_file = exp_dir / "metrics.json"

        # Load existing metrics
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {'history': []}

        # Add timestamp
        metrics['timestamp'] = datetime.utcnow().isoformat(timespec="seconds")
        if step is not None:
            metrics['step'] = step

        all_metrics['history'].append(metrics)

        # Keep summary of latest metrics
        all_metrics['latest'] = metrics

        save_json(all_metrics, metrics_file)

    def log_training_history(
        self,
        experiment_id: str,
        history: Dict[str, List[float]]
    ):
        """
        Log complete training history.

        Args:
            experiment_id: Experiment ID
            history: Training history from model.fit() (e.g., {'loss': [...], 'val_loss': [...]})
        """
        exp_dir = self.experiments_dir / experiment_id
        history_path = exp_dir / "training_history.json"
        save_json(history, history_path)

    def log_final_results(
        self,
        experiment_id: str,
        results: Dict[str, Any]
    ):
        """
        Log final experiment results.

        Args:
            experiment_id: Experiment ID
            results: Final results (val metrics, test metrics, model paths)
        """
        exp_dir = self.experiments_dir / experiment_id
        results_path = exp_dir / "final_results.json"

        results['completed_at'] = datetime.utcnow().isoformat(timespec="seconds")
        save_json(results, results_path)

        # Update registry
        for exp in self.registry['experiments']:
            if exp['experiment_id'] == experiment_id:
                exp['final_results'] = results
                exp['status'] = 'completed'
                break
        self._save_registry()

        print(f"Logged final results for experiment: {experiment_id}")

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment details by ID."""
        for exp in self.registry['experiments']:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None

    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None
    ) -> List[Dict]:
        """
        List all experiments, optionally filtered by tags or status.

        Args:
            tags: Filter by tags (returns experiments with ANY of these tags)
            status: Filter by status ('created', 'configured', 'completed')

        Returns:
            List of experiment dictionaries
        """
        experiments = self.registry['experiments']

        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp.get('tags', []) for tag in tags)
            ]

        if status:
            experiments = [
                exp for exp in experiments
                if exp.get('status') == status
            ]

        return experiments

    def compare_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare experiments side-by-side.

        Args:
            experiment_ids: List of experiment IDs to compare (None = all completed)
            metrics: List of metrics to compare (None = all available)

        Returns:
            DataFrame with experiment comparison
        """
        if experiment_ids is None:
            # Get all completed experiments
            experiments = self.list_experiments(status='completed')
        else:
            experiments = [self.get_experiment(eid) for eid in experiment_ids]
            experiments = [e for e in experiments if e is not None]

        if not experiments:
            print("No experiments found for comparison")
            return pd.DataFrame()

        # Build comparison table
        rows = []
        for exp in experiments:
            row = {
                'experiment_id': exp['experiment_id'],
                'name': exp['name'],
                'created_at': exp['created_at'],
                'tags': ', '.join(exp.get('tags', []))
            }

            # Add config info
            if 'config' in exp:
                config = exp['config']
                if 'train' in config:
                    row['model_type'] = config['train'].get('model_type', 'N/A')
                    row['base_model'] = config['train'].get('base_model', 'N/A')
                    row['batch_size'] = config['train'].get('batch_size', 'N/A')
                    row['learning_rate'] = config['train'].get('learning_rate', 'N/A')
                    row['epochs'] = config['train'].get('epochs', 'N/A')

            # Add final results
            if 'final_results' in exp:
                results = exp['final_results']
                row['val_accuracy'] = results.get('val_accuracy', 'N/A')
                row['val_top5_accuracy'] = results.get('val_top5_accuracy', 'N/A')
                row['val_loss'] = results.get('val_loss', 'N/A')
                row['epochs_trained'] = results.get('epochs_trained', 'N/A')

            rows.append(row)

        df = pd.DataFrame(rows)

        # Filter by requested metrics if specified
        if metrics:
            cols_to_keep = ['experiment_id', 'name', 'created_at', 'tags'] + metrics
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[cols_to_keep]

        return df

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate markdown report of all experiments.

        Args:
            output_path: Path to save report (default: experiments_dir/report.md)

        Returns:
            Report as string
        """
        if output_path is None:
            output_path = self.experiments_dir / "experiments_report.md"

        report = []
        report.append("# Dog Breed Classification - Experiment Report")
        report.append(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"\nTotal experiments: {len(self.registry['experiments'])}")

        # Summary by status
        status_counts = {}
        for exp in self.registry['experiments']:
            status = exp.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        report.append("\n## Experiment Status")
        for status, count in sorted(status_counts.items()):
            report.append(f"- {status}: {count}")

        # Compare completed experiments
        completed_exps = self.list_experiments(status='completed')
        if completed_exps:
            report.append("\n## Completed Experiments")
            df = self.compare_experiments()
            report.append("\n" + df.to_markdown(index=False))

            # Best performing models
            if 'val_accuracy' in df.columns and not df['val_accuracy'].isna().all():
                df_sorted = df.sort_values('val_accuracy', ascending=False)
                report.append("\n### Top 5 Models by Validation Accuracy")
                report.append("\n" + df_sorted.head(5)[
                    ['name', 'model_type', 'base_model', 'val_accuracy', 'val_top5_accuracy']
                ].to_markdown(index=False))

        # Individual experiment details
        report.append("\n## Experiment Details")
        for exp in self.registry['experiments']:
            report.append(f"\n### {exp['name']} ({exp['experiment_id']})")
            report.append(f"- **Status**: {exp.get('status', 'unknown')}")
            report.append(f"- **Created**: {exp['created_at']}")
            if exp.get('tags'):
                report.append(f"- **Tags**: {', '.join(exp['tags'])}")
            if exp.get('description'):
                report.append(f"- **Description**: {exp['description']}")

            if 'final_results' in exp:
                results = exp['final_results']
                report.append(f"- **Val Accuracy**: {results.get('val_accuracy', 'N/A'):.4f}")
                report.append(f"- **Val Top-5 Accuracy**: {results.get('val_top5_accuracy', 'N/A'):.4f}")
                report.append(f"- **Val Loss**: {results.get('val_loss', 'N/A'):.4f}")
                report.append(f"- **Epochs Trained**: {results.get('epochs_trained', 'N/A')}")

        report_text = '\n'.join(report)

        # Save report
        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"\nGenerated experiment report: {output_path}")
        return report_text


def get_tracker(experiments_dir: str = "experiments") -> ExperimentTracker:
    """
    Get or create experiment tracker.

    Args:
        experiments_dir: Directory for experiment logs

    Returns:
        ExperimentTracker instance
    """
    return ExperimentTracker(Path(experiments_dir))
