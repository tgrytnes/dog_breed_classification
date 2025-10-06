"""
CLI tool for managing and comparing experiments.

Usage:
    python -m dbc.experiments list             # List all experiments
    python -m dbc.experiments compare          # Compare all experiments
    python -m dbc.experiments report           # Generate markdown report
"""

import argparse
from pathlib import Path
from .experiment_tracker import get_tracker


def list_experiments(args):
    """List all experiments."""
    tracker = get_tracker(args.experiments_dir)

    # Filter by status if provided
    status = args.status if hasattr(args, 'status') else None
    tags = args.tags.split(',') if hasattr(args, 'tags') and args.tags else None

    experiments = tracker.list_experiments(tags=tags, status=status)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiment(s):\n")
    print(f"{'ID':<25} {'Name':<30} {'Status':<12} {'Tags'}")
    print("-" * 90)

    for exp in experiments:
        exp_id = exp['experiment_id']
        name = exp['name'][:28] + '..' if len(exp['name']) > 30 else exp['name']
        status = exp.get('status', 'unknown')
        tags_str = ', '.join(exp.get('tags', []))[:30]

        print(f"{exp_id:<25} {name:<30} {status:<12} {tags_str}")

    print()


def compare_experiments(args):
    """Compare experiments side-by-side."""
    tracker = get_tracker(args.experiments_dir)

    # Get experiment IDs if provided
    experiment_ids = args.ids.split(',') if hasattr(args, 'ids') and args.ids else None

    df = tracker.compare_experiments(experiment_ids=experiment_ids)

    if df.empty:
        print("No experiments found for comparison.")
        return

    print("\nExperiment Comparison:\n")
    print(df.to_string(index=False))
    print()

    # Show best models
    if 'val_accuracy' in df.columns and not df['val_accuracy'].isna().all():
        best_idx = df['val_accuracy'].idxmax()
        best_exp = df.iloc[best_idx]
        print(f"\n🏆 Best Model: {best_exp['name']}")
        print(f"   Validation Accuracy: {best_exp['val_accuracy']:.4f}")
        print(f"   Top-5 Accuracy: {best_exp.get('val_top5_accuracy', 'N/A')}")
        print()


def generate_report(args):
    """Generate markdown report."""
    tracker = get_tracker(args.experiments_dir)

    output_path = Path(args.output) if hasattr(args, 'output') and args.output else None
    report = tracker.generate_report(output_path)

    print(report)


def show_experiment(args):
    """Show detailed information about a specific experiment."""
    tracker = get_tracker(args.experiments_dir)
    experiment = tracker.get_experiment(args.id)

    if not experiment:
        print(f"Experiment not found: {args.id}")
        return

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment['name']}")
    print(f"{'='*60}")
    print(f"ID: {experiment['experiment_id']}")
    print(f"Status: {experiment.get('status', 'unknown')}")
    print(f"Created: {experiment['created_at']}")
    if experiment.get('tags'):
        print(f"Tags: {', '.join(experiment['tags'])}")
    if experiment.get('description'):
        print(f"Description: {experiment['description']}")

    # Show config
    if 'config' in experiment:
        print(f"\n{'Configuration:'}")
        config = experiment['config']
        if 'train' in config:
            train = config['train']
            print(f"  Model: {train.get('model_type', 'N/A')}")
            if train.get('model_type') == 'transfer':
                print(f"  Base Model: {train.get('base_model', 'N/A')}")
            print(f"  Batch Size: {train.get('batch_size', 'N/A')}")
            print(f"  Learning Rate: {train.get('learning_rate', 'N/A')}")
            print(f"  Epochs: {train.get('epochs', 'N/A')}")

    # Show final results
    if 'final_results' in experiment:
        print(f"\n{'Final Results:'}")
        results = experiment['final_results']
        print(f"  Val Loss: {results.get('val_loss', 'N/A'):.4f}")
        print(f"  Val Accuracy: {results.get('val_accuracy', 'N/A'):.4f}")
        print(f"  Val Top-5 Accuracy: {results.get('val_top5_accuracy', 'N/A'):.4f}")
        print(f"  Epochs Trained: {results.get('epochs_trained', 'N/A')}")
        if 'checkpoint_path' in results:
            print(f"  Best Model: {results['checkpoint_path']}")

    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dog Breed Classification - Experiment Management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--experiments-dir',
        default='experiments',
        help='Directory containing experiments (default: experiments)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument('--status', help='Filter by status (created, configured, completed)')
    list_parser.add_argument('--tags', help='Filter by tags (comma-separated)')
    list_parser.set_defaults(func=list_experiments)

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--ids', help='Comma-separated experiment IDs (default: all completed)')
    compare_parser.set_defaults(func=compare_experiments)

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate markdown report')
    report_parser.add_argument('--output', help='Output file path (default: experiments/experiments_report.md)')
    report_parser.set_defaults(func=generate_report)

    # Show command
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('id', help='Experiment ID')
    show_parser.set_defaults(func=show_experiment)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the command
    args.func(args)


if __name__ == "__main__":
    main()
