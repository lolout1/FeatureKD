#!/usr/bin/env python3
"""Aggregate SE comparison results from multiple model runs."""

import json
import pandas as pd
from pathlib import Path
import sys

def aggregate_results(results_dir: str):
    """Aggregate all model results into a comparison table."""
    results_path = Path(results_dir)
    metrics_dir = results_path / "metrics"

    if not metrics_dir.exists():
        print(f"Error: {metrics_dir} not found")
        return None

    results = []

    for model_dir in sorted(metrics_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        scores_file = model_dir / "scores.csv"
        if not scores_file.exists():
            print(f"Warning: {model_dir.name} - no scores.csv yet")
            continue

        df = pd.read_csv(scores_file)

        # Get data rows (exclude Average)
        df_data = df[df['test_subject'] != 'Average']

        if len(df_data) == 0:
            print(f"Warning: {model_dir.name} - no fold data")
            continue

        # Calculate statistics
        summary = {
            'model': model_dir.name,
            'folds': len(df_data),
            'accuracy_mean': df_data['test_accuracy'].mean(),
            'accuracy_std': df_data['test_accuracy'].std(),
            'f1_mean': df_data['test_f1_score'].mean(),
            'f1_std': df_data['test_f1_score'].std(),
            'precision_mean': df_data['test_precision'].mean(),
            'precision_std': df_data['test_precision'].std(),
            'recall_mean': df_data['test_recall'].mean(),
            'recall_std': df_data['test_recall'].std(),
        }

        # Add AUC if available
        if 'test_auc' in df_data.columns:
            summary['auc_mean'] = df_data['test_auc'].mean()
            summary['auc_std'] = df_data['test_auc'].std()

        results.append(summary)

        # Save individual summary
        summary_file = model_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    if not results:
        print("No results found")
        return None

    # Sort by F1 score descending
    results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)

    # Print comparison table
    print("")
    print("=" * 100)
    print(f"{'Model':<30} {'Folds':>6} {'Accuracy':>16} {'F1 Score':>16} {'Precision':>16} {'Recall':>16}")
    print("=" * 100)

    for r in results:
        print(f"{r['model']:<30} {r['folds']:>6} "
              f"{r['accuracy_mean']:>6.2f} ± {r['accuracy_std']:>5.2f} "
              f"{r['f1_mean']:>6.2f} ± {r['f1_std']:>5.2f} "
              f"{r['precision_mean']:>6.2f} ± {r['precision_std']:>5.2f} "
              f"{r['recall_mean']:>6.2f} ± {r['recall_std']:>5.2f}")

    print("=" * 100)

    # Save combined comparison
    comparison_file = results_path / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump({'models': results, 'results_dir': str(results_path)}, f, indent=2)

    # Save as CSV too
    comparison_df = pd.DataFrame(results)
    comparison_csv = results_path / "comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    print(f"\nResults saved to:")
    print(f"  - {comparison_file}")
    print(f"  - {comparison_csv}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent se_comparison directory
        results_path = Path("results")
        se_dirs = sorted(results_path.glob("se_comparison*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if se_dirs:
            results_dir = str(se_dirs[0])
            print(f"Using most recent: {results_dir}")
        else:
            results_dir = "results/se_comparison"

    aggregate_results(results_dir)
