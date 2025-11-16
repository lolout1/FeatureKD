"""
Standalone Results Analyzer for Dual-Stream Model Comparison

Use this script to analyze and visualize results from completed model runs.

Usage:
    python analyze_dualstream_results.py --work-dir work_dir/dualstream_comparison_TIMESTAMP
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def setup_argparser():
    parser = argparse.ArgumentParser(description='Analyze dual-stream comparison results')
    parser.add_argument('--work-dir', type=str, required=True,
                       help='Path to the comparison work directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['baseline', 'shared', 'light', 'asymmetric'],
                       help='Model names to analyze')
    return parser


def load_model_results(work_dir, model_name):
    """Load results for a single model"""
    model_dir = os.path.join(work_dir, model_name)

    if not os.path.exists(model_dir):
        print(f"Warning: {model_dir} not found, skipping {model_name}")
        return None

    results = {}

    # Load scores
    scores_path = os.path.join(model_dir, 'scores.csv')
    if os.path.exists(scores_path):
        results['scores'] = pd.read_csv(scores_path)
        results['scores']['model'] = model_name
    else:
        print(f"Warning: No scores.csv found for {model_name}")
        return None

    # Load training log
    log_path = os.path.join(model_dir, 'training_log.csv')
    if os.path.exists(log_path):
        results['training_log'] = pd.read_csv(log_path)
        results['training_log']['model'] = model_name
    else:
        results['training_log'] = None

    return results


def print_detailed_statistics(all_results):
    """Print detailed statistical comparison"""
    print("\n" + "="*100)
    print("DETAILED STATISTICAL COMPARISON")
    print("="*100 + "\n")

    metrics = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_auc']
    metric_names = {
        'test_accuracy': 'Accuracy',
        'test_f1_score': 'F1-Score',
        'test_precision': 'Precision',
        'test_recall': 'Recall',
        'test_auc': 'AUC'
    }

    for metric in metrics:
        print(f"\n{metric_names[metric]} (%):")
        print("-" * 80)

        stats = []
        for model_name, results in all_results.items():
            if results and results['scores'] is not None:
                # Exclude "Average" row
                data = results['scores'][results['scores']['test_subject'] != 'Average'][metric]

                stats.append({
                    'Model': model_name,
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Median': data.median(),
                    'Q1': data.quantile(0.25),
                    'Q3': data.quantile(0.75)
                })

        if stats:
            stats_df = pd.DataFrame(stats)
            stats_df = stats_df.round(2)
            print(stats_df.to_string(index=False))

    print("\n" + "="*100 + "\n")


def plot_training_curves(all_results, output_dir):
    """Plot training curves for all models"""
    print("Generating training curve plots...")

    # For each metric, plot all models together
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('accuracy', 'Accuracy (%)'),
        ('f1_score', 'F1-Score (%)')
    ]

    for metric_key, metric_name in metrics_to_plot:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        for phase_idx, phase in enumerate(['train', 'val', 'test']):
            ax = axes[phase_idx]

            for model_name, results in all_results.items():
                if results and results['training_log'] is not None:
                    log = results['training_log']
                    phase_data = log[log['phase'] == phase]

                    if not phase_data.empty and metric_key in phase_data.columns:
                        # Aggregate across folds
                        grouped = phase_data.groupby('epoch')[metric_key].agg(['mean', 'std'])

                        ax.plot(grouped.index, grouped['mean'], label=model_name, linewidth=2)
                        ax.fill_between(grouped.index,
                                       grouped['mean'] - grouped['std'],
                                       grouped['mean'] + grouped['std'],
                                       alpha=0.2)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{phase.capitalize()} {metric_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'training_curves_{metric_key}.png'), dpi=300)
        plt.close()

    print(f"✓ Training curves saved to {output_dir}/")


def create_parameter_efficiency_plot(all_results, output_dir):
    """Plot accuracy vs model parameters"""
    print("Creating parameter efficiency plot...")

    # Model parameter counts (approximate)
    param_counts = {
        'baseline': 110000,
        'shared': 15000,
        'light': 25000,
        'asymmetric': 35000
    }

    plot_data = []
    for model_name, results in all_results.items():
        if results and results['scores'] is not None:
            avg_row = results['scores'][results['scores']['test_subject'] == 'Average']
            if not avg_row.empty:
                plot_data.append({
                    'Model': model_name,
                    'Parameters': param_counts.get(model_name, 0),
                    'Test Accuracy': avg_row['test_accuracy'].values[0],
                    'Test F1': avg_row['test_f1_score'].values[0]
                })

    if plot_data:
        df = pd.DataFrame(plot_data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Accuracy vs Parameters
        for idx, row in df.iterrows():
            ax1.scatter(row['Parameters'], row['Test Accuracy'],
                       s=300, alpha=0.7, label=row['Model'])
            ax1.text(row['Parameters'], row['Test Accuracy'],
                    f"  {row['Model']}\n  ({row['Parameters']/1000:.0f}K)",
                    fontsize=9, va='center')

        ax1.set_xlabel('Number of Parameters', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_title('Parameter Efficiency: Accuracy vs Model Size', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)

        # F1 vs Parameters
        for idx, row in df.iterrows():
            ax2.scatter(row['Parameters'], row['Test F1'],
                       s=300, alpha=0.7, label=row['Model'])
            ax2.text(row['Parameters'], row['Test F1'],
                    f"  {row['Model']}\n  ({row['Parameters']/1000:.0f}K)",
                    fontsize=9, va='center')

        ax2.set_xlabel('Number of Parameters', fontsize=12)
        ax2.set_ylabel('Test F1-Score (%)', fontsize=12)
        ax2.set_title('Parameter Efficiency: F1-Score vs Model Size', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_efficiency.png'), dpi=300)
        plt.close()

        print(f"✓ Parameter efficiency plot saved")


def identify_best_model(all_results):
    """Identify best model for each metric"""
    print("\n" + "="*100)
    print("BEST MODEL FOR EACH METRIC")
    print("="*100 + "\n")

    metrics = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_auc']
    metric_names = {
        'test_accuracy': 'Accuracy',
        'test_f1_score': 'F1-Score',
        'test_precision': 'Precision',
        'test_recall': 'Recall',
        'test_auc': 'AUC'
    }

    for metric in metrics:
        best_model = None
        best_value = -np.inf

        for model_name, results in all_results.items():
            if results and results['scores'] is not None:
                avg_row = results['scores'][results['scores']['test_subject'] == 'Average']
                if not avg_row.empty:
                    value = avg_row[metric].values[0]
                    if value > best_value:
                        best_value = value
                        best_model = model_name

        print(f"{metric_names[metric]:15s}: {best_model:15s} ({best_value:.2f}%)")

    print("\n" + "="*100 + "\n")


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    if not os.path.exists(args.work_dir):
        print(f"Error: Work directory not found: {args.work_dir}")
        return

    print("\n" + "="*100)
    print("ANALYZING DUAL-STREAM MODEL COMPARISON RESULTS")
    print("="*100)
    print(f"Work directory: {args.work_dir}\n")

    # Load results for all models
    all_results = {}
    for model_name in args.models:
        results = load_model_results(args.work_dir, model_name)
        if results:
            all_results[model_name] = results
            print(f"✓ Loaded results for: {model_name}")

    if not all_results:
        print("\nError: No results loaded. Check work directory and model names.")
        return

    # Create output directory for analysis
    analysis_dir = os.path.join(args.work_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"\n{'='*100}")
    print("RUNNING ANALYSIS...")
    print(f"{'='*100}\n")

    # Print detailed statistics
    print_detailed_statistics(all_results)

    # Identify best models
    identify_best_model(all_results)

    # Create visualizations
    plot_training_curves(all_results, analysis_dir)
    create_parameter_efficiency_plot(all_results, analysis_dir)

    print(f"\n{'='*100}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*100}")
    print(f"\nResults saved in: {analysis_dir}/")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
