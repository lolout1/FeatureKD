#!/usr/bin/env python3
"""
Unified IMU Model Comparison Script

Compares three models with fair feature engineering:
1. TransModel: Accelerometer-only with SMV (4 channels) - BASELINE
2. IMU 8-channel: Acc+Gyro with SMV and magnitude (8 channels) - NO DTW
3. IMU 8-channel+DTW: Same as #2 but with gyro→acc alignment - WITH DTW

All results logged to single CSV with per-fold and aggregate statistics.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def get_timestamp():
    """Generate timestamp in format: 05am-11-16-2025"""
    now = datetime.now()
    hour_12 = now.strftime("%I")  # 12-hour format
    am_pm = now.strftime("%p").lower()  # am/pm
    month_day_year = now.strftime("%m-%d-%Y")
    return f"{hour_12}{am_pm}-{month_day_year}"


def run_model(config_path, work_dir, device, num_epochs, batch_size):
    """
    Run a single model configuration using main.py

    Args:
        config_path: Path to YAML config file
        work_dir: Directory to save results
        device: GPU device ID
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        dict: Results dictionary with metrics
    """
    print(f"\n{'='*80}")
    print(f"Running: {config_path}")
    print(f"Work directory: {work_dir}")
    print(f"{'='*80}\n")

    # Run training
    cmd = [
        'python', 'main.py',
        '--config', config_path,
        '--work-dir', work_dir,
        '--device', str(device),
        '--num-epoch', str(num_epochs),
        '--batch-size', str(batch_size),
        '--phase', 'train'
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {config_path}: {e}")
        return None

    # Load results
    summary_path = os.path.join(work_dir, 'summary_statistics.csv')
    scores_path = os.path.join(work_dir, 'scores.csv')
    per_fold_path = os.path.join(work_dir, 'per_fold_detailed.csv')

    results = {}

    # Load summary statistics
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        # Extract mean metrics
        if 'mean' in df_summary['statistic'].values:
            mean_row = df_summary[df_summary['statistic'] == 'mean'].iloc[0]
            results['test_accuracy_mean'] = mean_row['test_accuracy']
            results['test_f1_mean'] = mean_row['test_f1_score']
            results['test_precision_mean'] = mean_row['test_precision']
            results['test_recall_mean'] = mean_row['test_recall']
            results['test_auc_mean'] = mean_row['test_auc']
            results['val_accuracy_mean'] = mean_row['val_accuracy']
            results['val_f1_mean'] = mean_row['val_f1_score']

        # Extract std metrics
        if 'std' in df_summary['statistic'].values:
            std_row = df_summary[df_summary['statistic'] == 'std'].iloc[0]
            results['test_accuracy_std'] = std_row['test_accuracy']
            results['test_f1_std'] = std_row['test_f1_score']

        # Extract overfitting gap
        if 'mean' in df_summary['statistic'].values:
            mean_row = df_summary[df_summary['statistic'] == 'mean'].iloc[0]
            results['overfitting_gap_acc'] = mean_row.get('overfitting_gap_accuracy', 0)
            results['overfitting_gap_f1'] = mean_row.get('overfitting_gap_f1', 0)

    # Load per-fold results for detailed analysis
    if os.path.exists(per_fold_path):
        results['per_fold_path'] = per_fold_path
        results['per_fold_data'] = pd.read_csv(per_fold_path)

    return results


def aggregate_results(all_results, output_dir):
    """
    Aggregate results from all models into comparison CSVs

    Args:
        all_results: List of (model_name, results_dict) tuples
        output_dir: Directory to save comparison results
    """
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Create summary comparison table
    summary_data = []
    for model_name, results in all_results:
        if results is None:
            continue
        summary_data.append({
            'Model': model_name,
            'Test Accuracy (%)': f"{results.get('test_accuracy_mean', 0):.2f}",
            'Test F1 (%)': f"{results.get('test_f1_mean', 0):.2f}",
            'Test Precision (%)': f"{results.get('test_precision_mean', 0):.2f}",
            'Test Recall (%)': f"{results.get('test_recall_mean', 0):.2f}",
            'Test AUC (%)': f"{results.get('test_auc_mean', 0):.2f}",
            'Val Accuracy (%)': f"{results.get('val_accuracy_mean', 0):.2f}",
            'Val F1 (%)': f"{results.get('val_f1_mean', 0):.2f}",
            'Acc Std (%)': f"{results.get('test_accuracy_std', 0):.2f}",
            'F1 Std (%)': f"{results.get('test_f1_std', 0):.2f}",
            'Overfit Gap Acc (%)': f"{results.get('overfitting_gap_acc', 0):.2f}",
            'Overfit Gap F1 (%)': f"{results.get('overfitting_gap_f1', 0):.2f}",
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(comparison_dir, 'model_comparison_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✓ Saved comparison summary: {summary_path}")

    # Create per-fold comparison (all models)
    per_fold_data = []
    for model_name, results in all_results:
        if results is None or 'per_fold_data' not in results:
            continue
        df_fold = results['per_fold_data'].copy()
        df_fold['model'] = model_name
        per_fold_data.append(df_fold)

    if per_fold_data:
        df_all_folds = pd.concat(per_fold_data, ignore_index=True)
        per_fold_path = os.path.join(comparison_dir, 'all_models_per_fold.csv')
        df_all_folds.to_csv(per_fold_path, index=False)
        print(f"✓ Saved per-fold comparison: {per_fold_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*100}\n")
    print(df_summary.to_string(index=False))
    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description='Run IMU model comparison')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--num-epochs', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--base-work-dir', type=str, default='work_dir', help='Base work directory')

    args = parser.parse_args()

    # Generate timestamp for this run
    timestamp = get_timestamp()
    base_work_dir = os.path.join(args.base_work_dir, timestamp)
    os.makedirs(base_work_dir, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"IMU MODEL COMPARISON - {timestamp}")
    print(f"{'='*100}")
    print(f"Base work directory: {base_work_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*100}\n")

    # Define models to compare
    models = [
        {
            'name': 'TransModel_AccOnly_4ch',
            'config': 'config/smartfallmm/transformer.yaml',
            'description': 'Baseline: Acc+SMV only (4 channels)'
        },
        {
            'name': 'IMU_8channel',
            'config': 'config/smartfallmm/imu_8channel.yaml',
            'description': 'Acc+Gyro with SMV+magnitude (8 channels, no DTW)'
        },
        {
            'name': 'IMU_8channel_DTW',
            'config': 'config/smartfallmm/imu_8channel_dtw.yaml',
            'description': 'Acc+Gyro with SMV+magnitude+DTW alignment (8 channels)'
        }
    ]

    # Run all models
    all_results = []
    for model_info in models:
        model_name = model_info['name']
        config_path = model_info['config']
        description = model_info['description']

        print(f"\n{'#'*100}")
        print(f"# Model: {model_name}")
        print(f"# {description}")
        print(f"{'#'*100}\n")

        work_dir = os.path.join(base_work_dir, model_name)

        results = run_model(
            config_path=config_path,
            work_dir=work_dir,
            device=args.device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )

        all_results.append((model_name, results))

    # Aggregate and save comparison results
    aggregate_results(all_results, base_work_dir)

    print(f"\n{'='*100}")
    print("ALL MODELS COMPLETED!")
    print(f"{'='*100}")
    print(f"Results saved to: {base_work_dir}")
    print(f"Comparison summary: {os.path.join(base_work_dir, 'comparison', 'model_comparison_summary.csv')}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
