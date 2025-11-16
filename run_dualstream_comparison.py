"""
LOSO Cross-Validation Comparison Script for Dual-Stream Architectures

Runs all 3 dual-stream models sequentially with LOSO cross-validation:
- Option 1: Shared-Weight Dual Stream (~15K params)
- Option 2: Lightweight Independent Streams (~25K params)
- Option 3: Asymmetric Dual Stream (~35K params)

Each model is trained with Leave-One-Subject-Out (LOSO) cross-validation
using all 30 young participants from the SmartFall dataset.

Usage:
    python run_dualstream_comparison.py --device 0

Results are saved in:
    - work_dir/dualstream_comparison_results/
    - Each model gets its own subdirectory with full LOSO results
    - Final comparison CSV and plots generated at the end
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import yaml
import shutil


def setup_argparser():
    parser = argparse.ArgumentParser(description='Run dual-stream model comparison with LOSO CV')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID (default: 0 for NVIDIA A100)')
    parser.add_argument('--num-epochs', type=int, default=80,
                       help='Number of epochs per fold (default: 80)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--work-dir', type=str, default='work_dir/dualstream_comparison',
                       help='Base directory for results')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline model (only run dual-stream models)')
    parser.add_argument('--models', type=str, nargs='+',
                       choices=['baseline', 'shared', 'light', 'asymmetric', 'all'],
                       default=['all'],
                       help='Which models to run (default: all)')

    # Preprocessing arguments
    parser.add_argument('--enable-normalization', type=str, default='true',
                       help='Enable per-window normalization (true/false, default: true)')
    parser.add_argument('--enable-filtering', type=str, default='true',
                       help='Enable Butterworth low-pass filter (true/false, default: true)')
    parser.add_argument('--filter-cutoff', type=float, default=5.5,
                       help='Butterworth filter cutoff frequency in Hz (default: 5.5)')
    parser.add_argument('--filter-fs', type=float, default=25.0,
                       help='Butterworth filter sampling rate in Hz (default: 25)')
    return parser


def parse_bool(value):
    """Parse boolean string values"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)


def update_yaml_preprocessing(config_path, enable_normalization, enable_filtering,
                              filter_cutoff, filter_fs, output_path):
    """
    Update preprocessing settings in a YAML config file

    Args:
        config_path: Path to original YAML config
        enable_normalization: Boolean for normalization
        enable_filtering: Boolean for Butterworth filter
        filter_cutoff: Filter cutoff frequency
        filter_fs: Filter sampling rate
        output_path: Path to save modified config

    Returns:
        Path to modified config file
    """
    # Load the YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update preprocessing settings in dataset_args
    if 'dataset_args' not in config:
        config['dataset_args'] = {}

    config['dataset_args']['enable_normalization'] = enable_normalization
    config['dataset_args']['enable_filtering'] = enable_filtering
    config['dataset_args']['filter_cutoff'] = filter_cutoff
    config['dataset_args']['filter_fs'] = filter_fs

    # Save the modified config
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated config saved to: {output_path}")
    print(f"  - Normalization: {enable_normalization}")
    print(f"  - Filtering: {enable_filtering}")
    print(f"  - Filter cutoff: {filter_cutoff} Hz")
    print(f"  - Filter fs: {filter_fs} Hz")

    return output_path


def get_models_to_run(args):
    """Determine which models to run based on args"""
    if 'all' in args.models:
        models = ['baseline', 'shared', 'light', 'asymmetric']
    else:
        models = args.models

    if args.skip_baseline and 'baseline' in models:
        models.remove('baseline')

    return models


def run_model(config_path, work_dir, device, num_epochs, batch_size):
    """
    Run a single model with LOSO cross-validation using main.py

    Args:
        config_path: Path to the YAML config file
        work_dir: Directory to save results
        device: GPU device ID
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {config_path}")
    print(f"Work dir: {work_dir}")
    print(f"Device: cuda:{device}")
    print(f"{'='*80}\n")

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
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ Successfully completed: {config_path}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {config_path}: {e}\n")
        return False


def load_results(work_dir, model_name):
    """Load results from a completed model run"""
    scores_path = os.path.join(work_dir, 'scores.csv')
    training_log_path = os.path.join(work_dir, 'training_log.csv')

    results = {}

    # Store paths for later use
    results['scores_path'] = scores_path
    results['training_log_path'] = training_log_path

    if os.path.exists(scores_path):
        results['scores'] = pd.read_csv(scores_path)
        results['scores']['model'] = model_name
    else:
        print(f"Warning: {scores_path} not found")
        results['scores'] = None

    if os.path.exists(training_log_path):
        results['training_log'] = pd.read_csv(training_log_path)
        results['training_log']['model'] = model_name
    else:
        print(f"Warning: {training_log_path} not found")
        results['training_log'] = None

    return results


def create_comparison_plots(all_results, output_dir):
    """Create comparison plots for all models"""
    os.makedirs(output_dir, exist_ok=True)

    # Combine all scores
    all_scores = []
    for model_name, results in all_results.items():
        if results['scores'] is not None:
            df = results['scores'].copy()
            # Filter out the "Average" row for plotting
            df = df[df['test_subject'] != 'Average']
            all_scores.append(df)

    if not all_scores:
        print("No scores to plot")
        return

    combined_scores = pd.concat(all_scores, ignore_index=True)

    # 1. Test Accuracy Comparison (Box plot)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_scores, x='model', y='test_accuracy')
    plt.title('Test Accuracy Distribution Across Models (LOSO CV)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_boxplot.png'), dpi=300)
    plt.close()

    # 2. Test F1-Score Comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=combined_scores, x='model', y='test_f1_score')
    plt.title('Test F1-Score Distribution Across Models (LOSO CV)', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison_boxplot.png'), dpi=300)
    plt.close()

    # 3. Multiple Metrics Comparison (Bar plot with error bars)
    metrics = ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall']
    metric_names = ['Accuracy', 'F1-Score', 'Precision', 'Recall']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        # Calculate mean and std for each model
        summary = combined_scores.groupby('model')[metric].agg(['mean', 'std']).reset_index()

        x_pos = np.arange(len(summary))
        ax.bar(x_pos, summary['mean'], yerr=summary['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(summary['model'], rotation=15)
        ax.set_ylabel(f'{metric_name} (%)', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_title(f'{metric_name} (Mean ± Std)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison_bar.png'), dpi=300)
    plt.close()

    # 4. Per-subject performance comparison (heatmap)
    for metric in ['test_accuracy', 'test_f1_score']:
        pivot_data = combined_scores.pivot(index='test_subject', columns='model', values=metric)

        plt.figure(figsize=(12, 20))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=pivot_data.mean().mean(),
                   cbar_kws={'label': f'{metric.replace("test_", "").replace("_", " ").title()} (%)'})
        plt.title(f'Per-Subject {metric.replace("test_", "").replace("_", " ").title()} Comparison',
                 fontsize=14, fontweight='bold')
        plt.ylabel('Test Subject', fontsize=12)
        plt.xlabel('Model', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_subject_{metric}_heatmap.png'), dpi=300)
        plt.close()

    print(f"\n✓ Plots saved to {output_dir}/\n")


def create_summary_table(all_results, output_dir):
    """Create summary comparison table"""
    os.makedirs(output_dir, exist_ok=True)
    summary_data = []

    for model_name, results in all_results.items():
        if results['scores'] is not None:
            # Get the "Average" row
            avg_row = results['scores'][results['scores']['test_subject'] == 'Average']

            if not avg_row.empty:
                summary_data.append({
                    'Model': model_name,
                    'Test Accuracy (%)': avg_row['test_accuracy'].values[0],
                    'Test F1 (%)': avg_row['test_f1_score'].values[0],
                    'Test Precision (%)': avg_row['test_precision'].values[0],
                    'Test Recall (%)': avg_row['test_recall'].values[0],
                    'Test AUC (%)': avg_row['test_auc'].values[0],
                    'Val Accuracy (%)': avg_row['val_accuracy'].values[0],
                    'Val F1 (%)': avg_row['val_f1_score'].values[0]
                })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.round(2)

        # Save to CSV
        csv_path = os.path.join(output_dir, 'model_comparison_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"\n✓ Summary table saved to {csv_path}\n")

        # Print to console
        print("\n" + "="*100)
        print("FINAL COMPARISON SUMMARY (LOSO Cross-Validation Averages)")
        print("="*100)
        print(summary_df.to_string(index=False))
        print("="*100 + "\n")

        return summary_df

    return None


def create_per_subject_comparison(all_results, output_dir):
    """
    Create detailed per-subject comparison across all models.
    Uses enhanced per_fold_detailed.csv files if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Try to load per_fold_detailed.csv for each model
    per_fold_data = {}
    for model_name, results in all_results.items():
        per_fold_file = os.path.join(os.path.dirname(results['scores_path']), 'per_fold_detailed.csv')
        if os.path.exists(per_fold_file):
            try:
                df = pd.read_csv(per_fold_file)
                per_fold_data[model_name] = df
                print(f"  ✓ Loaded per-fold details for {model_name}")
            except Exception as e:
                print(f"  ✗ Failed to load per-fold details for {model_name}: {e}")

    if not per_fold_data:
        print("  ! No per-fold detailed files found, skipping per-subject comparison")
        return None

    # Build comparison table
    all_subjects = set()
    for df in per_fold_data.values():
        all_subjects.update(df['test_subject'].astype(str).unique())

    all_subjects = sorted(list(all_subjects), key=lambda x: int(x) if x.isdigit() else 999)

    # Create comparison dataframe
    comparison_rows = []
    for subject in all_subjects:
        row = {'subject': subject}

        for model_name, df in per_fold_data.items():
            subject_row = df[df['test_subject'].astype(str) == subject]

            if not subject_row.empty:
                row[f'{model_name}_test_acc'] = subject_row['test_accuracy'].values[0]
                row[f'{model_name}_test_f1'] = subject_row['test_f1_score'].values[0]
                row[f'{model_name}_val_f1'] = subject_row['val_f1_score'].values[0]

                if 'overfitting_gap_f1' in subject_row.columns:
                    row[f'{model_name}_gap_f1'] = subject_row['overfitting_gap_f1'].values[0]

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'cross_model_per_subject.csv')
    comparison_df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\n✓ Cross-model per-subject comparison saved to {csv_path}\n")

    return comparison_df


def print_enhanced_summary(all_results, summary_df, comparison_df):
    """
    Print enhanced summary with per-fold statistics.
    """
    print("\n" + "="*100)
    print("ENHANCED RESULTS SUMMARY")
    print("="*100)

    # Print average performance (already shown in summary_df)
    print("\nCross-Model Averages:")
    print(summary_df.to_string(index=False))

    # Calculate and print variance statistics
    print("\n" + "-"*100)
    print("Performance Variance Analysis (Test F1 Standard Deviation):")
    print("-"*100)

    variance_data = []
    for model_name, results in all_results.items():
        # Try to load per_fold_detailed to get std
        per_fold_file = os.path.join(os.path.dirname(results['scores_path']), 'per_fold_detailed.csv')
        if os.path.exists(per_fold_file):
            try:
                df = pd.read_csv(per_fold_file)
                test_f1_std = df['test_f1_score'].std()
                test_f1_min = df['test_f1_score'].min()
                test_f1_max = df['test_f1_score'].max()

                variance_data.append({
                    'Model': model_name,
                    'Std Dev (%)': test_f1_std,
                    'Min (%)': test_f1_min,
                    'Max (%)': test_f1_max,
                    'Range (%)': test_f1_max - test_f1_min
                })
            except:
                pass

    if variance_data:
        variance_df = pd.DataFrame(variance_data)
        variance_df = variance_df.round(2)
        print(variance_df.to_string(index=False))
        print("\nInterpretation: Lower Std Dev = More consistent performance across subjects")

    print("\n" + "="*100 + "\n")


def save_model_info(all_results, output_dir):
    """Save model architecture information"""
    os.makedirs(output_dir, exist_ok=True)
    model_info = {
        'baseline': {
            'name': 'IMUTransformer (Baseline)',
            'architecture': 'Concatenated 6-channel input',
            'params_approx': '~110K',
            'config': 'config/smartfallmm/imu_student.yaml'
        },
        'shared': {
            'name': 'Shared-Weight Dual Stream',
            'architecture': 'Shared encoder for acc + gyro',
            'params_approx': '~15K',
            'config': 'config/smartfallmm/imu_dualstream_shared.yaml'
        },
        'light': {
            'name': 'Lightweight Independent Streams',
            'architecture': 'Separate 8d encoders per modality',
            'params_approx': '~25K',
            'config': 'config/smartfallmm/imu_dualstream_light.yaml'
        },
        'asymmetric': {
            'name': 'Asymmetric Dual Stream',
            'architecture': 'Acc (16d, 2 layers) + Gyro (8d, 1 layer)',
            'params_approx': '~35K',
            'config': 'config/smartfallmm/imu_dualstream_asymmetric.yaml'
        }
    }

    info_path = os.path.join(output_dir, 'model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"✓ Model info saved to {info_path}\n")


def main():
    parser = setup_argparser()
    args = parser.parse_args()

    # Parse preprocessing boolean values
    enable_normalization = parse_bool(args.enable_normalization)
    enable_filtering = parse_bool(args.enable_filtering)

    # Create base work directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_work_dir = f"{args.work_dir}_{timestamp}"
    os.makedirs(base_work_dir, exist_ok=True)

    # Create a config directory for modified configs
    config_dir = os.path.join(base_work_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    print("\n" + "="*80)
    print("DUAL-STREAM ARCHITECTURE COMPARISON - LOSO CROSS-VALIDATION")
    print("="*80)
    print(f"Base work directory: {base_work_dir}")
    print(f"GPU Device: cuda:{args.device} (NVIDIA A100)")
    print(f"Epochs per fold: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nPreprocessing Configuration:")
    print(f"  - Normalization: {enable_normalization}")
    print(f"  - Butterworth Filter: {enable_filtering}")
    print(f"  - Filter Cutoff: {args.filter_cutoff} Hz")
    print(f"  - Filter Sampling Rate: {args.filter_fs} Hz")
    print("="*80 + "\n")

    # Define original model configurations
    original_configs = {
        'baseline': 'config/smartfallmm/imu_student.yaml',
        'shared': 'config/smartfallmm/imu_dualstream_shared.yaml',
        'light': 'config/smartfallmm/imu_dualstream_light.yaml',
        'asymmetric': 'config/smartfallmm/imu_dualstream_asymmetric.yaml'
    }

    # Determine which models to run
    models_to_run = get_models_to_run(args)

    print(f"Models to run: {', '.join(models_to_run)}\n")

    # Update configs with preprocessing settings
    print("Updating configurations with preprocessing settings...\n")
    model_configs = {}
    for model_name in models_to_run:
        original_config = original_configs[model_name]
        updated_config = os.path.join(config_dir, f"{model_name}.yaml")

        model_configs[model_name] = update_yaml_preprocessing(
            config_path=original_config,
            enable_normalization=enable_normalization,
            enable_filtering=enable_filtering,
            filter_cutoff=args.filter_cutoff,
            filter_fs=args.filter_fs,
            output_path=updated_config
        )
        print()

    # Run each model
    all_results = {}
    successful_runs = []

    for model_name in models_to_run:
        config_path = model_configs[model_name]
        model_work_dir = os.path.join(base_work_dir, model_name)

        print(f"\n{'#'*80}")
        print(f"# Starting: {model_name.upper()}")
        print(f"{'#'*80}\n")

        success = run_model(
            config_path=config_path,
            work_dir=model_work_dir,
            device=args.device,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size
        )

        if success:
            successful_runs.append(model_name)
            # Load results
            results = load_results(model_work_dir, model_name)
            all_results[model_name] = results
            print(f"✓ {model_name} completed successfully")
        else:
            print(f"✗ {model_name} failed")

    # Generate comparison plots and summary
    if len(successful_runs) > 0:
        print("\n" + "="*80)
        print("GENERATING COMPARISON RESULTS")
        print("="*80 + "\n")

        comparison_dir = os.path.join(base_work_dir, 'comparison')

        # Create summary table
        summary_df = create_summary_table(all_results, comparison_dir)

        # Create per-subject comparison
        comparison_df = create_per_subject_comparison(all_results, comparison_dir)

        # Print enhanced summary with variance analysis
        if summary_df is not None:
            print_enhanced_summary(all_results, summary_df, comparison_df)

        # Create comparison plots
        create_comparison_plots(all_results, comparison_dir)

        # Save model info
        save_model_info(all_results, comparison_dir)

        print("\n" + "="*80)
        print("ALL TASKS COMPLETED!")
        print("="*80)
        print(f"\nResults saved in: {base_work_dir}/")
        print(f"  - Individual model results: {base_work_dir}/<model_name>/")
        print(f"  - Comparison plots and summary: {base_work_dir}/comparison/")
        print("\nSuccessfully completed models:")
        for model in successful_runs:
            print(f"  ✓ {model}")
        print("\n" + "="*80 + "\n")
    else:
        print("\n✗ No models completed successfully\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
