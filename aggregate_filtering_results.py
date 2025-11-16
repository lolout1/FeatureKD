"""
Aggregate Results from Filtering Experiments

This script aggregates results from all 7 configurations in run_filtering.sh
and generates comprehensive CSV files with fold-level metrics and summaries.

Usage:
    python aggregate_filtering_results.py --work-dir work_dir/filtering_TIMESTAMP
    python aggregate_filtering_results.py --work-dir work_dir/filtering_TIMESTAMP --output-dir results/filtering_comparison
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import re


class FilteringResultsAggregator:
    """Aggregates results from multiple filtering experiment configurations."""

    def __init__(self, work_dir: str, output_dir: Optional[str] = None):
        """
        Initialize aggregator

        Args:
            work_dir: Base work directory containing all experiment subdirectories
            output_dir: Directory to save aggregated results (default: work_dir/aggregated)
        """
        self.work_dir = Path(work_dir)
        if not self.work_dir.exists():
            raise ValueError(f"Work directory does not exist: {work_dir}")

        self.output_dir = Path(output_dir) if output_dir else self.work_dir / "aggregated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define expected configuration names
        self.config_names = [
            "TransModel_Baseline",
            "TransModel_MotionFilter",
            "IMU_AccOnly_MotionFilter",
            "IMU_AccGyro_Raw",
            "IMU_AccGyro_HardFilter",
            "IMU_AccGyro_Adaptive",
            "IMU_Madgwick_Fusion"
        ]

        print(f"Work directory: {self.work_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Expected configurations: {len(self.config_names)}")

    def find_experiment_dirs(self) -> Dict[str, Path]:
        """
        Find experiment directories for each configuration

        Returns:
            Dictionary mapping config name to directory path
        """
        experiment_dirs = {}

        for config_name in self.config_names:
            config_dir = self.work_dir / config_name
            if config_dir.exists() and config_dir.is_dir():
                experiment_dirs[config_name] = config_dir
                print(f"  ✓ Found: {config_name}")
            else:
                print(f"  ✗ Missing: {config_name}")

        return experiment_dirs

    def extract_fold_results(self, experiment_dir: Path) -> Optional[pd.DataFrame]:
        """
        Extract fold-level results from an experiment directory

        Args:
            experiment_dir: Path to experiment directory

        Returns:
            DataFrame with fold-level metrics or None if not found
        """
        # Check for epoch_test_score.csv (standard output from main.py LOSO mode)
        csv_path = experiment_dir / "epoch_test_score.csv"

        if not csv_path.exists():
            print(f"    Warning: No epoch_test_score.csv found in {experiment_dir.name}")
            return None

        try:
            df = pd.read_csv(csv_path)
            print(f"    Loaded {len(df)} fold results from {experiment_dir.name}")
            return df
        except Exception as e:
            print(f"    Error loading {csv_path}: {e}")
            return None

    def standardize_columns(self, df: pd.DataFrame, config_name: str) -> pd.DataFrame:
        """
        Standardize column names and add configuration identifier

        Args:
            df: DataFrame with fold results
            config_name: Name of configuration

        Returns:
            Standardized DataFrame
        """
        df = df.copy()

        # Add configuration name
        df.insert(0, 'config', config_name)

        # Rename columns to standard format if needed
        column_mapping = {
            'subject': 'fold_subject',
            'Subject': 'fold_subject',
            'test_subject': 'fold_subject',
            'fold': 'fold_number',
            'Fold': 'fold_number',
            'accuracy': 'test_accuracy',
            'Accuracy': 'test_accuracy',
            'f1': 'test_f1_score',
            'F1': 'test_f1_score',
            'f1_score': 'test_f1_score',
            'precision': 'test_precision',
            'Precision': 'test_precision',
            'recall': 'test_recall',
            'Recall': 'test_recall',
            'loss': 'test_loss',
            'Loss': 'test_loss',
            'auc': 'test_auc',
            'AUC': 'test_auc'
        }

        df.rename(columns=column_mapping, inplace=True)

        # Ensure we have fold identifier
        if 'fold_number' not in df.columns and 'fold_subject' in df.columns:
            # Create fold numbers from 1 to N
            df['fold_number'] = range(1, len(df) + 1)

        return df

    def aggregate_all_results(self) -> pd.DataFrame:
        """
        Aggregate results from all configurations into a single DataFrame

        Returns:
            DataFrame with all fold-level results
        """
        print("\nAggregating results...")
        experiment_dirs = self.find_experiment_dirs()

        all_results = []

        for config_name, exp_dir in experiment_dirs.items():
            print(f"\n  Processing: {config_name}")
            fold_df = self.extract_fold_results(exp_dir)

            if fold_df is not None:
                fold_df = self.standardize_columns(fold_df, config_name)
                all_results.append(fold_df)
                print(f"    ✓ Added {len(fold_df)} folds")

        if not all_results:
            raise ValueError("No results found in any experiment directory")

        # Concatenate all results
        combined_df = pd.concat(all_results, ignore_index=True)

        print(f"\n✓ Combined {len(combined_df)} total fold results from {len(all_results)} configurations")

        return combined_df

    def compute_summary_statistics(self, fold_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics (mean, std, min, max) for each configuration

        Args:
            fold_df: DataFrame with fold-level results

        Returns:
            DataFrame with summary statistics
        """
        metric_columns = [
            col for col in fold_df.columns
            if any(metric in col.lower() for metric in ['accuracy', 'f1', 'precision', 'recall', 'loss', 'auc'])
        ]

        summary_data = []

        for config in fold_df['config'].unique():
            config_df = fold_df[fold_df['config'] == config]
            summary_row = {'config': config, 'num_folds': len(config_df)}

            for metric in metric_columns:
                if metric in config_df.columns:
                    values = config_df[metric].dropna()
                    if len(values) > 0:
                        summary_row[f'{metric}_mean'] = values.mean()
                        summary_row[f'{metric}_std'] = values.std()
                        summary_row[f'{metric}_min'] = values.min()
                        summary_row[f'{metric}_max'] = values.max()
                        summary_row[f'{metric}_median'] = values.median()

            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)

        return summary_df

    def create_comparison_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a clean comparison table with key metrics

        Args:
            summary_df: DataFrame with summary statistics

        Returns:
            Clean comparison table
        """
        # Select key metrics
        comparison_columns = ['config', 'num_folds']

        # Add main metrics if they exist
        for metric in ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_loss', 'test_auc']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in summary_df.columns:
                # Create formatted column: "mean ± std"
                summary_df[metric] = summary_df.apply(
                    lambda row: f"{row[mean_col]:.2f} ± {row[std_col]:.2f}"
                    if pd.notna(row.get(std_col)) else f"{row[mean_col]:.2f}",
                    axis=1
                )
                comparison_columns.append(metric)

        comparison_df = summary_df[comparison_columns].copy()

        return comparison_df

    def save_results(self, fold_df: pd.DataFrame, summary_df: pd.DataFrame, comparison_df: pd.DataFrame):
        """
        Save all results to CSV files

        Args:
            fold_df: DataFrame with fold-level results
            summary_df: DataFrame with summary statistics
            comparison_df: Clean comparison table
        """
        print("\nSaving results...")

        # 1. Complete fold-level results
        fold_path = self.output_dir / "all_folds_comprehensive.csv"
        fold_df.to_csv(fold_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved comprehensive fold results: {fold_path}")
        print(f"    Rows: {len(fold_df)}, Columns: {len(fold_df.columns)}")

        # 2. Summary statistics
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"  ✓ Saved summary statistics: {summary_path}")
        print(f"    Configurations: {len(summary_df)}")

        # 3. Clean comparison table
        comparison_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  ✓ Saved comparison table: {comparison_path}")

        # 4. Create per-config fold tables
        per_config_dir = self.output_dir / "per_config"
        per_config_dir.mkdir(exist_ok=True)

        for config in fold_df['config'].unique():
            config_df = fold_df[fold_df['config'] == config]
            config_path = per_config_dir / f"{config}_folds.csv"
            config_df.to_csv(config_path, index=False, float_format='%.4f')

        print(f"  ✓ Saved per-config fold tables: {per_config_dir}/")

    def generate_report(self, fold_df: pd.DataFrame, summary_df: pd.DataFrame, comparison_df: pd.DataFrame):
        """
        Generate a text report summarizing the results

        Args:
            fold_df: DataFrame with fold-level results
            summary_df: DataFrame with summary statistics
            comparison_df: Clean comparison table
        """
        report_path = self.output_dir / "filtering_experiments_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("FILTERING AND QUALITY ASSESSMENT EXPERIMENTS - COMPREHENSIVE REPORT\n")
            f.write("="*100 + "\n\n")

            f.write(f"Work Directory: {self.work_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Total Configurations: {len(fold_df['config'].unique())}\n")
            f.write(f"Total Folds: {len(fold_df)}\n\n")

            f.write("="*100 + "\n")
            f.write("MODEL COMPARISON (Test Set Performance)\n")
            f.write("="*100 + "\n\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            f.write("="*100 + "\n")
            f.write("DETAILED SUMMARY STATISTICS\n")
            f.write("="*100 + "\n\n")

            # Print key metrics for each config
            for config in summary_df['config']:
                f.write(f"\n{config}:\n")
                f.write("-" * len(config) + "\n")

                config_stats = summary_df[summary_df['config'] == config].iloc[0]

                # Test metrics
                for metric in ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_loss']:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    if mean_col in config_stats:
                        f.write(f"  {metric:20s}: {config_stats[mean_col]:6.2f} ± {config_stats[std_col]:5.2f}\n")

                f.write(f"  {'num_folds':20s}: {config_stats['num_folds']:.0f}\n")

            f.write("\n" + "="*100 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*100 + "\n\n")

            # Find best performing model
            if 'test_f1_score_mean' in summary_df.columns:
                best_f1_idx = summary_df['test_f1_score_mean'].idxmax()
                best_model = summary_df.loc[best_f1_idx, 'config']
                best_f1 = summary_df.loc[best_f1_idx, 'test_f1_score_mean']
                f.write(f"Best F1 Score: {best_model} ({best_f1:.2f}%)\n\n")

            if 'test_accuracy_mean' in summary_df.columns:
                best_acc_idx = summary_df['test_accuracy_mean'].idxmax()
                best_model = summary_df.loc[best_acc_idx, 'config']
                best_acc = summary_df.loc[best_acc_idx, 'test_accuracy_mean']
                f.write(f"Best Accuracy: {best_model} ({best_acc:.2f}%)\n\n")

            f.write("="*100 + "\n")
            f.write("FILES GENERATED\n")
            f.write("="*100 + "\n\n")
            f.write(f"1. Comprehensive fold results: all_folds_comprehensive.csv\n")
            f.write(f"2. Summary statistics: summary_statistics.csv\n")
            f.write(f"3. Model comparison: model_comparison.csv\n")
            f.write(f"4. Per-config fold tables: per_config/\n")
            f.write(f"5. This report: filtering_experiments_report.txt\n\n")

            f.write("="*100 + "\n")

        print(f"\n✓ Generated report: {report_path}")

    def run(self):
        """Execute the aggregation pipeline"""
        print("\n" + "="*100)
        print("FILTERING EXPERIMENTS RESULTS AGGREGATION")
        print("="*100)

        # Aggregate all results
        fold_df = self.aggregate_all_results()

        # Compute summary statistics
        summary_df = self.compute_summary_statistics(fold_df)

        # Create comparison table
        comparison_df = self.create_comparison_table(summary_df)

        # Save all results
        self.save_results(fold_df, summary_df, comparison_df)

        # Generate report
        self.generate_report(fold_df, summary_df, comparison_df)

        print("\n" + "="*100)
        print("✓ AGGREGATION COMPLETE")
        print("="*100)
        print(f"\nAll results saved to: {self.output_dir}")
        print("\nKey files:")
        print(f"  - all_folds_comprehensive.csv  (complete fold-level data)")
        print(f"  - summary_statistics.csv       (mean, std, min, max for each config)")
        print(f"  - model_comparison.csv         (clean comparison table)")
        print(f"  - filtering_experiments_report.txt  (comprehensive text report)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from filtering experiments (run_filtering.sh)"
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='Base work directory containing experiment results (e.g., work_dir/filtering_TIMESTAMP)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for aggregated results (default: WORK_DIR/aggregated)'
    )

    args = parser.parse_args()

    # Run aggregation
    aggregator = FilteringResultsAggregator(
        work_dir=args.work_dir,
        output_dir=args.output_dir
    )
    aggregator.run()


if __name__ == "__main__":
    main()
