#!/usr/bin/env python3
"""
Aggregate Results from Motion Filtering Model Comparison

This script aggregates results from all 4 models in run_motion_filtering_comparison.sh
and generates comprehensive CSV files with fold-level metrics and summaries.

Usage:
    python aggregate_motion_filtering_results.py --work-dir work_dir/motion_filtering_comparison_TIMESTAMP
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class MotionFilteringAggregator:
    """Aggregates results from motion filtering model comparison."""

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

        # Define expected model names
        self.model_names = [
            "Model1_TransModel_AccOnly_MotionFiltered",
            "Model2_IMUTransformer_AccOnly_MotionFiltered",
            "Model3_DualStream_AccGyro_MotionFiltered",
            "Model4_Madgwick_Fusion_MotionFiltered"
        ]

        print(f"Work directory: {self.work_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Expected models: {len(self.model_names)}")

    def find_experiment_dirs(self) -> Dict[str, Path]:
        """Find experiment directories for each model"""
        experiment_dirs = {}

        for model_name in self.model_names:
            model_dir = self.work_dir / model_name
            if model_dir.exists() and model_dir.is_dir():
                experiment_dirs[model_name] = model_dir
                print(f"  ✓ Found: {model_name}")
            else:
                print(f"  ✗ Missing: {model_name}")

        return experiment_dirs

    def extract_fold_results(self, experiment_dir: Path) -> Optional[pd.DataFrame]:
        """Extract fold-level results from an experiment directory"""
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

    def standardize_columns(self, df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Add model name and standardize column names"""
        df = df.copy()
        df['model'] = model_name

        # Standardize column names
        column_mapping = {
            'Subject': 'test_subject',
            'Accuracy': 'accuracy',
            'F1 Score': 'f1_score',
            'Precision': 'precision',
            'Recall': 'recall',
            'AUC': 'auc'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]

        return df

    def aggregate_all_results(self) -> pd.DataFrame:
        """Aggregate results from all models"""
        print("\nAggregating results from all models...")
        all_results = []

        experiment_dirs = self.find_experiment_dirs()

        for model_name, experiment_dir in experiment_dirs.items():
            fold_results = self.extract_fold_results(experiment_dir)
            if fold_results is not None:
                fold_results = self.standardize_columns(fold_results, model_name)
                all_results.append(fold_results)

        if not all_results:
            raise ValueError("No results found to aggregate")

        combined_df = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal rows aggregated: {len(combined_df)}")
        return combined_df

    def compute_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics per model"""
        print("\nComputing summary statistics...")

        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        summaries = []

        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            summary = {'model': model}

            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        summary[f'{metric}_mean'] = values.mean()
                        summary[f'{metric}_std'] = values.std()
                        summary[f'{metric}_min'] = values.min()
                        summary[f'{metric}_max'] = values.max()
                    else:
                        summary[f'{metric}_mean'] = np.nan
                        summary[f'{metric}_std'] = np.nan
                        summary[f'{metric}_min'] = np.nan
                        summary[f'{metric}_max'] = np.nan

            summary['num_folds'] = len(model_data)
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)

        # Sort by F1 score (descending)
        if 'f1_score_mean' in summary_df.columns:
            summary_df = summary_df.sort_values('f1_score_mean', ascending=False)

        return summary_df

    def create_comparison_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Create simplified comparison table"""
        print("\nCreating model comparison table...")

        comparison = summary_df[['model']].copy()

        # Add key metrics with formatting
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in summary_df.columns and std_col in summary_df.columns:
                comparison[metric] = summary_df.apply(
                    lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}",
                    axis=1
                )

        comparison['num_folds'] = summary_df['num_folds']

        return comparison

    def generate_report(self, summary_df: pd.DataFrame):
        """Generate text report"""
        print("\nGenerating report...")

        report_path = self.output_dir / "motion_filtering_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MOTION FILTERING MODEL COMPARISON - RESULTS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("Models compared:\n")
            f.write("  1. TransModel (Baseline) - Acc-only, TransModel architecture\n")
            f.write("  2. IMUTransformer - Acc-only, IMUTransformer architecture\n")
            f.write("  3. Dual Stream - Acc+gyro, separate encoders, fused\n")
            f.write("  4. Madgwick Fusion - Acc+gyro → orientation angles\n\n")

            f.write("Optimizations applied:\n")
            f.write("  ✓ Motion filtering (threshold=10.0, min_axes=2)\n")
            f.write("  ✓ Low-pass filtering (5.5Hz cutoff)\n")
            f.write("  ✓ Class-aware stride (ADL=10, Fall=32)\n")
            f.write("  ✓ Validation: 40-50% ADLs\n")
            f.write("  ✓ Strong regularization (dropout 0.5-0.7)\n\n")

            f.write("="*80 + "\n")
            f.write("SUMMARY RESULTS (sorted by F1 score)\n")
            f.write("="*80 + "\n\n")

            for idx, row in summary_df.iterrows():
                f.write(f"{idx+1}. {row['model']}\n")
                if 'f1_score_mean' in row:
                    f.write(f"   F1 Score:  {row['f1_score_mean']:.4f} ± {row['f1_score_std']:.4f}\n")
                if 'accuracy_mean' in row:
                    f.write(f"   Accuracy:  {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}\n")
                if 'precision_mean' in row:
                    f.write(f"   Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}\n")
                if 'recall_mean' in row:
                    f.write(f"   Recall:    {row['recall_mean']:.4f} ± {row['recall_std']:.4f}\n")
                if 'auc_mean' in row:
                    f.write(f"   AUC:       {row['auc_mean']:.4f} ± {row['auc_std']:.4f}\n")
                f.write(f"   Folds:     {int(row['num_folds'])}\n\n")

            f.write("="*80 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("="*80 + "\n\n")

            f.write("1. Architecture Impact (Models 1 vs 2):\n")
            f.write("   - Both use acc-only with motion filtering\n")
            f.write("   - Comparison shows architecture impact\n\n")

            f.write("2. Modality Impact (Models 2 vs 3 vs 4):\n")
            f.write("   - Model 2: Acc-only (4ch)\n")
            f.write("   - Model 3: Acc+gyro separate (8ch)\n")
            f.write("   - Model 4: Acc+gyro fused (7ch)\n")
            f.write("   - Shows gyro value despite noise\n\n")

            f.write("3. Fusion Strategy (Models 3 vs 4):\n")
            f.write("   - Model 3: Late fusion (separate encoders)\n")
            f.write("   - Model 4: Early fusion (Madgwick AHRS)\n")
            f.write("   - Comparison of fusion approaches\n\n")

            f.write("="*80 + "\n")

        print(f"Report saved to: {report_path}")

    def run(self):
        """Run full aggregation pipeline"""
        print("\n" + "="*80)
        print("MOTION FILTERING RESULTS AGGREGATION")
        print("="*80 + "\n")

        # Aggregate all results
        all_results_df = self.aggregate_all_results()

        # Save comprehensive fold-level results
        comprehensive_path = self.output_dir / "all_folds_comprehensive.csv"
        all_results_df.to_csv(comprehensive_path, index=False)
        print(f"\n✓ Saved comprehensive results: {comprehensive_path}")

        # Compute summary statistics
        summary_df = self.compute_summary_statistics(all_results_df)

        # Save summary statistics
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved summary statistics: {summary_path}")

        # Create comparison table
        comparison_df = self.create_comparison_table(summary_df)

        # Save comparison table
        comparison_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Saved model comparison: {comparison_path}")

        # Generate report
        self.generate_report(summary_df)

        print("\n" + "="*80)
        print("AGGREGATION COMPLETE")
        print("="*80)
        print(f"\nResults saved in: {self.output_dir}")
        print("\nFiles generated:")
        print(f"  - {comprehensive_path.name}")
        print(f"  - {summary_path.name}")
        print(f"  - {comparison_path.name}")
        print(f"  - motion_filtering_report.txt")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from motion filtering model comparison"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="Base work directory containing experiment results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (default: work-dir/aggregated)"
    )

    args = parser.parse_args()

    try:
        aggregator = MotionFilteringAggregator(args.work_dir, args.output_dir)
        aggregator.run()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
