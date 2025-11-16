#!/usr/bin/env python3
"""
Comprehensive Aggregation for Motion Filtering Ablation Study
Research-grade results for scientific publication

Generates:
1. scores_comprehensive.csv - ALL folds with train/val/test subjects
2. model_comparison.csv - Side-by-side comparison with averages
3. summary_statistics.csv - Detailed stats per model/filtering condition
4. motion_filtering_report.txt - Full analysis report

Usage:
    python aggregate_motionfiltering_results.py --work-dir work_dir/motionfiltering_TIMESTAMP
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MotionFilteringAggregator:
    """Comprehensive aggregator for motion filtering ablation study"""

    def __init__(self, work_dir: str, output_dir: Optional[str] = None):
        self.work_dir = Path(work_dir)
        if not self.work_dir.exists():
            raise ValueError(f"Work directory does not exist: {work_dir}")

        self.output_dir = Path(output_dir) if output_dir else self.work_dir / "aggregated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define expected experiments: (name, model, filtering)
        self.experiments = [
            ("Model1_TransModel_NoFilter", "TransModel", "No"),
            ("Model1_TransModel_WithFilter", "TransModel", "Yes"),
            ("Model2_IMUTransformer_NoFilter", "IMUTransformer", "No"),
            ("Model2_IMUTransformer_WithFilter", "IMUTransformer", "Yes"),
            ("Model3_DualStream_NoFilter", "DualStream", "No"),
            ("Model3_DualStream_WithFilter", "DualStream", "Yes"),
            ("Model4_Madgwick_NoFilter", "Madgwick", "No"),
            ("Model4_Madgwick_WithFilter", "Madgwick", "Yes"),
        ]

        print(f"Work directory: {self.work_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Expected experiments: {len(self.experiments)}")

    def extract_fold_results(self, experiment_dir: Path) -> Optional[pd.DataFrame]:
        """Extract fold-level results from epoch_test_score.csv"""
        csv_path = experiment_dir / "epoch_test_score.csv"

        if not csv_path.exists():
            print(f"    Warning: No epoch_test_score.csv in {experiment_dir.name}")
            return None

        try:
            df = pd.read_csv(csv_path)
            print(f"    Loaded {len(df)} folds from {experiment_dir.name}")
            return df
        except Exception as e:
            print(f"    Error loading {csv_path}: {e}")
            return None

    def extract_subject_info(self, config_path: Path) -> Dict[str, List[int]]:
        """Extract train/val/test subjects from config or logs"""
        # Try to read from config
        subjects_info = {
            'all_subjects': [],
            'val_subjects': [],
            'train_subjects': [],
            'test_subjects': []
        }

        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    subjects_info['all_subjects'] = config.get('subjects', [])
                    subjects_info['val_subjects'] = config.get('validation_subjects', [])
            except Exception as e:
                print(f"    Warning: Could not read config {config_path}: {e}")

        return subjects_info

    def aggregate_comprehensive_scores(self) -> pd.DataFrame:
        """
        Generate comprehensive scores CSV with ALL fold-level details

        Columns:
        - model: Model name (TransModel, IMUTransformer, DualStream, Madgwick)
        - low_pass_filter: Yes/No
        - fold: Fold number
        - test_subject: Subject ID held out for testing
        - val_subjects: Validation subject IDs
        - train_subjects: Training subject IDs
        - accuracy, f1_score, precision, recall, auc: Performance metrics
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE SCORES")
        print("="*80)

        all_rows = []

        for exp_name, model_name, filtering in self.experiments:
            exp_dir = self.work_dir / exp_name

            if not exp_dir.exists():
                print(f"\n✗ Missing: {exp_name}")
                continue

            print(f"\n✓ Processing: {exp_name}")

            # Extract fold results
            fold_df = self.extract_fold_results(exp_dir)
            if fold_df is None:
                continue

            # Find config file (check both temp and original)
            config_path = exp_dir.parent / f"{exp_name}_temp.yaml"
            if not config_path.exists():
                # Try to find original config
                if "TransModel" in exp_name:
                    config_path = Path("config/smartfallmm/transmodel_motionfilter.yaml")
                elif "IMUTransformer" in exp_name:
                    config_path = Path("config/smartfallmm/imu_transformer_motionfilter.yaml")
                elif "DualStream" in exp_name:
                    config_path = Path("config/smartfallmm/dualstream_optimal_motionfilter.yaml")
                elif "Madgwick" in exp_name:
                    config_path = Path("config/smartfallmm/madgwick_motionfilter.yaml")

            # Extract subject info
            subjects_info = self.extract_subject_info(config_path)
            all_subjects = subjects_info['all_subjects']
            val_subjects = subjects_info['val_subjects']

            # Process each fold
            for idx, row in fold_df.iterrows():
                test_subject = row.get('Subject', idx + 1)

                # Determine train subjects (all - val - test)
                if all_subjects:
                    train_subjects = [s for s in all_subjects
                                      if s not in val_subjects and s != test_subject]
                else:
                    train_subjects = []

                # Create row for comprehensive CSV
                result_row = {
                    'model': model_name,
                    'low_pass_filter': filtering,
                    'fold': idx + 1,
                    'test_subject': test_subject,
                    'val_subjects': str(val_subjects),
                    'train_subjects': str(train_subjects) if train_subjects else 'N/A',
                    'accuracy': row.get('Accuracy', np.nan),
                    'f1_score': row.get('F1 Score', np.nan),
                    'precision': row.get('Precision', np.nan),
                    'recall': row.get('Recall', np.nan),
                    'auc': row.get('AUC', np.nan)
                }

                all_rows.append(result_row)

        # Create DataFrame
        comprehensive_df = pd.DataFrame(all_rows)

        # Sort by model, filtering, fold
        comprehensive_df = comprehensive_df.sort_values(['model', 'low_pass_filter', 'fold'])

        print(f"\nTotal rows: {len(comprehensive_df)}")
        return comprehensive_df

    def compute_summary_statistics(self, comprehensive_df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean/std for each model×filtering combination"""
        print("\n" + "="*80)
        print("COMPUTING SUMMARY STATISTICS")
        print("="*80)

        metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        summaries = []

        for (model, filtering), group in comprehensive_df.groupby(['model', 'low_pass_filter']):
            summary = {
                'model': model,
                'low_pass_filter': filtering,
                'num_folds': len(group)
            }

            for metric in metrics:
                values = group[metric].dropna()
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

            summaries.append(summary)

        summary_df = pd.DataFrame(summaries)
        summary_df = summary_df.sort_values(['model', 'low_pass_filter'])

        return summary_df

    def create_comparison_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create side-by-side comparison table (research-grade format)

        Format:
        | Model | Filtering | Accuracy | F1 | Precision | Recall | AUC |
        | TransModel | No  | 0.85±0.03 | ... |
        | TransModel | Yes | 0.87±0.02 | ... |
        | ... |
        """
        print("\n" + "="*80)
        print("CREATING MODEL COMPARISON TABLE")
        print("="*80)

        comparison = summary_df[['model', 'low_pass_filter', 'num_folds']].copy()

        # Format metrics as "mean ± std"
        for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'auc']:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in summary_df.columns and std_col in summary_df.columns:
                comparison[metric] = summary_df.apply(
                    lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
                    if not pd.isna(row[mean_col]) else "N/A",
                    axis=1
                )

        return comparison

    def generate_report(self, comprehensive_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Generate detailed text report for publication"""
        print("\n" + "="*80)
        print("GENERATING REPORT")
        print("="*80)

        report_path = self.output_dir / "motion_filtering_report.txt"

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MOTION FILTERING ABLATION STUDY - COMPREHENSIVE REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("Study Design:\n")
            f.write("  - 4 Models × 2 Filtering Conditions = 8 Experiments\n")
            f.write("  - Cross-validation: Leave-One-Subject-Out (LOSO)\n")
            f.write("  - Total folds per experiment: ~30\n\n")

            f.write("Models:\n")
            f.write("  1. TransModel - Acc-only baseline (4ch)\n")
            f.write("  2. IMUTransformer - Acc-only with adaptive architecture (4ch)\n")
            f.write("  3. DualStream - Asymmetric dual-stream, acc+gyro (8ch)\n")
            f.write("  4. Madgwick - Sensor fusion to orientation (7ch)\n\n")

            f.write("Filtering Conditions:\n")
            f.write("  - WITHOUT low-pass: Raw sensor data (baseline)\n")
            f.write("  - WITH low-pass:\n")
            f.write("      Accelerometer: LOW-pass 5.5 Hz (removes high-freq noise)\n")
            f.write("      Gyroscope: HIGH-pass 0.5 Hz (removes low-freq drift)\n")
            f.write("      4th-order Butterworth, 25 Hz sampling\n\n")

            f.write("="*80 + "\n")
            f.write("RESULTS SUMMARY (sorted by F1 score)\n")
            f.write("="*80 + "\n\n")

            # Sort by F1 score descending
            sorted_summary = summary_df.sort_values('f1_score_mean', ascending=False)

            for idx, row in sorted_summary.iterrows():
                f.write(f"{idx+1}. {row['model']} (Filtering: {row['low_pass_filter']})\n")
                if not pd.isna(row['f1_score_mean']):
                    f.write(f"   F1 Score:  {row['f1_score_mean']:.4f} ± {row['f1_score_std']:.4f}\n")
                if not pd.isna(row['accuracy_mean']):
                    f.write(f"   Accuracy:  {row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f}\n")
                if not pd.isna(row['precision_mean']):
                    f.write(f"   Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}\n")
                if not pd.isna(row['recall_mean']):
                    f.write(f"   Recall:    {row['recall_mean']:.4f} ± {row['recall_std']:.4f}\n")
                if not pd.isna(row['auc_mean']):
                    f.write(f"   AUC:       {row['auc_mean']:.4f} ± {row['auc_std']:.4f}\n")
                f.write(f"   Folds:     {int(row['num_folds'])}\n\n")

            f.write("="*80 + "\n")
            f.write("FILTERING IMPACT ANALYSIS\n")
            f.write("="*80 + "\n\n")

            # Compute filtering impact for each model
            for model in ['TransModel', 'IMUTransformer', 'DualStream', 'Madgwick']:
                no_filter = summary_df[(summary_df['model'] == model) &
                                        (summary_df['low_pass_filter'] == 'No')]
                with_filter = summary_df[(summary_df['model'] == model) &
                                          (summary_df['low_pass_filter'] == 'Yes')]

                if len(no_filter) > 0 and len(with_filter) > 0:
                    f.write(f"{model}:\n")
                    for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                        no_val = no_filter.iloc[0][f'{metric}_mean']
                        with_val = with_filter.iloc[0][f'{metric}_mean']
                        if not pd.isna(no_val) and not pd.isna(with_val):
                            delta = with_val - no_val
                            pct_change = (delta / no_val) * 100
                            sign = "+" if delta >= 0 else ""
                            f.write(f"  {metric.capitalize()}: {sign}{delta:.4f} ({sign}{pct_change:.2f}%)\n")
                    f.write("\n")

            f.write("="*80 + "\n")

        print(f"Report saved: {report_path}")

    def run(self):
        """Run full aggregation pipeline"""
        print("\n" + "="*80)
        print("MOTION FILTERING ABLATION - RESULTS AGGREGATION")
        print("="*80)

        # 1. Generate comprehensive scores
        comprehensive_df = self.aggregate_comprehensive_scores()
        comp_path = self.output_dir / "scores_comprehensive.csv"
        comprehensive_df.to_csv(comp_path, index=False)
        print(f"\n✓ Saved: {comp_path}")

        # 2. Compute summary statistics
        summary_df = self.compute_summary_statistics(comprehensive_df)
        summary_path = self.output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved: {summary_path}")

        # 3. Create comparison table
        comparison_df = self.create_comparison_table(summary_df)
        comparison_path = self.output_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"✓ Saved: {comparison_path}")

        # 4. Generate report
        self.generate_report(comprehensive_df, summary_df)

        # 5. Display comparison table
        print("\n" + "="*80)
        print("MODEL COMPARISON (Quick View)")
        print("="*80)
        print(comparison_df.to_string(index=False))

        print("\n" + "="*80)
        print("AGGREGATION COMPLETE")
        print("="*80)
        print(f"\nAll results saved in: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate motion filtering ablation study results"
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
        help="Output directory (default: work-dir/aggregated)"
    )

    args = parser.parse_args()

    try:
        aggregator = MotionFilteringAggregator(args.work_dir, args.output_dir)
        aggregator.run()
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
