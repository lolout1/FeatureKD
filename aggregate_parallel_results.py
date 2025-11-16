#!/usr/bin/env python3
"""
Aggregate results from parallel LOSO training.

After running parallel LOSO jobs, this script collects all fold results
and creates a unified summary just like the sequential version.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def aggregate_results(base_work_dir, output_dir=None):
    """
    Aggregate results from multiple parallel folds.

    Args:
        base_work_dir: Base directory containing fold_* subdirectories
        output_dir: Where to save aggregated results (default: base_work_dir/aggregated)
    """

    print("="*80)
    print("AGGREGATING PARALLEL LOSO RESULTS")
    print("="*80)
    print(f"Base directory: {base_work_dir}")
    print()

    # Find all fold directories
    fold_dirs = sorted(glob.glob(os.path.join(base_work_dir, "fold_*")))

    if not fold_dirs:
        print("ERROR: No fold directories found!")
        return

    print(f"Found {len(fold_dirs)} fold directories")

    # Collect results from each fold
    all_results = []

    for fold_dir in fold_dirs:
        # Extract test subject from directory name
        fold_name = os.path.basename(fold_dir)
        test_subject = fold_name.replace('fold_', '')

        # Look for results CSV
        csv_files = glob.glob(os.path.join(fold_dir, "**/per_fold_detailed.csv"), recursive=True)

        if not csv_files:
            print(f"  Warning: No results found for {fold_name}")
            continue

        # Read the results
        df = pd.read_csv(csv_files[0])

        # Add to collection
        all_results.append(df)
        print(f"  ✓ Loaded results for test subject {test_subject}")

    if not all_results:
        print("\nERROR: No results could be loaded!")
        return

    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Sort by fold/test subject
    if 'test_subject' in combined_df.columns:
        combined_df = combined_df.sort_values('test_subject')
    elif 'fold' in combined_df.columns:
        combined_df = combined_df.sort_values('fold')

    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(base_work_dir, "aggregated")
    os.makedirs(output_dir, exist_ok=True)

    # Save combined results
    combined_csv = os.path.join(output_dir, "all_folds_combined.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n✓ Saved combined results: {combined_csv}")

    # Calculate summary statistics
    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns

    # Calculate mean and std for each metric
    summary = {}
    for col in numeric_cols:
        if col not in ['fold', 'test_subject']:
            summary[col + '_mean'] = combined_df[col].mean()
            summary[col + '_std'] = combined_df[col].std()

    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(output_dir, "summary_statistics.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved summary statistics: {summary_csv}")

    # Print key metrics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    key_metrics = ['test_f1', 'test_acc', 'test_precision', 'test_recall', 'test_auc']
    for metric in key_metrics:
        if metric in combined_df.columns:
            mean_val = combined_df[metric].mean()
            std_val = combined_df[metric].std()
            print(f"{metric.upper():20s}: {mean_val:6.2f} ± {std_val:5.2f}%")

    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("="*80)


def main():
    """Main function."""

    if len(sys.argv) < 2:
        print("Usage: python aggregate_parallel_results.py <base_work_dir> [output_dir]")
        print("\nExample:")
        print("  python aggregate_parallel_results.py work_dir/parallel_loso_imu_8channel")
        sys.exit(1)

    base_work_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    aggregate_results(base_work_dir, output_dir)


if __name__ == "__main__":
    main()
