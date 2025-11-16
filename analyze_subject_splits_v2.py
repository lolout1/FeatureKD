#!/usr/bin/env python3
"""
Enhanced subject analysis that uses actual preprocessing pipeline.
This ensures ADL ratio calculations match what actually gets used during training.

Purpose:
- Load data through the actual preprocessing pipeline
- Account for filtering (length mismatches, quality checks, etc.)
- Calculate realistic fall/ADL ratios based on data that survives preprocessing
- Recommend optimal validation subjects with ~60% ADL ratio
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.loader import UTDDatasetBuilder


def analyze_preprocessed_data(config_path='config/smartfallmm/imu_8channel.yaml'):
    """
    Analyze subject data using actual preprocessing pipeline.

    Args:
        config_path: Path to a config file to use for preprocessing settings

    Returns:
        DataFrame with per-subject statistics after preprocessing
    """

    print("="*80)
    print("PREPROCESSING-AWARE SUBJECT ANALYSIS")
    print("="*80)
    print()

    # Load config
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_args = config.get('dataset_args', {})
    subjects = config.get('subjects', [])

    print(f"Found {len(subjects)} subjects to analyze")
    print(f"Preprocessing settings:")
    print(f"  - Mode: {dataset_args.get('mode', 'sliding_window')}")
    print(f"  - Max length: {dataset_args.get('max_length', 128)}")
    print(f"  - Modalities: {dataset_args.get('modalities', [])}")
    print(f"  - Enable filtering: {dataset_args.get('enable_filtering', False)}")
    print(f"  - Enable gyro quality check: {dataset_args.get('enable_gyro_quality_check', False)}")
    print(f"  - Discard mismatched: {dataset_args.get('discard_mismatched_modalities', False)}")
    print()

    # Create builder for each subject individually
    subject_stats = {}

    for subject_id in subjects:
        print(f"Processing subject {subject_id}...", end=' ')

        try:
            # Create builder with full preprocessing pipeline
            builder = UTDDatasetBuilder(
                dataset=config.get('dataset', 'smartfallmm'),
                **dataset_args
            )

            # Process this subject's data
            from utils.dataset import split_by_subjects
            data = split_by_subjects(builder, [subject_id], fuse=False, print_validation=False)

            # Get statistics from builder
            stats = builder.subject_modality_stats[str(subject_id)]

            # Calculate ratios
            total_windows = stats['window_count']
            fall_windows = stats['fall_windows']
            adl_windows = stats['adl_windows']

            fall_ratio = fall_windows / total_windows if total_windows > 0 else 0
            adl_ratio = adl_windows / total_windows if total_windows > 0 else 0

            subject_stats[subject_id] = {
                'subject_id': subject_id,
                'total_trials': stats['total_trials'],
                'valid_trials': stats['valid_trials'],
                'skipped_trials': stats['total_trials'] - stats['valid_trials'],
                'total_windows': total_windows,
                'fall_windows': fall_windows,
                'adl_windows': adl_windows,
                'fall_ratio': fall_ratio,
                'adl_ratio': adl_ratio,
                'fall:adl_ratio': f"1:{adl_windows/fall_windows:.2f}" if fall_windows > 0 else "N/A",
                # Skip reasons
                'skip_missing': stats['skipped_missing_modality'],
                'skip_mismatch': stats['skipped_length_mismatch'],
                'skip_short': stats['skipped_too_short'],
                'skip_dtw': stats['skipped_dtw_length_mismatch'],
                'skip_gyro_hard': stats['skipped_poor_gyro_hard'],
                'skip_gyro_adaptive': stats['skipped_poor_gyro_adaptive'],
            }

            print(f"✓ (windows: {total_windows}, valid trials: {stats['valid_trials']}/{stats['total_trials']})")

        except Exception as e:
            print(f"✗ Error: {e}")
            subject_stats[subject_id] = {
                'subject_id': subject_id,
                'total_trials': 0,
                'valid_trials': 0,
                'skipped_trials': 0,
                'total_windows': 0,
                'fall_windows': 0,
                'adl_windows': 0,
                'fall_ratio': 0,
                'adl_ratio': 0,
                'fall:adl_ratio': 'N/A',
                'skip_missing': 0,
                'skip_mismatch': 0,
                'skip_short': 0,
                'skip_dtw': 0,
                'skip_gyro_hard': 0,
                'skip_gyro_adaptive': 0,
            }

    # Convert to DataFrame
    df = pd.DataFrame(list(subject_stats.values()))
    df = df.sort_values('subject_id')

    return df


def recommend_validation_subjects_preprocessed(df, target_adl_ratio=0.60, tolerance=0.05, min_windows=50):
    """
    Recommend validation subject pairs based on preprocessed data.

    Args:
        df: DataFrame with subject statistics (after preprocessing)
        target_adl_ratio: Target ADL ratio (default: 0.60 for 60% ADLs)
        tolerance: Acceptable deviation from target ratio (default: 0.05)
        min_windows: Minimum number of windows required (default: 50)

    Returns:
        List of tuples: [(subject1, subject2), combined_stats]
    """

    # Filter subjects with sufficient data
    valid_subjects = df[df['total_windows'] >= min_windows].copy()

    print(f"\nFound {len(valid_subjects)} subjects with >= {min_windows} windows")

    # Find all pairs that give target ADL ratio
    recommendations = []

    for i, row1 in valid_subjects.iterrows():
        for j, row2 in valid_subjects.iterrows():
            if row1['subject_id'] >= row2['subject_id']:
                continue  # Avoid duplicates

            # Combine statistics
            combined_fall = row1['fall_windows'] + row2['fall_windows']
            combined_adl = row1['adl_windows'] + row2['adl_windows']
            combined_total = combined_fall + combined_adl

            if combined_total == 0:
                continue

            combined_adl_ratio = combined_adl / combined_total

            # Check if within tolerance
            if abs(combined_adl_ratio - target_adl_ratio) <= tolerance:
                recommendations.append({
                    'subject1': int(row1['subject_id']),
                    'subject2': int(row2['subject_id']),
                    'combined_fall': combined_fall,
                    'combined_adl': combined_adl,
                    'combined_total': combined_total,
                    'adl_ratio': combined_adl_ratio,
                    'deviation': abs(combined_adl_ratio - target_adl_ratio),
                    's1_windows': row1['total_windows'],
                    's2_windows': row2['total_windows'],
                })

    # Sort by deviation from target
    recommendations.sort(key=lambda x: x['deviation'])

    return recommendations


def main():
    """Main analysis function."""

    # Use IMU 8-channel config as reference (has both acc+gyro, quality checks, etc.)
    config_path = 'config/smartfallmm/imu_8channel.yaml'

    print("Analyzing subjects with actual preprocessing pipeline...")
    print("This may take a few minutes...\n")

    df = analyze_preprocessed_data(config_path)

    print("\n" + "="*80)
    print("PER-SUBJECT STATISTICS (AFTER PREPROCESSING)")
    print("="*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)

    # Display main stats
    display_cols = ['subject_id', 'total_trials', 'valid_trials', 'total_windows',
                    'fall_windows', 'adl_windows', 'fall_ratio', 'adl_ratio', 'fall:adl_ratio']
    print(df[display_cols].to_string(index=False))

    # Overall statistics
    total_fall = df['fall_windows'].sum()
    total_adl = df['adl_windows'].sum()
    total_windows = df['total_windows'].sum()
    overall_adl_ratio = total_adl / total_windows if total_windows > 0 else 0

    print("\n" + "="*80)
    print("OVERALL STATISTICS (AFTER PREPROCESSING)")
    print("="*80)
    print(f"Total windows: {total_windows}")
    print(f"Total fall windows: {total_fall}")
    print(f"Total ADL windows: {total_adl}")
    print(f"Overall fall:ADL ratio: 1:{total_adl/total_fall:.2f}" if total_fall > 0 else "N/A")
    print(f"Overall ADL ratio: {overall_adl_ratio:.2%}")
    print()

    # Recommend validation subject pairs
    print("\n" + "="*80)
    print("VALIDATION SUBJECT PAIR RECOMMENDATIONS (TARGET: 60% ADLs)")
    print("="*80)

    recommendations = recommend_validation_subjects_preprocessed(
        df,
        target_adl_ratio=0.60,
        tolerance=0.05,
        min_windows=50
    )

    if recommendations:
        print(f"\nFound {len(recommendations)} subject pairs with 60% ± 5% ADL ratio:")
        print("\nTop 10 recommendations:")
        print("-"*80)

        for i, rec in enumerate(recommendations[:10], 1):
            print(f"{i}. Subjects [{rec['subject1']}, {rec['subject2']}]:")
            print(f"   - Combined: {rec['combined_fall']} falls + {rec['combined_adl']} ADLs = {rec['combined_total']} windows")
            print(f"   - ADL ratio: {rec['adl_ratio']:.2%} (deviation: {rec['deviation']:.2%})")
            print(f"   - Individual windows: S{rec['subject1']}={rec['s1_windows']}, S{rec['subject2']}={rec['s2_windows']}")
            print()

        # Show the best recommendation
        best = recommendations[0]
        print("="*80)
        print("RECOMMENDED VALIDATION SUBJECTS:")
        print(f"  [{best['subject1']}, {best['subject2']}] - {best['adl_ratio']:.2%} ADLs")
        print("="*80)
    else:
        print("\nNo subject pairs found with target ADL ratio!")
        print("Consider widening the tolerance or lowering min_windows requirement.")

    # Show skip statistics
    print("\n" + "="*80)
    print("SKIP STATISTICS SUMMARY")
    print("="*80)
    skip_cols = ['subject_id', 'total_trials', 'valid_trials', 'skip_mismatch',
                 'skip_short', 'skip_dtw', 'skip_gyro_hard', 'skip_gyro_adaptive']
    subjects_with_skips = df[df['skipped_trials'] > 0]

    if len(subjects_with_skips) > 0:
        print(f"\n{len(subjects_with_skips)} subjects with skipped trials:")
        print(subjects_with_skips[skip_cols].to_string(index=False))
    else:
        print("\nNo subjects with skipped trials.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
