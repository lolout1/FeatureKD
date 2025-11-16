#!/usr/bin/env python3
"""
Find optimal validation subjects that maintain 40-50% ADL ratio
for BOTH accelerometer-only and acc+gyro configurations.

This ensures validation consistency across different model types
with motion filtering enabled.
"""

import os
import sys
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from utils.loader import DatasetBuilder
from utils.dataset import SmartFallMM, split_by_subjects


def quick_analyze_subject(subject_id, config_type='acc_only', enable_motion_filtering=True):
    """
    Quickly analyze a single subject with specified config.

    Args:
        subject_id: Subject ID to analyze
        config_type: 'acc_only' or 'acc_gyro'
        enable_motion_filtering: Whether to apply motion filtering

    Returns:
        dict with fall/adl window counts
    """

    # Configuration templates
    base_config = {
        'dataset': 'smartfallmm',
        'mode': 'sliding_window',
        'max_length': 128,
        'task': 'fd',
        'age_group': ['young'],
        'sensors': ['watch'],
        'use_skeleton': False,
        'enable_filtering': True,        # Enable low-pass filter
        'filter_cutoff': 5.5,
        'filter_fs': 25,
        'enable_normalization': False,
        'discard_mismatched_modalities': True,
        'enable_motion_filtering': enable_motion_filtering,
        'motion_threshold': 10.0,
        'motion_min_axes': 2,
    }

    configs = {
        'acc_only': {
            **base_config,
            'modalities': ['accelerometer'],
            'length_sensitive_modalities': ['accelerometer'],
        },
        'acc_gyro': {
            **base_config,
            'modalities': ['accelerometer', 'gyroscope'],
            'length_sensitive_modalities': ['accelerometer', 'gyroscope'],
            'enable_gyro_alignment': True,  # Enable DTW for acc+gyro
        }
    }

    try:
        # Get config and remove 'dataset' key
        config = configs[config_type].copy()
        config.pop('dataset', None)

        # Create dataset
        sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data'))
        sm_dataset.pipe_line(
            age_group=config['age_group'],
            modalities=config['modalities'],
            sensors=config['sensors']
        )

        # Create builder
        builder = DatasetBuilder(sm_dataset, **config)

        # Process subject
        data = split_by_subjects(builder, [subject_id], fuse=False, print_validation=False)

        stats = builder.subject_modality_stats[str(subject_id)]

        return {
            'subject_id': subject_id,
            'total_windows': stats['window_count'],
            'fall_windows': stats['fall_windows'],
            'adl_windows': stats['adl_windows'],
            'fall_ratio': stats['fall_windows'] / stats['window_count'] if stats['window_count'] > 0 else 0,
            'adl_ratio': stats['adl_windows'] / stats['window_count'] if stats['window_count'] > 0 else 0,
            'motion_total': stats.get('motion_total_windows', 0),
            'motion_rejected': stats.get('motion_rejected_windows', 0),
        }
    except Exception as e:
        print(f"  Error analyzing subject {subject_id}: {e}")
        return {
            'subject_id': subject_id,
            'total_windows': 0,
            'fall_windows': 0,
            'adl_windows': 0,
            'fall_ratio': 0,
            'adl_ratio': 0,
            'motion_total': 0,
            'motion_rejected': 0,
        }


def find_optimal_pairs(subjects, target_adl_min=0.40, target_adl_max=0.50, min_windows=50):
    """
    Find validation subject pairs that work for both acc-only and acc+gyro.

    Args:
        subjects: List of subject IDs to consider
        target_adl_min: Minimum ADL ratio (0.40 = 40%)
        target_adl_max: Maximum ADL ratio (0.50 = 50%)
        min_windows: Minimum windows required per subject

    Returns:
        List of recommended pairs with statistics for both configs
    """

    print("="*80)
    print("FINDING OPTIMAL VALIDATION SUBJECTS (40-50% ADLs)")
    print("="*80)
    print(f"Target: {target_adl_min:.0%} - {target_adl_max:.0%} ADLs")
    print(f"Analyzing {len(subjects)} subjects for BOTH acc-only and acc+gyro configs...")
    print(f"Motion filtering: ENABLED (threshold=10.0, min_axes=2)")
    print(f"Low-pass filter: ENABLED (cutoff=5.5Hz, fs=25Hz)")
    print()

    # Analyze all subjects for both configs
    acc_only_stats = {}
    acc_gyro_stats = {}

    print("Analyzing accelerometer-only configuration:")
    for subj in subjects:
        print(f"  Subject {subj}...", end=' ')
        stats = quick_analyze_subject(subj, 'acc_only')
        acc_only_stats[subj] = stats
        print(f"✓ ({stats['total_windows']} windows, {stats['adl_ratio']:.1%} ADLs)")

    print("\nAnalyzing accelerometer + gyroscope configuration (with DTW):")
    for subj in subjects:
        print(f"  Subject {subj}...", end=' ')
        stats = quick_analyze_subject(subj, 'acc_gyro')
        acc_gyro_stats[subj] = stats
        print(f"✓ ({stats['total_windows']} windows, {stats['adl_ratio']:.1%} ADLs)")

    print("\n" + "="*80)
    print("FINDING COMPATIBLE PAIRS (40-50% ADLs)")
    print("="*80)

    # Find pairs that work for BOTH configs
    recommendations = []

    for i, subj1 in enumerate(subjects):
        for subj2 in subjects[i+1:]:

            # Check if both subjects have enough windows in both configs
            if (acc_only_stats[subj1]['total_windows'] < min_windows or
                acc_only_stats[subj2]['total_windows'] < min_windows or
                acc_gyro_stats[subj1]['total_windows'] < min_windows or
                acc_gyro_stats[subj2]['total_windows'] < min_windows):
                continue

            # Calculate combined stats for acc-only
            acc_only_fall = acc_only_stats[subj1]['fall_windows'] + acc_only_stats[subj2]['fall_windows']
            acc_only_adl = acc_only_stats[subj1]['adl_windows'] + acc_only_stats[subj2]['adl_windows']
            acc_only_total = acc_only_fall + acc_only_adl
            acc_only_adl_ratio = acc_only_adl / acc_only_total if acc_only_total > 0 else 0

            # Calculate combined stats for acc+gyro
            acc_gyro_fall = acc_gyro_stats[subj1]['fall_windows'] + acc_gyro_stats[subj2]['fall_windows']
            acc_gyro_adl = acc_gyro_stats[subj1]['adl_windows'] + acc_gyro_stats[subj2]['adl_windows']
            acc_gyro_total = acc_gyro_fall + acc_gyro_adl
            acc_gyro_adl_ratio = acc_gyro_adl / acc_gyro_total if acc_gyro_total > 0 else 0

            # Check if BOTH configs meet the target (40-50% ADLs)
            acc_only_ok = target_adl_min <= acc_only_adl_ratio <= target_adl_max
            acc_gyro_ok = target_adl_min <= acc_gyro_adl_ratio <= target_adl_max

            if acc_only_ok and acc_gyro_ok:
                # Both configs meet target!
                target_center = (target_adl_min + target_adl_max) / 2
                max_deviation = max(
                    abs(acc_only_adl_ratio - target_center),
                    abs(acc_gyro_adl_ratio - target_center)
                )

                recommendations.append({
                    'subjects': [subj1, subj2],
                    'acc_only_adl_ratio': acc_only_adl_ratio,
                    'acc_only_total': acc_only_total,
                    'acc_only_fall': acc_only_fall,
                    'acc_only_adl': acc_only_adl,
                    'acc_gyro_adl_ratio': acc_gyro_adl_ratio,
                    'acc_gyro_total': acc_gyro_total,
                    'acc_gyro_fall': acc_gyro_fall,
                    'acc_gyro_adl': acc_gyro_adl,
                    'max_deviation': max_deviation,
                    'avg_adl_ratio': (acc_only_adl_ratio + acc_gyro_adl_ratio) / 2,
                })

    # Sort by max deviation (lower is better - closer to 45%)
    recommendations.sort(key=lambda x: x['max_deviation'])

    return recommendations, acc_only_stats, acc_gyro_stats


def update_config_files(validation_subjects):
    """
    Update validation_subjects in all motion filtering config files.

    Args:
        validation_subjects: List of subject IDs to use for validation
    """
    config_files = [
        'config/smartfallmm/transmodel_motionfilter.yaml',
        'config/smartfallmm/imu_transformer_motionfilter.yaml',
        'config/smartfallmm/dualstream_optimal_motionfilter.yaml',
        'config/smartfallmm/madgwick_motionfilter.yaml',
    ]

    print("\n" + "="*80)
    print("UPDATING CONFIG FILES")
    print("="*80)
    print(f"Setting validation_subjects to: {validation_subjects}\n")

    for config_file in config_files:
        config_path = os.path.join(os.getcwd(), config_file)

        if not os.path.exists(config_path):
            print(f"⚠ Skipping {config_file} (not found)")
            continue

        try:
            # Read the file
            with open(config_path, 'r') as f:
                content = f.read()

            # Replace validation_subjects line using regex
            # Match: validation_subjects: [any content]
            pattern = r'validation_subjects:\s*\[[^\]]*\]'
            replacement = f'validation_subjects: {validation_subjects}'

            new_content = re.sub(pattern, replacement, content)

            # Write back
            with open(config_path, 'w') as f:
                f.write(new_content)

            print(f"✓ Updated {config_file}")

        except Exception as e:
            print(f"❌ Error updating {config_file}: {e}")

    print("\n" + "="*80)
    print("✓ CONFIG FILES UPDATED")
    print("="*80)


def main():
    """Main function."""

    # Standard subjects used in experiments
    subjects = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    recommendations, acc_only_stats, acc_gyro_stats = find_optimal_pairs(
        subjects,
        target_adl_min=0.40,
        target_adl_max=0.50,
        min_windows=50
    )

    print(f"\nFound {len(recommendations)} compatible pairs!\n")

    if recommendations:
        print("="*80)
        print("TOP 10 RECOMMENDATIONS (40-50% ADLs for BOTH acc-only and acc+gyro)")
        print("="*80)

        for i, rec in enumerate(recommendations[:10], 1):
            subjs = rec['subjects']
            print(f"\n{i}. Subjects {subjs}:")
            print(f"   Acc-only:  {rec['acc_only_fall']} falls + {rec['acc_only_adl']} ADLs = {rec['acc_only_total']} windows ({rec['acc_only_adl_ratio']:.1%} ADLs)")
            print(f"   Acc+gyro:  {rec['acc_gyro_fall']} falls + {rec['acc_gyro_adl']} ADLs = {rec['acc_gyro_total']} windows ({rec['acc_gyro_adl_ratio']:.1%} ADLs)")
            print(f"   Deviation from 45%: ±{rec['max_deviation']:.1%}")
            print(f"   Average ADL ratio: {rec['avg_adl_ratio']:.1%}")

        # Show the best recommendation
        best = recommendations[0]
        print("\n" + "="*80)
        print("✓ RECOMMENDED VALIDATION SUBJECTS (40-50% ADLs):")
        print("="*80)
        print(f"  {best['subjects']}")
        print(f"  - Acc-only:  {best['acc_only_adl_ratio']:.1%} ADLs ({best['acc_only_total']} windows)")
        print(f"  - Acc+gyro:  {best['acc_gyro_adl_ratio']:.1%} ADLs ({best['acc_gyro_total']} windows)")
        print(f"  - Deviation from 45%: ±{best['max_deviation']:.1%}")
        print("="*80)

        # Save results
        results_df = pd.DataFrame(recommendations[:20])
        results_df.to_csv('validation_subjects_40pct_adl.csv', index=False)
        print(f"\n✓ Saved top 20 recommendations to: validation_subjects_40pct_adl.csv")

        # Automatically update config files with the best validation subjects
        update_config_files(best['subjects'])

    else:
        print("="*80)
        print("❌ NO COMPATIBLE PAIRS FOUND IN 40-50% RANGE")
        print("="*80)
        print("Showing individual subject stats for manual selection:")
        print("\nAcc-only configuration:")
        for subj, stats in sorted(acc_only_stats.items()):
            if stats['total_windows'] >= 30:
                print(f"  Subject {subj}: {stats['total_windows']:3d} windows, {stats['adl_ratio']:.1%} ADLs")

        print("\nAcc+gyro configuration:")
        for subj, stats in sorted(acc_gyro_stats.items()):
            if stats['total_windows'] >= 30:
                print(f"  Subject {subj}: {stats['total_windows']:3d} windows, {stats['adl_ratio']:.1%} ADLs")

    print()


if __name__ == "__main__":
    main()
