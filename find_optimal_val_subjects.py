#!/usr/bin/env python3
"""
Find optimal validation subjects that maintain 60-65% ADL ratio
for BOTH accelerometer-only and acc+gyro configurations.

This ensures validation consistency across different model types.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.loader import DatasetBuilder
from utils.dataset import SmartFallMM, split_by_subjects


def quick_analyze_subject(subject_id, config_type='acc_only'):
    """
    Quickly analyze a single subject with specified config.

    Args:
        subject_id: Subject ID to analyze
        config_type: 'acc_only' or 'acc_gyro'

    Returns:
        dict with fall/adl window counts
    """

    # Configuration templates
    configs = {
        'acc_only': {
            'dataset': 'smartfallmm',
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd',
            'modalities': ['accelerometer'],
            'age_group': ['young'],
            'sensors': ['watch'],
            'use_skeleton': False,
            'enable_filtering': False,
            'enable_normalization': False,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer'],
        },
        'acc_gyro': {
            'dataset': 'smartfallmm',
            'mode': 'sliding_window',
            'max_length': 128,
            'task': 'fd',
            'modalities': ['accelerometer', 'gyroscope'],
            'age_group': ['young'],
            'sensors': ['watch'],
            'use_skeleton': False,
            'enable_filtering': False,
            'enable_normalization': False,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer', 'gyroscope'],
        }
    }

    try:
        # Get config and remove 'dataset' key (it's not needed for DatasetBuilder)
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
        }


def find_optimal_pairs(subjects, target_adl=0.60, tolerance=0.05, min_windows=50):
    """
    Find validation subject pairs that work for both acc-only and acc+gyro.

    Args:
        subjects: List of subject IDs to consider
        target_adl: Target ADL ratio (0.60 = 60%)
        tolerance: Acceptable deviation
        min_windows: Minimum windows required per subject

    Returns:
        List of recommended pairs with statistics for both configs
    """

    print("="*80)
    print("FINDING OPTIMAL VALIDATION SUBJECTS")
    print("="*80)
    print(f"Target: {target_adl:.0%} ± {tolerance:.0%} ADLs")
    print(f"Analyzing {len(subjects)} subjects for BOTH acc-only and acc+gyro configs...")
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

    print("\nAnalyzing accelerometer + gyroscope configuration:")
    for subj in subjects:
        print(f"  Subject {subj}...", end=' ')
        stats = quick_analyze_subject(subj, 'acc_gyro')
        acc_gyro_stats[subj] = stats
        print(f"✓ ({stats['total_windows']} windows, {stats['adl_ratio']:.1%} ADLs)")

    print("\n" + "="*80)
    print("FINDING COMPATIBLE PAIRS")
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

            # Check if BOTH configs meet the target
            acc_only_ok = abs(acc_only_adl_ratio - target_adl) <= tolerance
            acc_gyro_ok = abs(acc_gyro_adl_ratio - target_adl) <= tolerance

            if acc_only_ok and acc_gyro_ok:
                # Both configs meet target!
                max_deviation = max(
                    abs(acc_only_adl_ratio - target_adl),
                    abs(acc_gyro_adl_ratio - target_adl)
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

    # Sort by max deviation (lower is better)
    recommendations.sort(key=lambda x: x['max_deviation'])

    return recommendations, acc_only_stats, acc_gyro_stats


def main():
    """Main function."""

    # Standard subjects used in experiments
    subjects = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    recommendations, acc_only_stats, acc_gyro_stats = find_optimal_pairs(
        subjects,
        target_adl=0.60,
        tolerance=0.05,
        min_windows=50
    )

    print(f"\nFound {len(recommendations)} compatible pairs!\n")

    if recommendations:
        print("="*80)
        print("TOP 10 RECOMMENDATIONS (work for BOTH acc-only and acc+gyro)")
        print("="*80)

        for i, rec in enumerate(recommendations[:10], 1):
            subjs = rec['subjects']
            print(f"\n{i}. Subjects {subjs}:")
            print(f"   Acc-only:  {rec['acc_only_fall']} falls + {rec['acc_only_adl']} ADLs = {rec['acc_only_total']} windows ({rec['acc_only_adl_ratio']:.1%} ADLs)")
            print(f"   Acc+gyro:  {rec['acc_gyro_fall']} falls + {rec['acc_gyro_adl']} ADLs = {rec['acc_gyro_total']} windows ({rec['acc_gyro_adl_ratio']:.1%} ADLs)")
            print(f"   Max deviation from 60%: {rec['max_deviation']:.1%}")
            print(f"   Average ADL ratio: {rec['avg_adl_ratio']:.1%}")

        # Show the best recommendation
        best = recommendations[0]
        print("\n" + "="*80)
        print("✓ RECOMMENDED VALIDATION SUBJECTS:")
        print("="*80)
        print(f"  {best['subjects']}")
        print(f"  - Acc-only:  {best['acc_only_adl_ratio']:.1%} ADLs ({best['acc_only_total']} windows)")
        print(f"  - Acc+gyro:  {best['acc_gyro_adl_ratio']:.1%} ADLs ({best['acc_gyro_total']} windows)")
        print(f"  - Deviation: ±{best['max_deviation']:.1%}")
        print("="*80)

        # Save results
        results_df = pd.DataFrame(recommendations[:20])
        results_df.to_csv('optimal_validation_subjects.csv', index=False)
        print(f"\n✓ Saved top 20 recommendations to: optimal_validation_subjects.csv")

    else:
        print("="*80)
        print("❌ NO COMPATIBLE PAIRS FOUND")
        print("="*80)
        print("Try widening tolerance or lowering min_windows requirement.")
        print("\nShowing individual subject stats for manual selection:")
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
