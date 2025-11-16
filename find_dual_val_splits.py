#!/usr/bin/env python3
"""
Find optimal validation subjects for BOTH motion-filtering and non-motion-filtering scenarios.

This ensures proper ADL ratios across all experimental configurations:
- Non-motion-filtering: Target 60% ADLs
- Motion-filtering: Target 45-50% ADLs (minimum, since motion filtering removes many ADLs)
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from utils.loader import DatasetBuilder
from utils.dataset import SmartFallMM, split_by_subjects


def analyze_subject_with_config(subject_id, config_name, config_dict):
    """
    Analyze a single subject with specific configuration.

    Args:
        subject_id: Subject ID to analyze
        config_name: Name of config (for logging)
        config_dict: Configuration dictionary

    Returns:
        dict with statistics
    """
    try:
        # Remove 'dataset' key to avoid conflicts
        config = config_dict.copy()
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

        total = stats['window_count']
        fall = stats['fall_windows']
        adl = stats['adl_windows']

        return {
            'subject_id': subject_id,
            'total_windows': total,
            'fall_windows': fall,
            'adl_windows': adl,
            'fall_ratio': fall / total if total > 0 else 0,
            'adl_ratio': adl / total if total > 0 else 0,
            'valid_trials': stats['valid_trials'],
            'total_trials': stats['total_trials'],
        }
    except Exception as e:
        print(f"    Error: {e}")
        return {
            'subject_id': subject_id,
            'total_windows': 0,
            'fall_windows': 0,
            'adl_windows': 0,
            'fall_ratio': 0,
            'adl_ratio': 0,
            'valid_trials': 0,
            'total_trials': 0,
        }


def find_optimal_pairs_multi_config(subjects, configs, min_windows=40):
    """
    Find validation pairs that work across multiple configurations.

    Args:
        subjects: List of subject IDs
        configs: Dict of {config_name: config_dict}
        min_windows: Minimum windows per subject

    Returns:
        dict of results per config and cross-config recommendations
    """

    # Analyze all subjects for all configs
    results_per_config = {}

    for config_name, config_dict in configs.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {config_name}")
        print(f"{'='*80}")

        subject_stats = {}
        for subj in subjects:
            print(f"  Subject {subj}...", end=' ')
            stats = analyze_subject_with_config(subj, config_name, config_dict)
            subject_stats[subj] = stats
            print(f"✓ ({stats['total_windows']} windows, {stats['adl_ratio']:.1%} ADLs)")

        results_per_config[config_name] = subject_stats

    # Find pairs that work for ALL configs
    print(f"\n{'='*80}")
    print("FINDING CROSS-CONFIG COMPATIBLE PAIRS")
    print(f"{'='*80}")

    # Define targets per config
    config_targets = {
        'acc_only_no_filter': (0.60, 0.05),  # 60% ± 5% ADLs
        'acc_only_motion_filter': (0.45, 0.05),  # 45% ± 5% ADLs (minimum)
        'acc_gyro_no_filter': (0.60, 0.05),  # 60% ± 5% ADLs
        'acc_gyro_motion_filter': (0.45, 0.05),  # 45% ± 5% ADLs (minimum)
    }

    recommendations = []

    for i, subj1 in enumerate(subjects):
        for subj2 in subjects[i+1:]:

            # Check if both subjects have sufficient data in ALL configs
            valid_in_all = True
            for config_name, stats_dict in results_per_config.items():
                if (stats_dict[subj1]['total_windows'] < min_windows or
                    stats_dict[subj2]['total_windows'] < min_windows):
                    valid_in_all = False
                    break

            if not valid_in_all:
                continue

            # Calculate combined stats for each config
            combined_stats = {}
            max_deviations = {}

            all_pass = True
            for config_name, stats_dict in results_per_config.items():
                stats1 = stats_dict[subj1]
                stats2 = stats_dict[subj2]

                combined_fall = stats1['fall_windows'] + stats2['fall_windows']
                combined_adl = stats1['adl_windows'] + stats2['adl_windows']
                combined_total = combined_fall + combined_adl

                if combined_total == 0:
                    all_pass = False
                    break

                combined_adl_ratio = combined_adl / combined_total

                combined_stats[config_name] = {
                    'fall': combined_fall,
                    'adl': combined_adl,
                    'total': combined_total,
                    'adl_ratio': combined_adl_ratio,
                }

                # Check if meets target for this config
                target, tolerance = config_targets.get(config_name, (0.60, 0.05))
                deviation = abs(combined_adl_ratio - target)
                max_deviations[config_name] = deviation

                if deviation > tolerance:
                    all_pass = False
                    break

            if not all_pass:
                continue

            # This pair works for all configs!
            recommendations.append({
                'subjects': [subj1, subj2],
                'stats_per_config': combined_stats,
                'max_deviation': max(max_deviations.values()),
                'avg_deviation': np.mean(list(max_deviations.values())),
            })

    # Sort by average deviation (lower is better)
    recommendations.sort(key=lambda x: x['avg_deviation'])

    return results_per_config, recommendations


def main():
    """Main function."""

    # Standard subjects
    subjects = [29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 43, 44, 45, 46, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

    # Define configurations to test
    configs = {
        'acc_only_no_filter': {
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
            'enable_motion_filtering': False,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer'],
        },
        'acc_only_motion_filter': {
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
            'enable_motion_filtering': True,  # MOTION FILTERING ENABLED
            'motion_threshold': 0.05,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer'],
        },
        'acc_gyro_no_filter': {
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
            'enable_motion_filtering': False,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer', 'gyroscope'],
        },
        'acc_gyro_motion_filter': {
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
            'enable_motion_filtering': True,  # MOTION FILTERING ENABLED
            'motion_threshold': 0.05,
            'discard_mismatched_modalities': True,
            'length_sensitive_modalities': ['accelerometer', 'gyroscope'],
        },
    }

    print("="*80)
    print("DUAL VALIDATION SPLIT ANALYSIS")
    print("="*80)
    print("\nFinding validation subjects that work for:")
    print("  1. Non-motion-filtering (target: ~60% ADLs)")
    print("  2. Motion-filtering (target: ~45-50% ADLs minimum)")
    print("\nThis will take a few minutes...")

    # Find optimal pairs
    results_per_config, recommendations = find_optimal_pairs_multi_config(
        subjects, configs, min_windows=30
    )

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    if recommendations:
        print(f"\nFound {len(recommendations)} subject pairs that work for ALL configurations!")
        print(f"\nTop 10 recommendations:")
        print("-"*80)

        for i, rec in enumerate(recommendations[:10], 1):
            subjs = rec['subjects']
            print(f"\n{i}. Subjects {subjs}:")
            print(f"   Max deviation: {rec['max_deviation']:.2%}, Avg deviation: {rec['avg_deviation']:.2%}")

            for config_name, stats in rec['stats_per_config'].items():
                print(f"   {config_name:30s}: {stats['adl_ratio']:5.1%} ADLs ({stats['total']:3d} windows)")

        # Recommend the best
        best = recommendations[0]
        print(f"\n{'='*80}")
        print("RECOMMENDED VALIDATION SUBJECTS (WORKS FOR ALL CONFIGS):")
        print(f"{'='*80}")
        print(f"  {best['subjects']}")
        print(f"\n  Performance:")
        for config_name, stats in best['stats_per_config'].items():
            print(f"    {config_name:30s}: {stats['adl_ratio']:5.1%} ADLs ({stats['adl']} ADL + {stats['fall']} fall = {stats['total']} total)")

        # Save results
        results_df = pd.DataFrame([{
            'subjects': str(rec['subjects']),
            'max_deviation': rec['max_deviation'],
            'avg_deviation': rec['avg_deviation'],
            **{f"{config}_adl_ratio": rec['stats_per_config'][config]['adl_ratio']
               for config in configs.keys()},
            **{f"{config}_total": rec['stats_per_config'][config]['total']
               for config in configs.keys()},
        } for rec in recommendations[:20]])

        results_df.to_csv('dual_val_split_results.csv', index=False)
        print(f"\n✓ Saved results to: dual_val_split_results.csv")

    else:
        print("\n❌ NO COMPATIBLE PAIRS FOUND")
        print("Try widening tolerance or lowering min_windows requirement.")

        # Show best per config
        for config_name in configs.keys():
            print(f"\n{config_name}:")
            print("  Top subjects by ADL ratio:")
            stats_list = [(sid, stats) for sid, stats in results_per_config[config_name].items()
                          if stats['total_windows'] >= 30]
            stats_list.sort(key=lambda x: abs(x[1]['adl_ratio'] - 0.50), reverse=False)

            for sid, stats in stats_list[:10]:
                print(f"    Subject {sid}: {stats['total_windows']:3d} windows, {stats['adl_ratio']:.1%} ADLs")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
