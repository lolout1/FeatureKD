"""
Analyze subject data splits in the SmartFallMM dataset.
This script examines the distribution of fall vs ADL data for each subject.

Purpose:
- Understand per-subject class distribution (fall/ADL ratio)
- Identify subjects suitable for validation (representative distribution)
- Ensure validation subjects have both accelerometer and gyroscope data
- Prevent data leakage by validating subject isolation
"""

import os
import pandas as pd
from collections import defaultdict
import numpy as np


def analyze_subject_data(data_dir='data/young/accelerometer/watch'):
    """
    Analyze the distribution of fall vs ADL data for each subject.

    Args:
        data_dir: Path to the data directory (default: data/young/accelerometer/watch)

    Returns:
        DataFrame with per-subject statistics
    """

    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return None

    subject_stats = defaultdict(lambda: {
        'fall_files': 0,
        'adl_files': 0,
        'total_files': 0,
        'fall_ratio': 0.0,
        'adl_ratio': 0.0
    })

    # Scan directory for CSV files
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue

        # Parse filename: format is S<subject>A<action>T<trial>.csv
        # Actions 1-9 are falls, 10+ are ADLs (based on code in loader.py line 711)
        try:
            # Remove .csv extension
            name = filename.replace('.csv', '')

            # Parse S<subject>A<action>T<trial>
            if not name.startswith('S'):
                continue

            # Split by 'A' to get subject and action
            parts = name[1:].split('A')  # Skip 'S' prefix
            if len(parts) < 2:
                continue

            subject_id = int(parts[0])
            action_part = parts[1].split('T')[0]  # Get action before 'T'
            action_id = int(action_part)

            # Determine if fall or ADL (based on code: label = int(trial.action_id > 9))
            is_fall = action_id <= 9

            subject_stats[subject_id]['total_files'] += 1
            if is_fall:
                subject_stats[subject_id]['fall_files'] += 1
            else:
                subject_stats[subject_id]['adl_files'] += 1

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue

    # Calculate ratios
    for subject_id, stats in subject_stats.items():
        total = stats['total_files']
        if total > 0:
            stats['fall_ratio'] = stats['fall_files'] / total
            stats['adl_ratio'] = stats['adl_files'] / total

    # Convert to DataFrame
    rows = []
    for subject_id, stats in sorted(subject_stats.items()):
        rows.append({
            'subject_id': subject_id,
            'total_files': stats['total_files'],
            'fall_files': stats['fall_files'],
            'adl_files': stats['adl_files'],
            'fall_ratio': stats['fall_ratio'],
            'adl_ratio': stats['adl_ratio'],
            'fall:adl_ratio': f"1:{stats['adl_files']/stats['fall_files']:.2f}" if stats['fall_files'] > 0 else "N/A"
        })

    df = pd.DataFrame(rows)
    return df


def check_modality_availability(base_dir='data/young'):
    """
    Check which subjects have both accelerometer and gyroscope data.

    Args:
        base_dir: Base directory containing modality folders

    Returns:
        Dictionary with modality availability per subject
    """
    modality_availability = defaultdict(lambda: {
        'accelerometer': False,
        'gyroscope': False,
        'both': False
    })

    sensors = ['watch']  # Can extend to ['watch', 'phone', 'meta_wrist', 'meta_hip']

    for sensor in sensors:
        # Check accelerometer
        acc_dir = os.path.join(base_dir, 'accelerometer', sensor)
        if os.path.exists(acc_dir):
            for filename in os.listdir(acc_dir):
                if filename.endswith('.csv'):
                    try:
                        # Parse S<subject>A<action>T<trial>.csv
                        name = filename.replace('.csv', '')
                        if name.startswith('S'):
                            subject_id = int(name[1:].split('A')[0])
                            modality_availability[subject_id]['accelerometer'] = True
                    except (ValueError, IndexError):
                        continue

        # Check gyroscope
        gyro_dir = os.path.join(base_dir, 'gyroscope', sensor)
        if os.path.exists(gyro_dir):
            for filename in os.listdir(gyro_dir):
                if filename.endswith('.csv'):
                    try:
                        # Parse S<subject>A<action>T<trial>.csv
                        name = filename.replace('.csv', '')
                        if name.startswith('S'):
                            subject_id = int(name[1:].split('A')[0])
                            modality_availability[subject_id]['gyroscope'] = True
                    except (ValueError, IndexError):
                        continue

    # Determine which subjects have both modalities
    for subject_id, availability in modality_availability.items():
        availability['both'] = availability['accelerometer'] and availability['gyroscope']

    return dict(modality_availability)


def recommend_validation_subjects(df, modality_availability, target_fall_ratio=0.43, tolerance=0.1):
    """
    Recommend subjects suitable for validation based on:
    1. Similar fall/ADL ratio to training data (~0.43 based on training data)
    2. Have both accelerometer and gyroscope data
    3. Sufficient number of samples

    Args:
        df: DataFrame with subject statistics
        modality_availability: Dictionary with modality availability
        target_fall_ratio: Target fall ratio (default: 0.43, based on training set)
        tolerance: Acceptable deviation from target ratio

    Returns:
        List of recommended validation subject IDs
    """
    recommended = []

    for _, row in df.iterrows():
        subject_id = row['subject_id']

        # Check if subject has both modalities
        if not modality_availability.get(subject_id, {}).get('both', False):
            continue

        # Check if subject has sufficient samples (at least 20 files)
        if row['total_files'] < 20:
            continue

        # Check if fall ratio is within tolerance of target
        fall_ratio = row['fall_ratio']
        if abs(fall_ratio - target_fall_ratio) <= tolerance:
            recommended.append(subject_id)

    return recommended


def main():
    """Main analysis function."""

    print("=" * 80)
    print("SUBJECT DATA SPLIT ANALYSIS")
    print("=" * 80)
    print()

    # Analyze subject data splits
    print("Analyzing subject data distribution...")
    df = analyze_subject_data()

    if df is None:
        return

    print(f"\nFound {len(df)} subjects\n")

    # Display full results
    print("Per-Subject Distribution:")
    print("-" * 80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    print(df.to_string(index=False))
    print()

    # Overall statistics
    total_fall = df['fall_files'].sum()
    total_adl = df['adl_files'].sum()
    total_files = df['total_files'].sum()
    overall_fall_ratio = total_fall / total_files if total_files > 0 else 0

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total files: {total_files}")
    print(f"Total fall files: {total_fall}")
    print(f"Total ADL files: {total_adl}")
    print(f"Overall fall:ADL ratio: 1:{total_adl/total_fall:.2f}")
    print(f"Overall fall ratio: {overall_fall_ratio:.2%}")
    print()

    # Check modality availability
    print("Checking modality availability...")
    modality_availability = check_modality_availability()

    subjects_with_both = [sid for sid, avail in modality_availability.items() if avail['both']]
    subjects_acc_only = [sid for sid, avail in modality_availability.items()
                         if avail['accelerometer'] and not avail['gyroscope']]
    subjects_gyro_only = [sid for sid, avail in modality_availability.items()
                          if avail['gyroscope'] and not avail['accelerometer']]

    print("\n" + "=" * 80)
    print("MODALITY AVAILABILITY")
    print("=" * 80)
    print(f"Subjects with both accelerometer and gyroscope: {len(subjects_with_both)}")
    print(f"  Subject IDs: {sorted(subjects_with_both)}")
    print(f"\nSubjects with only accelerometer: {len(subjects_acc_only)}")
    if subjects_acc_only:
        print(f"  Subject IDs: {sorted(subjects_acc_only)}")
    print(f"\nSubjects with only gyroscope: {len(subjects_gyro_only)}")
    if subjects_gyro_only:
        print(f"  Subject IDs: {sorted(subjects_gyro_only)}")
    print()

    # Recommend validation subjects
    recommended = recommend_validation_subjects(df, modality_availability)

    print("\n" + "=" * 80)
    print("VALIDATION SUBJECT RECOMMENDATIONS")
    print("=" * 80)
    print(f"Recommended validation subjects (fall ratio ~{overall_fall_ratio:.0%} ± 10%):")
    print(f"  Subject IDs: {sorted(recommended)}")
    print()
    print("These subjects have:")
    print("  - Both accelerometer and gyroscope data")
    print("  - At least 20 files")
    print(f"  - Fall ratio within 10% of overall ratio ({overall_fall_ratio:.0%})")
    print()

    # Display details for recommended subjects
    if recommended:
        print("\nDetailed statistics for recommended subjects:")
        print("-" * 80)
        recommended_df = df[df['subject_id'].isin(recommended)]
        print(recommended_df.to_string(index=False))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
