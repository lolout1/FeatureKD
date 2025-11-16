#!/usr/bin/env python3
"""
Quick verification that validation split [38, 44] has the expected ADL ratio.
"""

import yaml
from utils.dataset import SmartFallMM, split_by_subjects
from utils.loader import DatasetBuilder
import os

def verify_split(config_path):
    """Verify validation split for a given config."""

    print(f"\nTesting: {config_path}")
    print("-" * 80)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_args = config.get('dataset_args', {})
    val_subjects = config.get('validation_subjects', [])

    print(f"Validation subjects: {val_subjects}")
    print(f"Modalities: {dataset_args.get('modalities', [])}")

    # Create dataset
    dataset_args_copy = dataset_args.copy()
    dataset_args_copy.pop('dataset', None)

    sm_dataset = SmartFallMM(root_dir=os.path.join(os.getcwd(), 'data'))
    sm_dataset.pipe_line(
        age_group=dataset_args['age_group'],
        modalities=dataset_args['modalities'],
        sensors=dataset_args['sensors']
    )

    # Create builder
    builder = DatasetBuilder(sm_dataset, **dataset_args_copy)

    # Process validation subjects
    val_data = split_by_subjects(builder, val_subjects, fuse=False, print_validation=False)

    # Calculate stats
    total_fall = 0
    total_adl = 0
    for subj in val_subjects:
        stats = builder.subject_modality_stats[str(subj)]
        total_fall += stats['fall_windows']
        total_adl += stats['adl_windows']

    total_windows = total_fall + total_adl
    adl_ratio = total_adl / total_windows if total_windows > 0 else 0

    print(f"\nResults:")
    print(f"  Total windows: {total_windows}")
    print(f"  Fall windows: {total_fall}")
    print(f"  ADL windows: {total_adl}")
    print(f"  ADL ratio: {adl_ratio:.1%}")

    # Check if within target (55-65%)
    if 0.55 <= adl_ratio <= 0.65:
        print(f"  ✓ PASS - Within target range (55-65%)")
        return True
    else:
        print(f"  ✗ FAIL - Outside target range (got {adl_ratio:.1%}, expected 55-65%)")
        return False


def main():
    """Test both acc-only and acc+gyro configs."""

    print("="*80)
    print("VALIDATION SPLIT VERIFICATION")
    print("="*80)

    configs_to_test = [
        ('Acc-only', 'config/smartfallmm/transformer.yaml'),
        ('Acc+Gyro', 'config/smartfallmm/imu_8channel.yaml'),
    ]

    results = {}
    for name, config_path in configs_to_test:
        try:
            results[name] = verify_split(config_path)
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results[name] = False

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15s}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Validation split is correctly configured!")
    else:
        print("✗ SOME TESTS FAILED - Check configuration")
    print("="*80)


if __name__ == "__main__":
    main()
