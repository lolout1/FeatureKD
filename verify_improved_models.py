"""Quick sanity check for dual-stream parameter counts."""

import torch
import sys
sys.path.insert(0, '/mmfs1/home/sww35/FeatureKD')

from Models.imu_transformer import IMUTransformer
from Models.imu_dual_stream_shared import DualStreamSharedIMU
from Models.imu_dual_stream_light import DualStreamLightIMU
from Models.imu_dual_stream_asymmetric import DualStreamAsymmetricIMU


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


print("="*80)
print("BASELINE MODEL")
print("="*80)
baseline = IMUTransformer(
    imu_frames=128,
    imu_channels=6,
    num_classes=1,
    num_layers=2,
    embed_dim=32,
    num_heads=2,
    dropout=0.6
)
baseline_params, _ = count_parameters(baseline)
print(f"Parameters: {baseline_params:,}")

print("\n" + "="*80)
print("ORIGINAL VS IMPROVED DUAL-STREAM MODELS")
print("="*80)

# Shared V1 vs V2
print("\n1. SHARED DUAL-STREAM")
print("-" * 80)
shared_v1 = DualStreamSharedIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    num_layers=1, embed_dim=16, num_heads=2, dropout=0.65
)
shared_v1_params, _ = count_parameters(shared_v1)
print(f"V1: {shared_v1_params:,} ({shared_v1_params/baseline_params*100:.1f}% of baseline)")

shared_v2 = DualStreamSharedIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    num_layers=2, embed_dim=32, num_heads=4, dropout=0.6
)
shared_v2_params, _ = count_parameters(shared_v2)
print(f"V2: {shared_v2_params:,} ({shared_v2_params/baseline_params*100:.1f}% of baseline)")
print(f"Increase: {shared_v2_params/shared_v1_params:.1f}x parameters")

# Light V1 vs V2
print("\n2. LIGHT DUAL-STREAM")
print("-" * 80)
light_v1 = DualStreamLightIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    num_layers=1, stream_dim=8, num_heads=2, dropout=0.65
)
light_v1_params, _ = count_parameters(light_v1)
print(f"V1: {light_v1_params:,} ({light_v1_params/baseline_params*100:.1f}% of baseline)")

light_v2 = DualStreamLightIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    num_layers=2, stream_dim=16, num_heads=4, dropout=0.6
)
light_v2_params, _ = count_parameters(light_v2)
print(f"V2: {light_v2_params:,} ({light_v2_params/baseline_params*100:.1f}% of baseline)")
print(f"Increase: {light_v2_params/light_v1_params:.1f}x parameters")

# Asymmetric V1 vs V2
print("\n3. ASYMMETRIC DUAL-STREAM")
print("-" * 80)
asym_v1 = DualStreamAsymmetricIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    acc_layers=2, gyro_layers=1, acc_dim=16, gyro_dim=8,
    num_heads=2, dropout=0.6
)
asym_v1_params, _ = count_parameters(asym_v1)
print(f"V1: {asym_v1_params:,} ({asym_v1_params/baseline_params*100:.1f}% of baseline)")

asym_v2 = DualStreamAsymmetricIMU(
    imu_frames=128, imu_channels=6, num_classes=1,
    acc_layers=3, gyro_layers=2, acc_dim=32, gyro_dim=16,
    num_heads=4, dropout=0.6
)
asym_v2_params, _ = count_parameters(asym_v2)
print(f"V2: {asym_v2_params:,} ({asym_v2_params/baseline_params*100:.1f}% of baseline)")
print(f"Increase: {asym_v2_params/asym_v1_params:.1f}x parameters")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Baseline parameters: {baseline_params:,}")
print("Original model ratios:")
print(f"  Shared:      {shared_v1_params:,} ({shared_v1_params/baseline_params*100:>5.1f}% of baseline)")
print(f"  Light:       {light_v1_params:,} ({light_v1_params/baseline_params*100:>5.1f}% of baseline)")
print(f"  Asymmetric:  {asym_v1_params:,} ({asym_v1_params/baseline_params*100:>5.1f}% of baseline)")
print("Improved model ratios:")
print(f"  Shared V2:   {shared_v2_params:,} ({shared_v2_params/baseline_params*100:>5.1f}% of baseline)")
print(f"  Light V2:    {light_v2_params:,} ({light_v2_params/baseline_params*100:>5.1f}% of baseline)")
print(f"  Asym V2:     {asym_v2_params:,} ({asym_v2_params/baseline_params*100:>5.1f}% of baseline)")
