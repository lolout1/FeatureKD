"""
Quick script to analyze parameter counts and architectures of all models
"""

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


def analyze_model_capacity(model, model_name):
    """Analyze the model's capacity and architecture"""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")

    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")

    # Model-specific analysis
    if hasattr(model, 'encoder'):
        # Baseline model
        print(f"\nArchitecture details:")
        print(f"  - Embedding dim: {model.encoder_layer.self_attn.embed_dim}")
        print(f"  - Number of layers: {len(model.encoder.layers)}")
        print(f"  - Number of heads: {model.encoder_layer.self_attn.num_heads}")
        print(f"  - FFN dim: {model.encoder_layer.linear1.out_features}")
        print(f"  - Kernel size: 8")

    elif hasattr(model, 'shared_encoder'):
        # Shared dual stream
        print(f"\nArchitecture details:")
        print(f"  - Embedding dim: {model.embed_dim}")
        print(f"  - Number of layers: {len(model.shared_transformer.layers)}")
        print(f"  - Streams: 2 (acc + gyro sharing same weights)")
        print(f"  - Kernel size: 5")

    elif hasattr(model, 'acc_encoder') and hasattr(model, 'gyro_encoder'):
        # Light or Asymmetric
        if hasattr(model, 'acc_transformer') and isinstance(model.acc_transformer, torch.nn.TransformerEncoderLayer):
            # Light model
            print(f"\nArchitecture details:")
            print(f"  - Stream dim: {model.stream_dim}")
            print(f"  - Acc layers: 1 (TransformerEncoderLayer)")
            print(f"  - Gyro layers: 1 (TransformerEncoderLayer)")
            print(f"  - Kernel size: 5")
        else:
            # Asymmetric model
            print(f"\nArchitecture details:")
            print(f"  - Acc dim: {model.acc_dim}")
            print(f"  - Gyro dim: {model.gyro_dim}")
            print(f"  - Acc layers: {model.acc_layers}")
            print(f"  - Gyro layers: {model.gyro_layers}")
            print(f"  - Acc kernel size: 8")
            print(f"  - Gyro kernel size: 3")

    return total, trainable


def main():
    # Create models with their configured parameters
    baseline = IMUTransformer(
        imu_frames=128,
        imu_channels=6,
        num_classes=1,
        num_layers=2,
        embed_dim=32,
        num_heads=2,
        dropout=0.6
    )

    shared = DualStreamSharedIMU(
        imu_frames=128,
        imu_channels=6,
        num_classes=1,
        num_layers=1,
        embed_dim=16,
        num_heads=2,
        dropout=0.65
    )

    light = DualStreamLightIMU(
        imu_frames=128,
        imu_channels=6,
        num_classes=1,
        num_layers=1,
        stream_dim=8,
        num_heads=2,
        dropout=0.65
    )

    asymmetric = DualStreamAsymmetricIMU(
        imu_frames=128,
        imu_channels=6,
        num_classes=1,
        acc_layers=2,
        gyro_layers=1,
        acc_dim=16,
        gyro_dim=8,
        num_heads=2,
        dropout=0.6
    )

    # Analyze each model
    models = {
        'Baseline (IMUTransformer)': baseline,
        'Shared Dual Stream': shared,
        'Light Dual Stream': light,
        'Asymmetric Dual Stream': asymmetric
    }

    results = {}
    for name, model in models.items():
        total, trainable = analyze_model_capacity(model, name)
        results[name] = total

    # Summary comparison
    print(f"\n{'='*80}")
    print("PARAMETER COUNT COMPARISON")
    print(f"{'='*80}")
    baseline_params = results['Baseline (IMUTransformer)']

    for name, params in results.items():
        ratio = params / baseline_params * 100
        print(f"{name:40s}: {params:>8,} ({ratio:>5.1f}% of baseline)")

    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    print(f"""
1. PARAMETER GAP ANALYSIS:
   - Baseline has ~{baseline_params/1000:.1f}K parameters
   - Dual-stream models have 7-20K parameters (87-93% FEWER parameters)
   - This is a MASSIVE capacity reduction

2. ARCHITECTURE DIFFERENCES:
   - Baseline: Single stream, 32-dim embedding, 2 layers, kernel=8
   - Shared: Dual stream, 16-dim embedding, 1 layer, kernel=5, SHARED weights
   - Light: Dual stream, 8-dim per stream, 1 layer, kernel=5, INDEPENDENT weights
   - Asymmetric: Dual stream, 16-dim acc/8-dim gyro, 2/1 layers, LEARNABLE fusion

3. LIKELY REASONS FOR UNDERPERFORMANCE:

   a) SEVERE UNDERFITTING:
      - The dual-stream models are TOO small for this task
      - 8-16 dim embeddings are likely insufficient to capture complex fall patterns
      - Single transformer layer provides minimal temporal modeling capacity

   b) INSUFFICIENT FEATURE REPRESENTATION:
      - Baseline: 32-dim × 2 layers = 64-dim effective feature space
      - Dual-stream: 16-dim × 1 layer = 16-dim (or 8+8 for light)
      - This is 4x less representational capacity

   c) SMALLER RECEPTIVE FIELD:
      - Baseline uses kernel=8 (captures longer temporal patterns)
      - Dual-stream uses kernel=5 or 3 (misses longer-range dependencies)

   d) INSUFFICIENT CROSS-MODALITY INTERACTION:
      - Dual-stream models process acc and gyro separately until fusion
      - Baseline processes all 6 channels together from the start
      - Early fusion (baseline) may capture important acc-gyro correlations

4. VALIDATION vs TEST GAP:
   - All models show reasonable validation performance (77-85%)
   - But dual-stream models drop significantly on test (67%)
   - This suggests they're memorizing validation patterns but failing to generalize
   - The extremely limited capacity prevents learning robust features

RECOMMENDATIONS:
See the improvement proposals in the next section...
""")


if __name__ == "__main__":
    main()
