#!/bin/bash
# Filtering and Quality Assessment Experiments
# Tests 7 configurations: baseline + motion filtering + 5 filtering/fusion approaches
#
# Configurations tested:
#   1. Baseline: TransModel (acc-only, 4ch, no filtering)
#   2. TransModel + Motion Filter (4ch with Android app logic, TransModel architecture)
#   3. IMU Acc-only + Motion Filter (4ch with Android app logic, IMUTransformer)
#   4. IMU Raw Acc+Gyro (6ch, no quality filter)
#   5. IMU Acc+Gyro + Hard Quality Filter (6ch, SNR threshold)
#   6. IMU Acc+Gyro + Adaptive Fallback (6ch→4ch based on quality)
#   7. IMU Acc+Orientation + Madgwick Fusion (7ch)

set -e  # Exit on error
set -o pipefail

echo "========================================================================"
echo "FILTERING AND QUALITY ASSESSMENT EXPERIMENTS"
echo "========================================================================"
echo ""
echo "This script will run 7 configurations with LOSO cross-validation:"
echo "  1. Baseline: TransModel (acc-only, 4ch, no filtering)"
echo "  2. TransModel + Motion Filter (4ch with Android app logic, TransModel)"
echo "  3. IMU Acc-only + Motion Filter (4ch with Android app logic, IMUTransformer)"
echo "  4. IMU Raw Acc+Gyro (6ch, no quality filter)"
echo "  5. IMU Acc+Gyro + Hard Quality Filter (6ch, SNR threshold)"
echo "  6. IMU Acc+Gyro + Adaptive Fallback (6ch→4ch based on quality)"
echo "  7. IMU Acc+Orientation + Madgwick Fusion (7ch)"
echo ""
echo "Each config uses Leave-One-Subject-Out (LOSO) cross-validation"
echo "Expected runtime: 14-21 hours on NVIDIA A100"
echo ""
echo "========================================================================"
echo ""

# Configuration
DEVICE=0
NUM_EPOCHS=80
TIMESTAMP=$(date +"%I%p-%m-%d-%Y" | tr '[:upper:]' '[:lower:]')
BASE_WORK_DIR="work_dir/filtering_${TIMESTAMP}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --work-dir)
            BASE_WORK_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device <N>       GPU device ID (default: 0)"
            echo "  --epochs <N>       Number of epochs (default: 80)"
            echo "  --work-dir <PATH>  Base work directory (default: work_dir/filtering_TIMESTAMP)"
            echo "  --help             Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  GPU Device: $DEVICE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Base Work Directory: $BASE_WORK_DIR"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure CUDA is available."
fi

# Create base work directory
mkdir -p "$BASE_WORK_DIR"

# Define configurations
declare -a CONFIGS=(
    "transformer.yaml:TransModel_Baseline"
    "transformer_motion_filtered.yaml:TransModel_MotionFilter"
    "imu_acc_only_filtered.yaml:IMU_AccOnly_MotionFilter"
    "imu_acc_gyro_raw.yaml:IMU_AccGyro_Raw"
    "imu_acc_gyro_quality_hard.yaml:IMU_AccGyro_HardFilter"
    "imu_acc_gyro_quality_adaptive.yaml:IMU_AccGyro_Adaptive"
    "imu_madgwick_fusion.yaml:IMU_Madgwick_Fusion"
)

echo "========================================================================"
echo "Starting experiments..."
echo "========================================================================"
echo ""

# Run each configuration
for i in "${!CONFIGS[@]}"; do
    CONFIG_ENTRY="${CONFIGS[$i]}"
    CONFIG_FILE=$(echo "$CONFIG_ENTRY" | cut -d':' -f1)
    CONFIG_NAME=$(echo "$CONFIG_ENTRY" | cut -d':' -f2)

    WORK_DIR="${BASE_WORK_DIR}/${CONFIG_NAME}"
    CONFIG_PATH="config/smartfallmm/${CONFIG_FILE}"

    echo "------------------------------------------------------------------------"
    echo "[$((i+1))/7] Running: $CONFIG_NAME"
    echo "------------------------------------------------------------------------"
    echo "  Config: $CONFIG_PATH"
    echo "  Work Dir: $WORK_DIR"
    echo "  Start Time: $(date)"
    echo ""

    # Check if config exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $CONFIG_PATH"
        echo "Skipping this experiment..."
        echo ""
        continue
    fi

    # Run training with LOSO cross-validation
    python main.py \
        --config "$CONFIG_PATH" \
        --work-dir "$WORK_DIR" \
        --device "$DEVICE" \
        --num-epoch "$NUM_EPOCHS" \
        2>&1 | tee "${WORK_DIR}.log"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "  ✓ Completed successfully"
        echo "  End Time: $(date)"
    else
        echo ""
        echo "  ✗ Failed with exit code: $EXIT_CODE"
        echo "  Check log: ${WORK_DIR}.log"
    fi

    echo ""
done

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo ""
echo "Results directory: $BASE_WORK_DIR"
echo ""
echo "Individual experiment results:"
for i in "${!CONFIGS[@]}"; do
    CONFIG_ENTRY="${CONFIGS[$i]}"
    CONFIG_NAME=$(echo "$CONFIG_ENTRY" | cut -d':' -f2)
    echo "  [$((i+1))] ${CONFIG_NAME}:"
    echo "      - Work Dir: ${BASE_WORK_DIR}/${CONFIG_NAME}/"
    echo "      - Log: ${BASE_WORK_DIR}/${CONFIG_NAME}.log"
done
echo ""

# Generate summary report
SUMMARY_FILE="${BASE_WORK_DIR}/experiment_summary.txt"
echo "Generating summary report: $SUMMARY_FILE"
echo ""

cat > "$SUMMARY_FILE" << EOF
======================================================================
FILTERING AND QUALITY ASSESSMENT EXPERIMENTS - SUMMARY
======================================================================

Timestamp: $TIMESTAMP
Device: GPU $DEVICE
Epochs per config: $NUM_EPOCHS

Configurations tested:
  1. TransModel Baseline (acc-only, 4ch, no filtering)
  2. TransModel + Motion Filter (acc-only, 4ch, Android app logic, TransModel)
  3. IMU Acc-only + Motion Filter (4ch, Android app logic, IMUTransformer)
  4. IMU Raw Acc+Gyro (6ch, no quality filter)
  5. IMU Acc+Gyro + Hard Quality Filter (6ch, SNR ≥ 1.0)
  6. IMU Acc+Gyro + Adaptive Fallback (6ch→4ch based on quality)
  7. IMU Acc+Orientation + Madgwick Fusion (7ch)

Results:
------------------------------------------------------------------------

EOF

# Append individual results to summary
for i in "${!CONFIGS[@]}"; do
    CONFIG_ENTRY="${CONFIGS[$i]}"
    CONFIG_NAME=$(echo "$CONFIG_ENTRY" | cut -d':' -f2)
    WORK_DIR="${BASE_WORK_DIR}/${CONFIG_NAME}"

    echo "[$((i+1))] ${CONFIG_NAME}" >> "$SUMMARY_FILE"
    echo "    Status: $([ -d "$WORK_DIR" ] && echo "Completed" || echo "Failed/Skipped")" >> "$SUMMARY_FILE"
    echo "    Work Dir: ${WORK_DIR}/" >> "$SUMMARY_FILE"
    echo "    Log: ${WORK_DIR}.log" >> "$SUMMARY_FILE"

    # Try to extract final metrics if available
    if [ -f "${WORK_DIR}/results.txt" ]; then
        echo "    Metrics:" >> "$SUMMARY_FILE"
        tail -20 "${WORK_DIR}/results.txt" | grep -E "(F1|Accuracy|Precision|Recall)" >> "$SUMMARY_FILE" 2>/dev/null || echo "    (Metrics parsing failed)" >> "$SUMMARY_FILE"
    fi
    echo "" >> "$SUMMARY_FILE"
done

cat >> "$SUMMARY_FILE" << EOF
======================================================================
ANALYSIS NOTES
======================================================================

Expected findings:

1. Motion filtering comparison (Configs 2 vs 3):
   - Config 2 (TransModel + Motion): Tests motion filtering with standard architecture
   - Config 3 (IMU + Motion): Tests motion filtering with IMUTransformer architecture
   - Both should improve precision by reducing false positives on quiet periods
   - Direct comparison to assess architecture impact vs. filtering impact

2. Raw acc+gyro (Config 4) likely to underperform baseline due to
   poor gyroscope quality (SNR < 1.0 for 76% of Group 2 subjects).

3. Hard quality filter (Config 5) trades sample size for quality:
   - Retains only ~35% of trials (SNR ≥ 1.0)
   - May improve if quality > quantity
   - Risk of overfitting due to reduced data

4. Adaptive fallback (Config 6) preserves all samples:
   - Uses acc+gyro (6ch) when gyro quality is good
   - Falls back to acc-only (4ch) when gyro is noisy
   - Should outperform both hard filter and raw baseline

5. Madgwick fusion (Config 7) transforms gyro to orientation:
   - 7 channels: [smv, ax, ay, az, roll, pitch, yaw]
   - Corrects gyroscope drift via accelerometer reference
   - Zhang et al. (2024) achieved 97.13% with Madgwick+ResNet
   - Deeper architecture (3 layers) for orientation learning

Comparison metrics:
  - F1 Score (primary): Balance of precision and recall
  - Precision: False positive rate (critical for fall detection)
  - Recall: False negative rate (must detect actual falls)
  - Per-fold variance: Generalization across subjects

Next steps:
  1. Compare F1 scores across all 7 configs
  2. Analyze per-fold results for stability
  3. Compare motion filtering impact (Configs 2 vs 3) on precision/recall
  4. Evaluate architecture impact (TransModel vs IMUTransformer) with motion filtering
  5. Check if Madgwick fusion outperforms raw gyro
  6. Determine if quality filtering helps or hurts overall performance

======================================================================
EOF

echo "Summary report generated: $SUMMARY_FILE"
echo ""

# Run aggregation script to generate comprehensive CSV files
echo "========================================================================"
echo "AGGREGATING RESULTS"
echo "========================================================================"
echo ""
echo "Generating comprehensive CSV files with fold-level metrics..."
echo ""

python aggregate_filtering_results.py --work-dir "$BASE_WORK_DIR"

AGGREGATION_EXIT_CODE=$?

if [ $AGGREGATION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "  ✓ Aggregation completed successfully"
    echo ""
    echo "Results available at:"
    echo "  - Comprehensive fold results: ${BASE_WORK_DIR}/aggregated/all_folds_comprehensive.csv"
    echo "  - Summary statistics: ${BASE_WORK_DIR}/aggregated/summary_statistics.csv"
    echo "  - Model comparison: ${BASE_WORK_DIR}/aggregated/model_comparison.csv"
    echo "  - Full report: ${BASE_WORK_DIR}/aggregated/filtering_experiments_report.txt"
else
    echo ""
    echo "  ✗ Aggregation failed (exit code: $AGGREGATION_EXIT_CODE)"
    echo "  You can manually run: python aggregate_filtering_results.py --work-dir $BASE_WORK_DIR"
fi

echo ""
echo "========================================================================"
echo "To view summary:"
echo "  cat ${BASE_WORK_DIR}/aggregated/filtering_experiments_report.txt"
echo ""
echo "To view comparison table:"
echo "  cat ${BASE_WORK_DIR}/aggregated/model_comparison.csv"
echo ""
echo "========================================================================"
