#!/bin/bash
# Motion Filtering Model Comparison
# Tests 4 models with motion filtering, low-pass filtering, and class-aware stride
#
# Models tested:
#   1. TransModel (Baseline) - acc-only, 4ch, motion filtered
#   2. IMUTransformer - acc-only, 4ch, motion filtered
#   3. Dual Stream Fusion - acc+gyro, 4ch each, separate encoders
#   4. Madgwick Fusion - acc+gyro → orientation, 7ch
#
# Optimizations applied:
#   - Low-pass filtering (5.5Hz cutoff) to reduce noise-based false positives
#   - Class-aware stride (stride=10 for ADLs, stride=32 for falls) to balance dataset
#   - Validation subjects with 40-50% ADLs for realistic ADL detection evaluation
#   - DTW alignment for acc+gyro models (acc as ground truth)
#   - Strong regularization to prevent overfitting on ~1000 samples
#
# Expected runtime: 8-12 hours on NVIDIA A100

set -e  # Exit on error
set -o pipefail

echo "========================================================================"
echo "MOTION FILTERING MODEL COMPARISON"
echo "========================================================================"
echo ""
echo "This script compares 4 models with motion filtering enabled:"
echo "  1. TransModel (Baseline) - Acc-only with motion filtering"
echo "  2. IMUTransformer - Acc-only with IMUTransformer architecture"
echo "  3. Dual Stream Fusion - Separate acc/gyro encoders, fused"
echo "  4. Madgwick Fusion - Sensor fusion to orientation angles"
echo ""
echo "Optimizations:"
echo "  - Low-pass filter (5.5Hz) to reduce false positives"
echo "  - Class-aware stride (ADL stride=10, Fall stride=32)"
echo "  - Validation: 40-50% ADLs (subjects 48, 57)"
echo "  - DTW alignment for models 3-4 (acc = ground truth)"
echo ""
echo "Each model uses Leave-One-Subject-Out (LOSO) cross-validation"
echo "Expected runtime: 8-12 hours on NVIDIA A100"
echo ""
echo "========================================================================"
echo ""

# Configuration
DEVICE=0
NUM_EPOCHS=80
TIMESTAMP=$(date +"%I%p-%m-%d-%Y" | tr '[:upper:]' '[:lower:]')
BASE_WORK_DIR="work_dir/motion_filtering_comparison_${TIMESTAMP}"

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
            echo "  --work-dir <PATH>  Base work directory (default: work_dir/motion_filtering_comparison_TIMESTAMP)"
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
    "transformer_motion_filtered.yaml:Model1_TransModel_AccOnly_MotionFiltered"
    "imu_acc_only_filtered.yaml:Model2_IMUTransformer_AccOnly_MotionFiltered"
    "imu_dualstream_motion_filtered.yaml:Model3_DualStream_AccGyro_MotionFiltered"
    "imu_madgwick_motion_filtered.yaml:Model4_Madgwick_Fusion_MotionFiltered"
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
    echo "[$((i+1))/4] Running: $CONFIG_NAME"
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
MOTION FILTERING MODEL COMPARISON - SUMMARY
======================================================================

Timestamp: $TIMESTAMP
Device: GPU $DEVICE
Epochs per model: $NUM_EPOCHS

Models tested (all with motion filtering):
  1. TransModel (Baseline) - Acc-only, 4ch, TransModel architecture
  2. IMUTransformer - Acc-only, 4ch, IMUTransformer architecture
  3. Dual Stream Fusion - Acc+gyro, 8ch (4+4), separate encoders
  4. Madgwick Fusion - Acc+gyro → orientation, 7ch

Optimizations applied to ALL models:
  ✓ Motion filtering (threshold=10.0, min_axes=2)
  ✓ Low-pass filtering (5.5Hz cutoff, 25Hz sampling)
  ✓ Class-aware stride (ADL=10, Fall=32)
  ✓ Validation subjects: 40-50% ADLs [48, 57]
  ✓ Strong regularization (dropout 0.5-0.7, weight_decay 1e-3)

Model-specific optimizations:
  Model 3 (Dual Stream):
    - DTW alignment: gyro aligned to acc (ground truth)
    - Ultra-lightweight: 1 layer, 8d per stream
    - Strong dropout: 0.7

  Model 4 (Madgwick):
    - DTW alignment before fusion
    - Reduced architecture: 2 layers, 64d embed (vs 3 layers, 96d)
    - Madgwick beta=0.1 (optimal for 30Hz)

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

1. Model Architecture Impact (Models 1 vs 2):
   - Both use acc-only, 4ch, motion filtering
   - TransModel uses standard transformer architecture
   - IMUTransformer uses channel-adaptive architecture
   - Comparison shows architecture impact independent of modality

2. Modality Impact (Models 2 vs 3 vs 4):
   - Model 2: Acc-only (4ch)
   - Model 3: Acc+Gyro separate encoders (8ch)
   - Model 4: Acc+Gyro fused to orientation (7ch)
   - Shows whether gyro adds value despite noise issues

3. Fusion Strategy Impact (Models 3 vs 4):
   - Model 3: Separate encoders, late fusion
   - Model 4: Sensor fusion (Madgwick), early fusion
   - Comparison of fusion strategies for noisy gyro data

4. Overfitting Mitigation:
   - All models use strong regularization
   - Models 3-4 optimized for small dataset (~1000 samples)
   - Val/test gap should be smaller with optimizations

5. ADL Detection (40-50% ADL validation set):
   - Class-aware stride should improve ADL recall
   - Low-pass filtering should reduce false positives
   - Validation set designed to test ADL detection capability

Comparison metrics:
  - F1 Score (primary): Balance of precision and recall
  - ADL Recall: Critical for motion filtering evaluation
  - Precision: False positive rate (falls detected on ADLs)
  - Val/Test Gap: Overfitting indicator
  - Per-fold variance: Generalization across subjects

Next steps:
  1. Run aggregation script: python aggregate_motion_filtering_results.py --work-dir $BASE_WORK_DIR
  2. Compare F1 scores across all 4 models
  3. Analyze ADL recall improvement from class-aware stride
  4. Check val/test gap reduction from regularization
  5. Compare fusion strategies (Models 3 vs 4)
  6. Evaluate low-pass filtering impact on false positives

======================================================================
EOF

echo "Summary report generated: $SUMMARY_FILE"
echo ""

# Run aggregation script
echo "========================================================================"
echo "AGGREGATING RESULTS"
echo "========================================================================"
echo ""
echo "Generating comprehensive CSV files with fold-level metrics..."
echo ""

python aggregate_motion_filtering_results.py --work-dir "$BASE_WORK_DIR"

AGGREGATION_EXIT_CODE=$?

if [ $AGGREGATION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "  ✓ Aggregation completed successfully"
    echo ""
    echo "Results available at:"
    echo "  - Comprehensive fold results: ${BASE_WORK_DIR}/aggregated/all_folds_comprehensive.csv"
    echo "  - Summary statistics: ${BASE_WORK_DIR}/aggregated/summary_statistics.csv"
    echo "  - Model comparison: ${BASE_WORK_DIR}/aggregated/model_comparison.csv"
    echo "  - Full report: ${BASE_WORK_DIR}/aggregated/motion_filtering_report.txt"
else
    echo ""
    echo "  ✗ Aggregation failed (exit code: $AGGREGATION_EXIT_CODE)"
    echo "  You can manually run: python aggregate_motion_filtering_results.py --work-dir $BASE_WORK_DIR"
fi

echo ""
echo "========================================================================"
echo "To view summary:"
echo "  cat ${BASE_WORK_DIR}/aggregated/motion_filtering_report.txt"
echo ""
echo "To view comparison table:"
echo "  cat ${BASE_WORK_DIR}/aggregated/model_comparison.csv"
echo ""
echo "========================================================================"
