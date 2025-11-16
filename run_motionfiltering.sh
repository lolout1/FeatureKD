#!/bin/bash
# Motion Filtering Ablation Study: Low-Pass Filter Impact
# Scientific research-grade logging for publication
#
# Tests 4 models × 2 filtering conditions = 8 experiments
# Models: TransModel, IMUTransformer, DualStream Optimal, Madgwick Fusion
# Conditions: WITHOUT low-pass filtering, WITH low-pass filtering
#
# Low-pass parameters (research-based):
#   - Accelerometer: 5.5 Hz cutoff (human motion < 5 Hz)
#   - Gyroscope: 4.0 Hz cutoff (tighter for drift/noise)
#   - 4th-order Butterworth filter (25 Hz sampling)
#
# Expected runtime: 16-24 hours on NVIDIA A100 (LOSO cross-validation)

set -e  # Exit on error
set -o pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE=0
NUM_EPOCHS=80
TIMESTAMP=$(date +"%I%p-%m-%d-%Y" | tr '[:upper:]' '[:lower:]')
BASE_WORK_DIR="work_dir/motionfiltering_${TIMESTAMP}"

echo "========================================================================"
echo "MOTION FILTERING ABLATION STUDY - LOW-PASS FILTER IMPACT"
echo "========================================================================"
echo ""
echo "Study Design:"
echo "  - 4 Models × 2 Filtering Conditions = 8 Experiments"
echo "  - Each experiment: Leave-One-Subject-Out (LOSO) cross-validation"
echo "  - Total runtime: 16-24 hours on NVIDIA A100"
echo ""
echo "Models Tested:"
echo "  1. TransModel (Baseline) - Acc-only, 4ch"
echo "  2. IMUTransformer - Acc-only, 4ch, adaptive architecture"
echo "  3. Optimal Dual Stream - Acc+gyro, 8ch, asymmetric fusion"
echo "  4. Madgwick Fusion - Acc+gyro→orientation, 7ch"
echo ""
echo "Filtering Conditions:"
echo "  A. WITHOUT low-pass filter (baseline)"
echo "  B. WITH low-pass filter (research-based)"
echo "     - Accelerometer: 5.5 Hz cutoff"
echo "     - Gyroscope: 4.0 Hz cutoff"
echo "     - 4th-order Butterworth, 25 Hz sampling"
echo ""
echo "Common Optimizations (all experiments):"
echo "  ✓ Motion filtering (threshold=10.0, min_axes=2)"
echo "  ✓ Class-aware stride (ADL=10, Fall=32)"
echo "  ✓ DTW alignment for Models 3-4 (acc = ground truth)"
echo "  ✓ Validation: 40-50% ADLs"
echo ""
echo "========================================================================"
echo ""

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
            echo "  --work-dir <PATH>  Base work directory (default: work_dir/motionfiltering_TIMESTAMP)"
            echo "  --help             Show this help message"
            echo ""
            echo "Output:"
            echo "  - Individual model results: <work-dir>/<model>_<filtering>/"
            echo "  - Comprehensive CSV: <work-dir>/aggregated/scores_comprehensive.csv"
            echo "  - Summary report: <work-dir>/aggregated/motion_filtering_report.txt"
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
echo "  Epochs per experiment: $NUM_EPOCHS"
echo "  Base Work Directory: $BASE_WORK_DIR"
echo "  Timestamp: $TIMESTAMP"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure CUDA is available."
fi

# Create base work directory
mkdir -p "$BASE_WORK_DIR"

# Save configuration
cat > "${BASE_WORK_DIR}/experiment_config.txt" << EOF
MOTION FILTERING ABLATION STUDY CONFIGURATION
Generated: $(date)

Study Design:
  - Models: 4
  - Filtering conditions: 2 (without/with low-pass)
  - Total experiments: 8
  - Cross-validation: Leave-One-Subject-Out (LOSO)

Hardware:
  - Device: GPU $DEVICE
  - Expected runtime: 16-24 hours on NVIDIA A100

Training Parameters:
  - Epochs: $NUM_EPOCHS
  - Optimizer: AdamW
  - Learning rate: 1e-3
  - Weight decay: 1e-3
  - Batch size: 64

Low-Pass Filter Parameters (when enabled):
  - Accelerometer cutoff: 5.5 Hz
  - Gyroscope cutoff: 4.0 Hz
  - Filter type: 4th-order Butterworth
  - Sampling rate: 25 Hz

Motion Filtering (all experiments):
  - Threshold: 10.0 m/s²
  - Minimum axes: 2

Class-Aware Stride (all experiments):
  - Fall stride: 32 (~75% overlap)
  - ADL stride: 10 (~92% overlap, 3x more samples)

References:
  - Low-pass filtering: Smartphone fall detection literature (5 Hz cutoff)
  - Madgwick fusion: Zhang et al. (2024) - 97.13% accuracy
  - DTW alignment: FastDTW for multi-core efficiency
EOF

# ============================================================================
# EXPERIMENT DEFINITIONS
# ============================================================================

# Define experiments: config_file:experiment_name:filtering_enabled
declare -a EXPERIMENTS=(
    "transmodel_motionfilter.yaml:Model1_TransModel_NoFilter:false"
    "transmodel_motionfilter.yaml:Model1_TransModel_WithFilter:true"
    "imu_transformer_motionfilter.yaml:Model2_IMUTransformer_NoFilter:false"
    "imu_transformer_motionfilter.yaml:Model2_IMUTransformer_WithFilter:true"
    "dualstream_optimal_motionfilter.yaml:Model3_DualStream_NoFilter:false"
    "dualstream_optimal_motionfilter.yaml:Model3_DualStream_WithFilter:true"
    "madgwick_motionfilter.yaml:Model4_Madgwick_NoFilter:false"
    "madgwick_motionfilter.yaml:Model4_Madgwick_WithFilter:true"
)

echo "========================================================================"
echo "STARTING EXPERIMENTS (8 total)"
echo "========================================================================"
echo ""

# Run each experiment
for i in "${!EXPERIMENTS[@]}"; do
    EXPERIMENT="${EXPERIMENTS[$i]}"
    CONFIG_FILE=$(echo "$EXPERIMENT" | cut -d':' -f1)
    EXPERIMENT_NAME=$(echo "$EXPERIMENT" | cut -d':' -f2)
    FILTERING_ENABLED=$(echo "$EXPERIMENT" | cut -d':' -f3)

    WORK_DIR="${BASE_WORK_DIR}/${EXPERIMENT_NAME}"
    CONFIG_PATH="config/smartfallmm/${CONFIG_FILE}"

    # Determine filtering status for logging
    if [ "$FILTERING_ENABLED" = "true" ]; then
        FILTER_STATUS="WITH low-pass filter (Acc: 5.5 Hz, Gyro: 4.0 Hz)"
    else
        FILTER_STATUS="WITHOUT low-pass filter (raw data)"
    fi

    echo "------------------------------------------------------------------------"
    echo "[$((i+1))/8] ${EXPERIMENT_NAME}"
    echo "------------------------------------------------------------------------"
    echo "  Config: $CONFIG_PATH"
    echo "  Work Dir: $WORK_DIR"
    echo "  Filtering: $FILTER_STATUS"
    echo "  Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if config exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "  ✗ ERROR: Config file not found: $CONFIG_PATH"
        echo "  Skipping this experiment..."
        echo ""
        continue
    fi

    # Create temporary config with filtering enabled/disabled
    TEMP_CONFIG="${WORK_DIR}_temp.yaml"
    mkdir -p "$(dirname "$TEMP_CONFIG")"

    # Copy config and modify filtering setting
    cp "$CONFIG_PATH" "$TEMP_CONFIG"
    sed -i "s/enable_filtering: False/enable_filtering: ${FILTERING_ENABLED}/" "$TEMP_CONFIG"
    sed -i "s/enable_filtering: false/enable_filtering: ${FILTERING_ENABLED}/" "$TEMP_CONFIG"

    # Run training with LOSO cross-validation
    echo "  [$(date '+%H:%M:%S')] Starting training..."
    python main.py \
        --config "$TEMP_CONFIG" \
        --work-dir "$WORK_DIR" \
        --device "$DEVICE" \
        --num-epoch "$NUM_EPOCHS" \
        2>&1 | tee "${WORK_DIR}.log"

    EXIT_CODE=${PIPESTATUS[0]}

    # Save metadata
    cat > "${WORK_DIR}/experiment_metadata.txt" << EOF
Experiment: ${EXPERIMENT_NAME}
Config: ${CONFIG_FILE}
Filtering: ${FILTER_STATUS}
Start: $(head -1 "${WORK_DIR}.log" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' || echo "N/A")
Status: $([ $EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED (exit code: $EXIT_CODE)")
EOF

    # Clean up temp config
    rm -f "$TEMP_CONFIG"

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "  ✓ Completed successfully"
        echo "  End Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  Log: ${WORK_DIR}.log"
    else
        echo ""
        echo "  ✗ FAILED with exit code: $EXIT_CODE"
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
for i in "${!EXPERIMENTS[@]}"; do
    EXPERIMENT="${EXPERIMENTS[$i]}"
    EXPERIMENT_NAME=$(echo "$EXPERIMENT" | cut -d':' -f2)
    echo "  [$((i+1))/8] ${EXPERIMENT_NAME}"
    echo "       Work Dir: ${BASE_WORK_DIR}/${EXPERIMENT_NAME}/"
    echo "       Log: ${BASE_WORK_DIR}/${EXPERIMENT_NAME}.log"
done
echo ""

# ============================================================================
# RESULTS AGGREGATION
# ============================================================================

echo "========================================================================"
echo "AGGREGATING RESULTS"
echo "========================================================================"
echo ""
echo "Generating comprehensive analysis with fold-level details..."
echo ""

python aggregate_motionfiltering_results.py --work-dir "$BASE_WORK_DIR"

AGGREGATION_EXIT_CODE=$?

if [ $AGGREGATION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Aggregation completed successfully"
    echo ""
    echo "Results available at:"
    echo "  - Comprehensive scores: ${BASE_WORK_DIR}/aggregated/scores_comprehensive.csv"
    echo "  - Model comparison: ${BASE_WORK_DIR}/aggregated/model_comparison.csv"
    echo "  - Summary statistics: ${BASE_WORK_DIR}/aggregated/summary_statistics.csv"
    echo "  - Full report: ${BASE_WORK_DIR}/aggregated/motion_filtering_report.txt"
    echo ""
    echo "Quick view (model averages):"
    if [ -f "${BASE_WORK_DIR}/aggregated/model_comparison.csv" ]; then
        cat "${BASE_WORK_DIR}/aggregated/model_comparison.csv"
    fi
else
    echo ""
    echo "✗ Aggregation failed (exit code: $AGGREGATION_EXIT_CODE)"
    echo "You can manually run: python aggregate_motionfiltering_results.py --work-dir $BASE_WORK_DIR"
fi

echo ""
echo "========================================================================"
echo "STUDY COMPLETE - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""
echo "To view detailed results:"
echo "  cat ${BASE_WORK_DIR}/aggregated/motion_filtering_report.txt"
echo ""
echo "To view comprehensive scores (all folds):"
echo "  cat ${BASE_WORK_DIR}/aggregated/scores_comprehensive.csv"
echo ""
echo "========================================================================"
