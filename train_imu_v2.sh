#!/bin/bash

# Enhanced trainer for the IMU student models.

set -euo pipefail

CONFIG_FILE="./config/smartfallmm/imu_student.yaml"
AUTO_TEST=true
DEVICE=0
VERBOSE=false
SKIP_TRAIN=false
WORK_DIR_SUFFIX=""
LOG_FILE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
NC='\033[0m'

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -c, --config FILE       Path to YAML config file (default: ./config/smartfallmm/imu_student.yaml)
    -t, --auto-test BOOL    Automatically run testing after training (default: true)
    -d, --device ID         GPU device ID (default: 0)
    -s, --skip-train        Skip training, only run testing (requires existing weights)
    -w, --work-dir-suffix   Suffix to append to work directory name
    -v, --verbose           Enable verbose logging
    -h, --help              Display this help message

Examples:
    # Train and test automatically
    $0 --config ./config/smartfallmm/imu_student.yaml --auto-test true

    # Train only, skip testing
    $0 --auto-test false

    # Test only with existing weights
    $0 --skip-train --work-dir-suffix "2025-11-10_18-34-14"

    # Verbose mode with custom device
    $0 --verbose --device 1

EOF
}

log_message() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - ${message}"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}"
            ;;
        *)
            echo "[${level}] ${timestamp} - ${message}"
            ;;
    esac

    if [ -n "$LOG_FILE" ]; then
        echo "[${level}] ${timestamp} - ${message}" >> "$LOG_FILE"
    fi
}

validate_config() {
    local config=$1

    log_message INFO "Validating configuration file: $config"

    if [ ! -f "$config" ]; then
        log_message ERROR "Config file not found: $config"
        return 1
    fi

    if ! python3 -c "import yaml,sys; yaml.safe_load(open('$config'))" 2>/dev/null; then
        log_message ERROR "Invalid YAML format in config file"
        return 1
    fi

    local required_fields=("model" "dataset" "model_args" "dataset_args")
    for field in "${required_fields[@]}"; do
        if ! python3 -c "import yaml,sys; config=yaml.safe_load(open('$config')); sys.exit(0 if '$field' in config else 1)" 2>/dev/null; then
            log_message ERROR "Missing required field in config: $field"
            return 1
        fi
    done

    log_message SUCCESS "Configuration validation passed"
    return 0
}

extract_config_value() {
    local config=$1
    local key=$2
    local default=$3

    python3 -c "import yaml,sys; config=yaml.safe_load(open('$config')); print(config.get('$key', '$default'))" 2>/dev/null || echo "$default"
}

print_header() {
    local title=$1
    local width=60

    echo ""
    echo "$(printf '=%.0s' $(seq 1 $width))"
    echo "$title"
    echo "$(printf '=%.0s' $(seq 1 $width))"
}

train_model() {
    local config=$1
    local work_dir=$2
    local weights=$3
    local device=$4

    print_header "TRAINING PHASE"

    log_message INFO "Starting training process..."
    log_message INFO "Configuration: $config"
    log_message INFO "Output directory: $work_dir"
    log_message INFO "Weight name: $weights"
    log_message INFO "Device: GPU $device"

    local train_start=$(date +%s)

    if $VERBOSE; then
        python3 main.py \
            --config "$config" \
            --work-dir "$work_dir" \
            --model-saved-name "$weights" \
            --device "$device" \
            --include-val True \
            --phase 'train'
    else
        python3 main.py \
            --config "$config" \
            --work-dir "$work_dir" \
            --model-saved-name "$weights" \
            --device "$device" \
            --include-val True \
            --phase 'train' 2>&1 | tee -a "$LOG_FILE"
    fi

    local train_status=$?
    local train_end=$(date +%s)
    local train_duration=$((train_end - train_start))

    if [ $train_status -eq 0 ]; then
        log_message SUCCESS "Training completed successfully!"
        log_message INFO "Training duration: $(printf '%02d:%02d:%02d' $((train_duration/3600)) $((train_duration%3600/60)) $((train_duration%60)))"
        log_message INFO "Model weights saved to: $work_dir/$weights"
        return 0
    else
        log_message ERROR "Training failed with exit code: $train_status"
        return 1
    fi
}

test_model() {
    local config=$1
    local work_dir=$2
    local weights=$3
    local device=$4

    print_header "TESTING PHASE"

    log_message INFO "Starting testing process..."

    if [ ! -d "$work_dir" ]; then
        log_message ERROR "Work directory not found: $work_dir"
        log_message ERROR "Please train the model first or specify correct work directory"
        return 1
    fi

    local test_start=$(date +%s)

    if $VERBOSE; then
        python3 main.py \
            --config "$config" \
            --work-dir "$work_dir" \
            --weights "$work_dir/$weights" \
            --device "$device" \
            --phase 'test'
    else
        python3 main.py \
            --config "$config" \
            --work-dir "$work_dir" \
            --weights "$work_dir/$weights" \
            --device "$device" \
            --phase 'test' 2>&1 | tee -a "$LOG_FILE"
    fi

    local test_status=$?
    local test_end=$(date +%s)
    local test_duration=$((test_end - test_start))

    if [ $test_status -eq 0 ]; then
        log_message SUCCESS "Testing completed successfully!"
        log_message INFO "Testing duration: $(printf '%02d:%02d:%02d' $((test_duration/3600)) $((test_duration%3600/60)) $((test_duration%60)))"
        log_message INFO "Results saved to: $work_dir"
        return 0
    else
        log_message ERROR "Testing failed with exit code: $test_status"
        return 1
    fi
}

generate_summary() {
    local work_dir=$1
    local sensor=$2

    print_header "EXPERIMENT SUMMARY"

    echo "Experiment Details:"
    echo "  - Model: IMUTransformer"
    echo "  - Sensor: $sensor"
    echo "  - Input: 6 channels (accelerometer + gyroscope)"
    echo "  - Age Group: Young participants"
    echo "  - Window Size: 128 samples"
    echo "  - Work Directory: $work_dir"
    echo ""

    if [ -f "$work_dir/scores.csv" ]; then
        log_message INFO "Performance Results:"
        echo ""

        # Display results using Python
        python3 << EOF
import pandas as pd
try:
    df = pd.read_csv('$work_dir/scores.csv', index_col=0)
    print(df.to_string())
    print("\n" + "="*60)

    # Get average row (last row)
    if len(df) > 0:
        avg_row = df.iloc[-1]
        print("\n${GREEN}Overall Performance:${NC}")
        print(f"  Accuracy:  {avg_row['accuracy']:.2f}%")
        print(f"  F1-Score:  {avg_row['f1_score']:.2f}%")
        print(f"  Precision: {avg_row['precision']:.2f}%")
        print(f"  Recall:    {avg_row['recall']:.2f}%")
        print(f"  AUC:       {avg_row['auc']:.2f}%")
except Exception as e:
    print(f"Error reading results: {e}")
EOF
    else
        log_message WARNING "Results file not found: $work_dir/scores.csv"
    fi

    echo ""
    echo "Configuration Tips:"
    echo "  - To reduce overfitting: increase dropout (currently 0.5)"
    echo "  - To reduce model size: decrease embed_dim (currently 32)"
    echo "  - To speed up training: reduce num_layers (currently 2)"
    echo ""
    echo "Available Sensors:"
    echo "  - watch (wrist-worn smartwatch)"
    echo "  - phone (smartphone in pocket)"
    echo "  - meta_wrist (Meta Quest wrist sensor)"
    echo "  - meta_hip (Meta Quest hip sensor)"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--auto-test)
            AUTO_TEST="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -s|--skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        -w|--work-dir-suffix)
            WORK_DIR_SUFFIX="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_message ERROR "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

SCRIPT_START=$(date +%s)
print_header "IMU FALL DETECTION - TRAINING PIPELINE"

log_message INFO "Script started at $(date '+%Y-%m-%d %H:%M:%S')"
log_message INFO "Configuration: $CONFIG_FILE"
log_message INFO "Auto-test: $AUTO_TEST"
log_message INFO "Device: GPU $DEVICE"
log_message INFO "Skip training: $SKIP_TRAIN"

if ! validate_config "$CONFIG_FILE"; then
    log_message ERROR "Configuration validation failed"
    exit 1
fi

SENSOR=$(python3 -c "import yaml,sys; config=yaml.safe_load(open('$CONFIG_FILE')); print(config.get('dataset_args', {}).get('sensors', ['watch'])[0])" 2>/dev/null || echo "watch")

WEIGHTS="imu_student_best"
if [ -n "$WORK_DIR_SUFFIX" ]; then
    WORK_DIR="exps/smartfall_imu/student/${SENSOR}_acc_gyro_6ch_young_${WORK_DIR_SUFFIX}"
else
    WORK_DIR="exps/smartfall_imu/student/${SENSOR}_acc_gyro_6ch_young"
fi

mkdir -p "$(dirname "$WORK_DIR")"
LOG_FILE="${WORK_DIR}_$(date '+%Y%m%d_%H%M%S').log"

log_message INFO "Sensor: $SENSOR"
log_message INFO "Modalities: Accelerometer + Gyroscope (6-channel IMU)"
log_message INFO "Log file: $LOG_FILE"

if ! $SKIP_TRAIN; then
    if ! train_model "$CONFIG_FILE" "$WORK_DIR" "$WEIGHTS" "$DEVICE"; then
        log_message ERROR "Pipeline failed during training"
        exit 1
    fi
else
    log_message WARNING "Skipping training phase (--skip-train enabled)"
fi

if [ "$AUTO_TEST" = "true" ] || $SKIP_TRAIN; then
    if ! test_model "$CONFIG_FILE" "$WORK_DIR" "$WEIGHTS" "$DEVICE"; then
        log_message ERROR "Pipeline failed during testing"
        exit 1
    fi
else
    log_message INFO "Skipping testing phase (--auto-test=false)"
fi

generate_summary "$WORK_DIR" "$SENSOR"

SCRIPT_END=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END - SCRIPT_START))

print_header "PIPELINE COMPLETED"
log_message SUCCESS "Total execution time: $(printf '%02d:%02d:%02d' $((TOTAL_DURATION/3600)) $((TOTAL_DURATION%3600/60)) $((TOTAL_DURATION%60)))"
log_message INFO "All logs saved to: $LOG_FILE"
log_message INFO "Results available in: $WORK_DIR"

exit 0
