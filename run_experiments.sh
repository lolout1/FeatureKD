#!/bin/bash
# Quick Start Script for Dual-Stream Architecture Comparison
# Runs all 4 models (baseline + 3 dual-stream) with LOSO cross-validation on A100

set -e  # Exit on error

echo "========================================================================"
echo "DUAL-STREAM ARCHITECTURE COMPARISON - LOSO CROSS-VALIDATION"
echo "========================================================================"
echo ""
echo "This script will run 4 models with Leave-One-Subject-Out validation:"
echo "  1. Baseline (IMUTransformer) - ~110K params"
echo "  2. Shared-Weight Dual Stream - ~15K params"
echo "  3. Lightweight Independent Streams - ~25K params"
echo "  4. Asymmetric Dual Stream - ~35K params"
echo ""
echo "Each model will be trained on 29 subjects (1 for test, 1 for val, 27 for train)"
echo "Expected runtime: 12-18 hours on NVIDIA A100"
echo ""
echo "========================================================================"
echo ""

# Configuration
DEVICE=0
NUM_EPOCHS=80
BATCH_SIZE=64
WORK_DIR="work_dir/dualstream_comparison"
ENABLE_NORMALIZATION="false"
ENABLE_FILTERING="false"
FILTER_CUTOFF=5.5
FILTER_FS=25
SKIP_BASELINE=0

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
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --enable-normalization)
            ENABLE_NORMALIZATION="$2"
            shift 2
            ;;
        --enable-filtering)
            ENABLE_FILTERING="$2"
            shift 2
            ;;
        --filter-cutoff)
            FILTER_CUTOFF="$2"
            shift 2
            ;;
        --filter-fs)
            FILTER_FS="$2"
            shift 2
            ;;
        --skip-baseline)
            SKIP_BASELINE=1
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --device <N>                GPU device ID (default: 0)"
            echo "  --epochs <N>                Number of epochs (default: 80)"
            echo "  --batch-size <N>            Batch size (default: 64)"
            echo "  --work-dir <PATH>           Work directory (default: work_dir/dualstream_comparison)"
            echo "  --enable-normalization <BOOL>  Enable per-window normalization (default: true)"
            echo "  --enable-filtering <BOOL>   Enable Butterworth filter (default: true)"
            echo "  --filter-cutoff <FLOAT>     Filter cutoff frequency in Hz (default: 5.5)"
            echo "  --filter-fs <FLOAT>         Filter sampling rate in Hz (default: 25)"
            echo "  --models <LIST>             Space-separated models to run: baseline shared light asymmetric all (default: all)"
            echo "  --skip-baseline             Skip baseline model (alternative to --models)"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --enable-normalization true --enable-filtering true"
            echo "  $0 --models baseline shared --epochs 100"
            echo "  $0 --filter-cutoff 6.0 --filter-fs 30"
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
echo "  Batch Size: $BATCH_SIZE"
echo "  Work Directory: $WORK_DIR"
echo "  Preprocessing:"
echo "    - Normalization: $ENABLE_NORMALIZATION"
echo "    - Butterworth Filter: $ENABLE_FILTERING"
echo "    - Filter Cutoff: $FILTER_CUTOFF Hz"
echo "    - Filter Sampling Rate: $FILTER_FS Hz"
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure CUDA is available."
fi

echo "========================================================================"
echo "Starting experiments..."
echo "========================================================================"
echo ""

# Build the command with all arguments
CMD="python run_dualstream_comparison.py \
    --device $DEVICE \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --work-dir $WORK_DIR \
    --enable-normalization $ENABLE_NORMALIZATION \
    --enable-filtering $ENABLE_FILTERING \
    --filter-cutoff $FILTER_CUTOFF \
    --filter-fs $FILTER_FS"

# Add optional flags
if [ $SKIP_BASELINE -eq 1 ]; then
    CMD="$CMD --skip-baseline"
fi

if [ ! -z "$MODELS" ]; then
    CMD="$CMD --models $MODELS"
fi

# Run the comparison script
eval $CMD

echo ""
echo "========================================================================"
echo "EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo ""
echo "Results saved in: $WORK_DIR"
echo ""
echo "To analyze results, run:"
echo "  python analyze_dualstream_results.py --work-dir <work_dir_timestamp>"
echo ""
echo "========================================================================"
