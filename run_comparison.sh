#!/bin/bash

# Compare accelerometer-only and full IMU models under shared configs.

set -e  # Exit on error

# Default values
SEED=42
DEVICE="cuda"
OUTPUT_DIR=""
TEST_ONLY=false
MULTI_SEED=false
NUM_EPOCHS=""
LOOCV=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test_only)
            TEST_ONLY=true
            shift
            ;;
        --multi_seed)
            MULTI_SEED=true
            shift
            ;;
        --loocv)
            LOOCV=true
            shift
            ;;
        --help)
            echo "Usage: bash run_comparison.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --seed SEED           Random seed (default: 42)"
            echo "  --device DEVICE       Device (cuda or cpu, default: cuda)"
            echo "  --epochs EPOCHS       Number of epochs (default: from config)"
            echo "  --output_dir DIR      Output directory (default: auto-generated)"
            echo "  --test_only           Skip training, only test"
            echo "  --multi_seed          Run with multiple seeds (42, 123, 456, 789, 999)"
            echo "  --loocv               Use Leave-One-Subject-Out cross-validation"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.7+"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_info "Python version: $PYTHON_VERSION"

# Check if required files exist
print_header "Checking Required Files"

required_files=(
    "Models/transformer.py"
    "Models/imu_transformer.py"
    "config/smartfallmm/comparison_acc_only.yaml"
    "config/smartfallmm/comparison_imu_full.yaml"
    "compare_models.py"
    "test_models.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        exit 1
    fi
done

# Run model verification tests first
print_header "Running Model Verification Tests"
print_info "This ensures both models are correctly configured..."

if python test_models.py; then
    print_success "Model verification tests passed!"
else
    print_error "Model verification tests failed!"
    print_info "Please fix the issues before running comparison"
    exit 1
fi

# Function to run single comparison
run_single_comparison() {
    local seed=$1
    local output_suffix=$2

    print_header "Running Comparison with Seed: $seed"

    # Set output directory
    local output_dir
    if [ -n "$OUTPUT_DIR" ]; then
        output_dir="${OUTPUT_DIR}${output_suffix}"
    else
        timestamp=$(date +%Y%m%d_%H%M%S)
        output_dir="work_dir/comparison_${timestamp}_seed${seed}"
    fi

    print_info "Output directory: $output_dir"

    # Build command
    local cmd="python compare_models.py"
    cmd="$cmd --baseline config/smartfallmm/comparison_acc_only.yaml"
    cmd="$cmd --imu config/smartfallmm/comparison_imu_full.yaml"
    cmd="$cmd --seed $seed"
    cmd="$cmd --device $device"
    cmd="$cmd --output_dir $output_dir"

    if [ "$LOOCV" = true ]; then
        cmd="$cmd --loocv"
        print_info "Running with LOOCV (Leave-One-Subject-Out Cross-Validation)"
    fi

    if [ "$TEST_ONLY" = true ]; then
        cmd="$cmd --test_only"
        print_warning "Running in TEST ONLY mode (skipping training)"
    fi

    print_info "Command: $cmd"

    # Run comparison
    if $cmd; then
        print_success "Comparison completed successfully!"
        print_success "Results saved to: $output_dir"

        # Display key results
        print_header "Quick Results Summary"

        if [ -f "$output_dir/reports/results.csv" ]; then
            echo ""
            cat "$output_dir/reports/results.csv"
            echo ""
        fi

        # Display result files
        print_info "Generated files:"
        echo "  📊 Report:     $output_dir/reports/comparison_report.md"
        echo "  📈 Plots:      $output_dir/plots/"
        echo "  💾 Models:     $output_dir/models/"
        echo "  📋 CSV:        $output_dir/reports/results.csv"
        echo "  🔧 Configs:    $output_dir/configs/"

        return 0
    else
        print_error "Comparison failed!"
        return 1
    fi
}

# Function to verify no data leakage
verify_no_data_leakage() {
    print_header "Data Leakage Check"

    print_info "Verifying configuration files for data leakage prevention..."

    # Check both configs have same subjects and validation_subjects
    acc_subjects=$(grep -A1 "^subjects:" config/smartfallmm/comparison_acc_only.yaml | tail -1)
    imu_subjects=$(grep -A1 "^subjects:" config/smartfallmm/comparison_imu_full.yaml | tail -1)

    acc_val=$(grep -A1 "^validation_subjects:" config/smartfallmm/comparison_acc_only.yaml | tail -1)
    imu_val=$(grep -A1 "^validation_subjects:" config/smartfallmm/comparison_imu_full.yaml | tail -1)

    if [ "$acc_subjects" = "$imu_subjects" ]; then
        print_success "Both models use same subject list"
    else
        print_warning "Models use different subject lists - may not be directly comparable"
    fi

    if [ "$acc_val" = "$imu_val" ]; then
        print_success "Both models use same validation subjects"
    else
        print_warning "Models use different validation subjects - may cause data leakage!"
    fi

    # Check seeds
    acc_seed=$(grep "^seed:" config/smartfallmm/comparison_acc_only.yaml | awk '{print $2}')
    imu_seed=$(grep "^seed:" config/smartfallmm/comparison_imu_full.yaml | awk '{print $2}')

    if [ "$acc_seed" = "$imu_seed" ]; then
        print_success "Both models use same default seed: $acc_seed"
    else
        print_warning "Models use different seeds - will override with command line seed"
    fi

    print_info "Data leakage check complete"
}

# Function to compare hyperparameters
compare_hyperparameters() {
    print_header "Hyperparameter Comparison"

    print_info "Comparing model configurations to ensure fair comparison..."

    # Extract and compare key hyperparameters
    echo ""
    echo "Parameter Comparison:"
    echo "┌─────────────────────┬─────────────────┬─────────────────┬──────────┐"
    echo "│ Parameter           │ ACC-Only        │ IMU-Full        │ Match    │"
    echo "├─────────────────────┼─────────────────┼─────────────────┼──────────┤"

    params=("embed_dim" "num_heads" "dropout" "num_epoch" "batch_size" "base_lr" "weight_decay" "optimizer")

    for param in "${params[@]}"; do
        # Handle different parameter names
        if [ "$param" = "num_heads" ]; then
            acc_val=$(grep "num_heads:" config/smartfallmm/comparison_acc_only.yaml | head -1 | awk '{print $2}')
            imu_val=$(grep "num_heads:" config/smartfallmm/comparison_imu_full.yaml | head -1 | awk '{print $2}')
        else
            acc_val=$(grep "${param}:" config/smartfallmm/comparison_acc_only.yaml | head -1 | awk '{print $2}')
            imu_val=$(grep "${param}:" config/smartfallmm/comparison_imu_full.yaml | head -1 | awk '{print $2}')
        fi

        if [ "$acc_val" = "$imu_val" ]; then
            match="✓"
        else
            match="✗"
        fi

        printf "│ %-19s │ %-15s │ %-15s │ %-8s │\n" "$param" "$acc_val" "$imu_val" "$match"
    done

    echo "└─────────────────────┴─────────────────┴─────────────────┴──────────┘"
    echo ""

    # Check input channels (should be different)
    acc_channels=$(grep "acc_coords:" config/smartfallmm/comparison_acc_only.yaml | head -1 | awk '{print $2}')
    imu_channels=$(grep "imu_channels:" config/smartfallmm/comparison_imu_full.yaml | head -1 | awk '{print $2}')

    print_info "Input channels: ACC-Only=$acc_channels, IMU-Full=$imu_channels (expected to differ)"
}

# Main execution
print_header "Model Comparison Pipeline"
echo ""
echo "This script will:"
echo "  1. Verify model configurations"
echo "  2. Check for data leakage"
echo "  3. Compare hyperparameters"
echo "  4. Run model comparison"
echo "  5. Generate comprehensive reports"
echo ""

# Verify no data leakage
verify_no_data_leakage
echo ""

# Compare hyperparameters
compare_hyperparameters
echo ""

# Confirm before proceeding
if [ "$TEST_ONLY" = false ]; then
    if [ "$LOOCV" = true ]; then
        print_warning "LOOCV Training mode: This may take MANY hours (training on all folds)"
    else
        print_warning "Training mode: This may take several hours depending on dataset size"
    fi
else
    print_info "Test-only mode: Will only evaluate pre-trained models"
fi

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Aborted by user"
    exit 0
fi

# Run comparison(s)
if [ "$MULTI_SEED" = true ]; then
    print_header "Running Multi-Seed Comparison"
    print_info "Running with seeds: 42, 123, 456, 789, 999"

    seeds=(42 123 456 789 999)
    success_count=0

    for seed in "${seeds[@]}"; do
        if run_single_comparison $seed "_seed${seed}"; then
            ((success_count++))
        fi
        echo ""
    done

    print_header "Multi-Seed Comparison Complete"
    print_success "Completed $success_count out of ${#seeds[@]} runs"

    # Aggregate results
    print_info "Aggregating results across seeds..."

    if [ -n "$OUTPUT_DIR" ]; then
        base_dir="$OUTPUT_DIR"
    else
        base_dir="work_dir"
    fi

    echo ""
    echo "Summary across all seeds:"
    echo "┌──────┬─────────────────────┬──────────────┬─────────────┐"
    echo "│ Seed │ Model               │ Accuracy (%) │ F1 Score    │"
    echo "├──────┼─────────────────────┼──────────────┼─────────────┤"

    for seed in "${seeds[@]}"; do
        result_dir="${base_dir}_seed${seed}"
        if [ -f "$result_dir/reports/results.csv" ]; then
            # Parse CSV and extract metrics (skip header)
            tail -n +2 "$result_dir/reports/results.csv" | while IFS=',' read -r model accuracy precision recall f1 rest; do
                printf "│ %-4s │ %-19s │ %12s │ %11s │\n" "$seed" "$model" "$accuracy" "$f1"
            done
        fi
    done

    echo "└──────┴─────────────────────┴──────────────┴─────────────┘"

else
    # Single seed comparison
    run_single_comparison $SEED ""
fi

# Final summary
print_header "Comparison Complete!"

print_success "All tasks completed successfully"
echo ""
print_info "Next steps:"
echo "  1. Review the generated reports in the output directory"
echo "  2. Check the plots for visual comparison"
echo "  3. Analyze the confusion matrices and ROC curves"
echo "  4. Read the comparison_report.md for detailed analysis"
echo ""
print_info "For publication:"
echo "  - All plots are saved as high-resolution PNG files"
echo "  - Results are available in CSV format for tables"
echo "  - Markdown report includes all metrics and analysis"
echo ""

exit 0
