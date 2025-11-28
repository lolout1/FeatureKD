#!/bin/bash
# SE Module Comparison Experiment
# Compares baseline vs SE+TemporalAttention models for acc-only and acc+gyro

set -e

EXPERIMENT_NAME="se_comparison_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="results/${EXPERIMENT_NAME}"
LOG_DIR="${RESULTS_DIR}/logs"
METRICS_DIR="${RESULTS_DIR}/metrics"

mkdir -p "${LOG_DIR}" "${METRICS_DIR}"

# Model configurations
declare -A CONFIGS
CONFIGS["transformer_baseline"]="config/smartfallmm/transformer_baseline.yaml"
CONFIGS["transformer_se"]="config/smartfallmm/transformer_se.yaml"
CONFIGS["imu_transformer_baseline"]="config/smartfallmm/imu_transformer_baseline.yaml"
CONFIGS["imu_transformer_se"]="config/smartfallmm/imu_transformer_se.yaml"

# Device selection
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DEVICE=0
    export CUDA_VISIBLE_DEVICES=0
    echo "Using CUDA device 0"
else
    DEVICE="cpu"
    NUM_CORES=$(nproc)
    export OMP_NUM_THREADS=$NUM_CORES
    export MKL_NUM_THREADS=$NUM_CORES
    echo "Using CPU with ${NUM_CORES} cores"
fi

echo "=========================================="
echo "SE Comparison Experiment: ${EXPERIMENT_NAME}"
echo "Results directory: ${RESULTS_DIR}"
echo "Device: ${DEVICE}"
echo "=========================================="

run_experiment() {
    local model_name=$1
    local config_path=$2
    local model_work_dir="${METRICS_DIR}/${model_name}"

    echo ""
    echo "=========================================="
    echo "Running: ${model_name}"
    echo "Config: ${config_path}"
    echo "=========================================="

    python main.py \
        --config "${config_path}" \
        --work-dir "${model_work_dir}" \
        --device "${DEVICE}" \
        --phase train \
        2>&1 | tee "${LOG_DIR}/${model_name}.log"

    if [ -f "${model_work_dir}/scores.csv" ]; then
        echo ""
        echo "${model_name} completed. Extracting summary..."
        python -c "
import pandas as pd
import json

model_name = '${model_name}'
work_dir = '${model_work_dir}'

df = pd.read_csv(f'{work_dir}/scores.csv')
df_data = df[df['test_subject'] != 'Average']

if len(df_data) == 0:
    print(f'{model_name} - No valid folds completed')
else:
    summary = {
        'model': model_name,
        'num_folds': len(df_data),
        'test': {
            'accuracy': {'mean': float(df_data['test_accuracy'].mean()), 'std': float(df_data['test_accuracy'].std())},
            'f1': {'mean': float(df_data['test_f1_score'].mean()), 'std': float(df_data['test_f1_score'].std())},
            'precision': {'mean': float(df_data['test_precision'].mean()), 'std': float(df_data['test_precision'].std())},
            'recall': {'mean': float(df_data['test_recall'].mean()), 'std': float(df_data['test_recall'].std())}
        }
    }

    with open(f'{work_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'{model_name} - Test F1: {summary[\"test\"][\"f1\"][\"mean\"]:.2f} +/- {summary[\"test\"][\"f1\"][\"std\"]:.2f}')
"
    else
        echo "${model_name} - No scores.csv found"
    fi
}

# Run all experiments
for model_name in "${!CONFIGS[@]}"; do
    run_experiment "${model_name}" "${CONFIGS[$model_name]}"
done

# Final comparison
echo ""
echo "=========================================="
echo "FINAL COMPARISON"
echo "=========================================="

python -c "
import json
from pathlib import Path

metrics_dir = Path('${METRICS_DIR}')
results = []

for model_dir in metrics_dir.iterdir():
    if model_dir.is_dir():
        summary_file = model_dir / 'summary.json'
        if summary_file.exists():
            with open(summary_file) as f:
                results.append(json.load(f))

if not results:
    print('No results found')
else:
    print('')
    print('=' * 90)
    print(f'{\"Model\":<30} {\"Test Acc\":>14} {\"Test F1\":>14} {\"Test Prec\":>14} {\"Test Rec\":>14}')
    print('=' * 90)

    for r in sorted(results, key=lambda x: x['test']['f1']['mean'], reverse=True):
        name = r['model']
        t = r['test']
        print(f'{name:<30} {t[\"accuracy\"][\"mean\"]:>6.2f}+/-{t[\"accuracy\"][\"std\"]:>5.2f} {t[\"f1\"][\"mean\"]:>6.2f}+/-{t[\"f1\"][\"std\"]:>5.2f} {t[\"precision\"][\"mean\"]:>6.2f}+/-{t[\"precision\"][\"std\"]:>5.2f} {t[\"recall\"][\"mean\"]:>6.2f}+/-{t[\"recall\"][\"std\"]:>5.2f}')

    print('=' * 90)

    combined = {'models': results}
    with open('${RESULTS_DIR}/comparison.json', 'w') as f:
        json.dump(combined, f, indent=2)
"

echo ""
echo "Experiment complete. Results saved to: ${RESULTS_DIR}"
echo "  - Per-model results: ${METRICS_DIR}/<model>/scores.csv"
echo "  - Model summaries: ${METRICS_DIR}/<model>/summary.json"
echo "  - Final comparison: ${RESULTS_DIR}/comparison.json"
