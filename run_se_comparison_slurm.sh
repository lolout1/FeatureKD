#!/bin/bash
# SE Comparison - Submit 4 parallel CPU jobs + aggregation

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/se_comparison_${TIMESTAMP}"

mkdir -p ${RESULTS_DIR}/logs ${RESULTS_DIR}/metrics/{transformer_baseline,transformer_se,imu_transformer_baseline,imu_transformer_se}

echo "=========================================="
echo "SE Comparison Experiment"
echo "Results: ${RESULTS_DIR}"
echo "=========================================="
echo ""
echo "Submitting 4 jobs to parallel partition..."

JOB1=$(sbatch --parsable -J se_tb -p parallel -N1 -n1 -c48 --exclusive --mem=128G -t 24:00:00 -o ${RESULTS_DIR}/logs/transformer_baseline.out -e ${RESULTS_DIR}/logs/transformer_baseline.err run_single_experiment.sh transformer_baseline config/smartfallmm/transformer_baseline.yaml ${RESULTS_DIR})
echo "  transformer_baseline: Job ${JOB1}"

JOB2=$(sbatch --parsable -J se_ts -p parallel -N1 -n1 -c48 --exclusive --mem=128G -t 24:00:00 -o ${RESULTS_DIR}/logs/transformer_se.out -e ${RESULTS_DIR}/logs/transformer_se.err run_single_experiment.sh transformer_se config/smartfallmm/transformer_se.yaml ${RESULTS_DIR})
echo "  transformer_se: Job ${JOB2}"

JOB3=$(sbatch --parsable -J se_ib -p parallel -N1 -n1 -c48 --exclusive --mem=128G -t 24:00:00 -o ${RESULTS_DIR}/logs/imu_transformer_baseline.out -e ${RESULTS_DIR}/logs/imu_transformer_baseline.err run_single_experiment.sh imu_transformer_baseline config/smartfallmm/imu_transformer_baseline.yaml ${RESULTS_DIR})
echo "  imu_transformer_baseline: Job ${JOB3}"

JOB4=$(sbatch --parsable -J se_is -p parallel -N1 -n1 -c48 --exclusive --mem=128G -t 24:00:00 -o ${RESULTS_DIR}/logs/imu_transformer_se.out -e ${RESULTS_DIR}/logs/imu_transformer_se.err run_single_experiment.sh imu_transformer_se config/smartfallmm/imu_transformer_se.yaml ${RESULTS_DIR})
echo "  imu_transformer_se: Job ${JOB4}"

# Submit aggregation job that runs after all 4 complete
AGG_JOB=$(sbatch --parsable -J se_agg -p parallel -N1 -n1 -c4 --mem=8G -t 00:30:00 \
    --dependency=afterany:${JOB1}:${JOB2}:${JOB3}:${JOB4} \
    -o ${RESULTS_DIR}/logs/aggregate.out -e ${RESULTS_DIR}/logs/aggregate.err \
    run_aggregate.sh ${RESULTS_DIR})
echo ""
echo "  aggregate: Job ${AGG_JOB} (runs after all complete)"

echo ""
echo "=========================================="
echo "Monitor: squeue -u \$USER"
echo "Logs: tail -f ${RESULTS_DIR}/logs/*.out"
echo "=========================================="
