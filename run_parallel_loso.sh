#!/bin/bash
#SBATCH --job-name=loso_parallel
#SBATCH --output=logs/loso_%A_%a.out
#SBATCH --error=logs/loso_%A_%a.err
#SBATCH --array=0-28%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# SLURM array wrapper for LOSO training; each task owns one fold.

set -e

# Configuration
CONFIG_FILE=${1:-"config/smartfallmm/imu_8channel.yaml"}
DEVICE=0
NUM_EPOCHS=80
BATCH_SIZE=64

echo "========================================================================"
echo "PARALLEL LOSO - SLURM Array Job"
echo "========================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Config: ${CONFIG_FILE}"
echo "Device: ${DEVICE}"
echo "========================================================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# All subjects for LOSO
SUBJECTS=(29 30 31 32 34 35 36 37 38 39 43 44 45 46 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63)

# Get the test subject for this array task
TEST_SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "This task will handle test subject: ${TEST_SUBJECT}"
echo ""

# Extract config basename for work directory
CONFIG_BASENAME=$(basename ${CONFIG_FILE} .yaml)
WORK_DIR="work_dir/parallel_loso_${CONFIG_BASENAME}/fold_${TEST_SUBJECT}"

echo "Work directory: ${WORK_DIR}"
echo ""

# Run training for this single fold
python main.py \
    --config ${CONFIG_FILE} \
    --device ${DEVICE} \
    --num_epoch ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --work-dir ${WORK_DIR} \
    --phase train \
    --single-fold ${TEST_SUBJECT}

echo ""
echo "========================================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} (test subject ${TEST_SUBJECT}) completed!"
echo "========================================================================"
