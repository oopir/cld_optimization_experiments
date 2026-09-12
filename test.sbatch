#!/bin/bash
#SBATCH --job-name=cld-test
#SBATCH --output=slurm_logs/%x-%A_%a.log
#SBATCH --chdir=/home/ofir.gaash/research/code/tmp/
#SBATCH --account=acct-danielsoudry
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:03:00

set -euo pipefail

source /home/ofir.gaash/miniforge3/etc/profile.d/conda.sh
conda activate cld-opt

config="configs/test_seed${SLURM_ARRAY_TASK_ID}.yaml"

sleep "${SLURM_ARRAY_TASK_ID}"
sleep "${SLURM_ARRAY_TASK_ID}"
echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| visible GPUs:', torch.cuda.device_count())"

# The active engine is single-device; do not replace this with torchrun.
python scripts/run_exp_from_config.py --config "$config"
