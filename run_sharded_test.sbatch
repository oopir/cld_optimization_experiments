#!/bin/bash
#SBATCH --job-name=cld-test
#SBATCH --output=slurm_logs/%x-%A_%a.log
#SBATCH --chdir=/home/ofir.gaash/research/from_11/ofirg/cld_optimization_experiments
#SBATCH --account=acct-danielsoudry
#SBATCH --partition=part-danielsoudry
#SBATCH --qos=qos-danielsoudry
#SBATCH --array=0-4
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00

set -euo pipefail

source /home/ofir.gaash/miniforge3/etc/profile.d/conda.sh
conda activate cld-opt

config="configs/sharded_slurm/test_seed${SLURM_ARRAY_TASK_ID}.yaml"

sleep "${SLURM_ARRAY_TASK_ID}"
echo "Job ${SLURM_JOB_ID}, array task ${SLURM_ARRAY_TASK_ID}, running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA build:', torch.version.cuda, '| visible GPUs:', torch.cuda.device_count())"

python scripts/run_sharded_exp_from_config.py --config "$config"
