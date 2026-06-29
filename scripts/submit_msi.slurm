#!/usr/bin/env bash
#SBATCH --job-name=sat-splat-distort
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/satsplat_%j.out
#SBATCH --error=logs/satsplat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=arun08sharma@gmail.com

set -euo pipefail

# Known-good MSI GPU env (a100-4). One-time setup, see scripts/MSI_RUNBOOK.md:
#   conda create -n satsplat python=3.11 && source activate satsplat && pip install -e .
#   then build the CUDA rasterizer fork: pip install -v -e cuda_rasterizer/
module load gcc/13.1.0-5z64cho
module load cuda/12.4
source activate ${PROJ_ENV:-satsplat}
unset LD_LIBRARY_PATH   # MSI modules clobber LD_LIBRARY_PATH and break conda CUDA libs
# v100 fallback (faster queue): --partition=v100 --gres=gpu:v100:1

EXP_NAME=${1:-dfc2019_jax}
OUT_DIR=outputs/${EXP_NAME}_${SLURM_JOB_ID}

mkdir -p logs "$OUT_DIR"

srun python -m sat_splat.training.train \
  --config-name $EXP_NAME \
  train.out_dir=$OUT_DIR \
  logging.wandb=true
