#!/usr/bin/env bash
#SBATCH --job-name=sat-splat-distort
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/satsplat_%j.out
#SBATCH --error=logs/satsplat_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=arun08sharma@gmail.com

set -euo pipefail

module load python/3.11
module load cuda/12.4
source ~/envs/sat-splat-distort/bin/activate

EXP_NAME=${1:-dfc2019_jax}
OUT_DIR=outputs/${EXP_NAME}_${SLURM_JOB_ID}

mkdir -p logs "$OUT_DIR"

srun python -m sat_splat.training.train \
  --config-name $EXP_NAME \
  train.out_dir=$OUT_DIR \
  logging.wandb=true
