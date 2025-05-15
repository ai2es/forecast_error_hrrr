#!/bin/bash
#SBATCH --job-name=__data
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --partition=a100
#SBATCH --time=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gpus-per-node=1
#SBATCH --array=1-18                # One job for each forecast hour
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%A_%a.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%A_%a.out

YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
MODEL="hrrr"

echo "[$(date)] Forecast Hour: $SLURM_ARRAY_TASK_ID"

apptainer run --nv \
  --bind /rdma/dgx-a100/NYSM:/home/aevans/nysm \
  --bind /rdma/xcitedb/AI2ES:/home/aevans/ai2es \
  /home/aevans/apptainer/rapids.sif \
  /home/aevans/inference_ai2es_forecast_err/src/clean_lstm_data.py \
  --fh $SLURM_ARRAY_TASK_ID \
  --year $YEAR \
  --month $MONTH \
  --day $DAY \
  --model $MODEL