#!/bin/bash
#SBATCH --job-name=__data
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --partition=a100
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
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

APPTAINER_PYTHON=/home/aevans/miniconda3/bin/python
SCRIPT=/home/aevans/inference_ai2es_forecast_err/src/clean_lstm_data.py

apptainer exec /home/aevans/apptainer/pytorch2.sif \
$APPTAINER_PYTHON $SCRIPT $SLURM_ARRAY_TASK_ID $YEAR $MONTH $DAY $MODEL