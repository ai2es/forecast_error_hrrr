#!/bin/bash
#SBATCH --job-name=__data
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --partition=a100
#SBATCH --time=0
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

# Dynamically get current date/time
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
HOUR=$(date +%H)
MODEL="hrrr"  # Or dynamically choose if needed

echo "[$(date)] Running job with:"
echo "YEAR=$YEAR, MONTH=$MONTH, DAY=$DAY, HOUR=$HOUR, MODEL=$MODEL"

# Path to your Python environment and script
APPTAINER_PYTHON=/home/aevans/miniconda3/bin/python
SCRIPT=/home/aevans/inference_ai2es_forecast_err/src/clean_lstm_data.py

# Run using Apptainer if needed
apptainer exec /home/aevans/apptainer/pytorch2.sif \
$APPTAINER_PYTHON $SCRIPT $SLURM_ARRAY_TASK_ID $YEAR $MONTH $DAY $MODEL