#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=0
#SBATCH --partition=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1         # Run all processes on a single node
#SBATCH --ntasks=1                  # Number of processes
#SBATCH --array=1-6  # Adjust to number of GPUs/stations
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

echo "[$(date)] Forecast Hour: $SLURM_ARRAY_TASK_ID"

# Run the training script for the selected climate division
apptainer run --nv /home/aevans/apptainer/rapids.sif /home/aevans/inference_ai2es_forecast_err/src/lstm_s2s_engine.py \
    --fh $SLURM_ARRAY_TASK_ID \
    --device_id $SLURM_ARRAY_TASK_ID