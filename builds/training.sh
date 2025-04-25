#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=0
#SBATCH --partition=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --array=0-2  # Adjust to number of GPUs/stations
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

CLIM_DIVS=("Champlain Valley" "Eastern Plateau" "Coastal")  
# Replace with your 4 climate divisions
# "St. Lawrence Valley" "Western Plateau" "Coastal" "Central Lakes" "Champlain Valley" "Eastern Plateau" "Great Lakes" "Hudson Valley" "Mohawk Valley" "Northern Plateau"
CLIM_DIV=${CLIM_DIVS[$SLURM_ARRAY_TASK_ID]}

# Run the training script for the selected climate division
apptainer run --nv /home/aevans/apptainer/pytorch2.sif /home/aevans/miniconda3/bin/python /home/aevans/inference_ai2es_forecast_err/src/engine_lstm_training.py \
    --clim_div $CLIM_DIV \
    --device_id $SLURM_ARRAY_TASK_ID