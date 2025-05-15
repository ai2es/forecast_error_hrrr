#!/bin/bash
#SBATCH --job-name=_-pipe_-
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=0
#SBATCH --partition=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Number of processes
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

# Clean data (will only run if it finishes successfully)
bash /home/aevans/inference_ai2es_forecast_err/builds/resample_nysm.sh && \
bash /home/aevans/inference_ai2es_forecast_err/builds/clean_meteo_data.sh && \
# Run the next script only if the clean data script finishes successfully
bash /home/aevans/inference_ai2es_forecast_err/builds/inference_cast.sh && \
# Run the next script only if the previous one finishes successfully
bash /home/aevans/inference_ai2es_forecast_err/builds/inference_cast_r2.sh && \
# Run the final script only if the previous one finishes successfully
bash /home/aevans/inference_ai2es_forecast_err/builds/inference_cast_r3.sh

