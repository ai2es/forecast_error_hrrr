#!/bin/bash
#SBATCH --job-name=__nysm
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --partition=a100
#SBATCH --time=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%A.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%A.out

YEAR=$(date +%Y)
MONTH=$(date +%m)

APPTAINER_PYTHON=/home/aevans/miniconda3/bin/python
SCRIPT=/home/aevans/inference_ai2es_forecast_err/src/data_cleaning/get_resampled_nysm_data.py

apptainer exec /home/aevans/apptainer/pytorch2.sif \
$APPTAINER_PYTHON $SCRIPT --year $YEAR --month $MONTH