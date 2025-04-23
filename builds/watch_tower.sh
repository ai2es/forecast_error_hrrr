#!/bin/bash
#SBATCH --job-name=--(O)~(O)--
#SBATCH --account=ai2es
#SBATCH --qos=long
#SBATCH --time=0
#SBATCH --partition=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --error=/home/aevans/slurm_error_logs/ai2es_error_%j.err
#SBATCH --output=/home/aevans/slurm_output_logs/ai2es_log_%j.out

TARGET_SCRIPT="/home/aevans/inference_ai2es_forecast_err/builds/pipeline.sh"

echo "[$(date)] WatchTower started. Running indefinitely..."

while true; do
    echo "[$(date)] Starting hourly job cycle..."

    start_epoch=$(date +%s)

    while true; do
        echo "[$(date)] Attempting to run $TARGET_SCRIPT"
        bash "$TARGET_SCRIPT"
        STATUS=$?

        if [ $STATUS -eq 0 ]; then
            echo "[$(date)] Job succeeded. Preparing to wait until next hour..."
            break
        else
            echo "[$(date)] Job failed (exit code $STATUS). Retrying in 10 minutes..."
            sleep 10m
        fi
    done

    now_epoch=$(date +%s)
    elapsed=$((now_epoch - start_epoch))
    wait_time=$((3600 - elapsed))

    if [ $wait_time -gt 0 ]; then
        echo "[$(date)] Sleeping for $wait_time seconds to align with the next hour."
        sleep $wait_time
    else
        echo "[$(date)] No wait needed. Restarting job cycle."
    fi
done