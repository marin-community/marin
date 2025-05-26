#!/bin/bash

export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

opt=$1
model_size=$2
chinchilla_ratio=$3

name="exp725_${opt}sweep_${model_size}_${chinchilla_ratio}"
for i in {1..30}; do
    log_file="logs/${name}.txt"
    if [ -f "$log_file" ] && grep -q "Succeeded LOL" "$log_file"; then
        echo "Job ${name} already succeeded. Skipping launch."
        continue
    fi
    python3 marin/run/ray_run.py -e WANDB_API_KEY $WANDB_API_KEY -- python experiments/optimizer_sweep/PhaseI_Bound/"${name}".py --force_run_failed True 2>&1 | tee logs/"${name}".txt 
    exit_code=$?
    # If the process failed (non-zero exit code), sleep for 20 seconds.
    if [ $exit_code -ne 0 ]; then
        sleep_time=$((RANDOM % 41))
    fi
    python experiments/optimizer_sweep/rewriter.py
done
