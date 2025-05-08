#!/bin/bash

export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

# Get optimizer list and model choice from arguments
optimizers=( soap )
model="520M"
chinchillas=( 2 4 8 )
for opt in "${optimizers[@]}"; do
    for chinchilla in "${chinchillas[@]}"; do
        # Define the experiment name using the current optimizer
        name="exp725_${opt}sweep_${model}_${chinchilla}"
        # Loop 30 times for the current optimizer
        for i in {1..30}; do
            # Check if log directory exists, if not create it
            mkdir -p logs
            
            log_file="logs/${name}.txt"
            if [ -f "$log_file" ] && grep -q "Succeeded LOL" "$log_file"; then
                echo "Job ${name} already succeeded. Skipping launch."
                continue
            fi
            
            python3 marin/run/ray_run.py -e WANDB_API_KEY $WANDB_API_KEY -- python optimizer_sweep/"${name}".py --force_run_failed True 2>&1 | tee "$log_file"
            exit_code=$?
            
            # If the process failed (non-zero exit code), sleep for a random time
            if [ $exit_code -ne 0 ]; then
                sleep_time=$((RANDOM % 41))
                echo "Process failed. Sleeping for ${sleep_time} seconds..."
                sleep "$sleep_time"
                python optimizer_sweep/stupid.py
            fi
            python optimizer_sweep/rewriter.py
        done
    done
done
