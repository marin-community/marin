#!/bin/bash

export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

optimizers=(mars muon scion)

# Loop over each optimizer in the list
for opt in "${optimizers[@]}"; do
    # Define the experiment name using the current optimizer
    name="exp725_${opt}sweep_130M_4"
    
    # Loop 10 times for the current optimizer
    for i in {1..30}; do
        # Start the ray_run process and capture its PID.
        log_file="logs/${name}.txt"
        if [ -f "$log_file" ] && grep -q "Succeeded LOL" "$log_file"; then
            echo "Job ${name} already succeeded. Skipping launch."
            continue
        fi
        python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/"${name}".py --force_run_failed True 2>&1 | tee logs/"${name}".txt 
        exit_code=$?
        # If the process failed (non-zero exit code), sleep for 20 seconds.
        if [ $exit_code -ne 0 ]; then
            sleep_time=$((RANDOM % 41))
            python optimizer_sweep/stupid.py
        fi
        python optimizer_sweep/rewriter.py
    done
done