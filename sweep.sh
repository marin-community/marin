#!/bin/bash

export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=1200

optimizers_list_1=(adamw lion mini)
optimizers_list_2=(nadamw mars cautious)
optimizers_list_3=(muon scion soap)
model_1='300M'
model_2='520M'

# Get optimizer list and model choice from arguments
optimizer_choice=$1
model_choice=$2

# Set optimizer list and model based on choices
if [ "$optimizer_choice" = "1" ]; then
    optimizers=("${optimizers_list_1[@]}")
elif [ "$optimizer_choice" = "2" ]; then
    optimizers=("${optimizers_list_2[@]}")
elif [ "$optimizer_choice" = "3" ]; then
    optimizers=("${optimizers_list_3[@]}")
else
    echo "Invalid optimizer choice. Please use 1, 2, or 3."
    exit 1
fi

if [ "$model_choice" = "1" ]; then
    model="$model_1"
elif [ "$model_choice" = "2" ]; then
    model="$model_2"
else
    echo "Invalid model choice. Please use 1 or 2."
    exit 1
fi

chinchillas=(2 4 8)
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
            
            python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/"${name}".py --force_run_failed True 2>&1 | tee "$log_file"
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
