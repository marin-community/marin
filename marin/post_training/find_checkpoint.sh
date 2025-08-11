#!/bin/bash

# Checkpoint Detection Script
# Finds the latest checkpoint in a given experiment directory

find_latest_checkpoint() {
    local run_name="$1"
    local base_path="gs://marin-us-central2/post_training/experiments/$run_name"
    
    echo "Checking for existing checkpoints in: $base_path" >&2
    
    # Check if any checkpoint directories exist
    local checkpoint_dirs=$(gsutil ls "$base_path/" 2>/dev/null | grep -E "/$run_name--[a-f0-9-]+/$" | head -20)
    
    if [[ -z "$checkpoint_dirs" ]]; then
        echo "No checkpoint directories found" >&2
        return 1
    fi
    
    local latest_checkpoint=""
    local latest_step=-1
    local latest_timestamp=0
    
    echo "Found checkpoint directories, analyzing..." >&2
    
    # Iterate through each UUID directory
    while IFS= read -r uuid_dir; do
        if [[ -z "$uuid_dir" ]]; then continue; fi
        
        # Remove trailing slash and extract UUID
        uuid_dir=${uuid_dir%/}
        local uuid=$(basename "$uuid_dir" | sed "s/^$run_name--//")
        
        echo "Checking UUID: $uuid" >&2
        
        # Check if checkpoints subdirectory exists
        local checkpoints_path="$uuid_dir/checkpoints"
        if ! gsutil ls "$checkpoints_path/" >/dev/null 2>&1; then
            echo "  No checkpoints subdirectory found" >&2
            continue
        fi
        
        # Find the highest step number in this UUID directory
        local step_dirs=$(gsutil ls "$checkpoints_path/" 2>/dev/null | grep -E "/step_[0-9]+/$")
        local max_step=-1
        local max_step_dir=""
        
        while IFS= read -r step_dir; do
            if [[ -z "$step_dir" ]]; then continue; fi
            
            local step_num=$(basename "${step_dir%/}" | sed 's/step_//')
            if [[ "$step_num" =~ ^[0-9]+$ ]] && [[ "$step_num" -gt "$max_step" ]]; then
                max_step="$step_num"
                max_step_dir="$step_dir"
            fi
        done <<< "$step_dirs"
        
        if [[ "$max_step" -ge 0 ]]; then
            echo "  Found max step: $max_step" >&2
            
            # Get timestamp of the step directory (use params.msgpack as reference)
            local params_file="${max_step_dir%/}/params.msgpack"
            local timestamp=$(gsutil stat "$params_file" 2>/dev/null | grep "Creation time:" | sed 's/Creation time:[[:space:]]*//' | xargs -I {} date -d "{}" +%s 2>/dev/null || echo "0")
            
            echo "  Timestamp: $timestamp" >&2
            
            # Check if this is the latest checkpoint overall
            if [[ "$max_step" -gt "$latest_step" ]] || 
               [[ "$max_step" -eq "$latest_step" && "$timestamp" -gt "$latest_timestamp" ]]; then
                latest_checkpoint="$max_step_dir"
                latest_step="$max_step"
                latest_timestamp="$timestamp"
                echo "  This is now the latest checkpoint" >&2
            fi
        fi
    done <<< "$checkpoint_dirs"
    
    if [[ -n "$latest_checkpoint" ]]; then
        # Remove trailing slash
        latest_checkpoint=${latest_checkpoint%/}
        echo "Latest checkpoint found: $latest_checkpoint (step $latest_step)" >&2
        
        # Verify required files exist
        local params_file="$latest_checkpoint/params.msgpack"
        local config_file="$latest_checkpoint/config.json"
        
        if gsutil stat "$params_file" >/dev/null 2>&1 && gsutil stat "$config_file" >/dev/null 2>&1; then
            echo "Checkpoint files verified" >&2
            # Output the checkpoint path for the caller
            echo "$latest_checkpoint"
            return 0
        else
            echo "ERROR: Required checkpoint files not found in $latest_checkpoint" >&2
            return 1
        fi
    else
        echo "No valid checkpoints found" >&2
        return 1
    fi
}

# If script is run directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "Usage: $0 <run_name>"
        echo "Example: $0 random_acts_of_pizza_ckpt"
        exit 1
    fi
    
    RUN_NAME="$1"
    if latest_checkpoint=$(find_latest_checkpoint "$RUN_NAME"); then
        echo "Latest checkpoint: $latest_checkpoint"
        echo "Params: $latest_checkpoint/params.msgpack"
        echo "Config: $latest_checkpoint/config.json"
        exit 0
    else
        echo "No checkpoints found"
        exit 1
    fi
fi
