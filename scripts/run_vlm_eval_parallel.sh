#!/bin/bash
# VLM Evaluation Script for MMMU
# Usage: ./scripts/run_vlm_eval_parallel.sh <checkpoint_path> [max_examples] [parallel_jobs]

set -e

# Configuration
CHECKPOINT=${1:-"gs://marin-us-east1/checkpoints/vlm-official-qwen3-1.7b-stage3_new-0-4e97cb/hf/vlm-official-qwen3-1.7b-stage3_new-0-4e97cb/step-39061/"}
MAX_EXAMPLES=${2:-""}  # Empty means all examples
PARALLEL_JOBS=${3:-1}  # Number of parallel jobs (default: 1 for sequential)

# Output directory
OUTPUT_DIR="vlm_eval_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# MMMU subtasks (30 subjects)
MMMU_TASKS=(
    "mmmu_val_Accounting"
    "mmmu_val_Agriculture"
    "mmmu_val_Architecture_and_Engineering"
    "mmmu_val_Art"
    "mmmu_val_Art_Theory"
    "mmmu_val_Basic_Medical_Science"
    "mmmu_val_Biology"
    "mmmu_val_Chemistry"
    "mmmu_val_Clinical_Medicine"
    "mmmu_val_Computer_Science"
    "mmmu_val_Design"
    "mmmu_val_Diagnostics_and_Laboratory_Medicine"
    "mmmu_val_Economics"
    "mmmu_val_Electronics"
    "mmmu_val_Energy_and_Power"
    "mmmu_val_Finance"
    "mmmu_val_Geography"
    "mmmu_val_History"
    "mmmu_val_Literature"
    "mmmu_val_Manage"
    "mmmu_val_Marketing"
    "mmmu_val_Materials"
    "mmmu_val_Math"
    "mmmu_val_Mechanical_Engineering"
    "mmmu_val_Music"
    "mmmu_val_Pharmacy"
    "mmmu_val_Physics"
    "mmmu_val_Psychology"
    "mmmu_val_Public_Health"
    "mmmu_val_Sociology"
)

# Function to run evaluation for a single task
run_eval() {
    local task=$1
    local log_file="$OUTPUT_DIR/${task}.log"
    local result_file="$OUTPUT_DIR/${task}.json"

    echo "[$(date '+%H:%M:%S')] Starting: $task"

    # Build command
    local cmd="uv run python -m levanter.main.eval_vlm \
        --hf_checkpoint $CHECKPOINT \
        --tokenizer_path $CHECKPOINT \
        --eval_harness.task_spec='[\"$task\"]' \
        --trainer.ray.auto_start_cluster=False"

    # Add max_examples if specified
    if [ -n "$MAX_EXAMPLES" ]; then
        cmd="$cmd --eval_harness.max_examples=$MAX_EXAMPLES"
    fi

    # Run and log
    if eval "$cmd" > "$log_file" 2>&1; then
        echo "[$(date '+%H:%M:%S')] Completed: $task"
        # Extract accuracy from log
        grep -E "(acc|accuracy)" "$log_file" | tail -1 >> "$OUTPUT_DIR/summary.txt"
    else
        echo "[$(date '+%H:%M:%S')] FAILED: $task (see $log_file)"
        echo "FAILED: $task" >> "$OUTPUT_DIR/summary.txt"
    fi
}

# Export function for parallel execution
export -f run_eval
export CHECKPOINT OUTPUT_DIR MAX_EXAMPLES

echo "=========================================="
echo "VLM Evaluation: MMMU"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Tasks: ${#MMMU_TASKS[@]}"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "=========================================="

# Run evaluations
if [ "$PARALLEL_JOBS" -eq 1 ]; then
    # Sequential execution
    for task in "${MMMU_TASKS[@]}"; do
        run_eval "$task"
    done
else
    # Parallel execution using GNU parallel (if available) or xargs
    if command -v parallel &> /dev/null; then
        printf '%s\n' "${MMMU_TASKS[@]}" | parallel -j "$PARALLEL_JOBS" run_eval {}
    else
        # Fallback: run in background with job control
        echo "Note: GNU parallel not found, using background jobs"
        running=0
        for task in "${MMMU_TASKS[@]}"; do
            run_eval "$task" &
            ((running++))
            if [ "$running" -ge "$PARALLEL_JOBS" ]; then
                wait -n  # Wait for any job to finish
                ((running--))
            fi
        done
        wait  # Wait for remaining jobs
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR/summary.txt" 2>/dev/null || echo "(No results yet)"
echo "=========================================="
