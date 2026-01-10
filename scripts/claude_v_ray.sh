#!/usr/bin/env bash
#
# Wrapper script to run ray_run.py with automatic retry on failure.
#
# Usage:
#   ./scripts/ray_run_retry.sh [OPTIONS] -- <ray_run args>
#
# Options:
#   --max-retries N    Maximum number of retry attempts (default: 3)
#
# Example:
#   ./scripts/ray_run_retry.sh --max-retries 5 -- \
#       --cluster us-central2 -- uv run python experiments/my_exp.py
#
# Exit codes from ray_run.py:
#   0 = SUCCEEDED
#   1 = FAILED
#   2 = STOPPED
#   3 = Unexpected status

set -euo pipefail

# Default configuration
MAX_RETRIES=3

# Parse wrapper options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--max-retries N] -- <ray_run args>" >&2
            exit 1
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Error: No ray_run.py arguments provided after '--'" >&2
    echo "Usage: $0 [--max-retries N] -- <ray_run args>" >&2
    exit 1
fi

# Store ray_run arguments
RAY_RUN_ARGS=("$@")

attempt=0

while true; do
    attempt=$((attempt + 1))
    echo "=============================================="
    echo "Attempt $attempt of $((MAX_RETRIES + 1))"
    echo "=============================================="

    set +e
    uv run python -m marin.run.ray_run "${RAY_RUN_ARGS[@]}"
    exit_code=$?
    set -e

    case $exit_code in
        0)
            echo "Job succeeded on attempt $attempt"
            exit 0
            ;;
        1)
            echo "Job failed (exit code 1)"
            ;;
        2)
            echo "Job was stopped (exit code 2)"
            # Don't retry stopped jobs - they were intentionally stopped
            echo "Not retrying stopped jobs."
            exit 2
            ;;
        *)
            echo "Job exited with unexpected code: $exit_code"
            ;;
    esac

    if [[ $attempt -gt $MAX_RETRIES ]]; then
        echo "Max retries ($MAX_RETRIES) exceeded. Giving up."
        exit $exit_code
    fi

    echo "Retrying immediately..."
done
