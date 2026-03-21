#!/bin/bash
# Bootstrap script for RL experiments inside Docker on TPU VMs.
# All deps (vllm-tpu, marin, levanter, etc.) are pre-installed in the Docker
# image via `uv sync --extra rl`. This script just runs the experiment.
set -e

cd /opt/marin

# Force unbuffered output so tracebacks are visible in docker logs
export PYTHONUNBUFFERED=1
# Print Python-level traceback on segfaults/fatal signals
export PYTHONFAULTHANDLER=1

exec python "$@"
