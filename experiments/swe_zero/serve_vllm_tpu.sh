#!/usr/bin/env bash
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Launch vLLM on TPU to serve Gemma 4 for SWE-ZERO rollout generation.
#
# Usage:
#   # On a v6e-4 TPU VM (E2B — smaller, faster):
#   bash experiments/swe_zero/serve_vllm_tpu.sh google/gemma-4-E2B-it 4
#
#   # On a v5p-8 TPU VM (E4B — larger):
#   bash experiments/swe_zero/serve_vllm_tpu.sh google/gemma-4-E4B-it 8
#
# The server will be available at http://localhost:8000/v1
# Test with:
#   curl http://localhost:8000/v1/models

set -euo pipefail

MODEL="${1:-google/gemma-4-E2B-it}"
TP_SIZE="${2:-4}"
PORT="${3:-8000}"
MAX_MODEL_LEN="${4:-16384}"

echo "=== SWE-ZERO vLLM-TPU Server ==="
echo "Model: ${MODEL}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "Port: ${PORT}"
echo "Max model length: ${MAX_MODEL_LEN}"

# vLLM-tpu environment settings
export SKIP_JAX_PRECOMPILE="${SKIP_JAX_PRECOMPILE:-0}"
export VLLM_XLA_CHECK_RECOMPILATION="${VLLM_XLA_CHECK_RECOMPILATION:-0}"

# Enable tool calling support
export VLLM_ENABLE_TOOL_USE="${VLLM_ENABLE_TOOL_USE:-1}"

echo "Starting vLLM server..."

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${PORT}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --chat-template auto \
    --dtype bfloat16 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95
