#!/usr/bin/env bash
# GB200x4 gate for the stack: B's module gate (routing, QuACK contract, layer scan, MoE pytest), then
# the model-level smoke. Submit with the hero env (cuda_async, fraction 0.75, ragged flags, overlap 1).
set -uo pipefail
bash autoresearch/loop-260930-mfu30/b/gpu_gate.sh
b_status=$?
python autoresearch/loop-260930-mfu30/stack/model_smoke.py
smoke_status=$?
echo "b_gate_status=${b_status} model_smoke_status=${smoke_status}"
exit $(( b_status || smoke_status ))
