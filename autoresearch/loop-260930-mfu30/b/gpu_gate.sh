#!/usr/bin/env bash
# Runs on a GB200x4 node: GPU MoE tests, then the control-vs-candidate routing gate.
set -uo pipefail
nvidia-smi -L
uv pip install --quiet pytest >/dev/null 2>&1 || echo "pytest install failed"
python -m pytest -o addopts="" -p no:cacheprovider -q lib/levanter/tests/grug/test_grugformer_moe.py 2>&1 | tail -25
test_status=${PIPESTATUS[0]}
python autoresearch/loop-260930-mfu30/b/routing_gate.py "$@"
gate_status=$?
echo "pytest_status=${test_status} gate_status=${gate_status}"
exit $(( test_status || gate_status ))
