#!/usr/bin/env bash
# GB200x4: the reorder gate (single-layer bitwise, then two scan pairs), then the GPU MoE tests.
set -uo pipefail
cd autoresearch/loop-260930-mfu30/b
python reorder_gate.py single
single_status=$?
python reorder_gate.py scan dme candidate
scan_status=$?
python reorder_gate.py scan pipe pipe_forward_order
pipe_status=$?
cd - >/dev/null
uv pip install --quiet pytest >/dev/null 2>&1 || echo "pytest install failed"
python -m pytest -o addopts="" -p no:cacheprovider -q -rf --tb=short lib/levanter/tests/grug/test_grugformer_moe.py lib/levanter/tests/kernels/test_quack_expert_mlp.py 2>&1 | tail -40
test_status=${PIPESTATUS[0]}
echo "single_status=${single_status} scan_status=${scan_status} pipe_status=${pipe_status} pytest_status=${test_status}"
exit $(( single_status || scan_status || pipe_status || test_status ))
