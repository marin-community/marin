#!/usr/bin/env bash
# GB200x4, from a checkout with the holdback: model/module/schedule gate, then the GPU MoE tests.
set -uo pipefail
python autoresearch/loop-260930-mfu30/b/holdback_gate.py
gate_status=$?
uv pip install --quiet pytest >/dev/null 2>&1 || echo "pytest install failed"
python -m pytest -o addopts="" -p no:cacheprovider -q -rf --tb=short lib/levanter/tests/grug/test_grugformer_moe.py 2>&1 | tail -20
test_status=${PIPESTATUS[0]}
echo "gate_status=${gate_status} pytest_status=${test_status}"
exit $(( gate_status || test_status ))
