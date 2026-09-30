#!/usr/bin/env bash
# Runs on a GB200x4 node: the routing comparison, the QuACK row contract, the scan comparison,
# then the GPU MoE tests.
set -uo pipefail
nvidia-smi -L
python autoresearch/loop-260930-mfu30/b/routing_gate.py "$@"
gate_status=$?
python autoresearch/loop-260930-mfu30/b/quack_contract.py
contract_status=$?
python autoresearch/loop-260930-mfu30/b/scan_compare.py
scan_status=$?
uv pip install --quiet pytest >/dev/null 2>&1 || echo "pytest install failed"
python -m pytest -o addopts="" -p no:cacheprovider -q -rf --tb=short lib/levanter/tests/grug/test_grugformer_moe.py 2>&1 | tail -80
test_status=${PIPESTATUS[0]}
echo "gate_status=${gate_status} contract_status=${contract_status} scan_status=${scan_status} pytest_status=${test_status}"
exit $(( gate_status || contract_status || scan_status || test_status ))
