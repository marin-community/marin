#!/usr/bin/env bash
# Validate Grug on Megatron, plus a CP2 FlashAttention policy update on the frozen runtime.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-env}"
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" development megatron

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "::: running the Grug Megatron gates and CP2 FlashAttention smoke"
cd "$REPOSITORY_ROOT"
JUNIT_XML="$REPOSITORY_ROOT/grug-megatron-junit.xml"
test_targets=(
  skyrl-train/tests/gpu/test_grug_megatron.py
  skyrl-train/tests/gpu/test_grug_sampler_trainer.py
  skyrl-train/tests/gpu/test_grug_levanter_parity.py
  skyrl-train/tests/gpu/test_megatron_worker.py::test_megatron_flash_attention_cp2_forward_backward
)
if [[ -n "${GRUG_MEGATRON_TESTS:-}" ]]; then
  test_targets=("$GRUG_MEGATRON_TESTS")
fi
"$PYTHON" "$REPOSITORY_ROOT/marinskyrl/environment_contract.py" \
  run-grug-gpu-gate "$REPOSITORY_ROOT" -- \
  "$PYTHON" -m pytest ${GRUG_MEGATRON_PYTEST_ARGS:--x} -s \
  --junitxml="$JUNIT_XML" \
  "${test_targets[@]}"
"$PYTHON" -c "import xml.etree.ElementTree as ET; cases = ET.parse('$JUNIT_XML').getroot().findall('.//testcase'); assert cases and all(case.find('skipped') is None for case in cases), 'a Grug Megatron gate did not execute'"
