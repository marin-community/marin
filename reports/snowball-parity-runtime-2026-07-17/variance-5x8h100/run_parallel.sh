#!/usr/bin/env bash

set -u -o pipefail

output_dir=".agents/tmp/snowball-parity-runtime-2026-07-17/variance-5x8h100"
node_id="tests/cluster/vllm/test_snowball_backend_parity.py::test_snowball_backend_matches_representative_goldens[vllm-gpu]"
pids=()

for run in 1 2 3 4 5; do
  (
    PYTHONUNBUFFERED=1 uv run pytest "$node_id" \
      -m cluster -o addopts= --import-mode=importlib -vv -s --durations=0 \
      --junitxml="$output_dir/run-${run}-junit.xml" 2>&1 \
      | tee "$output_dir/run-${run}-pytest.log"
  ) &
  pids+=("$!")
done

failures=0
for index in "${!pids[@]}"; do
  run=$((index + 1))
  if wait "${pids[$index]}"; then
    echo "variance run $run passed"
  else
    echo "variance run $run failed"
    failures=$((failures + 1))
  fi
done

echo "variance experiment complete: $failures/5 pytest invocations failed"
exit "$failures"
