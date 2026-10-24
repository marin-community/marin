#!/usr/bin/env bash
# Build the vendored SkyRL GPU runtime (the RL engines' vLLM fork) and run the group-routing benchmark.
set -euo pipefail
cd "$(dirname "$0")/../.."
started=$(date +%s)
bash vendor/skyrl/cloud/iris/bootstrap_runtime.sh vendor/skyrl /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron production
echo "{\"event\": \"runtime_ready\", \"seconds\": $(( $(date +%s) - started ))}"
source /tmp/skyrl-runtime.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8
/tmp/skyrl-env/bin/python experiments/benchmarks_local/group_routing_bench.py "$@"
