#!/usr/bin/env bash
# One task per serving node: build the vendored SkyRL GPU runtime, stage the Mochi metadata and tokenizer, then
# join the Ray gang; the head runs mochi_serving_ab.py with the arguments given to this script.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
skyrl="$root/vendor/skyrl"
bench="$root/experiments/benchmarks_local"
started=$(date +%s)
bash "$skyrl/cloud/iris/bootstrap_runtime.sh" "$skyrl" /tmp/skyrl-env /tmp/skyrl-runtime.sh megatron production
source /tmp/skyrl-runtime.sh
echo "MOCHI_AB_STAGE runtime_seconds=$(( $(date +%s) - started )) task=${IRIS_TASK_ID:-?}"
uv pip install --quiet --python /tmp/skyrl-env/bin/python py-spy || echo "MOCHI_AB_STAGE py-spy install failed"
export PYTHONPATH="$skyrl/skyrl-train:$skyrl/skyrl-gym:$skyrl:$bench"
export SKYRL_HOME="$skyrl"
/tmp/skyrl-env/bin/python "$bench/mochi_serving_ab.py" stage-metadata --source "$MOCHI_SOURCE" --dest /tmp/mochi-model
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -8
exec /tmp/skyrl-env/bin/python "$bench/mochi_serving_runtime.py" \
  --rendezvous-dir "$MOCHI_OUTPUT/rendezvous" --ray-log-dir "$MOCHI_OUTPUT/ray-logs" \
  -- /tmp/skyrl-env/bin/python "$bench/mochi_serving_ab.py" run --output "$MOCHI_OUTPUT" --source "$MOCHI_SOURCE" "$@"
