#!/usr/bin/env bash

set -euo pipefail

config="$1"
spill_dir="$2"
gate_report_prefix="$3"
shift 3

if [[ -n "$spill_dir" ]]; then
  if ! mkdir -p -- "$spill_dir" || [[ ! -d "$spill_dir" ]]; then
    printf '[rl-iris] Could not prepare local Ray spill directory %q before controller startup\n' "$spill_dir" >&2
    exit 1
  fi
fi

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export SKYRL_HOME="$PWD"
app_dir="$(cd ../.. && pwd)"
source cloud/iris/load_runtime_environment.sh "$SKYRL_HOME/.iris-runtime-env"
export PYTHONPATH="$app_dir:$SKYRL_HOME:$SKYRL_HOME/skyrl-train:${PYTHONPATH:-}"
export VLLM_USE_V1=1

python "$app_dir/experiments/post_training/nccl_transport_probe.py" \
  --mode gate --gate-coverage all-gpus --gate-report-prefix "$gate_report_prefix" "$@"
exec python cloud/iris/task_runtime.py --config "$config"
