#!/usr/bin/env bash

set -euo pipefail

config="$1"
gate_report_prefix="$2"
shift 2

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
app_dir="$(cd ../.. && pwd)"
source cloud/iris/load_runtime_environment.sh
load_runtime_environment "$(skyrl_task_runtime_file)"

python "$app_dir/experiments/post_training/nccl_transport_probe.py" \
  --mode gate --gate-coverage all-gpus --gate-report-prefix "$gate_report_prefix" "$@"
exec python cloud/iris/task_runtime.py --config "$config" --environment-prepared
