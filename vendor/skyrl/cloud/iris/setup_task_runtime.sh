#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
source cloud/iris/load_runtime_environment.sh
runtime_file="$(skyrl_task_runtime_file)"
bash cloud/iris/bootstrap_runtime.sh "$PWD" "$IRIS_VENV" "$runtime_file" "$1" production

if (( $# > 1 )); then
  load_runtime_environment "$runtime_file"
  shift
  exec "$@"
fi
