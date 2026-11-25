#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
bash cloud/iris/bootstrap_runtime.sh "$PWD" "$IRIS_VENV" "$PWD/.iris-runtime-env" "$1" production
source cloud/iris/load_runtime_environment.sh "$PWD/.iris-runtime-env"

if (( $# > 1 )); then
  shift
  exec "$@"
fi
