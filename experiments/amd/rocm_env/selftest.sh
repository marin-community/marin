#!/bin/bash
# selftest.sh PYTHON [--expect-devices N] [--anp-plugin PATH]: run selftest.py with PYTHON in a clean environment.
# The dynamic loader reads LD_LIBRARY_PATH only when the process starts, so it is set here: system ROCm entries
# removed (the venv ships its own ROCm libraries) and, with --anp-plugin, a directory holding the plugin prepended.
set -euo pipefail
PY=$1
shift
ANP_PLUGIN=
args=("$@")
while [ $# -gt 0 ]; do
  case $1 in
    --anp-plugin) ANP_PLUGIN=$2; shift 2 ;;
    *) shift ;;
  esac
done
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd: - || true)
if [ -n "$ANP_PLUGIN" ]; then
  [ -r "$ANP_PLUGIN" ] || { echo "ANP plugin $ANP_PLUGIN not found on $(hostname)" >&2; exit 1; }
  mkdir "$work/anp"
  ln -s "$(realpath "$ANP_PLUGIN")" "$work/anp/librccl-anp.so"
  LD_LIBRARY_PATH=$work/anp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export NCCL_NET_PLUGIN=librccl-anp.so
fi
export LD_LIBRARY_PATH
mkdir "$work/rccl" "$work/triton"
export NCCL_PROTO=Simple HSA_NO_SCRATCH_RECLAIM=1 XLA_PYTHON_CLIENT_PREALLOCATE=false TRITON_CACHE_DIR=$work/triton \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_DEBUG_FILE=$work/rccl/rccl.%h.%p.log
"$PY" "$(dirname "$0")/selftest.py" --rccl-log-dir "$work/rccl" "${args[@]}"
