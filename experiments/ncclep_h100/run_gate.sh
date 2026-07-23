#!/bin/bash
# Build TE locally once, then launch the fixed H100x8 NCCL_EP gate.

set -euo pipefail

TE_SHA=4adad4c218c115cd9af235fb3d4e13ef4cec55a8
NCCL_RUNTIME_VERSION=2.30.7
WORK=${WORK:-/tmp/ncclep-h100}
WARMUP=${WARMUP:-8}
ITERATIONS=${ITERATIONS:-30}
XLA_PREALLOC_FRACTION=${XLA_PREALLOC_FRACTION:-0.65}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage: bash experiments/ncclep_h100/run_gate.sh [--dry-run]

Environment overrides:
  WORK                     task-local build directory (/tmp/ncclep-h100)
  WARMUP                   warmup calls per benchmark (8)
  ITERATIONS               timed calls per benchmark (30)
  XLA_PREALLOC_FRACTION    XLA allocation fraction, must be <= 0.70 (0.65)
  NCCL_DEBUG               NCCL log level (WARN)

The transport shape, TE SHA, NCCL floor, command-buffer setting, and decision
threshold are fixed by the gate.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  cat <<EOF
NCCL_EP H100x8 gate dry run
  repository: $REPO_ROOT
  work dir: $WORK
  Transformer Engine: $TE_SHA
  build arch: 90
  NCCL runtime: $NCCL_RUNTIME_VERSION
  topology: one Iris task, H100x8, 8 processes x 1 GPU
  shape: EP8, 16384 tokens/rank, hidden 2560, top-k4, 64 experts, BF16, uniform
  timing: $WARMUP warmup, $ITERATIONS measured
  command buffers: disabled
  XLA preallocation fraction: $XLA_PREALLOC_FRACTION
  decision: fwd+bwd median <= 18.33144 ms
            (unpaired hard sanity bound; not an apples-to-apples ring comparison)

Runtime phases:
  1. Assemble a CUDA 13 + NCCL $NCCL_RUNTIME_VERSION toolchain from PyPI wheels.
  2. Build and install TE $TE_SHA once at NVTE_CUDA_ARCHS=90 with NCCL_EP.
  3. Keep TE's generated NCCL_EP JIT headers in the task-local source tree.
  4. Launch:
     python -m iris.runtime.multigpu --nproc 8 --devices-per-proc 1 -- \\
       python -u $SCRIPT_DIR/ep_transport_gate.py \\
       --warmup $WARMUP --iterations $ITERATIONS
  5. Emit one rank-0 JSON summary and exit 2 when the latency gate fails.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 64
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "FATAL: this gate requires x86_64, got $(uname -m)" >&2
  exit 1
fi
python - "$XLA_PREALLOC_FRACTION" <<'PY'
import sys

fraction = float(sys.argv[1])
if not 0.0 < fraction <= 0.70:
    raise SystemExit(f"XLA_PREALLOC_FRACTION must be in (0, 0.70], got {fraction}")
PY

mkdir -p "$WORK"
cd "$REPO_ROOT"

echo "=== setup: CUDA 13 runtime and JIT toolchain ==="
# shellcheck source=experiments/ncclep_h100/cuda_wheels_env.sh
source "$SCRIPT_DIR/cuda_wheels_env.sh"

echo "=== preflight: one H100x8 node ==="
mapfile -t gpu_rows < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)
if [[ "${#gpu_rows[@]}" -ne 8 ]]; then
  echo "FATAL: expected 8 GPUs, found ${#gpu_rows[@]}" >&2
  printf '  %s\n' "${gpu_rows[@]:-}" >&2
  exit 1
fi
for row in "${gpu_rows[@]}"; do
  if [[ "$row" != *H100* || "$row" != *9.0* ]]; then
    echo "FATAL: expected H100 compute capability 9.0, got '$row'" >&2
    exit 1
  fi
done
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
  echo "FATAL: active NVLink links were not detected" >&2
  exit 1
fi
command -v nvcc
command -v ptxas

NCCL_VERSION=$(python - <<'PY'
import ctypes

library = ctypes.CDLL("libnccl.so.2")
version = ctypes.c_int()
status = library.ncclGetVersion(ctypes.byref(version))
if status != 0:
    raise SystemExit(f"ncclGetVersion failed with status {status}")
if version.value < 23004:
    raise SystemExit(f"NCCL >= 2.30.4 required, got integer version {version.value}")
print(version.value)
PY
)
echo "runtime NCCL integer version: $NCCL_VERSION"

echo "=== build: task-local Transformer Engine wheel ==="
bash "$SCRIPT_DIR/build_te_wheel.sh"

JIT_INCLUDE=$(<"$WORK/nccl-ep-jit-include")
if [[ ! -d "$JIT_INCLUDE/nccl_ep" ]]; then
  echo "FATAL: generated NCCL_EP JIT source directory missing: $JIT_INCLUDE/nccl_ep" >&2
  exit 1
fi

export NCCL_EP_JIT_SOURCE_DIR="$JIT_INCLUDE/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$JIT_INCLUDE"
export NCCL_EP_JIT_CUDA_INCLUDE_DIR="$CUDA_HOME/include"
export NCCL_EP_JIT_CACHE_DIR="$WORK/nccl-ep-jit-cache"
export NCCL_EP_JIT_LOG=${NCCL_EP_JIT_LOG:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=1
export NVTE_EP_HANDLE_CACHE_SIZE=-1
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_PREALLOC_FRACTION"
export NCCLEP_NCCL_RUNTIME_VERSION="$NCCL_VERSION"
export NCCLEP_TE_SHA="$TE_SHA"
mkdir -p "$NCCL_EP_JIT_CACHE_DIR"

echo "=== run: eight supervised one-GPU processes ==="
exec python -m iris.runtime.multigpu --nproc 8 --devices-per-proc 1 -- \
  python -u "$SCRIPT_DIR/ep_transport_gate.py" \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
