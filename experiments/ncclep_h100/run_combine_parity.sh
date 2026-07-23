#!/bin/bash
# Build pinned TE, then isolate NCCL_EP dispatch/combine BF16 numerics.

set -euo pipefail

TE_SHA=4adad4c218c115cd9af235fb3d4e13ef4cec55a8
NCCL_RUNTIME_VERSION=2.30.7
WORK=${WORK:-/tmp/ncclep-h100-combine-parity}
XLA_PREALLOC_FRACTION=${XLA_PREALLOC_FRACTION:-0.65}
export WORK

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
  cat <<'EOF'
Usage: bash experiments/ncclep_h100/run_combine_parity.sh [--dry-run]

Environment overrides:
  WORK                     task-local build directory (/tmp/ncclep-h100-combine-parity)
  XLA_PREALLOC_FRACTION    XLA allocation fraction, must be <= 0.70 (0.65)
  NCCL_DEBUG               NCCL log level (WARN)

The topology, shape, seed, cases, TE SHA, NCCL version, and strict BF16
tolerances are fixed by the experiment.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  cat <<EOF
NCCL_EP BF16 dispatch/combine numerical isolation dry run
  repository: $REPO_ROOT
  work dir: $WORK
  Transformer Engine: $TE_SHA
  build arch: 90
  NCCL runtime: $NCCL_RUNTIME_VERSION
  topology: one Iris task, H100x8, 8 processes x 1 GPU
  fixed shape: EP8, 16384 tokens/rank, d2560, e64, BF16
  cases: top-k1/top-k4 identity and per-expert-scaled identity
  route weights: exact binary fractions; top-k4 scaled contributions are distinct
  dispatch checks: exact expert counts, token-bit sums, weighted token-bit sums
  output references: FP32 accumulation, forward-route BF16, reverse-route BF16
  output metrics: rtol=0.1, atol=0.0002, rel-L2, max/mean absolute error, BF16 ULP histogram
  FP32 probe: attempt FP32 combine input and label unsupported TE behavior cleanly
  promotion decision: none
  XLA preallocation fraction: $XLA_PREALLOC_FRACTION

Runtime phases:
  1. Assemble the CUDA 13 + NCCL $NCCL_RUNTIME_VERSION toolchain.
  2. Build and install one task-local TE $TE_SHA wheel at NVTE_CUDA_ARCHS=90.
  3. Launch eight supervised one-GPU processes.
  4. Compile and run all fixed numerical cases in one process group.
  5. Emit one structured rank-0 JSON result.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  usage >&2
  exit 64
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "FATAL: this experiment requires x86_64, got $(uname -m)" >&2
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

echo "=== run: fixed dispatch/combine numerical cases on eight one-GPU processes ==="
exec python -m iris.runtime.multigpu --nproc 8 --devices-per-proc 1 -- \
  python -u "$SCRIPT_DIR/ep_combine_parity.py"
