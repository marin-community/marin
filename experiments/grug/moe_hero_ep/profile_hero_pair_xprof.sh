#!/usr/bin/env bash
# XProf-only head-to-head capture at the hero LatentMoE shape, both arms on ONE rack.
#
# Both arms run sequentially inside a single Iris task set, under the same 16 GB200 nodes,
# so the rack-to-rack noise floor (~0.6-1%) that dominated the 2026-08-19 head-to-head
# cannot confound the comparison. Nsight is deliberately absent: Nsight and XProf both
# subscribe to CUPTI and Nsight wins, which in the 2026-08-08 r8 round produced XProf
# files with a host plane and no GPU device plane at all.
#
# Window: training steps 90..94 inclusive (`steps-90-to-95`), with the optimizer horizon
# kept at 100 so the LR schedule matches the head-to-head. MoK's step time only became
# statistically flat around step 85, so steps 80-84 (the previous round's window) sits
# inside MoK's still-converging regime.
set -euo pipefail

MOK_WHEEL=/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl
RUN_TAG=${RUN_TAG:?RUN_TAG must be set}
PY=${IRIS_PYTHON:?IRIS_PYTHON must be set by the Iris task}

echo "=== environment ==="
echo "host: $(hostname)"
echo "interpreter: $PY"
"$PY" -V
uname -m
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
echo "XLA_FLAGS=${XLA_FLAGS:-<unset>}"
echo "NCCL_DEBUG=${NCCL_DEBUG:-<unset>}"
echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-<unset>}"
echo "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-<unset>}"
echo "MARIN_PREFIX=${MARIN_PREFIX:-<unset>}"

echo "=== install $MOK_WHEEL ==="
ls -l "$MOK_WHEEL"
sha256sum "$MOK_WHEEL"
/bin/uv pip install --python "$PY" --link-mode symlink --no-deps "$MOK_WHEEL"

# `require_mok_available` shells out to nvcc, which the gpu extra ships as a wheel rather
# than on PATH. Resolve it once here so every child inherits it.
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if ! command -v nvcc >/dev/null 2>&1; then
  NVCC=$(find "$SITE/nvidia" -maxdepth 4 -name nvcc 2>/dev/null | head -1)
  if [ -n "${NVCC:-}" ]; then
    PATH="$(dirname "$NVCC"):$PATH"
    export PATH
  fi
fi
echo "nvcc: $(command -v nvcc || echo MISSING)"
"$PY" -c 'import mok, mok._C as native; print("mok", mok.__version__, "ffi abi", native.levanter_mok_ffi_abi_version())'

run_arm() {
  "$PY" -u -m iris.hooks.multigpu_main \
    --nproc 4 \
    --devices-per-proc 1 \
    -- \
    "$PY" -u -m experiments.grug.moe_hero_ep.dev_run "$@"
}

echo "=== ARM A: mok (dropless fused megakernel) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
run_arm \
  --run-id "mhprof-mok-${RUN_TAG}" \
  --backend mok \
  --num-steps 100 \
  --stop-after-steps 95 \
  --profile-start-step 90 \
  --profile-steps 5 \
  --profile-all-processes \
  --mok-minibatch-size 8192 \
  --mok-expert-placement contiguous

echo "=== ARM B: fixed_pooled_wave_all_to_all (capacity 1.15) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
run_arm \
  --run-id "mhprof-wave-${RUN_TAG}" \
  --backend fixed \
  --num-steps 100 \
  --stop-after-steps 95 \
  --profile-start-step 90 \
  --profile-steps 5 \
  --profile-all-processes

echo "=== both arms complete ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
