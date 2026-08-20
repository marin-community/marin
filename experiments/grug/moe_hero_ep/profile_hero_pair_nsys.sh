#!/usr/bin/env bash
# Nsight Systems head-to-head capture at the hero LatentMoE shape, both arms on ONE rack.
#
# Separate job from the XProf pass, per the r8 CUPTI lesson. XProf is still *enabled* here,
# but only as the capture-range trigger: nsys is run with `--capture-range=cudaProfilerApi
# --capture-range-end=stop`, and the thing that calls cuProfilerStart/Stop is XLA's GPU
# profiler when `jax.profiler.start_trace`/`stop_trace` run. Without it nsys would either
# trace the whole 95-step run (hundreds of GB) or collect nothing. The XProf artifacts this
# job uploads are the known-bad host-only kind and are NOT used; the XProf pass owns that.
#
# Only global rank 0 profiles (`--tasks first` selects IRIS_MULTIGPU_PROCESS_INDEX == 0,
# which is task 0 / local rank 0), and XProf's default process_index=0 matches it, so
# exactly one process carries both subscribers.
set -euo pipefail

MOK_WHEEL=/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl
RUN_TAG=${RUN_TAG:?RUN_TAG must be set}
PROFILE_ROOT=${PROFILE_ROOT:?PROFILE_ROOT must be set}
PY=${IRIS_PYTHON:?IRIS_PYTHON must be set by the Iris task}

echo "=== environment ==="
echo "host: $(hostname)"
echo "interpreter: $PY"
"$PY" -V
uname -m
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
nsys --version || echo "nsys unavailable"
echo "XLA_FLAGS=${XLA_FLAGS:-<unset>}"
echo "NCCL_DEBUG=${NCCL_DEBUG:-<unset>}"
echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-<unset>}"
echo "MARIN_PREFIX=${MARIN_PREFIX:-<unset>}"

echo "=== install $MOK_WHEEL ==="
ls -l "$MOK_WHEEL"
sha256sum "$MOK_WHEEL"
/bin/uv pip install --python "$PY" --link-mode symlink --no-deps "$MOK_WHEEL"

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
  local output_uri=$1
  shift
  "$PY" -u -m iris.hooks.multigpu_main \
    --nproc 4 \
    --devices-per-proc 1 \
    -- \
    bash -c 'exec "$IRIS_PYTHON" -m iris.hooks.nsys_main "$@"' iris-nsys \
    --tasks first \
    --capture-range \
    --output-uri "$output_uri" \
    -- \
    "$PY" -u -m experiments.grug.moe_hero_ep.dev_run_nsys "$@"
}

arm_status=0
echo "=== ARM A: mok (dropless fused megakernel) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
run_arm "$PROFILE_ROOT/mok" \
  --run-id "mhprofn-mok-${RUN_TAG}" \
  --backend mok \
  --num-steps 100 \
  --stop-after-steps 95 \
  --profile-start-step 90 \
  --profile-steps 5 \
  --mok-minibatch-size 8192 \
  --mok-expert-placement contiguous || { arm_status=1; echo "ARM A exited non-zero" >&2; }

echo "=== ARM B: fixed_pooled_wave_all_to_all (capacity 1.15) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
run_arm "$PROFILE_ROOT/wave" \
  --run-id "mhprofn-wave-${RUN_TAG}" \
  --backend fixed \
  --num-steps 100 \
  --stop-after-steps 95 \
  --profile-start-step 90 \
  --profile-steps 5 || { arm_status=1; echo "ARM B exited non-zero" >&2; }

echo "=== both arms complete (arm_status=$arm_status) ==="
date -u +"%Y-%m-%dT%H:%M:%SZ"
uv run --no-sync fsutil ls -l "$PROFILE_ROOT/mok/" || true
uv run --no-sync fsutil ls -l "$PROFILE_ROOT/wave/" || true
exit "$arm_status"
