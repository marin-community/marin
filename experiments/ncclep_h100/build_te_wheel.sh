#!/bin/bash
# Build the pinned Transformer Engine source once in the task-local workspace.

set -euo pipefail

PINNED_TE_SHA=4adad4c218c115cd9af235fb3d4e13ef4cec55a8
TE_SHA=${TE_SHA:-$PINNED_TE_SHA}
WORK=${WORK:-/tmp/ncclep-h100}
MAX_JOBS=${MAX_JOBS:-64}
NVTE_ENABLE_NCCL_EP_OVERFLOW_DROP_PATCH=${NVTE_ENABLE_NCCL_EP_OVERFLOW_DROP_PATCH:-0}
export MAX_JOBS

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OVERFLOW_DROP_PATCH="$SCRIPT_DIR/transformer_engine_jax_overflow_drop.patch"
OVERFLOW_DROP_VALIDATOR="$SCRIPT_DIR/validate_te_overflow_drop_patch.sh"

case "$NVTE_ENABLE_NCCL_EP_OVERFLOW_DROP_PATCH" in
  0 | 1) ;;
  *)
    echo "FATAL: NVTE_ENABLE_NCCL_EP_OVERFLOW_DROP_PATCH must be 0 or 1" >&2
    exit 64
    ;;
esac

if [[ -z "${CUDA_HOME:-}" || -z "${NCCL_HOME:-}" ]]; then
  echo "FATAL: source cuda_wheels_env.sh before build_te_wheel.sh" >&2
  exit 1
fi

TE_SOURCE="$WORK/transformer-engine"
WHEELHOUSE="$WORK/wheelhouse"
BUILD_LOG="$WORK/transformer-engine-build.log"

echo "=== build: install host build tools ==="
uv pip install pip cmake ninja "pybind11[global]" wheel setuptools packaging \
  "nvidia-cudnn-frontend>=1.25.0" flax
command -v cmake
command -v ninja
"$CUDA_HOME/bin/nvcc" --version

echo "=== build: fetch Transformer Engine @ $TE_SHA ==="
rm -rf "$TE_SOURCE" "$WHEELHOUSE"
mkdir -p "$TE_SOURCE" "$WHEELHOUSE"
git -C "$TE_SOURCE" init
git -C "$TE_SOURCE" remote add origin https://github.com/NVIDIA/TransformerEngine.git
git -C "$TE_SOURCE" fetch --depth 1 origin "$TE_SHA"
git -C "$TE_SOURCE" checkout --detach FETCH_HEAD
git -C "$TE_SOURCE" submodule update --init --recursive --depth 1 \
  || git -C "$TE_SOURCE" submodule update --init --recursive

if [[ "$NVTE_ENABLE_NCCL_EP_OVERFLOW_DROP_PATCH" == 1 ]]; then
  if [[ "$TE_SHA" != "$PINNED_TE_SHA" ]]; then
    echo "FATAL: overflow-drop patch requires TE $PINNED_TE_SHA, got $TE_SHA" >&2
    exit 1
  fi
  bash "$OVERFLOW_DROP_VALIDATOR" "$TE_SOURCE" --check
  git -C "$TE_SOURCE" apply --whitespace=error-all "$OVERFLOW_DROP_PATCH"
  bash "$OVERFLOW_DROP_VALIDATOR" "$TE_SOURCE" --patched
else
  echo "=== build: overflow-drop patch disabled; using pristine TE source ==="
fi

echo "=== build: JAX wheel with NCCL_EP for sm_90 ==="
export NVTE_FRAMEWORK=jax
export NVTE_CUDA_ARCHS=90
export NVTE_WITH_NCCL_EP=1

build_status=0
python -m pip wheel --no-build-isolation --no-deps -w "$WHEELHOUSE" "$TE_SOURCE" \
  >"$BUILD_LOG" 2>&1 || build_status=$?
if [[ "$build_status" -ne 0 ]]; then
  echo "FATAL: Transformer Engine wheel build failed (exit $build_status)" >&2
  grep -nE "FAILED:|fatal error|error:|Error limit|ptxas.*error|Killed" "$BUILD_LOG" \
    | grep -vE "nvcc -forward|/usr/bin/c\+\+ " | head -80 || true
  echo "=== final build log lines ===" >&2
  tail -80 "$BUILD_LOG" >&2
  exit "$build_status"
fi
tail -10 "$BUILD_LOG"

mapfile -t wheels < <(find "$WHEELHOUSE" -maxdepth 1 -name 'transformer_engine*.whl' -type f | sort)
if [[ "${#wheels[@]}" -ne 1 ]]; then
  echo "FATAL: expected one Transformer Engine wheel, found ${#wheels[@]}" >&2
  printf '  %s\n' "${wheels[@]:-}" >&2
  exit 1
fi
case "${wheels[0]}" in
  *x86_64.whl) ;;
  *)
    echo "FATAL: expected an x86_64 wheel, got ${wheels[0]}" >&2
    exit 1
    ;;
esac
sha256sum "${wheels[0]}"

JIT_INCLUDE="$TE_SOURCE/3rdparty/nccl-extensions/build/include"
if [[ ! -d "$JIT_INCLUDE/nccl_ep" ]]; then
  echo "FATAL: NCCL_EP JIT headers missing at $JIT_INCLUDE" >&2
  exit 1
fi

echo "=== build: install and probe local wheel ==="
uv pip install --force-reinstall --no-deps "${wheels[0]}"
python - <<'PY'
from transformer_engine.jax import ep

required = {"ep_bootstrap", "ep_dispatch", "ep_combine"}
missing = required.difference(dir(ep))
if missing:
    raise RuntimeError(f"Transformer Engine wheel lacks NCCL_EP symbols: {sorted(missing)}")
print("Transformer Engine NCCL_EP import probe passed")
PY

TE_CORE=$(find "$(python -c 'import os, transformer_engine; print(os.path.dirname(transformer_engine.__file__))')" \
  -name 'libtransformer_engine.so*' | head -1)
if [[ -z "$TE_CORE" ]]; then
  echo "FATAL: libtransformer_engine.so not found after wheel install" >&2
  exit 1
fi
if readelf -d "$TE_CORE" | grep NEEDED | grep -q '\.so\.12'; then
  echo "FATAL: Transformer Engine core links a CUDA 12 library" >&2
  readelf -d "$TE_CORE" | grep NEEDED >&2
  exit 1
fi

printf '%s\n' "${wheels[0]}" >"$WORK/te-wheel-path"
printf '%s\n' "$JIT_INCLUDE" >"$WORK/nccl-ep-jit-include"
echo "=== build complete ==="
