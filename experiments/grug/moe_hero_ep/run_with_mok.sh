#!/usr/bin/env bash
# Install the bundled MoK wheel into the task venv and run a module under the multi-GPU supervisor.
#
# Same shape as profile_pair.sh: the wheel ships inside the Iris bundle at /app, the task venv gets
# it with --no-deps (its declared torch pin is already satisfied by the gpu extra), and each of the
# four children owns one GB200. Arguments after the script are the child's argv following `-m`.
set -euo pipefail

WHEEL=${MOK_WHEEL:-/app/experiments/grug/moe_hero_ep/mixture_of_kittens-0.1.0-cp312-cp312-linux_aarch64.whl}
NPROC=${MOK_NPROC:-4}
PY=${IRIS_PYTHON:?IRIS_PYTHON must be set by the Iris task}

echo "=== environment ==="
echo "host: $(hostname)"
echo "interpreter: $PY"
"$PY" -V
uname -m
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"

echo "=== install $WHEEL ==="
ls -l "$WHEEL"
sha256sum "$WHEEL"
/bin/uv pip install --python "$PY" --link-mode symlink --no-deps "$WHEEL"

# `require_mok_available` shells out to nvcc, which the gpu extra ships as a wheel rather than on
# PATH. Resolve it once here so every child inherits it.
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

echo "=== run: $* ==="
exec "$PY" -m iris.hooks.multigpu_main --nproc "$NPROC" --devices-per-proc 1 -- "$PY" "$@"
