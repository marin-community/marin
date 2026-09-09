#!/usr/bin/env bash
# Run bench_ragged_expert_gemms.py inside an Iris GPU job, after upgrading cuDNN in the job's
# venv so XLA's cuDNN ragged-dot fusion (cuDNN >= 9.22) can engage. The lock still pins 9.19.
set -euo pipefail
PY="$(command -v python)"
echo "python=$PY"
# cuDNN's MoE grouped matmul backward is a cuBLASLt grouped GEMM that needs cuBLASLt >= 13.5
# (cudnn-frontend docs/operations/MoeGroupedMatmul.md); the lock carries 13.4.1.
uv pip install --python "$PY" "nvidia-cudnn-cu13==${CUDNN_VERSION:-9.25.1.1}" "nvidia-cublas==${CUBLAS_VERSION:-13.6.1.10}" 2>&1 | tail -4
if [ -n "${PJRT_WHEEL:-}" ]; then
  # A rebuilt plugin (URL or path). --no-deps: the stock siblings already satisfy its pins.
  uv pip install --python "$PY" --no-deps --reinstall "$PJRT_WHEEL" 2>&1 | tail -2
fi
# jax loads cuDNN by soname. Put the venv's CUDA 13 cuDNN first so the upgrade is the copy that
# maps, and print what actually mapped, with its runtime version, to prove which one engaged.
CUDNN_LIB="$(python -c 'import nvidia.cudnn, os; print(os.path.join(nvidia.cudnn.__path__[0], "lib"))')"
ls -la "$CUDNN_LIB" | head -5
export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<'PY'
import ctypes, importlib.metadata as m, re, jax
print("cudnn pkg", m.version("nvidia-cudnn-cu13"), "cublas pkg", m.version("nvidia-cublas"), "jax", jax.__version__, "pjrt", m.version("jax-cuda13-pjrt"))
print(jax.devices())
jax.numpy.ones(4).block_until_ready()
maps = open("/proc/self/maps").read()
paths = sorted({line.split()[-1] for line in maps.splitlines() if "cudnn" in line or "cublas" in line})
print("mapped cudnn:", paths)
for path in paths:
    if path.endswith("libcudnn.so.9") or re.search(r"libcudnn\.so\.9(\.\d+)*$", path):
        lib = ctypes.CDLL(path)
        lib.cudnnGetVersion.restype = ctypes.c_size_t
        print("cudnnGetVersion", path, lib.cudnnGetVersion())
PY
mkdir -p "${IRIS_OUTPUT_DIR:-/tmp/out}"
# One process per lowering: an abort inside XLA (a C++ CHECK, a std::out_of_range) would otherwise
# take every later variant with it. VARIANTS is comma-separated; the rest of "$@" passes through.
IFS=, read -r -a variants <<<"${VARIANTS:-quack,xla,xla-cudnn,xla-triton,xla-cudnn-triton,haliax,dense}"
for v in "${variants[@]}"; do
  python lib/levanter/scripts/bench/bench_ragged_expert_gemms.py --variants "$v" \
    --json "${IRIS_OUTPUT_DIR:-/tmp/out}/results-$v.json" "$@" || echo "== $v: process exited $?"
done
