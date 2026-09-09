#!/usr/bin/env bash
# Run probe_cudnn_moe_bwd.py inside an Iris GPU job with cuDNN 9.25 and cudnn-frontend's Python
# bindings installed after the sync. The frontend imports libcudnn by soname, so the venv's cuDNN
# goes first on the library path.
set -euo pipefail
PY="$(command -v python)"
uv pip install --python "$PY" "nvidia-cudnn-cu13==${CUDNN_VERSION:-9.25.1.1}" "nvidia-cudnn-frontend==${CUDNN_FE_VERSION:-1.28.0}" "nvidia-cublas==${CUBLAS_VERSION:-13.6.1.10}" 2>&1 | tail -4
CUDNN_LIB="$(python -c 'import nvidia.cudnn, os; print(os.path.join(nvidia.cudnn.__path__[0], "lib"))')"
export LD_LIBRARY_PATH="$CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python lib/levanter/scripts/bench/probe_cudnn_moe_bwd.py "$@"
