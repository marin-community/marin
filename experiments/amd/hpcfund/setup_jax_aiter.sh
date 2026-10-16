#!/bin/bash
# Build $JAX_AITER_HOME/venv (this repository, JAX 0.11.0 on TheRock 7.14) and install JAX-AITER into it.
# Runs inside the container through run_jax_aiter.sh, from the repository root. The container's system Python
# carries TheRock's ROCm packages, so the venv sees its site-packages.
set -euxo pipefail

: "${JAX_AITER_HOME:?run through run_jax_aiter.sh}"
JAX_AITER_COMMIT=c290cdd5437936b702402a7dca7ef52b0acad624
# Prebuilt MHA libraries for that commit's AITER pin (a3158e4). The commit bumped the pin without updating
# jit_assets.CACHE_ID, so jax-aiter-fetch-mha's default tag names the previous pin's release and fails validation.
JIT_LIBS_TAG=jit-libs-26a4ef9da0d91247
VENV=$JAX_AITER_HOME/venv
SRC=$JAX_AITER_HOME/jax-aiter

uv venv --allow-existing --system-site-packages --python /usr/bin/python3.12 "$VENV"
UV_PROJECT_ENVIRONMENT=$VENV uv sync --frozen
uv pip install --python "$VENV" \
    "jax==0.11.0" "jaxlib==0.11.0" "jax-rocm7-plugin==0.11.0.post1" "jax-rocm7-pjrt==0.11.0.post1" \
    cmake ninja pyyaml psutil pandas
# The CUDA plugin from uv sync would compete with the ROCm one for the GPU backend.
uv pip uninstall --python "$VENV" jax-cuda13-plugin jax-cuda13-pjrt || true

if [ ! -d "$SRC" ]; then
    git clone https://github.com/ROCm/jax-aiter "$SRC"
fi
git -C "$SRC" fetch origin "$JAX_AITER_COMMIT"
git -C "$SRC" checkout "$JAX_AITER_COMMIT"
git -C "$SRC" submodule update --init --recursive

# The prebuilt MHA libraries for this commit are fetched below, so build only the FFI shims, not the JIT libraries.
(
    cd "$SRC"
    export JA_ROOT_DIR=$SRC GPU_ARCHS=gfx950 AITER_SYMBOL_VISIBLE=1
    make -j"$(nproc)"
    make ja_mods -j"$(nproc)"
    uv pip install --python "$VENV" --no-build-isolation --no-deps .
)
"$VENV/bin/jax-aiter-fetch-mha" --tag "$JIT_LIBS_TAG"
"$VENV/bin/python" -c "import jax; from jax_aiter.mha import flash_attn_func; print(jax.__version__, jax.devices())"
