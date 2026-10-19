#!/usr/bin/env bash
set -euo pipefail

# CPython 3.12, Linux. TE supports x86_64 and aarch64; other recipes target x86_64.
package="${1:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}"
build_dir="$(realpath -m "${2:?usage: build_native.sh PACKAGE BUILD_DIRECTORY}")"
script_dir="$(cd "$(dirname "$0")" && pwd)"
source_subdir=.
python_version=3.12.14
architecture="$(uname -m)"
case "$architecture" in
    x86_64) ;;
    aarch64)
        if [[ "$package" != transformer-engine-torch ]]; then
            echo "Only transformer-engine-torch is qualified for aarch64 in this script" >&2
            exit 2
        fi
        ;;
    *) echo "unsupported build architecture: $architecture" >&2; exit 2 ;;
esac
max_jobs=2
package_environment=()
case "$package" in
    flash-attn)
        repository=Dao-AILab/flash-attention
        source_commit=060c9188beec3a8b62b33a3bfa6d5d2d44975fab
        package_environment+=(FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90)
        ;;
    causal-conv1d)
        repository=Dao-AILab/causal-conv1d
        source_commit=9e4ace0b1d53ede275308abf25f64a1fc04c5fd4
        package_environment+=(CAUSAL_CONV1D_FORCE_BUILD=TRUE)
        ;;
    mamba-ssm)
        repository=state-spaces/mamba
        source_commit=c5afbdf3bda1a09d68f65181ae3a43ec71079820
        package_environment+=(MAMBA_FORCE_BUILD=TRUE)
        ;;
    transformer-engine-torch)
        repository=NVIDIA/TransformerEngine
        source_commit=5e52befd5262c06289106338c308079d6adb391f
        source_subdir=transformer_engine/pytorch
        package_environment+=(NVTE_PYTORCH_FORCE_BUILD=TRUE NVTE_NO_LOCAL_VERSION=1 NVTE_BUILD_MAX_JOBS=1)
        max_jobs=1
        ;;
    *) echo "unsupported package: $package" >&2; exit 2 ;;
esac

mkdir -p "$build_dir"
uv venv --allow-existing --python "$python_version" "$build_dir/venv"
uv pip sync --python "$build_dir/venv/bin/python" \
    --link-mode copy \
    --extra-index-url https://download.pytorch.org/whl/cu132 \
    --index-strategy unsafe-best-match "$script_dir/native-cu132.txt"

if [[ ! -d "$build_dir/source" ]]; then
    git init --quiet "$build_dir/source"
    git -C "$build_dir/source" remote add origin "https://github.com/$repository.git"
    git -C "$build_dir/source" fetch --depth 1 origin "$source_commit"
    git -C "$build_dir/source" checkout --quiet --detach FETCH_HEAD
fi
test "$(git -C "$build_dir/source" rev-parse HEAD)" = "$source_commit"
if [[ -n "$(git -C "$build_dir/source" status --porcelain --untracked-files=all --ignore-submodules=none)" ]]; then
    echo "Build source must be clean: $build_dir/source" >&2
    exit 1
fi
git -C "$build_dir/source" submodule update --init --recursive
git -C "$build_dir/source" submodule foreach --quiet --recursive \
    'test -z "$(git status --porcelain --untracked-files=all --ignore-submodules=none)"'

virtual_env="$build_dir/venv"
site_packages="$virtual_env/lib/python${python_version%.*}/site-packages"
cuda_home="$site_packages/nvidia/cu13"
# NVIDIA's runtime wheel ships the SONAME but no development linker name.
ln -sf libcudart.so.13 "$cuda_home/lib/libcudart.so"
build_environment=(
    "VIRTUAL_ENV=$virtual_env"
    "CUDA_HOME=$cuda_home"
    "PATH=$virtual_env/bin:$cuda_home/bin:$PATH"
    "CPATH=$site_packages/nvidia/cudnn/include:$site_packages/nvidia/nccl/include"
    "NVCC_THREADS=1"
    "MAX_JOBS=$max_jobs"
    "${package_environment[@]}"
)
{
    printf 'package=%s\narchitecture=%s\nsource=%s\n' "$package" "$architecture" "$source_commit"
    sha256sum "$0" "$script_dir/native-cu132.txt"
    git -C "$build_dir/source" submodule status --recursive
    "$virtual_env/bin/python" --version
    uv --version
    "$cuda_home/bin/nvcc" --version
    c++ --version
    ldd --version
    "$virtual_env/bin/python" -c \
        'import torch; print("torch=" + torch.__version__); print("cxx11_abi=" + str(torch.compiled_with_cxx11_abi()))'
} > "$build_dir/BUILD_INFO"
uv pip freeze --python "$virtual_env/bin/python" > "$build_dir/BUILD_REQUIREMENTS.txt"
cat "$build_dir/BUILD_INFO"
env "${build_environment[@]}" uv build --wheel --no-build-isolation --python "$virtual_env/bin/python" \
    --out-dir "$build_dir/dist" "$build_dir/source/$source_subdir"
(cd "$build_dir/dist" && sha256sum -- *.whl) > "$build_dir/SHA256SUMS"
