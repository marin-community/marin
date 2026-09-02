#!/usr/bin/env bash
set -euo pipefail

: "${NCCL_TEST_VERSION:?set NCCL_TEST_VERSION}"
: "${IRIS_VENV:=$PWD/.venv}"
: "${IRIS_WORKDIR:=$PWD}"
JAX_VERSION=0.11.1.dev20260725
JAX_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
JAXPP_REVISION=7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9
JAX_TVM_FFI_REVISION=e238a28483123efc8f56b9de358c2fb8b8de77e5

uv pip install --python "$IRIS_VENV/bin/python" --link-mode=symlink --prerelease=allow --index "$JAX_INDEX" \
  "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" \
  "jax-cuda13-plugin[with-cuda]==$JAX_VERSION" \
  "jax-cuda13-pjrt==$JAX_VERSION" "nvidia-nccl-cu13==$NCCL_TEST_VERSION" \
  cupy-cuda13x

runtime_dir=$(mktemp -d)
trap 'rm -rf "$runtime_dir"' EXIT
jax_tvm_ffi_source="$runtime_dir/jax-tvm-ffi"
jaxpp_source="$runtime_dir/jaxpp"
git clone --quiet --filter=blob:none https://github.com/NVIDIA/jax-tvm-ffi.git "$jax_tvm_ffi_source"
git -C "$jax_tvm_ffi_source" checkout --quiet "$JAX_TVM_FFI_REVISION"
git -C "$jax_tvm_ffi_source" apply "$IRIS_WORKDIR/experiments/grug/moe/jax_tvm_ffi_multidevice.patch"
uv pip install --python "$IRIS_VENV/bin/python" --link-mode=symlink --force-reinstall --no-deps "$jax_tvm_ffi_source"

git clone --quiet --filter=blob:none https://github.com/NVIDIA/jaxpp.git "$jaxpp_source"
git -C "$jaxpp_source" checkout --quiet "$JAXPP_REVISION"
git -C "$jaxpp_source" apply --unidiff-zero "$IRIS_WORKDIR/experiments/grug/moe/jaxpp_jax_0_11_inline.patch"
uv pip install --python "$IRIS_VENV/bin/python" --link-mode=symlink --force-reinstall --no-deps "$jaxpp_source"
bash experiments/grug/moe/patch_cutlass_dsl_mlir_type_guard.sh "$IRIS_VENV/bin/python"

export JAXPP_SOURCE="$jaxpp_source"
export ISSUE_7655_REPRO_ONLY=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_gpu_ragged_all_to_all_mode=symmetric --xla_enable_nccl_symmetric_buffers_for_collectives=RaggedAllToAll --xla_gpu_nccl_termination_timeout_seconds=120"
set +e
timeout 180 "$IRIS_VENV/bin/python" -u \
  experiments/grug/moe/check_jaxpp_explicit_mpmd_std1f1b_ragged_parity.py
status=$?
set -e

if [[ $status -eq 0 ]]; then
  echo "NCCL $NCCL_TEST_VERSION: passed"
elif [[ $status -eq 124 ]]; then
  echo "NCCL $NCCL_TEST_VERSION: killed after 180s"
else
  echo "NCCL $NCCL_TEST_VERSION: failed with status $status"
fi
exit "$status"
