#!/bin/bash
# Run probe_recompute.py under each XLA flag configuration in a fresh process (flags bind at
# backend init). Installs the self-built PJRT wheel + matching nightly siblings first, the same
# overlay the arms use, so the compiler under test matches the rack arms.
#
# usage (iris GPU job, 4x GB200): WHEEL=s3://.../x.whl bash probe_recompute_driver.sh
set -euo pipefail
WHEEL="${WHEEL:?set WHEEL (s3 url of the self-built PJRT wheel)}"

NIGHTLY="https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry"
PYBIN="$(command -v python)"

python - <<PY
import fsspec
fs, path = fsspec.core.url_to_fs("${WHEEL}")
fs.get(path, "/tmp/pjrt-selfbuilt.whl")
print("fetched /tmp/pjrt-selfbuilt.whl")
PY
uv pip install --python "$PYBIN" --reinstall \
  "jax @ ${NIGHTLY}/jax/jax-0.11.2.dev20260821-py3-none-any.whl" \
  "jaxlib @ ${NIGHTLY}/jaxlib/jaxlib-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "jax-cuda13-plugin[with-cuda] @ ${NIGHTLY}/jax-cuda13-plugin/jax_cuda13_plugin-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "jax-cuda13-pjrt @ ${NIGHTLY}/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.2.dev20260821-py3-none-manylinux_2_27_aarch64.whl"
uv pip install --python "$PYBIN" --no-deps --reinstall /tmp/pjrt-selfbuilt.whl

PROBE="$(dirname "${BASH_SOURCE[0]}")/probe_recompute.py"
declare -A CONFIGS=(
  [A-none]=""
  [B-dk-only]="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true"
  [C-scoped-only]="--xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"
  [D-dk-scoped]="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall"
  [E-dk-global]="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_gpu_experimental_enable_nccl_symmetric_buffers=true"
)
for name in A-none B-dk-only C-scoped-only D-dk-scoped E-dk-global; do
  echo "==== CONFIG ${name} ===="
  XLA_FLAGS="${CONFIGS[$name]}" python "$PROBE" || echo "CONFIG ${name} FAILED"
done
echo PROBE_DRIVER_OK
