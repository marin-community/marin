#!/bin/bash
# Run ragged_a2a_bench.py under each XLA flag config with the given PJRT wheel overlay.
# usage (iris GPU job): WHEEL=s3://...whl CONFIGS="oneshot dk" bash bench_driver.sh
set -euo pipefail
WHEEL="${WHEEL:?}"
CONFIGS="${CONFIGS:?space-separated subset of: oneshot dk}"

NIGHTLY="https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry"
PYBIN="$(command -v python)"
mkdir -p /tmp/pjrt-wheel && rm -f /tmp/pjrt-wheel/*.whl
python - <<PY
import fsspec
fs, path = fsspec.core.url_to_fs("${WHEEL}")
dest = "/tmp/pjrt-wheel/" + path.rsplit("/", 1)[1]
fs.get(path, dest)
print("fetched", dest)
PY
uv pip install --python "$PYBIN" --reinstall \
  "jax @ ${NIGHTLY}/jax/jax-0.11.2.dev20260821-py3-none-any.whl" \
  "jaxlib @ ${NIGHTLY}/jaxlib/jaxlib-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "jax-cuda13-plugin[with-cuda] @ ${NIGHTLY}/jax-cuda13-plugin/jax_cuda13_plugin-0.11.2.dev20260821-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "jax-cuda13-pjrt @ ${NIGHTLY}/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.2.dev20260821-py3-none-manylinux_2_27_aarch64.whl"
uv pip install --python "$PYBIN" --no-deps --reinstall /tmp/pjrt-wheel/*.whl

BENCH="$(dirname "${BASH_SOURCE[0]}")/ragged_a2a_bench.py"
for cfg in $CONFIGS; do
  echo "==== BENCH ${cfg} ===="
  case "$cfg" in
    oneshot) FLAGS="" ;;
    dk) FLAGS="--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true --xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall" ;;
    *) echo "unknown config $cfg"; exit 1 ;;
  esac
  XLA_FLAGS="$FLAGS" python "$BENCH" || echo "BENCH ${cfg} FAILED"
done
echo BENCH_DRIVER_OK
