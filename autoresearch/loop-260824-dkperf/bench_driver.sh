#!/bin/bash
# Run ragged_a2a_bench.py (from loop-260823-dknative) with an arbitrary XLA_FLAGS string and a
# PJRT wheel overlay. One flag config per iris job: a second gang re-init in the same job grabs
# the dead coordinator endpoint.
#
# usage (iris GPU gang job): WHEEL=s3://...whl FLAGS="--xla_..." GANG=1 bash bench_driver.sh
set -euo pipefail
WHEEL="${WHEEL:?}"
FLAGS="${FLAGS:?full XLA_FLAGS string for this cell}"

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

BENCH_DIR="$(dirname "${BASH_SOURCE[0]}")"
echo "==== BENCH flags: ${FLAGS} ===="
if [ "${GANG:-0}" = "1" ]; then
  XLA_FLAGS="$FLAGS" python -m iris.hooks.multigpu_main --nproc 4 --devices-per-proc 1 -- \
    python "${BENCH_DIR}/bench_iris_entry.py" || echo "BENCH FAILED"
else
  XLA_FLAGS="$FLAGS" python "${BENCH_DIR}/ragged_a2a_bench.py" || echo "BENCH FAILED"
fi
echo BENCH_DRIVER_OK
