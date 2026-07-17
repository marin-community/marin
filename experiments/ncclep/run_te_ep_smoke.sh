#!/bin/bash
# Single-node NCCL_EP smoke: run TE's multi-process EP test suite (13 tests,
# dp2 x ep2 over 4 GPUs, one process per GPU) on a GB200 node (issue #7331,
# NCCLEP-003).
#
# Installs the stashed TE wheel (built by build_te_wheel.sh), upgrades runtime
# NCCL to the EP minimum, fetches the matching TE test file, and launches it
# with a localhost coordinator — TE's own launch shape, minus the PYTHONPATH
# override (which would shadow the installed wheel with the sourceless tree).
#
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 32 --memory 128g \
#     --extra gpu --timeout 3600 --job-name ncclep-smoke -- bash experiments/ncclep/run_te_ep_smoke.sh
set -euxo pipefail

TE_SHA=${TE_SHA:-68493d2d55ac37e540301467b278bdb1c2019e81}
NCCL_RUNTIME_VERSION=${NCCL_RUNTIME_VERSION:-2.30.7}
WHEEL_SRC=${WHEEL_SRC:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep/wheels/}
WORK=${WORK:-/tmp/ncclep-smoke}
TEST_TIMEOUT_S=${TEST_TIMEOUT_S:-300}
mkdir -p "$WORK"
cd "$WORK"

echo "=== fetch + install TE wheel ==="
uv pip install s3fs
python - "$WHEEL_SRC" <<'EOF'
import os, sys
import s3fs
fs = s3fs.S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
whls = sorted(fs.glob(sys.argv[1].rstrip("/") + "/*.whl"))
assert whls, f"no wheels under {sys.argv[1]}"
src = whls[-1]
dst = os.path.basename(src)
fs.get(src, dst)
print("fetched", src, "->", dst)
EOF
uv pip install ./transformer_engine*.whl "nvidia-nccl-cu13==${NCCL_RUNTIME_VERSION}"
uv pip install flax || true

SP=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')
NCCL_LIB_DIR=$(dirname "$(find "$SP" -name 'libnccl.so.2' | head -1)")
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -c "import ctypes; l=ctypes.CDLL('libnccl.so.2'); v=ctypes.c_int(); l.ncclGetVersion(ctypes.byref(v)); print('runtime nccl', v.value); assert v.value >= 23004"

echo "=== fetch TE test file @ $TE_SHA ==="
curl -fsSL "https://raw.githubusercontent.com/NVIDIA/TransformerEngine/${TE_SHA}/tests/jax/test_multi_process_ep.py" -o test_multi_process_ep.py

echo "=== launch 4-proc EP test (localhost coordinator) ==="
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_graph_min_graph_size=1"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NUM_RANKS=4
for ((i=1; i<NUM_RANKS; i++)); do
  timeout --foreground --signal=KILL "$TEST_TIMEOUT_S" \
    python test_multi_process_ep.py 127.0.0.1:12975 "$i" "$NUM_RANKS" > "rank_${i}.log" 2>&1 &
done
rc=0
timeout --foreground --signal=KILL "$TEST_TIMEOUT_S" \
  python test_multi_process_ep.py 127.0.0.1:12975 0 "$NUM_RANKS" 2>&1 | tee rank_0.log || rc=$?
wait || true

echo "=== rank logs ==="
for ((i=1; i<NUM_RANKS; i++)); do echo "--- rank $i ---"; tail -30 "rank_${i}.log" || true; done

grep -qE "^OK$|OK \(" rank_0.log || { echo "SMOKE FAILED (rank 0 summary missing/failed, rc=$rc)"; exit 1; }
echo "=== SMOKE PASSED ==="
