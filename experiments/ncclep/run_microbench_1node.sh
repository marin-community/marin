#!/bin/bash
# Single-node (4-GPU, dp1 x ep4) NCCL_EP transport microbench launcher
# (issue #7331, NCCLEP-004). Installs the stashed TE wheel, then runs
# ep_transport_microbench.py one-process-per-GPU with a localhost coordinator.
#
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 32 --memory 256g \
#     --extra gpu --timeout 3600 --job-name ncclep-microbench -- \
#     bash experiments/ncclep/run_microbench_1node.sh --ep 4 --backward
set -euxo pipefail

NCCL_RUNTIME_VERSION=${NCCL_RUNTIME_VERSION:-2.30.7}
WHEEL_SRC=${WHEEL_SRC:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep/wheels/}
WORK=${WORK:-/tmp/ncclep-microbench}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-900}
BENCH_ARGS=("$@")
mkdir -p "$WORK"
REPO_ROOT=$(pwd)
cd "$WORK"

uv pip install s3fs
python - "$WHEEL_SRC" <<'EOF'
import os, sys
import s3fs
fs = s3fs.S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
whls = sorted(fs.glob(sys.argv[1].rstrip("/") + "/*.whl"))
assert whls, f"no wheels under {sys.argv[1]}"
fs.get(whls[-1], os.path.basename(whls[-1]))
print("fetched", whls[-1])
EOF
uv pip install ./transformer_engine*.whl "nvidia-nccl-cu13==${NCCL_RUNTIME_VERSION}"
uv pip install flax || true

SP=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))')
NCCL_LIB_DIR=$(dirname "$(find "$SP" -name 'libnccl.so.2' | head -1)")
export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_graph_min_graph_size=1"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

BENCH="$REPO_ROOT/experiments/ncclep/ep_transport_microbench.py"
NUM_RANKS=4
for ((i=1; i<NUM_RANKS; i++)); do
  timeout --foreground --signal=KILL "$RUN_TIMEOUT_S" \
    python "$BENCH" 127.0.0.1:12975 "$i" "$NUM_RANKS" "${BENCH_ARGS[@]}" > "rank_${i}.log" 2>&1 &
done
rc=0
timeout --foreground --signal=KILL "$RUN_TIMEOUT_S" \
  python "$BENCH" 127.0.0.1:12975 0 "$NUM_RANKS" "${BENCH_ARGS[@]}" 2>&1 | tee rank_0.log || rc=$?
wait || true

for ((i=1; i<NUM_RANKS; i++)); do echo "--- rank $i tail ---"; tail -15 "rank_${i}.log" || true; done
grep -q "MICROBENCH DONE" rank_0.log || { echo "MICROBENCH FAILED rc=$rc"; exit 1; }
echo "=== MICROBENCH JOB DONE ==="
