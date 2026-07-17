#!/bin/bash
# NCCL_EP transport microbench under an iris gang, one process per GPU
# (issue #7331, NCCLEP-004/-005). Per-task setup runs once, then the iris
# multigpu supervisor spawns 4 children per node (supervised jax_init joins the
# global mesh via the endpoint registry).
#
# 1 node, EP4:
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 32 --memory 256g \
#     --extra gpu --timeout 3600 --job-name ncclep-mb-ep4 -- \
#     bash experiments/ncclep/run_microbench_gang.sh --ep 4 --backward
# 2 nodes, EP8 (first cross-node NCCL_EP):
#   ... --replicas 2 --job-name ncclep-mb-ep8 -- \
#     bash experiments/ncclep/run_microbench_gang.sh --ep 8 --backward
set -euxo pipefail

STASH=${STASH:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep}
WORK=${WORK:-/tmp/ncclep-mb}
BENCH_ARGS=("$@")
mkdir -p "$WORK"
REPO_ROOT=$(pwd)

pushd "$WORK"
source "$REPO_ROOT/experiments/ncclep/cuda_wheels_env.sh"

uv pip install s3fs
python - "$STASH" <<'EOF'
import os, sys
import s3fs
fs = s3fs.S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
stash = sys.argv[1].rstrip("/")
whls = sorted(fs.glob(stash + "/wheels/*.whl"))
assert whls, f"no wheels under {stash}/wheels/"
fs.get(whls[-1], os.path.basename(whls[-1]))
print("fetched", whls[-1])
fs.get(stash + "/jit/nccl-ep-jit-headers.tgz", "nccl-ep-jit-headers.tgz")
EOF
uv pip install ./transformer_engine*.whl
mkdir -p jit-include && tar -C jit-include -xzf nccl-ep-jit-headers.tgz
export NCCL_EP_JIT_SOURCE_DIR="$WORK/jit-include/nccl_ep"
export NCCL_EP_JIT_BUILD_INCLUDE_DIR="$WORK/jit-include"
export NCCL_EP_JIT_LOG=1
popd

export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_graph_min_graph_size=1"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

exec python -m iris.runtime.multigpu --nproc 4 -- \
  python "$REPO_ROOT/experiments/ncclep/ep_transport_microbench.py" "${BENCH_ARGS[@]}"
