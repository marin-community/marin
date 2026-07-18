#!/bin/bash
# Standalone MFU bench with the TE NCCL_EP backend under an iris gang, one
# process per GPU (issue #7331, NCCLEP-005/-006). Per-task setup (TE wheel +
# JIT toolchain) runs once per node, then the iris multigpu supervisor spawns
# 4 children; supervised jax_init joins the global mesh.
#
# Single-node EP4 smoke:
#   iris --cluster=marin job run --user mwittmann --target-cluster cw-us-east-08a \
#     --gpu GB200x4 --enable-extra-resources --cpu 32 --memory 256g \
#     --extra gpu --timeout 3600 --job-name ncclep-e2e-smoke -- \
#     bash experiments/ncclep/run_bench_gang.sh --run-id ep4-smoke --output-dir /tmp/out \
#       --moe-implementation nccl_ep --expert-parallelism 4 --num-gpus 4 --steps 6 --num-layers 4
# 64-GPU EP8 reference config: --replicas 16, --num-gpus 64, d5120 L48 b1024.
set -euxo pipefail

STASH=${STASH:-s3://marin-us-east-02a/marin/scratch/mwittmann/ncclep}
WORK=${WORK:-/tmp/ncclep-e2e}
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
export NCCL_EP_JIT_LOG=${NCCL_EP_JIT_LOG:-0}
popd

export XLA_PYTHON_CLIENT_ALLOCATOR=${XLA_PYTHON_CLIENT_ALLOCATOR:-cuda_async}
# No-drop EP recv buffers are large (ep x tokens x topk rows); default 0.75
# fraction leaves 48 GiB idle on a 186 GiB part.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# TE's EP integration disables command-buffer capture around the EP FFI ops;
# with capture on, XLA can run an EP op's host-side handle lookup before
# ep_prepare's cache insert (lookup_handle assertion). Unlimited handle cache
# is TE's own documented JAX workaround for handle_mem relocation.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}"
export NVTE_EP_HANDLE_CACHE_SIZE=-1
# Shared compile cache: retries below (and warm reruns) skip the ~15-min cold
# compile, isolating the intermittent first-execution CUBIN-load bug
# (B200MFU-036: no known flag mitigation; within-allocation passes are
# stochastic, so in-job sequential retries are the play).
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jax-compile-cache}

# uv's cached marin-levanter wheel goes stale when new source files are added
# without a version bump — shadow it with the bundled tree (stale-import
# lesson from B200MFU-014).
export PYTHONPATH="$REPO_ROOT/lib/levanter/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import levanter.grug._moe.ep_nccl as m; print('ep_nccl from', m.__file__)"

# Sequential in-job attempts against the intermittent CUBIN-load failure
# (B200MFU-036): every task loops in lockstep (gang-wide abort -> every task's
# attempt N fails -> all re-rendezvous at jax distributed init for N+1).
ATTEMPTS=${NCCLEP_BENCH_ATTEMPTS:-1}
rc=0
for attempt in $(seq 1 "$ATTEMPTS"); do
  echo "[bench attempt $attempt/$ATTEMPTS]"
  rc=0
  python -m iris.runtime.multigpu --nproc 4 -- \
    python "$REPO_ROOT/experiments/grug/moe/standalone/grug_moe_mfu.py" "${BENCH_ARGS[@]}" && break
  rc=$?
  echo "[bench attempt $attempt failed rc=$rc]"
  sleep 15
done
exit "$rc"
