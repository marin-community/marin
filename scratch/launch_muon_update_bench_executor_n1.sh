#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CLUSTER="${MUON_BENCH_CLUSTER:-cw-us-east-02a}"
DEFAULT_ENV_FILE="$HOME/.config/marin/marin-r2.env"
if [ ! -f "$DEFAULT_ENV_FILE" ] && [ -f "$HOME/.config/marin/cloudflare-r2.env" ]; then
    DEFAULT_ENV_FILE="$HOME/.config/marin/cloudflare-r2.env"
fi
ENV_FILE="${MARIN_R2_ENV_FILE:-$DEFAULT_ENV_FILE}"
KUBECONFIG_PATH="${KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"
RUN_ID="${RUN_ID:-MUON-BENCH-D2560-L2-D2E4-GROUPBOUNDARY-H1H3H5-N1-cw-$(date -u +%Y%m%d-%H%M)}"
MARIN_PREFIX="${MARIN_PREFIX:-s3://marin-na/tmp/ttl=7d}"

if [ -f "$ENV_FILE" ]; then
    R2_EXPORTS="$("${REPO_ROOT}/scripts/iris/cloudflare_r2_env.sh" "$ENV_FILE")"
else
    R2_EXPORTS="$("${REPO_ROOT}/scripts/iris/cloudflare_r2_env.sh")"
fi
eval "$R2_EXPORTS"
export KUBECONFIG="$KUBECONFIG_PATH"

cd "$REPO_ROOT"

exec uv run --package marin-iris --extra controller iris --cluster="$CLUSTER" \
    job run --no-wait \
    --memory=2G --disk=4G --cpu=1 --extra=cpu \
    -e MARIN_PREFIX "$MARIN_PREFIX" \
    -e RUN_ID "$RUN_ID" \
    -e AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID" \
    -e AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY" \
    -e AWS_ENDPOINT_URL "$AWS_ENDPOINT_URL" \
    -e AWS_ENDPOINT_URL_S3 "$AWS_ENDPOINT_URL_S3" \
    -e MUON_BENCH_LAYERS "${MUON_BENCH_LAYERS:-2}" \
    -e MUON_BENCH_NS4D_GROUP_SIZE "${MUON_BENCH_NS4D_GROUP_SIZE:-2}" \
    -e MUON_BENCH_NS4D_GROUP_AXIS "${MUON_BENCH_NS4D_GROUP_AXIS:-data}" \
    -e MUON_BENCH_REPLICA_AXIS "${MUON_BENCH_REPLICA_AXIS:-1}" \
    -e MUON_BENCH_DATA_AXIS "${MUON_BENCH_DATA_AXIS:-2}" \
    -e MUON_BENCH_EXPERT_AXIS "${MUON_BENCH_EXPERT_AXIS:-4}" \
    -e MUON_BENCH_MODEL_AXIS "${MUON_BENCH_MODEL_AXIS:-1}" \
    -e MUON_BENCH_DTYPE "${MUON_BENCH_DTYPE:-bf16}" \
    -e MUON_BENCH_NESTEROV "${MUON_BENCH_NESTEROV:-true}" \
    -e MUON_BENCH_NS_COMPUTE_DTYPE "${MUON_BENCH_NS_COMPUTE_DTYPE:-input}" \
    -e MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT "${MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT:-1}" \
    -e MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS "${MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS:-0}" \
    -e MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT "${MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT:-0}" \
    -e MUON_BENCH_ORTHOGONALIZATION_LAYOUT "${MUON_BENCH_ORTHOGONALIZATION_LAYOUT:-stack_batch_4d_sharded}" \
    -e MUON_BENCH_SWEEP_BACKEND_STEPS "${MUON_BENCH_SWEEP_BACKEND_STEPS:-1,3,5}" \
    -e MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES "${MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES:-512}" \
    -e MUON_BENCH_KINDS "${MUON_BENCH_KINDS:-muonh_update,muon_direction,ns4d_data_group_apply,expert_grouped_apply_boundary,expert_grouped_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply,ns4d_data_reshard_restore}" \
    -e MUON_BENCH_WARMUP "${MUON_BENCH_WARMUP:-1}" \
    -e MUON_BENCH_ITERS "${MUON_BENCH_ITERS:-3}" \
    -e MUON_BENCH_MODE "${MUON_BENCH_MODE:-both}" \
    -e MUON_BENCH_COMPILE_ONLY "${MUON_BENCH_COMPILE_ONLY:-false}" \
    -e MUON_BENCH_DISABLE_ABSTRACT_MESH "${MUON_BENCH_DISABLE_ABSTRACT_MESH:-true}" \
    -e MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES "${MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES:-false}" \
    -e MUON_BENCH_GPU_REPLICAS "${MUON_BENCH_GPU_REPLICAS:-1}" \
    -e MUON_BENCH_WORKER_CPU "${MUON_BENCH_WORKER_CPU:-8}" \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}" \
    -- python -m experiments.grug.moe.launch_cw_muon_update_bench
