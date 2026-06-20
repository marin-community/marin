#!/usr/bin/env bash
# Launch the May Recipe d=2560 Grug MoE CoreWeave H100 profile run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CLUSTER="cw-us-east-02a"
DEFAULT_ENV_FILE="$HOME/.config/marin/marin-r2.env"
if [ ! -f "$DEFAULT_ENV_FILE" ] && [ -f "$HOME/.config/marin/cloudflare-r2.env" ]; then
    DEFAULT_ENV_FILE="$HOME/.config/marin/cloudflare-r2.env"
fi
ENV_FILE="${MARIN_R2_ENV_FILE:-$DEFAULT_ENV_FILE}"
ENV_FILE_EXPLICIT=false
KUBECONFIG_PATH="${KUBECONFIG:-$HOME/.kube/coreweave-iris-gpu}"
MARIN_PREFIX="s3://marin-na/tmp/ttl=7d"
RUN_ID=""
SUBMIT=false
TASK_IMAGE="${MAY_TASK_IMAGE:-}"

GPU_REPLICAS=32
EXPERT_AXIS=8
REPLICA_AXIS=1
MODEL_AXIS=1
ALLOW_CROSS_NODE_EXPERT_AXIS="${MAY_ALLOW_CROSS_NODE_EXPERT_AXIS:-false}"
BATCH=256
SEQ_LEN=4096
SLIDING_WINDOW=2048
NUM_LAYERS=""
STEPS=30
TOTAL_TOKENS="${MAY_TOTAL_TOKENS:-10000000000000}"
CHECKPOINTS="none"
DATA="slimpajama"
REMAT="save_moe"
USE_PKO="${MAY_USE_PKO:-true}"
PKO_ON_LAST_LAYER="${MAY_PKO_ON_LAST_LAYER:-true}"
BLOCK_CROSS_DOCUMENT_ATTENTION="${MAY_BLOCK_CROSS_DOCUMENT_ATTENTION:-true}"
INPUT_EMBED_SHARDING="${MAY_INPUT_EMBED_SHARDING:-hidden_batch}"
OUTPUT_PROJ_SHARDING="${MAY_OUTPUT_PROJ_SHARDING:-lm_head}"
OPTIMIZER="${MAY_OPTIMIZER:-muonh}"
MUON_BACKEND_STEPS="${MAY_MUON_BACKEND_STEPS:-5}"
MUON_ORTHOGONALIZATION_LAYOUT="${MAY_MUON_ORTHOGONALIZATION_LAYOUT:-stack_batch_sharded}"
MUON_MAX_GROUPED_STACK_SIZE="${MAY_MUON_MAX_GROUPED_STACK_SIZE:-256}"
MUON_NS_COMPUTE_DTYPE="${MAY_MUON_NS_COMPUTE_DTYPE:-input}"
MUON_NESTEROV="${MAY_MUON_NESTEROV:-true}"
ASSERT_OPTIMIZER_SHARDING="${MAY_ASSERT_OPTIMIZER_SHARDING:-true}"
MATCH_OPTIMIZER_SHARDING="${MAY_MATCH_OPTIMIZER_SHARDING:-true}"
EXPERT_3D_OPTIMIZER="${MAY_EXPERT_3D_OPTIMIZER:-muonh}"
ORDINARY_2D_OPTIMIZER="${MAY_ORDINARY_2D_OPTIMIZER:-muonh}"
EXPERT_GROUPED_MUONH_GROUP_SIZE="${MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE:-}"
EXPERT_GROUPED_MUONH_PACKED_ENTRY="${MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY:-false}"
EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES="${MAY_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES:-false}"
MP="params=float32,compute=bfloat16,output=bfloat16"
LIVE_PARAM_MODE="param"
ATTENTION_IMPLEMENTATION="gpu_fa4_cute"
CE_IMPLEMENTATION=""
MOE_IMPLEMENTATION="${MAY_MOE_IMPLEMENTATION:-ring}"
TRACKER="wandb"
PROFILER_START=12
PROFILER_STEPS=8
ENABLE_HLO_PROTO="${MAY_PROFILER_ENABLE_HLO_PROTO:-false}"
HOST_TRACER_LEVEL="${MAY_PROFILER_HOST_TRACER_LEVEL:-1}"
PYTHON_TRACER_LEVEL="${MAY_PROFILER_PYTHON_TRACER_LEVEL:-0}"
DEVICE_TRACER_LEVEL="${MAY_PROFILER_DEVICE_TRACER_LEVEL:-}"
UPLOAD_PROFILER_ARTIFACT="${MAY_UPLOAD_PROFILER_ARTIFACT:-false}"
XLA_MEMORY_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
PALLAS_CE_AUTOTUNE_ON_MISS="${LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS:-false}"
NCCL_SOCKET_IFNAME_VALUE="${NCCL_SOCKET_IFNAME:-^ibs,ibp,lo,docker,veth,cilium,lxc}"
NCCL_SOCKET_FAMILY_VALUE="${NCCL_SOCKET_FAMILY:-AF_INET}"
WORKER_CPU=32
WATCH_INTERVAL=0
LOG_EVERY=1
LOG_JAXPRS="${MAY_LOG_JAXPRS:-false}"
LOG_XLA_HLO="${MAY_LOG_XLA_HLO:-false}"
SAVE_XLA_DUMPS="${MAY_SAVE_XLA_DUMPS:-false}"
XLA_FLAGS_VALUE="${MAY_XLA_FLAGS:-}"

usage() {
    cat <<'EOF'
Usage:
  experiments/grug/moe/run_cw_may_d2560.sh [options]

Options:
  --submit                  Submit the Iris dispatcher job. Without this, print a dry-run summary.
  --run-id RUN_ID           Use a fixed run id instead of generating one.
  --env-file PATH           Load R2 credentials from PATH (default: $MARIN_R2_ENV_FILE or ~/.config/marin/marin-r2.env).
  --prefix URI              MARIN_PREFIX for outputs (default: s3://marin-na/tmp/ttl=7d).
  --cluster NAME            Iris cluster name (default: cw-us-east-02a).
  --kubeconfig PATH         Kubeconfig path (default: $KUBECONFIG or ~/.kube/coreweave-iris-gpu).
  --task-image IMAGE        Task image override for child GPU jobs (sets MAY_TASK_IMAGE).
  --nodes N                 H100 node count / MAY_GPU_REPLICAS (default: 32).
  --worker-cpu N            MAY_CPU_PER_REPLICA for each H100 worker pod (default: 32).
  --expert-axis N           MAY_EXPERT_AXIS (default: 8).
  --replica-axis N          MAY_REPLICA_AXIS (default: 1).
  --model-axis N            MAY_MODEL_AXIS tensor/model-parallel axis size (default: 1).
  --allow-cross-node-expert-axis BOOL
                            MAY_ALLOW_CROSS_NODE_EXPERT_AXIS diagnostic toggle;
                            allows expert-axis groups to span workers (default: false).
  --batch N                 MAY_BATCH (default: 256).
  --seq-len N               MAY_SEQ_LEN (default: 4096).
  --sliding-window N        MAY_SLIDING_WINDOW for short layers (default: 2048).
  --layers N                MAY_NUM_LAYERS diagnostic override (default: May heuristic).
  --steps N                 MAY_STEPS (default: 30).
  --total-tokens N          MAY_TOTAL_TOKENS optimizer horizon (default: 10000000000000).
  --profiler-start N        MAY_PROFILER_START (default: 12).
  --profiler-steps N        MAY_PROFILER_STEPS (default: 8; set 0 to disable).
  --xla-memory-fraction F   XLA_PYTHON_CLIENT_MEM_FRACTION (default: 0.95).
  --nccl-socket-ifname IFACE NCCL_SOCKET_IFNAME for bootstrap/OOB sockets
                            (default: ^ibs,ibp,lo,docker,veth,cilium,lxc).
  --nccl-socket-family FAMILY NCCL_SOCKET_FAMILY (default: AF_INET).
  --pallas-ce-autotune-on-miss BOOL
                            LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS (default: false).
  --tracker NAME            MAY_TRACKER: wandb or json_logger (default: wandb).
  --data NAME               MAY_DATA: slimpajama, nemotron, or synthetic (default: slimpajama).
  --checkpoints MODE        MAY_CHECKPOINTS: none, local, or s3 (default: none).
  --remat MODE              MAY_REMAT: save_moe or recompute_all (default: save_moe).
  --use-pko BOOL            MAY_USE_PKO diagnostic toggle (default: true).
  --pko-on-last-layer BOOL  MAY_PKO_ON_LAST_LAYER diagnostic toggle (default: true).
  --block-cross-document-attention BOOL
                            MAY_BLOCK_CROSS_DOCUMENT_ATTENTION for synthetic data (default: true).
  --input-embed-sharding MODE MAY_INPUT_EMBED_SHARDING: hidden_batch or replicated (default: hidden_batch).
  --output-proj-sharding MODE MAY_OUTPUT_PROJ_SHARDING: lm_head or replicated (default: lm_head).
  --optimizer MODE        MAY_OPTIMIZER: muonh or sgd (default: muonh).
  --muon-backend-steps N  MAY_MUON_BACKEND_STEPS for MuonH (default: 5).
  --muon-orthogonalization-layout MODE
                            MAY_MUON_ORTHOGONALIZATION_LAYOUT: stack_batch_sharded or vmap_replicated (default: stack_batch_sharded).
  --muon-max-grouped-stack-size N
                            MAY_MUON_MAX_GROUPED_STACK_SIZE for grouped Muon (default: 256).
  --muon-ns-compute-dtype DTYPE
                            MAY_MUON_NS_COMPUTE_DTYPE: input, bf16, bfloat16, fp32, float32, fp16, or float16 (default: input).
  --muon-nesterov BOOL      MAY_MUON_NESTEROV: true or false (default: true).
  --assert-optimizer-sharding BOOL
                            MAY_ASSERT_OPTIMIZER_SHARDING diagnostic toggle (default: true).
  --match-optimizer-sharding BOOL
                            MAY_MATCH_OPTIMIZER_SHARDING diagnostic toggle for explicit optimizer reshards (default: true).
  --expert-3d-optimizer MODE MAY_EXPERT_3D_OPTIMIZER: muonh, adamh, or grouped_muonh (default: muonh).
  --ordinary-2d-optimizer MODE
                            MAY_ORDINARY_2D_OPTIMIZER: muonh, adamh, adam, or sgd for ordinary non-expert 2D weights (default: muonh).
  --expert-grouped-muonh-group-size N
                            MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE for grouped_muonh; empty chooses replica_dcn*data.
  --expert-grouped-muonh-packed-entry BOOL
                            MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY for grouped_muonh (default: false).
  --expert-grouped-muonh-chunk-local-boundaries BOOL
                            MAY_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES experimental grouped_muonh boundary mode (default: false).
  --mp POLICY               MAY_MP policy string.
  --live-param-mode MODE    MAY_LIVE_PARAM_MODE: param or compute_with_master (default: param).
  --attention NAME          MAY_ATTENTION_IMPLEMENTATION (default: gpu_fa4_cute).
  --ce-implementation NAME  MAY_CE_IMPLEMENTATION: pallas_gpu, xla, reference, or empty default.
  --moe-implementation NAME MAY_MOE_IMPLEMENTATION: ring, ragged_all_to_all, or deepep (default: ring).
  --watch-interval N        MAY_WATCH_INTERVAL; 0 disables grad/param watch stats (default: 0).
  --log-every N             MAY_LOG_EVERY train progress/scalar logging cadence (default: 1).
  --log-jaxprs BOOL         MAY_LOG_JAXPRS; true dumps JAXPRs (default: false).
  --log-xla-hlo BOOL        MAY_LOG_XLA_HLO; true dumps XLA HLO (default: false).
  --save-xla-dumps BOOL     MAY_SAVE_XLA_DUMPS; true uploads XLA_FLAGS dump dir to W&B (default: false).
  --xla-flags FLAGS         XLA_FLAGS forwarded to worker tasks; defaults to per-run HLO dumps when --save-xla-dumps true.
  -h, --help                Show this help.

This wrapper forwards explicit MAY_* environment variables to Iris; local shell
MAY_* values are not implicitly visible inside the remote job.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit)
            SUBMIT=true
            shift
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            ENV_FILE_EXPLICIT=true
            shift 2
            ;;
        --prefix)
            MARIN_PREFIX="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --kubeconfig)
            KUBECONFIG_PATH="$2"
            shift 2
            ;;
        --task-image)
            TASK_IMAGE="$2"
            shift 2
            ;;
        --nodes)
            GPU_REPLICAS="$2"
            shift 2
            ;;
        --worker-cpu)
            WORKER_CPU="$2"
            shift 2
            ;;
        --expert-axis)
            EXPERT_AXIS="$2"
            shift 2
            ;;
        --replica-axis)
            REPLICA_AXIS="$2"
            shift 2
            ;;
        --model-axis)
            MODEL_AXIS="$2"
            shift 2
            ;;
        --allow-cross-node-expert-axis)
            ALLOW_CROSS_NODE_EXPERT_AXIS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --sliding-window)
            SLIDING_WINDOW="$2"
            shift 2
            ;;
        --layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --total-tokens)
            TOTAL_TOKENS="$2"
            shift 2
            ;;
        --profiler-start)
            PROFILER_START="$2"
            shift 2
            ;;
        --profiler-steps)
            PROFILER_STEPS="$2"
            shift 2
            ;;
        --xla-memory-fraction)
            XLA_MEMORY_FRACTION="$2"
            shift 2
            ;;
        --nccl-socket-ifname)
            NCCL_SOCKET_IFNAME_VALUE="$2"
            shift 2
            ;;
        --nccl-socket-family)
            NCCL_SOCKET_FAMILY_VALUE="$2"
            shift 2
            ;;
        --pallas-ce-autotune-on-miss)
            PALLAS_CE_AUTOTUNE_ON_MISS="$2"
            shift 2
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        --data)
            DATA="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINTS="$2"
            shift 2
            ;;
        --remat)
            REMAT="$2"
            shift 2
            ;;
        --use-pko)
            USE_PKO="$2"
            shift 2
            ;;
        --pko-on-last-layer)
            PKO_ON_LAST_LAYER="$2"
            shift 2
            ;;
        --block-cross-document-attention)
            BLOCK_CROSS_DOCUMENT_ATTENTION="$2"
            shift 2
            ;;
        --input-embed-sharding)
            INPUT_EMBED_SHARDING="$2"
            shift 2
            ;;
        --output-proj-sharding)
            OUTPUT_PROJ_SHARDING="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --muon-backend-steps)
            MUON_BACKEND_STEPS="$2"
            shift 2
            ;;
        --muon-orthogonalization-layout)
            MUON_ORTHOGONALIZATION_LAYOUT="$2"
            shift 2
            ;;
        --muon-max-grouped-stack-size)
            MUON_MAX_GROUPED_STACK_SIZE="$2"
            shift 2
            ;;
        --muon-ns-compute-dtype)
            MUON_NS_COMPUTE_DTYPE="$2"
            shift 2
            ;;
        --muon-nesterov)
            MUON_NESTEROV="$2"
            shift 2
            ;;
        --assert-optimizer-sharding)
            ASSERT_OPTIMIZER_SHARDING="$2"
            shift 2
            ;;
        --match-optimizer-sharding)
            MATCH_OPTIMIZER_SHARDING="$2"
            shift 2
            ;;
        --expert-3d-optimizer)
            EXPERT_3D_OPTIMIZER="$2"
            shift 2
            ;;
        --ordinary-2d-optimizer)
            ORDINARY_2D_OPTIMIZER="$2"
            shift 2
            ;;
        --expert-grouped-muonh-group-size)
            EXPERT_GROUPED_MUONH_GROUP_SIZE="$2"
            shift 2
            ;;
        --expert-grouped-muonh-packed-entry)
            EXPERT_GROUPED_MUONH_PACKED_ENTRY="$2"
            shift 2
            ;;
        --expert-grouped-muonh-chunk-local-boundaries)
            EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES="$2"
            shift 2
            ;;
        --mp)
            MP="$2"
            shift 2
            ;;
        --live-param-mode)
            LIVE_PARAM_MODE="$2"
            shift 2
            ;;
        --attention)
            ATTENTION_IMPLEMENTATION="$2"
            shift 2
            ;;
        --ce-implementation)
            CE_IMPLEMENTATION="$2"
            shift 2
            ;;
        --moe-implementation)
            MOE_IMPLEMENTATION="$2"
            shift 2
            ;;
        --watch-interval)
            WATCH_INTERVAL="$2"
            shift 2
            ;;
        --log-every)
            LOG_EVERY="$2"
            shift 2
            ;;
        --log-jaxprs)
            LOG_JAXPRS="$2"
            shift 2
            ;;
        --log-xla-hlo)
            LOG_XLA_HLO="$2"
            shift 2
            ;;
        --save-xla-dumps)
            SAVE_XLA_DUMPS="$2"
            shift 2
            ;;
        --xla-flags)
            XLA_FLAGS_VALUE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$LIVE_PARAM_MODE" in
    param|compute_with_master)
        ;;
    *)
        echo "ERROR: --live-param-mode must be param or compute_with_master, got: $LIVE_PARAM_MODE" >&2
        exit 1
        ;;
esac

case "$INPUT_EMBED_SHARDING" in
    hidden_batch|replicated)
        ;;
    *)
        echo "ERROR: --input-embed-sharding must be hidden_batch or replicated, got: $INPUT_EMBED_SHARDING" >&2
        exit 1
        ;;
esac

case "$OUTPUT_PROJ_SHARDING" in
    lm_head|replicated)
        ;;
    *)
        echo "ERROR: --output-proj-sharding must be lm_head or replicated, got: $OUTPUT_PROJ_SHARDING" >&2
        exit 1
        ;;
esac

case "$OPTIMIZER" in
    muonh|sgd)
        ;;
    *)
        echo "ERROR: --optimizer must be muonh or sgd, got: $OPTIMIZER" >&2
        exit 1
        ;;
esac

case "$MUON_ORTHOGONALIZATION_LAYOUT" in
    stack_batch_sharded|stack_batch_4d_sharded|vmap_replicated)
        ;;
    *)
        echo "ERROR: --muon-orthogonalization-layout must be stack_batch_sharded, stack_batch_4d_sharded, or vmap_replicated, got: $MUON_ORTHOGONALIZATION_LAYOUT" >&2
        exit 1
        ;;
esac

case "$MUON_NS_COMPUTE_DTYPE" in
    input|bf16|bfloat16|fp32|float32|fp16|float16)
        ;;
    *)
        echo "ERROR: --muon-ns-compute-dtype must be input, bf16, bfloat16, fp32, float32, fp16, or float16, got: $MUON_NS_COMPUTE_DTYPE" >&2
        exit 1
        ;;
esac

case "$EXPERT_3D_OPTIMIZER" in
    muonh|adamh|grouped_muonh)
        ;;
    *)
        echo "ERROR: --expert-3d-optimizer must be muonh, adamh, or grouped_muonh, got: $EXPERT_3D_OPTIMIZER" >&2
        exit 1
        ;;
esac

case "$ORDINARY_2D_OPTIMIZER" in
    muonh|adamh|adam|sgd)
        ;;
    *)
        echo "ERROR: --ordinary-2d-optimizer must be muonh, adamh, adam, or sgd, got: $ORDINARY_2D_OPTIMIZER" >&2
        exit 1
        ;;
esac

if [ -f "$ENV_FILE" ] || [ "$ENV_FILE_EXPLICIT" = true ]; then
    R2_EXPORTS="$("${REPO_ROOT}/scripts/iris/cloudflare_r2_env.sh" "$ENV_FILE")"
else
    R2_EXPORTS="$("${REPO_ROOT}/scripts/iris/cloudflare_r2_env.sh")"
fi
eval "$R2_EXPORTS"
export KUBECONFIG="$KUBECONFIG_PATH"

if [ -z "$RUN_ID" ]; then
    RUN_ID="cw-may-d2560-profile-$(date -u +%Y%m%d-%H%M%S)"
fi

if [ "$SAVE_XLA_DUMPS" = true ]; then
    XLA_DUMP_FLAGS="--xla_dump_to=/tmp/xla_dumps/${RUN_ID} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"
    if [ -z "$XLA_FLAGS_VALUE" ]; then
        XLA_FLAGS_VALUE="$XLA_DUMP_FLAGS"
    elif [[ "$XLA_FLAGS_VALUE" != *"--xla_dump_to="* ]]; then
        XLA_FLAGS_VALUE="$XLA_FLAGS_VALUE $XLA_DUMP_FLAGS"
    fi
fi

RUN_TOKENS=$((BATCH * SEQ_LEN * STEPS))

ENV_ARGS=(
    -e MARIN_PREFIX "$MARIN_PREFIX"
    -e RUN_ID "$RUN_ID"
    -e AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID"
    -e AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY"
    -e AWS_ENDPOINT_URL "$AWS_ENDPOINT_URL"
    -e AWS_ENDPOINT_URL_S3 "$AWS_ENDPOINT_URL_S3"
    -e MAY_GPU_REPLICAS "$GPU_REPLICAS"
    -e MAY_CPU_PER_REPLICA "$WORKER_CPU"
    -e MAY_TASK_IMAGE "$TASK_IMAGE"
    -e MAY_EXPERT_AXIS "$EXPERT_AXIS"
    -e MAY_REPLICA_AXIS "$REPLICA_AXIS"
    -e MAY_MODEL_AXIS "$MODEL_AXIS"
    -e MAY_ALLOW_CROSS_NODE_EXPERT_AXIS "$ALLOW_CROSS_NODE_EXPERT_AXIS"
    -e MAY_BATCH "$BATCH"
    -e MAY_SEQ_LEN "$SEQ_LEN"
    -e MAY_SLIDING_WINDOW "$SLIDING_WINDOW"
    -e MAY_NUM_LAYERS "$NUM_LAYERS"
    -e MAY_STEPS "$STEPS"
    -e MAY_TOTAL_TOKENS "$TOTAL_TOKENS"
    -e MAY_CHECKPOINTS "$CHECKPOINTS"
    -e MAY_DATA "$DATA"
    -e MAY_REMAT "$REMAT"
    -e MAY_USE_PKO "$USE_PKO"
    -e MAY_PKO_ON_LAST_LAYER "$PKO_ON_LAST_LAYER"
    -e MAY_BLOCK_CROSS_DOCUMENT_ATTENTION "$BLOCK_CROSS_DOCUMENT_ATTENTION"
    -e MAY_INPUT_EMBED_SHARDING "$INPUT_EMBED_SHARDING"
    -e MAY_OUTPUT_PROJ_SHARDING "$OUTPUT_PROJ_SHARDING"
    -e MAY_OPTIMIZER "$OPTIMIZER"
    -e MAY_MUON_BACKEND_STEPS "$MUON_BACKEND_STEPS"
    -e MAY_MUON_ORTHOGONALIZATION_LAYOUT "$MUON_ORTHOGONALIZATION_LAYOUT"
    -e MAY_MUON_MAX_GROUPED_STACK_SIZE "$MUON_MAX_GROUPED_STACK_SIZE"
    -e MAY_MUON_NS_COMPUTE_DTYPE "$MUON_NS_COMPUTE_DTYPE"
    -e MAY_MUON_NESTEROV "$MUON_NESTEROV"
    -e MAY_ASSERT_OPTIMIZER_SHARDING "$ASSERT_OPTIMIZER_SHARDING"
    -e MAY_MATCH_OPTIMIZER_SHARDING "$MATCH_OPTIMIZER_SHARDING"
    -e MAY_EXPERT_3D_OPTIMIZER "$EXPERT_3D_OPTIMIZER"
    -e MAY_ORDINARY_2D_OPTIMIZER "$ORDINARY_2D_OPTIMIZER"
    -e MAY_EXPERT_GROUPED_MUONH_GROUP_SIZE "$EXPERT_GROUPED_MUONH_GROUP_SIZE"
    -e MAY_EXPERT_GROUPED_MUONH_PACKED_ENTRY "$EXPERT_GROUPED_MUONH_PACKED_ENTRY"
    -e MAY_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES "$EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES"
    -e MAY_MP "$MP"
    -e MAY_LIVE_PARAM_MODE "$LIVE_PARAM_MODE"
    -e MAY_ATTENTION_IMPLEMENTATION "$ATTENTION_IMPLEMENTATION"
    -e MAY_CE_IMPLEMENTATION "$CE_IMPLEMENTATION"
    -e MAY_MOE_IMPLEMENTATION "$MOE_IMPLEMENTATION"
    -e MAY_TRACKER "$TRACKER"
    -e MAY_PROFILER_START "$PROFILER_START"
    -e MAY_PROFILER_STEPS "$PROFILER_STEPS"
    -e MAY_PROFILER_ENABLE_HLO_PROTO "$ENABLE_HLO_PROTO"
    -e MAY_PROFILER_HOST_TRACER_LEVEL "$HOST_TRACER_LEVEL"
    -e MAY_PROFILER_PYTHON_TRACER_LEVEL "$PYTHON_TRACER_LEVEL"
    -e MAY_UPLOAD_PROFILER_ARTIFACT "$UPLOAD_PROFILER_ARTIFACT"
    -e MAY_WATCH_INTERVAL "$WATCH_INTERVAL"
    -e MAY_LOG_EVERY "$LOG_EVERY"
    -e MAY_LOG_JAXPRS "$LOG_JAXPRS"
    -e MAY_LOG_XLA_HLO "$LOG_XLA_HLO"
    -e MAY_SAVE_XLA_DUMPS "$SAVE_XLA_DUMPS"
    -e XLA_PYTHON_CLIENT_MEM_FRACTION "$XLA_MEMORY_FRACTION"
    -e LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS "$PALLAS_CE_AUTOTUNE_ON_MISS"
    -e NCCL_SOCKET_IFNAME "$NCCL_SOCKET_IFNAME_VALUE"
    -e NCCL_SOCKET_FAMILY "$NCCL_SOCKET_FAMILY_VALUE"
)

if [ -n "$XLA_FLAGS_VALUE" ]; then
    ENV_ARGS+=(-e XLA_FLAGS "$XLA_FLAGS_VALUE")
fi
if [ -n "$DEVICE_TRACER_LEVEL" ]; then
    ENV_ARGS+=(-e MAY_PROFILER_DEVICE_TRACER_LEVEL "$DEVICE_TRACER_LEVEL")
fi

for maybe_env in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT MAY_WANDB_GROUP TF_GPU_ALLOCATOR NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_DISABLE LEVANTER_PALLAS_GPU_CUSTOM_BWD_V_BLOCK_SIZE; do
    if [ -n "${!maybe_env:-}" ]; then
        ENV_ARGS+=(-e "$maybe_env" "${!maybe_env}")
    fi
done

CMD=(
    uv run --package marin-iris --extra controller iris --cluster="$CLUSTER"
    job run --no-wait
    --memory=2G --disk=4G --cpu=1 --extra=cpu
    "${ENV_ARGS[@]}"
    -- python -m experiments.grug.moe.launch_cw_may_d2560
)

if [ "$SUBMIT" != true ]; then
    cat <<EOF
Dry run: not submitting. Add --submit to launch.
cluster: $CLUSTER
run_id: $RUN_ID
kubeconfig: $KUBECONFIG
prefix: $MARIN_PREFIX
r2_endpoint: $AWS_ENDPOINT_URL
nodes: $GPU_REPLICAS
worker_cpu: $WORKER_CPU
task_image: ${TASK_IMAGE:-default}
mesh axes: replica=$REPLICA_AXIS expert=$EXPERT_AXIS model=$MODEL_AXIS
batch: $BATCH
seq_len: $SEQ_LEN
sliding_window: $SLIDING_WINDOW
layers: ${NUM_LAYERS:-default}
steps: $STEPS
run_tokens: $RUN_TOKENS
total_tokens: $TOTAL_TOKENS
tracker: $TRACKER
profiler: start=$PROFILER_START steps=$PROFILER_STEPS hlo_proto=$ENABLE_HLO_PROTO device_tracer=$DEVICE_TRACER_LEVEL upload_artifact=$UPLOAD_PROFILER_ARTIFACT
checkpoints: $CHECKPOINTS
xla_memory_fraction: $XLA_MEMORY_FRACTION
nccl_socket_ifname: $NCCL_SOCKET_IFNAME_VALUE
nccl_socket_family: $NCCL_SOCKET_FAMILY_VALUE
pallas_ce_autotune_on_miss: $PALLAS_CE_AUTOTUNE_ON_MISS
mp: $MP
live_param_mode: $LIVE_PARAM_MODE
watch_interval: $WATCH_INTERVAL
log_every: $LOG_EVERY
log_jaxprs: $LOG_JAXPRS
log_xla_hlo: $LOG_XLA_HLO
save_xla_dumps: $SAVE_XLA_DUMPS
xla_flags: ${XLA_FLAGS_VALUE:-unset}
use_pko: $USE_PKO
pko_on_last_layer: $PKO_ON_LAST_LAYER
block_cross_document_attention: $BLOCK_CROSS_DOCUMENT_ATTENTION
input_embed_sharding: $INPUT_EMBED_SHARDING
output_proj_sharding: $OUTPUT_PROJ_SHARDING
optimizer: $OPTIMIZER
muon_backend_steps: $MUON_BACKEND_STEPS
muon_orthogonalization_layout: $MUON_ORTHOGONALIZATION_LAYOUT
muon_max_grouped_stack_size: $MUON_MAX_GROUPED_STACK_SIZE
muon_ns_compute_dtype: $MUON_NS_COMPUTE_DTYPE
muon_nesterov: $MUON_NESTEROV
assert_optimizer_sharding: $ASSERT_OPTIMIZER_SHARDING
match_optimizer_sharding: $MATCH_OPTIMIZER_SHARDING
expert_3d_optimizer: $EXPERT_3D_OPTIMIZER
ordinary_2d_optimizer: $ORDINARY_2D_OPTIMIZER
expert_grouped_muonh_group_size: ${EXPERT_GROUPED_MUONH_GROUP_SIZE:-auto}
expert_grouped_muonh_packed_entry: $EXPERT_GROUPED_MUONH_PACKED_ENTRY
expert_grouped_muonh_chunk_local_boundaries: $EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES
attention: $ATTENTION_IMPLEMENTATION
ce_implementation: ${CE_IMPLEMENTATION:-default}
moe_implementation: $MOE_IMPLEMENTATION
data: $DATA

Command shape:
  uv run --package marin-iris --extra controller iris --cluster=$CLUSTER job run --no-wait ... -- python -m experiments.grug.moe.launch_cw_may_d2560
EOF
    exit 0
fi

if [ ! -f "$KUBECONFIG" ]; then
    echo "ERROR: kubeconfig not found: $KUBECONFIG" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "${CMD[@]}"
