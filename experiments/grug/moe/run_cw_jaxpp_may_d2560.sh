#!/usr/bin/env bash
# Launch a JaxPP May d=2560 Grug MoE profile run on CoreWeave H100s.

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
KUBECONFIG_PATH="${KUBECONFIG:-$HOME/.kube/coreweave-iris}"
MARIN_PREFIX="s3://marin-na/marin"
OBJECT_STORAGE_ENDPOINT="https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"
SUBMIT=false

RUN_ID=""
SCHEDULE="std_1f1b"
IMPLEMENTATION="auto"
EXPLICIT_MPMD_SCHEDULE_MODE="default"
EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT="bf16"
SONIC_FSDP_MATERIALIZATION="per_task"
PIPELINE="true"
PHYSICAL_STAGES=4
LOGICAL_STAGES=""
STAGE_LAYER_COUNTS=""
MICROBATCHES=8
NODES=4
GPUS_PER_REPLICA=8
EXPERT_AXIS=8
REPLICA_AXIS=1
BATCH=256
SEQ_LEN=4096
LAYERS=24
NUM_EXPERTS=256
TOP_K=4
VOCAB_SIZE=""
MOE_IMPLEMENTATION="ring"
RESEARCH_FP8_EXPERT_GEMM=false
ATTENTION_IMPLEMENTATION="${MAY_ATTENTION_IMPLEMENTATION:-}"
RAGGED_DOT_IMPLEMENTATION="${RAGGED_DOT_IMPL:-}"
RAGGED_DOT_BLOCK_K="${HALIAX_RAGGED_DOT_TRITON_BLOCK_K:-}"
RAGGED_DOT_NUM_WARPS="${HALIAX_RAGGED_DOT_TRITON_NUM_WARPS:-}"
LOSS_IMPLEMENTATION=""
CE_AUTOTUNE_ON_MISS="${LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS:-false}"
JAXPP_CONSERVATIVE_LOOP_CLUSTERING="${JAXPP_CONSERVATIVE_LOOP_CLUSTERING:-true}"
STEPS=20
DATA="synthetic"
TRACKER="wandb"
PROFILER_START=8
PROFILER_STEPS=0
WORKER_CPU=32
WORKER_RAM="256g"
WORKER_DISK="256g"
REMAT="save_moe"
MP="params=float32,compute=bfloat16,output=bfloat16"
JAXPP_REVISION="7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
JAX_NIGHTLY_VERSION="${JAX_NIGHTLY_VERSION:-}"
JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT="${JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT:-7200}"
JAXPP_CLIENT_TIMEOUT="${JAXPP_CLIENT_TIMEOUT:-7200000}"
DEEPEP_REVISION="7febc6e25660af0f54d95dd781ecdcd62265ecca"
DEEPEP_DISPATCH_NUM_THREADS="${DEEPEP_DISPATCH_NUM_THREADS:-512}"
XLA_MEMORY_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"

usage() {
    cat <<'EOF'
Usage:
  experiments/grug/moe/run_cw_jaxpp_may_d2560.sh [options]

Options:
  --submit                  Submit the Iris dispatcher job. Without this, print a dry-run summary.
  --run-id RUN_ID           Use a fixed run id instead of generating one.
  --schedule NAME           gpipe, std_1f1b, eager_1f1b, zero_bubble, interleaved_gpipe,
                            interleaved_1f1b, dualpipe_v, or kimi_k2 (default: std_1f1b).
  --implementation NAME     PP_IMPLEMENTATION: auto or explicit_mpmd (default: auto).
  --explicit-mpmd-schedule-mode NAME
                            default, transfer_priority, or input_gradient_first (default: default).
  --explicit-mpmd-pipeline-wire-format NAME
                            bf16 or fp8; fp8 uses packed E4M3 activations and E5M2 gradients
                            with explicit-MPMD std_1f1b only (default: bf16).
  --sonic-fsdp-materialization NAME
                            per_task or staged_per_step (default: per_task).
  --no-pipeline             Run the same model/backend without JaxPP for isolation.
  --physical-stages N       PP_MPMD_DIM / physical pipeline ranks (default: 4).
  --logical-stages N        PP_STAGES / logical pipeline stage cuts. Omit to infer per schedule.
  --stage-layer-counts CSV  Layers per logical stage, e.g. 7,6,6,5. Omit for an even split.
  --microbatches N          PP_MICROBATCHES (default: 8).
  --nodes N                 H100 node count / MAY_GPU_REPLICAS (default: 4).
  --gpus-per-replica N      H100 GPUs per Iris replica / MAY_GPUS_PER_REPLICA (default: 8).
  --expert-axis N           MAY_EXPERT_AXIS (default: 8).
  --layers N                MAY_NUM_LAYERS (default: 24).
  --experts N               MAY_NUM_EXPERTS (default: 256).
  --top-k N                 MAY_TOP_K (default: 4).
  --vocab-size N            MAY_VOCAB_SIZE. Omit to use the May heuristic default.
  --batch N                 MAY_BATCH (default: 256).
  --seq-len N               MAY_SEQ_LEN (default: 4096).
  --moe-implementation NAME ring, ring_quack_approx, ring_fused, ring_local_combine, ring_ppermute,
                            ragged_all_to_all, deepep, nccl_ep, nccl_ep_drop, scatter, or sonic
                            (default: ring).
  --research-fp8-expert-gemm
                            Use research-only FP8 routed expert GEMMs (default: disabled).
  --attention-implementation NAME
                            reference, gpu_fa4_cute, or gpu_fa4_thd.
                            Omit to use the model default.
  --ragged-dot-implementation NAME
                            auto, triton, or xla. Omit to use auto.
  --ragged-dot-block-k N     Pallas-Triton grouped-GEMM K tile: 32, 64, or 128.
  --ragged-dot-num-warps N  Pallas-Triton grouped-GEMM warps: 4 or 8.
  --loss-implementation NAME
                            Cross-entropy implementation: batched_xla, xla, or reference.
                            Omit to use Levanter default.
  --ce-autotune-on-miss BOOL
                            LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS (default: false).
  --conservative-loop-clustering BOOL
                            JAXPP_CONSERVATIVE_LOOP_CLUSTERING (default: true).
  --jax-init-timeout N      JAX distributed initialization timeout in seconds (default: 7200).
  --jaxpp-client-timeout-ms N
                            JaxPP coordination-client timeout in milliseconds (default: 7200000).
  --jax-nightly-version VERSION
                            Upgrade worker venvs to one exact public CUDA 13 nightly,
                            e.g. 0.11.1.dev20260725. Omit to keep the locked JAX version.
  --xla-memory-fraction N   XLA_PYTHON_CLIENT_MEM_FRACTION (default: 0.70).
  --remat NAME              recompute_all or save_moe (default: save_moe).
  --steps N                 MAY_STEPS (default: 20).
  --tracker NAME            MAY_TRACKER: wandb or json_logger (default: wandb).
  --data NAME               MAY_DATA: synthetic (default: synthetic).
  --profiler-steps N        MAY_PROFILER_STEPS (default: 0).
  --env-file PATH           Load R2 credentials from PATH.
  --prefix URI              MARIN_PREFIX for outputs (default: s3://marin-na/marin/).
  --cluster NAME            Iris cluster name (default: cw-us-east-02a).
  --kubeconfig PATH         Kubeconfig path (default: $KUBECONFIG or ~/.kube/coreweave-iris-gpu).
  -h, --help                Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit) SUBMIT=true; shift ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --schedule) SCHEDULE="$2"; shift 2 ;;
        --implementation) IMPLEMENTATION="$2"; shift 2 ;;
        --explicit-mpmd-schedule-mode) EXPLICIT_MPMD_SCHEDULE_MODE="$2"; shift 2 ;;
        --explicit-mpmd-pipeline-wire-format) EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT="$2"; shift 2 ;;
        --sonic-fsdp-materialization) SONIC_FSDP_MATERIALIZATION="$2"; shift 2 ;;
        --no-pipeline) PIPELINE="false"; shift ;;
        --physical-stages) PHYSICAL_STAGES="$2"; shift 2 ;;
        --logical-stages) LOGICAL_STAGES="$2"; shift 2 ;;
        --stage-layer-counts) STAGE_LAYER_COUNTS="$2"; shift 2 ;;
        --microbatches) MICROBATCHES="$2"; shift 2 ;;
        --nodes) NODES="$2"; shift 2 ;;
        --gpus-per-replica) GPUS_PER_REPLICA="$2"; shift 2 ;;
        --expert-axis) EXPERT_AXIS="$2"; shift 2 ;;
        --layers) LAYERS="$2"; shift 2 ;;
        --experts) NUM_EXPERTS="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --moe-implementation) MOE_IMPLEMENTATION="$2"; shift 2 ;;
        --research-fp8-expert-gemm) RESEARCH_FP8_EXPERT_GEMM=true; shift ;;
        --attention-implementation) ATTENTION_IMPLEMENTATION="$2"; shift 2 ;;
        --ragged-dot-implementation) RAGGED_DOT_IMPLEMENTATION="$2"; shift 2 ;;
        --ragged-dot-block-k) RAGGED_DOT_BLOCK_K="$2"; shift 2 ;;
        --ragged-dot-num-warps) RAGGED_DOT_NUM_WARPS="$2"; shift 2 ;;
        --loss-implementation) LOSS_IMPLEMENTATION="$2"; shift 2 ;;
        --ce-autotune-on-miss) CE_AUTOTUNE_ON_MISS="$2"; shift 2 ;;
        --conservative-loop-clustering) JAXPP_CONSERVATIVE_LOOP_CLUSTERING="$2"; shift 2 ;;
        --jax-init-timeout) JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT="$2"; shift 2 ;;
        --jaxpp-client-timeout-ms) JAXPP_CLIENT_TIMEOUT="$2"; shift 2 ;;
        --jax-nightly-version) JAX_NIGHTLY_VERSION="$2"; shift 2 ;;
        --xla-memory-fraction) XLA_MEMORY_FRACTION="$2"; shift 2 ;;
        --remat) REMAT="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --tracker) TRACKER="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --profiler-steps) PROFILER_STEPS="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; ENV_FILE_EXPLICIT=true; shift 2 ;;
        --prefix) MARIN_PREFIX="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --kubeconfig) KUBECONFIG_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$SCHEDULE" in
    gpipe|std_1f1b|eager_1f1b|zero_bubble|interleaved_gpipe|interleaved_1f1b|dualpipe_v|kimi_k2) ;;
    *)
        echo "ERROR: unsupported schedule: $SCHEDULE" >&2
        exit 1
        ;;
esac

case "$IMPLEMENTATION" in
    auto|explicit_mpmd) ;;
    *)
        echo "ERROR: unsupported implementation: $IMPLEMENTATION" >&2
        exit 1
        ;;
esac

case "$EXPLICIT_MPMD_SCHEDULE_MODE" in
    default|transfer_priority|input_gradient_first) ;;
    *)
        echo "ERROR: unsupported explicit MPMD schedule mode: $EXPLICIT_MPMD_SCHEDULE_MODE" >&2
        exit 1
        ;;
esac

case "$EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT" in
    bf16|fp8) ;;
    *)
        echo "ERROR: unsupported explicit MPMD pipeline wire format: $EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT" >&2
        exit 1
        ;;
esac

if [ "$EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT" = fp8 ]; then
    if [ "$PIPELINE" != true ] || [ "$IMPLEMENTATION" != explicit_mpmd ] || [ "$SCHEDULE" != std_1f1b ]; then
        echo "ERROR: fp8 pipeline wire format requires pipeline, --implementation explicit_mpmd, and --schedule std_1f1b" >&2
        exit 1
    fi
    if [ "$MICROBATCHES" -le 1 ]; then
        echo "ERROR: fp8 pipeline wire format requires --microbatches greater than 1" >&2
        exit 1
    fi
fi

if [ "$EXPLICIT_MPMD_SCHEDULE_MODE" = input_gradient_first ]; then
    if [ "$IMPLEMENTATION" != explicit_mpmd ] || [ "$SCHEDULE" != std_1f1b ]; then
        echo "ERROR: input_gradient_first requires --implementation explicit_mpmd --schedule std_1f1b" >&2
        exit 1
    fi
    if [ -n "$LOGICAL_STAGES" ]; then
        INPUT_GRADIENT_FIRST_STAGES="$LOGICAL_STAGES"
    else
        INPUT_GRADIENT_FIRST_STAGES="$PHYSICAL_STAGES"
    fi
    if [ "$INPUT_GRADIENT_FIRST_STAGES" -ne "$PHYSICAL_STAGES" ]; then
        echo "ERROR: input_gradient_first requires one logical stage per physical rank" >&2
        exit 1
    fi
    if [ "$MICROBATCHES" -lt "$INPUT_GRADIENT_FIRST_STAGES" ]; then
        echo "ERROR: input_gradient_first requires --microbatches >= pipeline stages" >&2
        exit 1
    fi
fi

case "$MOE_IMPLEMENTATION" in
    ring|ring_quack_approx|ring_fused|ring_local_combine|ring_ppermute|ragged_all_to_all|deepep|nccl_ep|nccl_ep_drop|scatter|sonic) ;;
    *)
        echo "ERROR: unsupported MoE implementation: $MOE_IMPLEMENTATION" >&2
        exit 1
        ;;
esac

if [ "$RESEARCH_FP8_EXPERT_GEMM" = true ]; then
    if [ "$PIPELINE" != true ] || [ "$IMPLEMENTATION" != explicit_mpmd ]; then
        echo "ERROR: research FP8 expert GEMMs require pipeline and --implementation explicit_mpmd" >&2
        exit 1
    fi
    case "$SCHEDULE" in
        gpipe|interleaved_gpipe|std_1f1b) ;;
        *)
            echo "ERROR: research FP8 expert GEMMs require gpipe, interleaved_gpipe, or std_1f1b" >&2
            exit 1
            ;;
    esac
    if [ "$MOE_IMPLEMENTATION" != ring ] || [ "$EXPERT_AXIS" -le 1 ]; then
        echo "ERROR: research FP8 expert GEMMs require --moe-implementation ring and --expert-axis greater than 1" >&2
        exit 1
    fi
fi

case "$SONIC_FSDP_MATERIALIZATION" in
    per_task|staged_per_step) ;;
    *)
        echo "ERROR: unsupported Sonic FSDP materialization mode: $SONIC_FSDP_MATERIALIZATION" >&2
        exit 1
        ;;
esac

if [ "$SONIC_FSDP_MATERIALIZATION" = staged_per_step ]; then
    if [ "$IMPLEMENTATION" != explicit_mpmd ] || [ "$SCHEDULE" != std_1f1b ]; then
        echo "ERROR: staged_per_step requires --implementation explicit_mpmd --schedule std_1f1b" >&2
        exit 1
    fi
    if [ "$MICROBATCHES" -le 1 ]; then
        echo "ERROR: staged_per_step requires --microbatches greater than 1" >&2
        exit 1
    fi
    if [ "$MOE_IMPLEMENTATION" != sonic ]; then
        echo "ERROR: staged_per_step requires --moe-implementation sonic" >&2
        exit 1
    fi
    if [ "$EXPERT_AXIS" -ne 1 ]; then
        echo "ERROR: staged_per_step requires --expert-axis 1 because Sonic does not support expert parallelism" >&2
        exit 1
    fi
fi

case "$REMAT" in
    recompute_all|save_moe) ;;
    *)
        echo "ERROR: unsupported remat mode: $REMAT" >&2
        exit 1
        ;;
esac

case "$ATTENTION_IMPLEMENTATION" in
    ""|reference|gpu_fa4_cute|gpu_fa4_thd) ;;
    *)
        echo "ERROR: unsupported attention implementation: $ATTENTION_IMPLEMENTATION" >&2
        exit 1
        ;;
esac

case "$RAGGED_DOT_IMPLEMENTATION" in
    ""|auto|triton|xla) ;;
    *)
        echo "ERROR: unsupported ragged dot implementation: $RAGGED_DOT_IMPLEMENTATION" >&2
        exit 1
        ;;
esac

case "$RAGGED_DOT_NUM_WARPS" in
    ""|4|8) ;;
    *)
        echo "ERROR: ragged dot num warps must be 4 or 8, got: $RAGGED_DOT_NUM_WARPS" >&2
        exit 1
        ;;
esac

case "$RAGGED_DOT_BLOCK_K" in
    ""|32|64|128) ;;
    *)
        echo "ERROR: ragged dot block K must be 32, 64, or 128, got: $RAGGED_DOT_BLOCK_K" >&2
        exit 1
        ;;
esac

case "$JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT" in
    ""|*[!0-9]*)
        echo "ERROR: JAX distributed initialization timeout must be a positive integer, got: $JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT" >&2
        exit 1
        ;;
esac
if [ "$JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT" -le 0 ]; then
    echo "ERROR: JAX distributed initialization timeout must be positive, got: $JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT" >&2
    exit 1
fi

case "$JAXPP_CLIENT_TIMEOUT" in
    ""|*[!0-9]*)
        echo "ERROR: JaxPP client timeout must be a positive integer, got: $JAXPP_CLIENT_TIMEOUT" >&2
        exit 1
        ;;
esac
if [ "$JAXPP_CLIENT_TIMEOUT" -le 0 ]; then
    echo "ERROR: JaxPP client timeout must be positive, got: $JAXPP_CLIENT_TIMEOUT" >&2
    exit 1
fi

if [ -n "$JAX_NIGHTLY_VERSION" ] && ! [[ "$JAX_NIGHTLY_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]{8}$ ]]; then
    echo "ERROR: JAX nightly version must look like 0.11.1.dev20260725, got: $JAX_NIGHTLY_VERSION" >&2
    exit 1
fi

R2_HELPER="${REPO_ROOT}/scripts/iris/cloudflare_r2_env.sh"
if [ -x "$R2_HELPER" ]; then
    if [ -f "$ENV_FILE" ] || [ "$ENV_FILE_EXPLICIT" = true ]; then
        R2_EXPORTS="$("$R2_HELPER" "$ENV_FILE")"
    else
        R2_EXPORTS="$("$R2_HELPER")"
    fi
    eval "$R2_EXPORTS"
elif [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
export KUBECONFIG="$KUBECONFIG_PATH"

if [ -z "$RUN_ID" ]; then
    RUN_ID="jaxpp-may-d2560-${SCHEDULE}-$(date -u +%Y%m%d-%H%M%S)"
fi

ENV_ARGS=(
    -e MARIN_PREFIX "$MARIN_PREFIX"
    -e RUN_ID "$RUN_ID"
    -e USER "${USER:-dlwh}"
    -e LOGNAME "${LOGNAME:-${USER:-dlwh}}"
    -e AWS_ENDPOINT_URL "$OBJECT_STORAGE_ENDPOINT"
    -e AWS_ENDPOINT_URL_S3 "$OBJECT_STORAGE_ENDPOINT"
    -e AWS_DEFAULT_REGION "auto"
    -e AWS_REGION "auto"
    -e MAY_GPU_REPLICAS "$NODES"
    -e MAY_GPUS_PER_REPLICA "$GPUS_PER_REPLICA"
    -e MAY_CPU_PER_REPLICA "$WORKER_CPU"
    -e MAY_WORKER_RAM "$WORKER_RAM"
    -e MAY_WORKER_DISK "$WORKER_DISK"
    -e MAY_EXPERT_AXIS "$EXPERT_AXIS"
    -e MAY_REPLICA_AXIS "$REPLICA_AXIS"
    -e MAY_BATCH "$BATCH"
    -e MAY_SEQ_LEN "$SEQ_LEN"
    -e MAY_NUM_LAYERS "$LAYERS"
    -e MAY_NUM_EXPERTS "$NUM_EXPERTS"
    -e MAY_TOP_K "$TOP_K"
    -e MAY_MOE_IMPLEMENTATION "$MOE_IMPLEMENTATION"
    -e MAY_RESEARCH_FP8_EXPERT_GEMM "$RESEARCH_FP8_EXPERT_GEMM"
    -e MAY_PIPELINE "$PIPELINE"
    -e MAY_STEPS "$STEPS"
    -e MAY_DATA "$DATA"
    -e MAY_TRACKER "$TRACKER"
    -e MAY_REMAT "$REMAT"
    -e MAY_MP "$MP"
    -e MAY_PROFILER_START "$PROFILER_START"
    -e MAY_PROFILER_STEPS "$PROFILER_STEPS"
    -e PP_IMPLEMENTATION "$IMPLEMENTATION"
    -e PP_EXPLICIT_MPMD_SCHEDULE_MODE "$EXPLICIT_MPMD_SCHEDULE_MODE"
    -e PP_EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT "$EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT"
    -e PP_SONIC_FSDP_MATERIALIZATION "$SONIC_FSDP_MATERIALIZATION"
    -e PP_SCHEDULE "$SCHEDULE"
    -e PP_MPMD_DIM "$PHYSICAL_STAGES"
    -e PP_MICROBATCHES "$MICROBATCHES"
    -e JAXPP_REVISION "$JAXPP_REVISION"
    -e JAX_NIGHTLY_VERSION "$JAX_NIGHTLY_VERSION"
    -e JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT "$JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT"
    -e JAXPP_CLIENT_TIMEOUT "$JAXPP_CLIENT_TIMEOUT"
    -e JAX_COMPILATION_CACHE_DIR "/tmp/jax-compilation-cache"
    -e GRUG_LOG_JAXPRS "${GRUG_LOG_JAXPRS:-false}"
    -e GRUG_LOG_XLA_HLO "${GRUG_LOG_XLA_HLO:-false}"
    -e LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS "$CE_AUTOTUNE_ON_MISS"
    -e JAXPP_CONSERVATIVE_LOOP_CLUSTERING "$JAXPP_CONSERVATIVE_LOOP_CLUSTERING"
    -e XLA_PYTHON_CLIENT_MEM_FRACTION "$XLA_MEMORY_FRACTION"
)

for maybe_env in \
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN \
    R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ENDPOINT_URL \
    CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_API_TOKEN; do
    if [ -n "${!maybe_env:-}" ]; then
        ENV_ARGS+=(-e "$maybe_env" "${!maybe_env}")
    fi
done

if [ -n "$LOGICAL_STAGES" ]; then
    ENV_ARGS+=(-e PP_STAGES "$LOGICAL_STAGES")
fi

if [ -n "$STAGE_LAYER_COUNTS" ]; then
    ENV_ARGS+=(-e PP_STAGE_LAYER_COUNTS "$STAGE_LAYER_COUNTS")
fi

if [ -n "$LOSS_IMPLEMENTATION" ]; then
    ENV_ARGS+=(-e MAY_LOSS_IMPLEMENTATION "$LOSS_IMPLEMENTATION")
fi

if [ -n "$ATTENTION_IMPLEMENTATION" ]; then
    ENV_ARGS+=(-e MAY_ATTENTION_IMPLEMENTATION "$ATTENTION_IMPLEMENTATION")
fi

if [ -n "$RAGGED_DOT_IMPLEMENTATION" ] && [ "$RAGGED_DOT_IMPLEMENTATION" != auto ]; then
    ENV_ARGS+=(-e RAGGED_DOT_IMPL "$RAGGED_DOT_IMPLEMENTATION")
fi

if [ -n "$RAGGED_DOT_NUM_WARPS" ]; then
    ENV_ARGS+=(-e HALIAX_RAGGED_DOT_TRITON_NUM_WARPS "$RAGGED_DOT_NUM_WARPS")
fi

if [ -n "$RAGGED_DOT_BLOCK_K" ]; then
    ENV_ARGS+=(-e HALIAX_RAGGED_DOT_TRITON_BLOCK_K "$RAGGED_DOT_BLOCK_K")
fi

if [ -n "$VOCAB_SIZE" ]; then
    ENV_ARGS+=(-e MAY_VOCAB_SIZE "$VOCAB_SIZE")
fi

if [ "$MOE_IMPLEMENTATION" = deepep ]; then
    ENV_ARGS+=(
        -e DEEPEP_SRC_ROOT "/tmp/DeepEP"
        -e DEEPEP_REVISION "$DEEPEP_REVISION"
        -e DEEPEP_CUDA_ARCH "sm_90"
        -e DEEPEP_DISPATCH_NUM_THREADS "$DEEPEP_DISPATCH_NUM_THREADS"
        -e MARIN_DEEPEP_CACHE_DIR "/tmp/marin-deepep-cache"
    )
fi

for maybe_env in \
    WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT MAY_WANDB_GROUP \
    IRIS_DEBUG_UV_SYNC GRUG_JAXPP_LOWER_EXPLICIT \
    GRUG_JAXPP_AUTO_EXPLICIT_IN_SHARDINGS GRUG_JAXPP_PATCH_CONST_SHARDINGS \
    GRUG_JAXPP_VALIDATE_TASK_PHASES \
    JAXPP_DISABLE_SCHEDULE_TASK_FUSION JAXPP_ENABLE_CHECK_JAXPR \
    JAXPP_ENABLE_TASK_JAXPR_DEDUPLICATION JAXPP_DIRECTIONAL_COMMUNICATORS \
    JAXPP_REUSE_RECV_BUFFERS \
    NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_HCA \
    TF_CPP_VMODULE TF_CPP_MAX_VLOG_LEVEL TF_GPU_ALLOCATOR XLA_FLAGS XLA_PYTHON_CLIENT_PREALLOCATE; do
    if [ -n "${!maybe_env:-}" ]; then
        ENV_ARGS+=(-e "$maybe_env" "${!maybe_env}")
    fi
done

CMD=(
    uv run --package marin-iris --extra controller iris --cluster="$CLUSTER"
    job run --no-wait
    --memory=2G --disk=4G --cpu=1 --extra=cpu
    "${ENV_ARGS[@]}"
    -- python -m experiments.grug.moe.launch_cw_jaxpp_may_d2560
)

if [ "$SUBMIT" != true ]; then
    cat <<EOF
Dry run: not submitting. Add --submit to launch.
cluster: $CLUSTER
run_id: $RUN_ID
kubeconfig: $KUBECONFIG
prefix: $MARIN_PREFIX
nodes: $NODES
gpus_per_replica: $GPUS_PER_REPLICA
schedule: $SCHEDULE
implementation: $IMPLEMENTATION
explicit_mpmd_schedule_mode: $EXPLICIT_MPMD_SCHEDULE_MODE
explicit_mpmd_pipeline_wire_format: $EXPLICIT_MPMD_PIPELINE_WIRE_FORMAT
sonic_fsdp_materialization: $SONIC_FSDP_MATERIALIZATION
pipeline: $PIPELINE
physical_stages: $PHYSICAL_STAGES
logical_stages: ${LOGICAL_STAGES:-inferred}
stage_layer_counts: ${STAGE_LAYER_COUNTS:-even}
microbatches: $MICROBATCHES
model: d2560 L${LAYERS} experts=${NUM_EXPERTS} top_k=${TOP_K} seq_len=${SEQ_LEN} vocab=${VOCAB_SIZE:-default}
batch: $BATCH
moe_implementation: $MOE_IMPLEMENTATION
research_fp8_expert_gemm: $RESEARCH_FP8_EXPERT_GEMM
attention_implementation: ${ATTENTION_IMPLEMENTATION:-default}
ragged_dot_implementation: ${RAGGED_DOT_IMPLEMENTATION:-auto}
ragged_dot_block_k: ${RAGGED_DOT_BLOCK_K:-32}
ragged_dot_num_warps: ${RAGGED_DOT_NUM_WARPS:-4}
loss_implementation: ${LOSS_IMPLEMENTATION:-default}
ce_autotune_on_miss: $CE_AUTOTUNE_ON_MISS
conservative_loop_clustering: $JAXPP_CONSERVATIVE_LOOP_CLUSTERING
jax_init_timeout: $JAX_DISTRIBUTED_INITIALIZATION_TIMEOUT
jaxpp_client_timeout_ms: $JAXPP_CLIENT_TIMEOUT
xla_memory_fraction: $XLA_MEMORY_FRACTION
remat: $REMAT
steps: $STEPS
tracker: $TRACKER
data: $DATA
jaxpp_revision: $JAXPP_REVISION
jax_nightly_version: ${JAX_NIGHTLY_VERSION:-locked}

Command shape:
  uv run --package marin-iris --extra controller iris --cluster=$CLUSTER job run --no-wait ... -- python -m experiments.grug.moe.launch_cw_jaxpp_may_d2560
EOF
    exit 0
fi

if [ ! -f "$KUBECONFIG" ]; then
    echo "ERROR: kubeconfig not found: $KUBECONFIG" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec "${CMD[@]}"
