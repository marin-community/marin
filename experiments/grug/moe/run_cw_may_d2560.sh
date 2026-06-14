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
MARIN_PREFIX="s3://marin-na/marin/"
RUN_ID=""
SUBMIT=false

GPU_REPLICAS=32
EXPERT_AXIS=8
REPLICA_AXIS=1
BATCH=256
SEQ_LEN=4096
STEPS=30
CHECKPOINTS="local"
DATA="slimpajama"
REMAT="save_moe"
MP="params=float32,compute=bfloat16,output=bfloat16"
TRACKER="wandb"
PROFILER_START=12
PROFILER_STEPS=8
ENABLE_HLO_PROTO=true
HOST_TRACER_LEVEL=1
PYTHON_TRACER_LEVEL=0

usage() {
    cat <<'EOF'
Usage:
  experiments/grug/moe/run_cw_may_d2560.sh [options]

Options:
  --submit                  Submit the Iris dispatcher job. Without this, print a dry-run summary.
  --run-id RUN_ID           Use a fixed run id instead of generating one.
  --env-file PATH           Load R2 credentials from PATH (default: $MARIN_R2_ENV_FILE or ~/.config/marin/marin-r2.env).
  --prefix URI              MARIN_PREFIX for outputs (default: s3://marin-na/marin/).
  --cluster NAME            Iris cluster name (default: cw-us-east-02a).
  --kubeconfig PATH         Kubeconfig path (default: $KUBECONFIG or ~/.kube/coreweave-iris-gpu).
  --nodes N                 H100 node count / MAY_GPU_REPLICAS (default: 32).
  --expert-axis N           MAY_EXPERT_AXIS (default: 8).
  --replica-axis N          MAY_REPLICA_AXIS (default: 1).
  --batch N                 MAY_BATCH (default: 256).
  --seq-len N               MAY_SEQ_LEN (default: 4096).
  --steps N                 MAY_STEPS (default: 30).
  --profiler-start N        MAY_PROFILER_START (default: 12).
  --profiler-steps N        MAY_PROFILER_STEPS (default: 8; set 0 to disable).
  --tracker NAME            MAY_TRACKER: wandb or json_logger (default: wandb).
  --data NAME               MAY_DATA: slimpajama or nemotron (default: slimpajama).
  --checkpoints MODE        MAY_CHECKPOINTS: local or s3 (default: local).
  --remat MODE              MAY_REMAT: save_moe or recompute_all (default: save_moe).
  --mp POLICY               MAY_MP policy string.
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
        --nodes)
            GPU_REPLICAS="$2"
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
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
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
        --mp)
            MP="$2"
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

ENV_ARGS=(
    -e MARIN_PREFIX "$MARIN_PREFIX"
    -e RUN_ID "$RUN_ID"
    -e AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID"
    -e AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY"
    -e AWS_ENDPOINT_URL "$AWS_ENDPOINT_URL"
    -e AWS_ENDPOINT_URL_S3 "$AWS_ENDPOINT_URL_S3"
    -e MAY_GPU_REPLICAS "$GPU_REPLICAS"
    -e MAY_EXPERT_AXIS "$EXPERT_AXIS"
    -e MAY_REPLICA_AXIS "$REPLICA_AXIS"
    -e MAY_BATCH "$BATCH"
    -e MAY_SEQ_LEN "$SEQ_LEN"
    -e MAY_STEPS "$STEPS"
    -e MAY_CHECKPOINTS "$CHECKPOINTS"
    -e MAY_DATA "$DATA"
    -e MAY_REMAT "$REMAT"
    -e MAY_MP "$MP"
    -e MAY_TRACKER "$TRACKER"
    -e MAY_PROFILER_START "$PROFILER_START"
    -e MAY_PROFILER_STEPS "$PROFILER_STEPS"
    -e MAY_PROFILER_ENABLE_HLO_PROTO "$ENABLE_HLO_PROTO"
    -e MAY_PROFILER_HOST_TRACER_LEVEL "$HOST_TRACER_LEVEL"
    -e MAY_PROFILER_PYTHON_TRACER_LEVEL "$PYTHON_TRACER_LEVEL"
)

for maybe_env in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT MAY_WANDB_GROUP; do
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
mesh axes: replica=$REPLICA_AXIS expert=$EXPERT_AXIS
batch: $BATCH
seq_len: $SEQ_LEN
steps: $STEPS
tracker: $TRACKER
profiler: start=$PROFILER_START steps=$PROFILER_STEPS hlo_proto=$ENABLE_HLO_PROTO
mp: $MP

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
