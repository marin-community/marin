#!/usr/bin/env bash
# Launch the Grug MoE CoreWeave scale recipe from launch_cw_scale.py.

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
MODE="smoke"
SUBMIT=false
RUN_ID=""
WORKER_CPU=32
TRACKER="json_logger"
CHECKPOINTS=""

usage() {
    cat <<'EOF'
Usage:
  experiments/grug/moe/run_cw_scale.sh [options]

Options:
  --submit              Submit the Iris dispatcher job. Without this, print a dry-run summary.
  --smoke               4-node sanity run (default): d1024, 16 layers, batch 64, 10 steps.
  --full                32-node 90B-total run: launcher defaults, batch 256, 50 steps.
  --env-file PATH       Load credentials from PATH (default: $MARIN_R2_ENV_FILE or ~/.config/marin/marin-r2.env).
  --run-id RUN_ID       Use a fixed run id instead of generating one.
  --prefix URI          MARIN_PREFIX for outputs (default: s3://marin-na/marin/).
  --cluster NAME        Iris cluster name (default: cw-us-east-02a).
  --kubeconfig PATH     Kubeconfig path (default: $KUBECONFIG or ~/.kube/coreweave-iris-gpu).
  --worker-cpu N        SCALE_CPU_PER_REPLICA for each H100 worker pod (default: 32).
  --tracker NAME        SCALE_TRACKER: json_logger or wandb (default: json_logger).
  --checkpoints MODE    SCALE_CHECKPOINTS: local or s3 (default: local for --smoke, launcher default for --full).
  -h, --help            Show this help.

Advanced shape/debug overrides:
  Set SCALE_GPU_REPLICAS, SCALE_HIDDEN_DIM, SCALE_NUM_LAYERS, SCALE_NUM_EXPERTS,
  SCALE_TOP_K, SCALE_BATCH, SCALE_SEQ_LEN, SCALE_STEPS, SCALE_CHECKPOINTS,
  SCALE_REMAT, SCALE_MP, SCALE_PROFILER_STEPS, or SCALE_PROFILER_START in the
  environment; this wrapper forwards them to the remote launcher.

Credential input:
  The env file may contain R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ENDPOINT_URL.
  If it instead contains CLOUDFLARE_ACCOUNT_ID/CLOUDFLARE_API_TOKEN, the script
  derives the R2 S3 credentials using scripts/iris/cloudflare_r2_env.sh.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --submit)
            SUBMIT=true
            shift
            ;;
        --smoke)
            MODE="smoke"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --env-file)
            ENV_FILE="$2"
            ENV_FILE_EXPLICIT=true
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
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
        --worker-cpu)
            WORKER_CPU="$2"
            shift 2
            ;;
        --tracker)
            TRACKER="$2"
            shift 2
            ;;
        --checkpoints)
            CHECKPOINTS="$2"
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
    RUN_ID="cw-grugmoe-${MODE}-$(date +%Y%m%d-%H%M%S)"
fi

COMMON_ENV=(
    -e MARIN_PREFIX "$MARIN_PREFIX"
    -e RUN_ID "$RUN_ID"
    -e AWS_ACCESS_KEY_ID "$AWS_ACCESS_KEY_ID"
    -e AWS_SECRET_ACCESS_KEY "$AWS_SECRET_ACCESS_KEY"
    -e AWS_ENDPOINT_URL "$AWS_ENDPOINT_URL"
    -e AWS_ENDPOINT_URL_S3 "$AWS_ENDPOINT_URL_S3"
    -e SCALE_CPU_PER_REPLICA "$WORKER_CPU"
    -e SCALE_TRACKER "$TRACKER"
)
if [ -n "$CHECKPOINTS" ]; then
    COMMON_ENV+=(-e SCALE_CHECKPOINTS "$CHECKPOINTS")
fi

OPTIONAL_SCALE_ENV_NAMES=(
    SCALE_GPU_REPLICAS
    SCALE_EXPERT_AXIS
    SCALE_REPLICA_AXIS
    SCALE_HIDDEN_DIM
    SCALE_NUM_LAYERS
    SCALE_NUM_EXPERTS
    SCALE_TOP_K
    SCALE_BATCH
    SCALE_SEQ_LEN
    SCALE_STEPS
    SCALE_CHECKPOINTS
    SCALE_REMAT
    SCALE_MP
    SCALE_PROFILER_STEPS
    SCALE_PROFILER_START
)
for optional_name in "${OPTIONAL_SCALE_ENV_NAMES[@]}"; do
    if [ "$optional_name" = "SCALE_CHECKPOINTS" ] && [ -n "$CHECKPOINTS" ]; then
        continue
    fi
    optional_value="${!optional_name-}"
    if [ -n "$optional_value" ]; then
        COMMON_ENV+=(-e "$optional_name" "$optional_value")
    fi
done

SCALE_ENV=()
if [ "$MODE" = "smoke" ]; then
    SCALE_ENV=(
        -e SCALE_GPU_REPLICAS 4
        -e SCALE_HIDDEN_DIM 1024
        -e SCALE_NUM_LAYERS 16
        -e SCALE_BATCH 64
        -e SCALE_STEPS 10
    )
    if [ -z "$CHECKPOINTS" ] && [ -z "${SCALE_CHECKPOINTS:-}" ]; then
        SCALE_ENV+=(-e SCALE_CHECKPOINTS local)
    fi
fi

DISPLAY_CHECKPOINTS="${CHECKPOINTS:-${SCALE_CHECKPOINTS:-}}"
if [ -z "$DISPLAY_CHECKPOINTS" ] && [ "$MODE" = "smoke" ]; then
    DISPLAY_CHECKPOINTS="local"
elif [ -z "$DISPLAY_CHECKPOINTS" ]; then
    DISPLAY_CHECKPOINTS="launcher default"
fi

CMD=(
    uv run --package marin-iris --extra controller iris --cluster="$CLUSTER"
    job run --no-wait
    --memory=2G --disk=4G --cpu=1 --extra=cpu
    "${COMMON_ENV[@]}"
)
if [ "${#SCALE_ENV[@]}" -gt 0 ]; then
    CMD+=("${SCALE_ENV[@]}")
fi
CMD+=(-- python -m experiments.grug.moe.launch_cw_scale)

if [ "$SUBMIT" != true ]; then
    cat <<EOF
Dry run: not submitting. Add --submit to launch.
cluster: $CLUSTER
mode: $MODE
run_id: $RUN_ID
kubeconfig: $KUBECONFIG
prefix: $MARIN_PREFIX
r2_endpoint: $AWS_ENDPOINT_URL
worker_cpu: $WORKER_CPU
tracker: $TRACKER
checkpoints: $DISPLAY_CHECKPOINTS

Command shape:
  uv run --package marin-iris --extra controller iris --cluster=$CLUSTER job run --no-wait ... -- python -m experiments.grug.moe.launch_cw_scale
EOF
    exit 0
fi

echo "Submitting Grug MoE CoreWeave scale run: $RUN_ID"
"${CMD[@]}"
