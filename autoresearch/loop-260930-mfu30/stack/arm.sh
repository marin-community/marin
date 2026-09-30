#!/usr/bin/env bash
# Submit one mfu30 rack arm from a checkout of the stack branch.
#
#   arm.sh <run-id> <iris-port> [--xla "<flags>"] [--env NAME=VALUE]... [--trace] [-- <launch_diagnostics args>]
#
# Protocol: seed 0, restore step-180000 of the pinned hero checkpoint, stop at 180060, interactive
# priority, no coordinator --timeout (it counts queue time). --trace profiles 180021-180023 for a
# PGLE build or an anatomy; scoring excludes those steps. XLA flags are merged with the hero
# defaults in experiments/grug/moe_hero_ep/train.py. Run from the checkout whose code the arm must
# run: the Iris bundle is the working tree.
set -euo pipefail
RUN="${1:?run id}"; PORT="${2:?iris port}"; shift 2
XLA=""; TRACE=0; EXTRA=(); ENV_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --xla) XLA="$2"; shift 2 ;;
    --trace) TRACE=1; shift ;;
    --env) ENV_ARGS+=(-e "${2%%=*}" "${2#*=}"); shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
CKPT=s3://marin-us-east-02a/marin/grug/hero-fa4sm100-nomask-step146k/2026.08.19.2/checkpoints/step-180000
cd "$(git rev-parse --show-toplevel)"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "refusing to submit from a dirty checkout; commit first" >&2; exit 1
fi
echo "submitting ${RUN} from $(git rev-parse --short HEAD) xla=[${XLA}] env=[${ENV_ARGS[*]:-}] trace=${TRACE} extra=[${EXTRA[*]:-}]"
WANDB_API_KEY=$(cat /run/user/1003/agenix/wandb-api-key)
ENVS=(-e WANDB_API_KEY "${WANDB_API_KEY}" -e WANDB_PROJECT marin_moe -e IRIS_PORT_JAX "${PORT}")
if [ -n "${XLA}" ]; then ENVS+=(-e XLA_FLAGS "${XLA}"); fi
ENVS+=("${ENV_ARGS[@]}")
PROFILE=()
if [ "${TRACE}" = 1 ]; then PROFILE=(--profile-start-step 180021 --profile-steps 3); fi
IRIS_USER=mwittmann uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive --cpu 2 --memory 8GB --disk 32GB \
  --job-name "${RUN}-coord" "${ENVS[@]}" -- \
  python -m experiments.grug.moe_hero_ep.launch_diagnostics --run-id "${RUN}" --seed 0 --num-steps 180060 \
  --schedule-steps 390251 --batch-size 1024 --gc-interval 100 --restore-from "${CKPT}" --version dev \
  "${PROFILE[@]}" "${EXTRA[@]}" --run
