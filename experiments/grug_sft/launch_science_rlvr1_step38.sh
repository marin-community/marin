#!/usr/bin/env bash
set -euo pipefail

MARIN_CHECKOUT=${MARIN_CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
SECRETS_ENV=${SECRETS_ENV:-}
VERSION=${1:-2026.09.25-v3}

if [[ -z ${WANDB_API_KEY:-} && -n $SECRETS_ENV && -f $SECRETS_ENV ]]; then
  WANDB_API_KEY=$(uv run --no-project python -c 'import sys; from dotenv import dotenv_values; print(dotenv_values(sys.argv[1]).get("WANDB_API_KEY") or "")' "$SECRETS_ENV")
fi
: "${WANDB_API_KEY:?Set WANDB_API_KEY or set SECRETS_ENV to a file containing it}"

cd "$MARIN_CHECKOUT"
if (( $# > 1 )); then
  mixes=("${@:2}")
else
  mixes=(balanced proof-first science-forward)
fi
for mix in "${mixes[@]}"; do
  case "$mix" in
    balanced|proof-first|science-forward) ;;
    *) echo "Unknown mix: $mix" >&2; exit 2 ;;
  esac
  uv run iris --cluster=cw-rno2a job run \
    --job-name "science-rlvr1-step38-${mix}-coord-${VERSION}" \
    --priority interactive --no-wait --enable-extra-resources \
    --cpu 4 --memory 16GB --disk 20GB --extra cpu \
    -e MARIN_PREFIX s3://marin-us-east-02a/marin \
    -e JAX_COMPILATION_CACHE_DIR "/tmp/science-rlvr1-step38-${mix}-${VERSION}-jax-cache" \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_MODE online -e NCCL_DEBUG WARN \
    -- python -m experiments.grug_sft.science_rlvr1_step38 --mix "$mix" --version "$VERSION"
done
