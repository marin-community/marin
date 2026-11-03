#!/bin/bash
# The acceptance measurement's launcher (README.md is the runbook). Run from anywhere; it works in this worktree.
#   launch.sh sample            draw sample.json from the finished n=25 run and freeze it at $ROOT/sample.json
#   launch.sh dry-run           render the three arms on the first 3 sampled proposals, no model (local)
#   launch.sh wave1             32 capabilities x 3 arms = 96 Iris jobs, one allocation each
#   launch.sh wave2 FROM TO     wave-2 capabilities with rank FROM..TO-1 (32 <= FROM < TO <= 192), 3 jobs each
#   launch.sh reduce [WAVE]     print the reducer's JSON for WAVE (1, 2 or all; default all)
# One job is one allocation (one author per 128 GB parent, the PC admission rule). Job names are deterministic
# (am-w<wave>-<arm>-r<rank>-<GEN>), so a duplicate submission is refused; relaunch a failed job with a new GEN.
# Even ranks use relay g, odd ranks relay f. Nothing here calls GLM; the jobs do. POLICY='{"solver.k": 4}' passes one
# --policy overlay to every job (and to dry-run); the manifest records it, so one run root holds one overlay.
set -euo pipefail
WT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$WT"
ROOT=${ROOT:-s3://marin-us-east-02a/marin/taskforge/acceptance-measure/20261102}
PROPOSALS=s3://marin-us-east-02a/marin/taskforge/proposals/curriculum-v3-n25-20261102/run
CATALOG=s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json
RELAY_EVEN=/muchanem/glm53-relay-rno2a-g-20261027T0414Z
RELAY_ODD=/muchanem/glm53-relay-rno2a-f-20261027T0147Z
ENV_FILE=/Users/k3sc0re/openathena/.env
GEN=${GEN:-g1}
POLICY_ARGS=()
[ -z "${POLICY:-}" ] || POLICY_ARGS=(--policy "$POLICY")
MODULE=experiments.post_training.capability_driven_envs.acceptance_measure
LOCAL=${LOCAL:-$HOME/.cache/taskforge-acceptance-measure}  # outside the worktree: untracked files ship in every Iris bundle
env_value() { sed -n "s/^$1=//p" "$ENV_FILE" | head -1 | tr -d '"'"'"; }
export CW_KEY_ID=$(env_value CW_STORAGE_KEY_ID) CW_KEY_SECRET=$(env_value CW_STORAGE_KEY_SECRET)
[ -n "$CW_KEY_ID" ] && [ -n "$CW_KEY_SECRET" ] || { echo "no CW storage key in $ENV_FILE" >&2; exit 1; }

secrets() {
  GLM_BULK_TOKEN=$(sed -n 's/^GLM_BULK_TOKEN=//p' /Users/k3sc0re/openathena/glm-infer/glm_bulk_token.txt)
  TASKFORGE_PARALLEL_KEY=$(env_value TASKFORGE_PARALLEL_KEY)
  TASKFORGE_REGISTRY_CREDENTIAL=$(uv run --frozen --package marin-taskforge-adapters python -c \
    'from taskforge_adapters.image_builder import registry_credential_json; print(registry_credential_json())')
  for v in GLM_BULK_TOKEN TASKFORGE_PARALLEL_KEY TASKFORGE_REGISTRY_CREDENTIAL; do
    [ -n "${!v}" ] || { echo "$v is empty" >&2; exit 1; }
  done
}

clean_head() {
  [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "commit first: the manifest records HEAD" >&2; exit 1; }
  GIT_SHA=$(git rev-parse HEAD)
}

submitted=0
submit() {  # wave rank shards arm
  local wave=$1 rank=$2 shards=$3 arm=$4 relay=$RELAY_EVEN reply
  [ $((rank % 2)) -eq 1 ] && relay=$RELAY_ODD
  reply=$(timeout 300 uv run --frozen iris --cluster=marin job run --no-wait \
    --job-name "am-w$wave-$(echo "$arm" | tr A-Z a-z)-r$(printf %03d "$rank")-$GEN" \
    --target-cluster cw-rno2a --cpu 32 --memory 128GB --disk 300GB --enable-extra-resources --no-preemptible \
    --max-retries 2 --timeout 259200 --secret-behavior isolate \
    -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" \
    -e TASKFORGE_REGISTRY_CREDENTIAL "$TASKFORGE_REGISTRY_CREDENTIAL" -e TASKFORGE_PARALLEL_KEY "$TASKFORGE_PARALLEL_KEY" \
    -e OMP_NUM_THREADS 32 -e RAYON_NUM_THREADS 32 -e CARGO_BUILD_JOBS 32 -e MAKEFLAGS j32 -- \
    uv run --frozen --package marin-taskforge-adapters python -u -m $MODULE.driver run ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} \
    --sample "$ROOT/sample.json" --output "$ROOT" --wave "$wave" --shard $((rank % shards)) --shards "$shards" \
    --arms "$arm" --git-sha "$GIT_SHA" --relay-job "$relay" --concurrency 1 --work-dir /tmp/acceptance-measure 2>&1) || true
  echo "$reply" | grep -oE "Job submitted: .*|.*rror.*" || true
  case "$reply" in *"Job submitted"*) submitted=$((submitted + 1)) ;; esac
}

case "${1:-}" in
  sample)
    mkdir -p "$LOCAL"
    out=$(uv run --frozen fsutil ls "$ROOT/sample.json" 2>/dev/null || true)
    [ -z "$out" ] || { echo "$ROOT/sample.json exists: the sample is frozen" >&2; exit 1; }
    uv run --frozen --package marin-taskforge-adapters python -m $MODULE.sample --proposals "$PROPOSALS" --catalog "$CATALOG" \
      --output "$LOCAL/sample.json"
    uv run --frozen fsutil cp "$LOCAL/sample.json" "$ROOT/sample.json" ;;
  dry-run)
    uv run --frozen --package marin-taskforge-adapters python -m $MODULE.driver dry-run ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} \
      --sample "$ROOT/sample.json" \
      --output "$LOCAL/dry-run" ;;
  wave1)
    clean_head; secrets
    for ((rank = 0; rank < 32; rank++)); do for arm in A B C; do submit 1 "$rank" 32 "$arm"; done; done
    echo "submitted $submitted of 96" ;;
  wave2)
    from=${2:?FROM}; to=${3:?TO}
    [ "$from" -ge 32 ] && [ "$to" -le 192 ] && [ "$from" -lt "$to" ] || { echo "need 32 <= FROM < TO <= 192" >&2; exit 2; }
    clean_head; secrets
    for ((rank = from; rank < to; rank++)); do for arm in A B C; do submit 2 "$rank" 160 "$arm"; done; done
    echo "submitted $submitted of $((3 * (to - from)))" ;;
  reduce)
    uv run --frozen --package marin-taskforge-adapters python -m $MODULE.reducer --output "$ROOT" --sample "$ROOT/sample.json" \
      --wave "${2:-all}" ;;
  *) echo "usage: launch.sh sample | dry-run | wave1 | wave2 FROM TO | reduce [WAVE]" >&2; exit 2 ;;
esac
