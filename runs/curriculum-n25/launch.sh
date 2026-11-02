#!/bin/bash
# Launch proposal shards on Iris (cw-rno2a, through the marin hub).
#   launch.sh smoke              offline startup checks (smoke.py); sends no completion
#   launch.sh pilot              2 capabilities x 25 slots in 1 shard into $ROOT/pilot
#   launch.sh full [SHARDS] [C]  all capabilities, SHARDS jobs (default 8) at C in flight each (default 40)
# POLICY_FILE=runs/curriculum-n25/policies/<name>.txt (relative to the worktree) replaces the stack's
# VERIFICATION_RULE; without it the stack's rule is used. A root holding results of another prompt refuses
# to start, so a different policy needs a different ROOT. Shards resume from S3; even shards use relay g,
# odd shards relay f. Relaunch under a NEW job name (keep post-mortems). Live limit: write_control.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$WT"
STAMP=${STAMP:-$(date -u +%m%d%H%M)}

if [ "${1:-}" = smoke ]; then
  exec uv run --frozen --package marin-taskforge runs/curriculum-n25/smoke.py "$CATALOG" "$RELAY_EVEN" "$RELAY_ODD"
fi

[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "worktree has uncommitted changes; commit first" >&2; exit 1; }
GIT_SHA=$(git rev-parse HEAD)
GLM_BULK_TOKEN=$(sed -n 's/^GLM_BULK_TOKEN=//p' /Users/k3sc0re/openathena/glm-infer/glm_bulk_token.txt)
[ -n "$GLM_BULK_TOKEN" ] || { echo "missing GLM bulk token" >&2; exit 1; }
POLICY_ARGS=()
if [ -n "${POLICY_FILE:-}" ]; then
  case "$POLICY_FILE" in /*) echo "POLICY_FILE must be relative to $WT" >&2; exit 1 ;; esac
  [ -f "$POLICY_FILE" ] || { echo "no policy file $WT/$POLICY_FILE" >&2; exit 1; }
  git ls-files --error-unmatch "$POLICY_FILE" >/dev/null || { echo "$POLICY_FILE is not committed" >&2; exit 1; }
  POLICY_ARGS=(--verification-rule-file "$POLICY_FILE")
fi
echo "git $GIT_SHA policy ${POLICY_FILE:-stack VERIFICATION_RULE} root $ROOT"

submitted=0
submit() {  # name shard shards concurrency out relay [extra args...]
  local name=$1 shard=$2 shards=$3 conc=$4 out=$5 relay=$6 reply; shift 6
  reply=$(timeout 300 uv run --frozen iris --cluster=marin job run --no-wait --job-name "$name" \
    --target-cluster cw-rno2a --cpu 4 --memory 16GB --disk 20GB --enable-extra-resources \
    --max-retries 5 --timeout 259200 \
    -e GLM_BULK_TOKEN "$GLM_BULK_TOKEN" -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" -- \
    uv run --frozen --package marin-taskforge python -u -m \
    experiments.post_training.capability_driven_envs.propose_curriculum \
    --catalog "$CATALOG" --output-root "$out" --shard "$shard" --shards "$shards" --slots 25 \
    --concurrency "$conc" --max-tokens "${MAX_TOKENS:-131072}" --relay-job "$relay" --git-sha "$GIT_SHA" \
    ${POLICY_ARGS[@]+"${POLICY_ARGS[@]}"} "$@" 2>&1) || true
  echo "$reply" | grep -oE "Job submitted: .*|.*rror.*" || true
  case "$reply" in *"Job submitted"*) submitted=$((submitted + 1)) ;; esac
}

case "${1:-}" in
  pilot)
    shards=1
    submit "tfprop-pilot-$STAMP" 0 1 60 "$ROOT/pilot" "$RELAY_EVEN" \
      --capability d01.algebra.recurrences --capability d27.reporting.cash_flows ;;
  full)
    shards=${2:-8} conc=${3:-40}
    for ((s = 0; s < shards; s++)); do
      relay=$RELAY_EVEN; [ $((s % 2)) -eq 1 ] && relay=$RELAY_ODD
      submit "tfprop-$STAMP-s$(printf %02d $s)of$shards" "$s" "$shards" "$conc" "$ROOT/run" "$relay"
    done ;;
  *) echo "usage: launch.sh smoke | pilot | full [SHARDS] [CONCURRENCY]" >&2; exit 2 ;;
esac
echo "submitted $submitted of $shards"
[ "$submitted" -eq "$shards" ]
