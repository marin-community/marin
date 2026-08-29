#!/usr/bin/env bash
# Submit one hero EP64 arm for the 8753-mfu loop (research/mcwitt/8753-mfu-loop, #8753 head base).
#
# Protocol (fixed for every iteration so arms stay comparable; see DESIGN.md):
#   - one NVL72 rack on cw-us-east-08a, PRODUCTION priority allowed with job RUNTIME <1h
#     (user 2026-08-28): ARM_TIMEOUT 28800s covers Kueue queue wait plus run (occupancy is
#     intrinsically bounded by the absolute NUM_STEPS stop); watchdog.sh COMPILE_CEILING=1800 +
#     STEP_BUDGET=1200 enforce the runtime cap from admission
#   - restore from the live hero's step-30000 checkpoint, mixture data (trained-router routing);
#     NUM_STEPS is the ABSOLUTE stop step (checkpoint step + window), never a relative count
#   - no checkpoints, no eval; profiler off on SCORED arms (a profile tail on an otherwise-scored
#     draw is allowed: PROFILE_STEPS>0 with PROFILE_START_STEP as an ABSOLUTE step past the
#     scoring window, e.g. 30021)
#   - no MARIN_PJRT_WHEEL overlay: the branch's uv.lock pins jax_cuda13_pjrt 0.11.1+marin.c9526e8c0272
#   - JAX_COMPILATION_CACHE_DIR rotated per run id (clique-init deadlock dodge); a RESUBMITTED
#     arm must use a fresh RID (and VERSION): reusing one merges W&B histories, reuses the
#     leader-populated compile cache (clique-deadlock recipe), and can vacuously reuse artifacts
#
# usage: RID=<run-id> VERSION=<calver> [REPO=<worktree>] [EXTRA_LAUNCH_ARGS="..."] arm.sh
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY}"
RID="${RID:?set RID}"
VERSION="${VERSION:?set VERSION -- bump it per arm or the artifact layer reuses the last run}"

# The watchdog addresses jobs at /mwittmann/...; make the submission land there no matter what
# the caller's shell has. (-e below only sets the remote env.)
export IRIS_USER=mwittmann

LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${LOOP_DIR}/../.." && pwd)}"

# Iris bundles the working tree, not HEAD: uncommitted edits (including another session writing
# to a shared checkout) would ship silently and break arm attribution. Only the loop dir's DATA
# files (logs, tsv, review transcripts) are exempt; uncommitted protocol scripts count as dirty,
# since arms.tsv records HEAD as the arm's provenance.
dirty="$(git -C "$REPO" status --porcelain | grep -Ev "autoresearch/.*\.(log|out|txt|tsv)$" || true)"
if [ -z "${ALLOW_DIRTY:-}" ] && [ -n "$dirty" ]; then
  echo "refusing to submit from a dirty tree (set ALLOW_DIRTY=1 to override):" >&2
  echo "$dirty" >&2
  exit 1
fi

# JAX_* env leaks into train tasks via dispatch forwarding, and this fork enables pipelined host
# offloading at optimization level O1+ regardless of its flag -- a leaked level would silently
# turn a control into an H10 treatment.
if [ -n "${JAX_OPTIMIZATION_LEVEL:-}" ]; then
  echo "JAX_OPTIMIZATION_LEVEL is set (${JAX_OPTIMIZATION_LEVEL}); unset it -- it changes compiler passes behind the arms' backs" >&2
  exit 1
fi

# A flag-based treatment must be deliberate: an ARM_XLA_FLAGS value leaking from the supervisor
# shell into a "control" draw silently runs the treatment and records nothing -- the null delta
# then reads as "dead lever". Controls run with TREATMENT unset and ARM_XLA_FLAGS empty.
if [ -n "${ARM_XLA_FLAGS:-}" ] && [ -z "${TREATMENT:-}" ]; then
  echo "ARM_XLA_FLAGS is set but TREATMENT=1 is not; refusing (is this a contaminated control?)" >&2
  exit 1
fi

# One RID/VERSION per submission, ever: a reused RID merges W&B histories, resurrects the
# leader-populated compile cache (clique-deadlock recipe), and can vacuously reuse artifacts.
if [ -f "${LOOP_DIR}/arms.tsv" ] && awk -F'\t' -v rid="$RID" -v ver="$VERSION" \
    'NR>1 && ($1==rid || $3==ver) {found=1} END {exit !found}' "${LOOP_DIR}/arms.tsv"; then
  echo "RID ${RID} or VERSION ${VERSION} already appears in arms.tsv; pick fresh ones" >&2
  exit 1
fi

PRIORITY="${PRIORITY:-production}"
# Covers Kueue queue wait PLUS the run: same-band gangs queue FIFO behind whatever holds the
# rack, and the coordinator's timeout clock runs while its child sits in the gate. Occupancy
# after admission is intrinsically bounded by NUM_STEPS (~30 steps past restore); the watchdog
# enforces the <1h-runtime rule from admission.
ARM_TIMEOUT="${ARM_TIMEOUT:-28800}"
NUM_STEPS="${NUM_STEPS:-30030}"
TRAINING_DATA="${TRAINING_DATA:-mixture}"
MASTER_PARAMS="${MASTER_PARAMS:-device}"
MOE_IMPL="${MOE_IMPL:-ragged_all_to_all}"
RESTORE_FROM="${RESTORE_FROM:-s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-30000}"
RESTORE_ARGS=()
if [ -n "$RESTORE_FROM" ]; then
  RESTORE_ARGS=(--restore-from "$RESTORE_FROM")
fi
# The learning-rate schedule's length; fixed across arms so losses stay comparable.
SCHEDULE_STEPS=4470000

read -r -a EXTRA <<<"${EXTRA_LAUNCH_ARGS:-}"

cd "$REPO"
uv run iris --config lib/iris/config/marin.yaml job run \
  --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority "${PRIORITY}" \
  --cpu 2 --memory 8GB --disk 32GB --timeout "${ARM_TIMEOUT}" \
  --job-name "${RID}-coord" \
  -e IRIS_USER mwittmann \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e WANDB_PROJECT marin_moe \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -e IRIS_PORT_JAX 32703 \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e JAX_COMPILATION_CACHE_DIR "s3://marin-us-east-02a/marin/tmp/ttl=30d/jaxcache/${RID}" \
  -e XLA_FLAGS "${ARM_XLA_FLAGS:-}" \
  -e TF_CPP_MIN_LOG_LEVEL 0 \
  -e TF_CPP_VMODULE "hlo_rematerialization=1,execution_stream_assignment=1,collective_pipeliner=1" \
  -- python -m experiments.grug.moe_hero_ep.launch_diagnostics \
     --run-id "${RID}" \
     --dp-racks 1 --num-steps "${NUM_STEPS}" --schedule-steps "${SCHEDULE_STEPS}" \
     --capacity-factor 1.15 \
     --moe-implementation "${MOE_IMPL}" \
     --processes-per-task 4 \
     --master-params "${MASTER_PARAMS}" \
     --training-data "${TRAINING_DATA}" \
     --profile-steps "${PROFILE_STEPS:-0}" --profile-start-step "${PROFILE_START_STEP:-30021}" \
     --watch-interval 0 --eval-every 0 \
     --no-save-checkpoints \
     "${RESTORE_ARGS[@]}" \
     "${EXTRA[@]}" \
     --version "${VERSION}" --run >"${LOOP_DIR}/${RID}-submit.log" 2>&1 \
  || { echo "submit FAILED, tail of ${RID}-submit.log:" >&2; tail -20 "${LOOP_DIR}/${RID}-submit.log" >&2; exit 1; }

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${RID}" "$(git -C "$REPO" rev-parse --short HEAD)" "${VERSION}" "${MOE_IMPL}" "${TRAINING_DATA}" \
  "${MASTER_PARAMS}" "${SCHEDULE_STEPS}" "${ARM_TIMEOUT}" "${RESTORE_FROM:-none}" \
  "${EXTRA_LAUNCH_ARGS:-none}" "${ARM_XLA_FLAGS:-none}" >> "${LOOP_DIR}/arms.tsv"

echo "submitted ${RID} at ${PRIORITY} priority (timeout ${ARM_TIMEOUT}s, commit $(git -C "$REPO" rev-parse --short HEAD), data ${TRAINING_DATA}, extra: ${EXTRA_LAUNCH_ARGS:-none}, xla_flags: ${ARM_XLA_FLAGS:-none})"
