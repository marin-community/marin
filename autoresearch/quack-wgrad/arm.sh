#!/usr/bin/env bash
# Submit one hero EP64 arm for the QuACK varlen-k weight-gradient A/B.
#
# Two trees, differing only in which kernel family computes dw13/dw2:
#   control    5be7ff19e2  #8753 head + the cuDNN grouped-Wgrad alignment fix
#   treatment  this branch  the same, with both weight gradients on QuACK's varlen-k GEMM
#
# The control carries the alignment fix because without it the cuDNN path computes weight
# gradients from an over-read (issue #8339), and a loss comparison against a tree that is
# silently wrong would mean nothing.
#
# Protocol is inherited verbatim from the 8753-mfu campaign so the numbers are comparable:
#   - one EP64 slice of an NVL72 rack on cw-us-east-08a, restore from the live hero's latest
#     durable step-42000 checkpoint,
#     mixture data; NUM_STEPS is the ABSOLUTE stop step, never a relative count
#   - score the run median of throughput/mfu over restore steps +5..+19, with drops and the
#     pointwise loss series as fidelity guards
#   - same-session interleaved draws only
#   - no checkpoints, no eval, profiler off on scored arms
#   - JAX_COMPILATION_CACHE_DIR rotated per run id (clique-init deadlock dodge); a RESUBMITTED
#     arm must use a fresh RID and VERSION
#
# usage: RID=<run-id> VERSION=<calver> [PORT=<int>] [REPO=<worktree>] arm.sh
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
# DK_CTAS_PER_SM became a KEPT setting (H11), so a control legitimately carries it. KEPT_FLAGS=1
# says "this value is the kept stack, not a treatment"; without either marker a stray value is
# still refused. An empty string reaches the plugin as atoi("")==0, which falls back to the
# default 8/SM grid -- so an unset knob means "kept default", never "CTAS=1 silently lost".
if [ -n "${DK_CTAS_PER_SM:-}" ] && [ -z "${TREATMENT:-}" ] && [ -z "${KEPT_FLAGS:-}" ]; then
  echo "DK_CTAS_PER_SM is set but neither TREATMENT=1 nor KEPT_FLAGS=1 is; refusing (contaminated control?)" >&2
  exit 1
fi
# The absolute-count knob (H21) overrides the per-SM one in the plugin, so it is always a
# treatment. It only exists in the H21 wheel: setting it without that wheel reaches a plugin that
# never reads it and scores as a vacuous null, which is indistinguishable from the lever failing.
if [ -n "${DK_CTAS_TOTAL:-}" ]; then
  if [ -z "${TREATMENT:-}" ]; then
    echo "DK_CTAS_TOTAL is set but TREATMENT=1 is not; refusing" >&2
    exit 1
  fi
  case "${EXTRA_LAUNCH_ARGS:-}" in
    *pjrt-h21-cta-absolute*) ;;
    *) echo "DK_CTAS_TOTAL is set but EXTRA_LAUNCH_ARGS does not carry the H21 wheel; that plugin does not read the variable and the arm would score as a vacuous null" >&2; exit 1 ;;
  esac
fi

# One RID/VERSION per submission, ever: a reused RID merges W&B histories, resurrects the
# leader-populated compile cache (clique-deadlock recipe), and can vacuously reuse artifacts.
if [ -f "${LOOP_DIR}/arms.tsv" ] && awk -F'\t' -v rid="$RID" -v ver="$VERSION" \
    'NR>1 && ($1==rid || $3==ver) {found=1} END {exit !found}' "${LOOP_DIR}/arms.tsv"; then
  echo "RID ${RID} or VERSION ${VERSION} already appears in arms.tsv; pick fresh ones" >&2
  exit 1
fi

PRIORITY="${PRIORITY:-interactive}"
# Covers Kueue queue wait PLUS the run: same-band gangs queue FIFO behind whatever holds the
# rack, and the coordinator's timeout clock runs while its child sits in the gate. Occupancy
# after admission is intrinsically bounded by NUM_STEPS (~30 steps past restore); the watchdog
# enforces the <1h-runtime rule from admission.
ARM_TIMEOUT="${ARM_TIMEOUT:-28800}"
NUM_STEPS="${NUM_STEPS:-42030}"
TRAINING_DATA="${TRAINING_DATA:-mixture}"
MASTER_PARAMS="${MASTER_PARAMS:-device}"
MOE_IMPL="${MOE_IMPL:-ragged_all_to_all}"
RESTORE_FROM="${RESTORE_FROM:-s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints/step-42000}"
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
  -e IRIS_PORT_JAX "${PORT:-32711}" \
  -e AWS_MAX_ATTEMPTS 25 -e AWS_RETRY_MODE adaptive \
  -e JAX_COMPILATION_CACHE_DIR "s3://hero-checkpoints/tmp/ttl=30d/jaxcache/${RID}" \
  -e XLA_FLAGS "${ARM_XLA_FLAGS:-}" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION "${MEM_FRACTION:-0.75}" \
  -e XLA_RAGGED_A2A_DK_CTAS_PER_SM "${DK_CTAS_PER_SM:-}" \
  -e XLA_RAGGED_A2A_DK_CTAS_TOTAL "${DK_CTAS_TOTAL:-}" \
  -e TF_CPP_MIN_LOG_LEVEL 0 \
  -e TF_CPP_VMODULE "hlo_rematerialization=1,execution_stream_assignment=1,collective_pipeliner=1,ragged_all_to_all_thunk=3" \
  -- python -m experiments.grug.moe_hero_ep.launch_diagnostics \
     --run-id "${RID}" \
     --dp-racks 1 --num-steps "${NUM_STEPS}" --schedule-steps "${SCHEDULE_STEPS}" \
     --capacity-factor 1.15 \
     --moe-implementation "${MOE_IMPL}" \
     --processes-per-task 4 \
     --master-params "${MASTER_PARAMS}" \
     --training-data "${TRAINING_DATA}" \
     --profile-steps "${PROFILE_STEPS:-0}" --profile-start-step "${PROFILE_START_STEP:-42021}" \
     --watch-interval 0 --eval-every 0 \
     --no-save-checkpoints \
     "${RESTORE_ARGS[@]}" \
     "${EXTRA[@]}" \
     --version "${VERSION}" --run >"${LOOP_DIR}/${RID}-submit.log" 2>&1 \
  || { echo "submit FAILED, tail of ${RID}-submit.log:" >&2; tail -20 "${LOOP_DIR}/${RID}-submit.log" >&2; exit 1; }

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${RID}" "$(git -C "$REPO" rev-parse --short HEAD)" "${VERSION}" "${MOE_IMPL}" "${TRAINING_DATA}" \
  "${MASTER_PARAMS}" "${SCHEDULE_STEPS}" "${ARM_TIMEOUT}" "${RESTORE_FROM:-none}" \
  "${EXTRA_LAUNCH_ARGS:-none}" "${ARM_XLA_FLAGS:-none}" "${MEM_FRACTION:-0.75}" "${DK_CTAS_PER_SM:-none}" \
  "${DK_CTAS_TOTAL:-none}" >> "${LOOP_DIR}/arms.tsv"

echo "submitted ${RID} at ${PRIORITY} priority (timeout ${ARM_TIMEOUT}s, commit $(git -C "$REPO" rev-parse --short HEAD), data ${TRAINING_DATA}, extra: ${EXTRA_LAUNCH_ARGS:-none}, xla_flags: ${ARM_XLA_FLAGS:-none}, memfrac: ${MEM_FRACTION:-0.75}, dk_ctas: ${DK_CTAS_PER_SM:-none}, dk_total: ${DK_CTAS_TOTAL:-none})"
