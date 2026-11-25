#!/bin/bash
# Scaled-down Snowball rehearsal on the 2-node Vultr MI355X cluster, all phases in one allocation:
#   K  pretrain 8K, 0-400, SIGKILL of one process after the step-150 checkpoint, auto-resumed by this script
#   D  batch doubling from K's step-300 checkpoint, 300-400
#   C  64K cooldown from D's step-400 checkpoint, 400-430, 2 microbatches, ZeRO-1
#   R  pretrain 8K, 0-400, uninterrupted, the reference K is compared against
# Submit from the checkout root:
#   sbatch -A openathena-ihodes -N2 -t 2:30:00 -J snowball-e2e -o logs/e2e/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/rehearsal.sbatch
# PHASES (default "K D C R") picks phases; TAG names the runs (default: the job id); C_RUN renames the cooldown run.
# STEPS overrides every phase's last step (a short smoke run: PHASES=R STEPS=30).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source experiments/amd/june_67b_a2b/vultr/launch_env.sh
DATA=$WORK/snowball-data/snowball_v2_subset20
VAL=$WORK/snowball-data/validation
OUT=$WORK/e2e-rehearsal/runs
TAG=${TAG:-$SLURM_JOB_ID}
PHASES=${PHASES:-K D C R}
MAX_RESTARTS=${MAX_RESTARTS:-3}
mkdir -p "$OUT" logs/e2e
BASE_XLA_FLAGS=$XLA_FLAGS
# Bitwise resume comparison needs deterministic reductions in K and R.
DET_XLA_FLAGS="$BASE_XLA_FLAGS --xla_gpu_deterministic_ops=true"

echo "commit $(cat COMMIT) nodes $SLURM_JOB_NODELIST phases '$PHASES' tag $TAG start $(date -u +%FT%T)"
env | grep -E '^NCCL_|^RCCL_' | sort
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# run_phase NAME XLA_FLAGS ARGS...: srun one process per node, restarting from the latest checkpoint on failure.
run_phase() {
  local name=$1 flags=$2; shift 2
  local attempt=0 rc=0
  while :; do
    echo "=== $name attempt $attempt start $(date -u +%FT%T)"
    XLA_FLAGS="$flags" srun_with_hang_recovery "logs/e2e/$name" \
      $PY -m experiments.amd.june_67b_a2b.rehearsal --log-dir "logs/e2e/$name-jsonlogs" --data-root "$DATA" \
        --validation-root "$VAL" --output-root "$OUT" ${STEPS:+--steps "$STEPS"} "$@"
    rc=$?
    echo "=== $name attempt $attempt exit $rc end $(date -u +%FT%T)"
    for t in 0 1; do mv -f "logs/e2e/$name-$t.log" "logs/e2e/$name-$t.attempt$attempt.log" 2>/dev/null; done
    [ $rc -eq 0 ] && return 0
    attempt=$((attempt + 1))
    [ $attempt -gt "$MAX_RESTARTS" ] && return $rc
  done
}

# kill_after_checkpoint RUN STEP DELAY: SIGKILL the training process on the second node DELAY s after the
# step-STEP checkpoint of RUN is committed.
kill_after_checkpoint() {
  local run=$1 step=$2 delay=$3
  while [ ! -f "$OUT/$run/checkpoints/step-$step/metadata.json" ]; do sleep 5; done
  sleep "$delay"
  echo "=== SIGKILL rehearsal on ${NODES[1]} at $(date -u +%FT%T)"
  srun --overlap --jobid "$SLURM_JOB_ID" -N1 -n1 -w "${NODES[1]}" pkill -KILL -f "june_67b_a2b.rehearsal --log-dir"
}

K=e2e-$TAG-K
D=e2e-$TAG-D
C=${C_RUN:-e2e-$TAG-C}
R=e2e-$TAG-R
status=0
for phase in $PHASES; do
  case $phase in
    K)
      kill_after_checkpoint "$K" 150 "${KILL_DELAY:-20}" &
      killer=$!
      run_phase "$K" "$DET_XLA_FLAGS" --phase pretrain --run-id "$K" || status=1
      kill "$killer" 2>/dev/null
      ;;
    D) run_phase "$D" "$BASE_XLA_FLAGS" --phase double --run-id "$D" \
         --initialize-from "$OUT/$K/checkpoints/step-300" || status=1 ;;
    C) run_phase "$C" "$BASE_XLA_FLAGS" --phase cooldown --run-id "$C" \
         --initialize-from "$OUT/$D/checkpoints/step-400" || status=1 ;;
    R) run_phase "$R" "$DET_XLA_FLAGS" --phase pretrain --run-id "$R" || status=1 ;;
  esac
done
echo "done status $status end $(date -u +%FT%T)"
exit $status
