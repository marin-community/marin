#!/bin/bash
# Run recovery_drill.py on one or two Vultr MI355X nodes in the Snowball launch environment, one process per node
# through srun_with_hang_recovery, restarting it from the run's latest checkpoint after any failure or hang, up to
# MAX_RESTARTS (default 3) times. DRILL_ARGS holds the drill's arguments past the data roots, e.g.
# "--size full --steps 60 --checkpoint-every 10 --nan-at 25 --damage-inputs-at 27".
# ABORT_AFTER_CHECKPOINT=N sends SIGABRT to the training process on the last node ABORT_DELAY (default 30) seconds after
# the step-N checkpoint commits, once. SCRATCH_RECLAIM_OFF=0 leaves HSA_NO_SCRATCH_RECLAIM unset. Submit from the
# checkout root:
#   DRILL_ARGS="..." sbatch -A openathena-ihodes -N2 -t 1:30:00 -J drill -o logs/drill/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/recovery_drill.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
source experiments/amd/june_67b_a2b/vultr/launch_env.sh
[ "${SCRATCH_RECLAIM_OFF:-1}" = 1 ] || unset HSA_NO_SCRATCH_RECLAIM
OUT=$WORK/corruption-recovery/runs
RUN=${RUN:-drill-$SLURM_JOB_ID}
MAX_RESTARTS=${MAX_RESTARTS:-3}
LOGS=logs/drill/$RUN
mkdir -p "$OUT" "$LOGS"

# abort_after_checkpoint STEP DELAY: SIGABRT the training process on the last node DELAY s after the step-STEP
# checkpoint commits.
abort_after_checkpoint() {
  local step=$1 delay=$2 node
  node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -1)
  while [ ! -f "$OUT/$RUN/checkpoints/step-$step/metadata.json" ]; do sleep 5; done
  sleep "$delay"
  echo "=== SIGABRT on $node at $(date -u +%FT%T)"
  srun --overlap --jobid "$SLURM_JOB_ID" -N1 -n1 -w "$node" pkill -ABRT -f "june_67b_a2b.vultr.recovery_drill"
}
if [ -n "${ABORT_AFTER_CHECKPOINT:-}" ]; then
  abort_after_checkpoint "$ABORT_AFTER_CHECKPOINT" "${ABORT_DELAY:-30}" &
fi

echo "commit $(cat COMMIT) nodes $SLURM_JOB_NODELIST run $RUN HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-unset} HANG_TIMEOUT=$HANG_TIMEOUT args [${DRILL_ARGS:-}] start $(date -u +%FT%T)"
attempt=0
while :; do
  echo "=== attempt $attempt start $(date -u +%FT%T)"
  srun_with_hang_recovery "$LOGS/attempt$attempt" \
    $PY -m experiments.amd.june_67b_a2b.vultr.recovery_drill --gpus-per-node 8 --replica-axis "$SLURM_JOB_NUM_NODES" \
      --run-id "$RUN" --output-root "$OUT" --log-dir "$LOGS/json" \
      --data-root "$WORK/snowball-data/snowball_v2_subset20" --validation-root "$WORK/snowball-data/validation" \
      --compilation-cache-dir "$WORK/jax-cache-drill" ${DRILL_ARGS:-} < /dev/null
  rc=$?
  echo "=== attempt $attempt exit $rc end $(date -u +%FT%T)"
  [ $rc -eq 0 ] && break
  attempt=$((attempt + 1))
  [ $attempt -gt "$MAX_RESTARTS" ] && break
done
echo "done rc $rc end $(date -u +%FT%T)"
exit $rc
