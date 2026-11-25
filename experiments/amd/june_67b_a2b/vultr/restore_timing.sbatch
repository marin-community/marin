#!/bin/bash
# Save the full 67B-A2B state (Adam) once to /work, then restore it three times, all in one allocation:
#   save     2 steps of the synthetic run, final checkpoint at step 2
#   cold     page cache of the checkpoint evicted on every node, no compilation cache
#   warm     right after cold, no compilation cache
#   warm-cc  right after warm, with the compilation cache warm's run filled
# Each restore trains steps 2 and 3 without saving; restore_timing.py logs the restore and first-step times.
# Submit from the checkout root:
#   sbatch -A openathena-ihodes -N2 -t 2:30:00 -J ckpt-restore -o logs/ckpt/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/restore_timing.sbatch
# KEEP_CHECKPOINT=1 leaves the checkpoint on /work (for the upload measurement); otherwise it is deleted at the end.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
# A restore never logs step 0, which is what ends the startup watch, so the watch must outlast a whole restore run.
export STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-7200}
source experiments/amd/june_67b_a2b/vultr/launch_env.sh
CKPT=$WORK/ckpt-budget/full-$SLURM_JOB_ID
CACHE=$WORK/ckpt-budget/jax-cache-$SLURM_JOB_ID
LOGS=logs/ckpt/$SLURM_JOB_ID
mkdir -p "$LOGS" "$CKPT"
ARGS=(--checkpoint-dir "$CKPT" --steps 2 --size full --seq-len 8192 --batch-size 16 --gpus-per-node 8
  --expert-axis 8 --replica-axis "$SLURM_JOB_NUM_NODES" --attention gpu_pallas_triton_flash --watch-interval 0
  --step-timeout 0 --log-dir "$LOGS/json")
echo "commit $(cat COMMIT) nodes $SLURM_JOB_NODELIST start $(date -u +%FT%T)"

run() {
  local name=$1; shift
  echo "=== $name start $(date -u +%FT%T)"
  srun_with_startup_retry "$LOGS/$name" $PY -m experiments.amd.june_67b_a2b.vultr.restore_timing --run-id "ckpt-$name" "$@"
  local rc=$?
  echo "=== $name exit $rc end $(date -u +%FT%T)"
  grep -h "timing " "$LOGS/$name-0.log"
  return $rc
}

run save --phase save "${ARGS[@]}" || exit 1
ls -la "$CKPT"
du -sb "$CKPT"
srun --ntasks-per-node=1 $PY -m experiments.amd.june_67b_a2b.vultr.evict_page_cache "$CKPT"
run cold --phase restore "${ARGS[@]}"
run warm --phase restore "${ARGS[@]}" --compilation-cache-dir "$CACHE"
run warm-cc --phase restore "${ARGS[@]}" --compilation-cache-dir "$CACHE"
rm -rf "$CACHE"
[ "${KEEP_CHECKPOINT:-0}" = 1 ] || rm -rf "$CKPT"
echo "done end $(date -u +%FT%T)"
