#!/bin/bash
# Scaled-down Snowball rehearsal on the 2-node Vultr MI355X cluster, all phases in one allocation:
#   K  pretrain 8K, 0-400, SIGKILL of one process after the step-150 checkpoint, auto-resumed by this script
#   D  batch doubling from K's step-300 checkpoint, 300-400
#   C  64K cooldown from D's step-400 checkpoint, 400-430, 2 microbatches, ZeRO-1
#   R  pretrain 8K, 0-400, uninterrupted, the reference K is compared against
# Submit from the checkout root:
#   sbatch -A openathena-ihodes -N2 -t 2:30:00 -J snowball-e2e -o logs/e2e/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/rehearsal.sbatch
# PHASES (default "K D C R") picks phases; TAG names the runs (default: the job id).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
ROOT=$PWD
WORK=/work/openathena-ihodes/asartran
DATA=$WORK/snowball-data/snowball_v2_subset20
VAL=$WORK/snowball-data/validation
OUT=$WORK/e2e-rehearsal/runs
TAG=${TAG:-$SLURM_JOB_ID}
PHASES=${PHASES:-K D C R}
MAX_RESTARTS=${MAX_RESTARTS:-3}
mkdir -p "$OUT" logs/e2e

pp=$ROOT
for d in "$ROOT"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp
PY=$WORK/rocm-extra-test/.venv/bin/python

# Checkpoints keep many files open at once; 1024 is too low.
ulimit -n 1048576 || ulimit -n "$(ulimit -Hn)"

export RAGGED_DOT_IMPL=triton XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd: -)
# Pensando ANP RCCL net plugin; only the plugin is exposed, not the rest of the system ROCm 7.2 libraries.
export NCCL_NET_PLUGIN=librccl-anp.so NCCL_IB_TC=104 NCCL_MIN_NCHANNELS=32 HSA_FORCE_FINE_GRAIN_PCIE=1 \
  NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IGNORE_CPU_AFFINITY=1 \
  RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
ANP_DIR=/tmp/$USER/anp-$SLURM_JOB_ID
srun --ntasks-per-node=1 bash -c "mkdir -p $ANP_DIR && ln -sf /opt/rocm-7.2.0/lib/librccl-anp.so $ANP_DIR/"
export LD_LIBRARY_PATH=$ANP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Tree+LL all-reduces across nodes can read a peer's input before it is written (2x loss on some steps).
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_DEBUG_SUBSYS=INIT,NET
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID XDG_CACHE_HOME=/tmp/$USER/xdg-$SLURM_JOB_ID
# /home is over quota; keep W&B's files on /work.
export WANDB_API_KEY=$(cat $WORK/secrets/wandb_api_key) WANDB_ENTITY=marin-community WANDB_RESUME=allow
export WANDB_DIR=$WORK/e2e-rehearsal/wandb WANDB_CACHE_DIR=$WORK/e2e-rehearsal/wandb-cache \
  WANDB_CONFIG_DIR=$WORK/e2e-rehearsal/wandb-config WANDB_DATA_DIR=$WORK/e2e-rehearsal/wandb-data
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
BASE_XLA_FLAGS="--xla_gpu_enable_command_buffer="
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
    XLA_FLAGS="$flags" srun --ntasks-per-node=1 --gpus-per-node=8 --kill-on-bad-exit=1 \
      --output="logs/e2e/$name-%t.log" \
      $PY -m experiments.amd.june_67b_a2b.rehearsal --log-dir "logs/e2e/$name-jsonlogs" --data-root "$DATA" \
        --validation-root "$VAL" --output-root "$OUT" "$@"
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
C=e2e-$TAG-C
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
