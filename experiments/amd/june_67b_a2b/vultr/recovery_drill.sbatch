#!/bin/bash
# Run recovery_drill.py on one or two Vultr MI355X nodes, one process per node, restarting it from the run's latest
# checkpoint after any failure, up to MAX_RESTARTS (default 3) times. DRILL_ARGS holds the drill's arguments past the
# data roots, e.g. "--size full --steps 60 --checkpoint-every 10 --nan-at 25 --abort-at 45 --abort-process 1".
# SCRATCH_RECLAIM_OFF=0 leaves HSA_NO_SCRATCH_RECLAIM unset (the launch environment sets it to 1). Submit from the
# checkout root:
#   DRILL_ARGS="..." sbatch -A openathena-ihodes -N2 -t 1:30:00 -J drill -o logs/drill/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/recovery_drill.sbatch
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
ROOT=$PWD
WORK=/work/openathena-ihodes/asartran
OUT=$WORK/corruption-recovery/runs
RUN=${RUN:-drill-$SLURM_JOB_ID}
MAX_RESTARTS=${MAX_RESTARTS:-3}
NNODES=$SLURM_JOB_NUM_NODES
LOGS=logs/drill/$RUN
mkdir -p "$OUT" "$LOGS"

pp=$ROOT
for d in "$ROOT"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp
PY=$WORK/rocm-extra-test/.venv/bin/python
ulimit -n "$(ulimit -Hn)"

# The launch environment of the MI355X runs.
export RAGGED_DOT_IMPL=triton XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd: -)
if [ "$NNODES" -gt 1 ]; then
  export NCCL_NET_PLUGIN=librccl-anp.so NCCL_IB_TC=104 NCCL_MIN_NCHANNELS=32 HSA_FORCE_FINE_GRAIN_PCIE=1 \
    NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IGNORE_CPU_AFFINITY=1 \
    RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
  ANP_DIR=/tmp/$USER/anp-$SLURM_JOB_ID
  srun --ntasks-per-node=1 bash -c "mkdir -p $ANP_DIR && ln -sf /opt/rocm-7.2.0/lib/librccl-anp.so $ANP_DIR/"
  export LD_LIBRARY_PATH=$ANP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
export NCCL_PROTO=${NCCL_PROTO:-Simple} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if [ "${SCRATCH_RECLAIM_OFF:-1}" = 1 ]; then export HSA_NO_SCRATCH_RECLAIM=1; fi
export XLA_FLAGS="--xla_gpu_enable_command_buffer=" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID XDG_CACHE_HOME=/tmp/$USER/xdg-$SLURM_JOB_ID

echo "commit $(cat COMMIT) nodes $SLURM_JOB_NODELIST run $RUN HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-unset} args [${DRILL_ARGS:-}] start $(date -u +%FT%T)"
attempt=0
while :; do
  echo "=== attempt $attempt start $(date -u +%FT%T)"
  srun --ntasks-per-node=1 --gpus-per-node=8 --kill-on-bad-exit=1 --output="$LOGS/attempt$attempt-%t.log" \
    $PY -m experiments.amd.june_67b_a2b.vultr.recovery_drill --gpus-per-node 8 --replica-axis "$NNODES" \
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
