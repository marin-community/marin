#!/bin/bash
# Count MI355X input-buffer corruption events with the train step's input check: NPROC (default 8) independent
# single-GPU recovery_drill.py processes on one node, 2 layers, batch 8, no collectives, no checkpoints. Each process
# reports every skipped step with the damaged arrays, and stops on a non-finite loss or state. Submit from the
# checkout root:
#   sbatch -A openathena-ihodes -N1 -t 0:50:00 -J crate -o logs/drill/%x-%j.out \
#     experiments/amd/june_67b_a2b/vultr/corruption_rate.sbatch
# STEPS (default 4000) per process; MINUTES (default 40) bounds each process; SCRATCH_RECLAIM_OFF=0 leaves
# HSA_NO_SCRATCH_RECLAIM unset (the launch environment sets it to 1).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
ROOT=$PWD
WORK=/work/openathena-ihodes/asartran
pp=$ROOT
for d in "$ROOT"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp RAGGED_DOT_IMPL=triton XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none
PY=$WORK/rocm-extra-test/.venv/bin/python
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd: -)
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
if [ "${SCRATCH_RECLAIM_OFF:-1}" = 1 ]; then export HSA_NO_SCRATCH_RECLAIM=1; fi
ulimit -n "$(ulimit -Hn)"
LOGS=logs/drill/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p "$LOGS"
echo "commit $(cat COMMIT) node $SLURM_JOB_NODELIST HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-unset} start $(date -u +%FT%T)"
for g in $(seq 0 $((${NPROC:-8} - 1))); do
  (
    unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    export HIP_VISIBLE_DEVICES=$g CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID-$g \
      XDG_CACHE_HOME=/tmp/$USER/xdg-$SLURM_JOB_ID-$g
    timeout $((${MINUTES:-40} * 60)) $PY -m experiments.amd.june_67b_a2b.vultr.recovery_drill --size full --layers 2 \
      --seq-len 8192 --batch-size 8 --expert-axis 1 --attention gpu_pallas_triton_flash --watch-interval 0 \
      --steps "${STEPS:-4000}" --run-id "$SLURM_JOB_NAME-$SLURM_JOB_ID-g$g" --log-dir "$LOGS/json" \
      --data-root "$WORK/snowball-data/snowball_v2_subset20" --validation-root "$WORK/snowball-data/validation" \
      --compilation-cache-dir "$WORK/jax-cache-drill" > "$LOGS/g$g.log" 2>&1 < /dev/null
    echo "g$g exit $? end $(date -u +%FT%T)"
  ) &
done
wait
echo "done $(date -u +%FT%T)"
