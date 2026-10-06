#!/bin/bash
# Run jax_matmul.py on GPU 0 of one AMD node. Arguments are passed to jax_matmul.py.
# Submit from the synced checkout root with MARIN_COMMIT exported; see README.md.
#SBATCH --nodes=1
#SBATCH --no-requeue
set -euo pipefail
: "${MARIN_COMMIT:?export MARIN_COMMIT=<sha> when submitting}"

agent_work="$WORK/agents/$(basename "$SLURM_SUBMIT_DIR")"
echo "commit=$MARIN_COMMIT job=$SLURM_JOB_ID node=$(hostname)"

export HIP_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
# Command buffers (HIP graphs) corrupt memory on this ROCm stack; see #9462.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}"
echo "XLA_FLAGS=$XLA_FLAGS"

"$agent_work/venv-jax/bin/python" "$SLURM_SUBMIT_DIR/experiments/amd_matmul_ceiling/jax_matmul.py" \
  --output "$agent_work/results/jax-matmul-$SLURM_JOB_ID.jsonl" \
  "$@"
