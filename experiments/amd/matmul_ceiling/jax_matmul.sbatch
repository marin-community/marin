#!/bin/bash
# Run jax_matmul.py on GPU 0 of one AMD node under rocprofv3, then add kernel-time rates from the trace.
# Arguments are passed to jax_matmul.py. Submit from the synced checkout root with MARIN_COMMIT exported; see README.md.
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

python="$agent_work/venv-jax/bin/python"
results="$agent_work/results/jax-matmul-$SLURM_JOB_ID.jsonl"
trace="$agent_work/results/jax-matmul-$SLURM_JOB_ID-trace"
"$agent_work/venv-jax/bin/rocprofv3" --kernel-trace --output-format csv -d "$trace" -- \
  "$python" "$SLURM_SUBMIT_DIR/experiments/amd/matmul_ceiling/jax_matmul.py" --output "$results" "$@"

cd "$SLURM_SUBMIT_DIR"
"$python" -m experiments.amd.matmul_ceiling.kernel_trace "$results" "$trace"
"$python" -m experiments.amd.matmul_ceiling.summarize "$results"
