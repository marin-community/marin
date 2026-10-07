#!/bin/bash
# Run MAMF-finder on GPU 0 of one AMD node. Arguments are passed to mamf-finder.py.
# Submit from the synced checkout root with MARIN_COMMIT exported; see README.md.
#SBATCH --nodes=1
#SBATCH --no-requeue
set -euo pipefail
: "${MARIN_COMMIT:?export MARIN_COMMIT=<sha> when submitting}"

agent_work="$WORK/agents/$(basename "$SLURM_SUBMIT_DIR")"
echo "commit=$MARIN_COMMIT job=$SLURM_JOB_ID node=$(hostname) mamf=$(cat "$agent_work/mamf/SOURCE_SHA")"

# TunableOp benchmarks every hipBLASLt solution per GEMM shape and keeps the fastest.
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME="$agent_work/results/tunableop-$SLURM_JOB_ID.csv"

"$agent_work/venv-torch/bin/python" -I "$agent_work/mamf/mamf-finder.py" \
  --output_file "$agent_work/results/mamf-$SLURM_JOB_ID.out" \
  --notes "commit=$MARIN_COMMIT job=$SLURM_JOB_ID node=$(hostname)" \
  "$@"
