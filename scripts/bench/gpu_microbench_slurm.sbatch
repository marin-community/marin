#!/bin/bash
#SBATCH --job-name=gpu-bench-bx00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

source /etc/profile
module load cuda/13.0.2

set -euo pipefail

cd "$HOME/marin"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer="

nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv
"$HOME/marin/.venv/bin/python" "$HOME/gpu-bench/gpu_microbench.py" \
  --iterations 5 \
  --warmup 2 \
  --gemm-sizes 8192,16384 \
  --hbm-size-mb 4096 \
  --collective-message-mb 1,8,64,256,512,1024
