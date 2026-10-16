#!/bin/bash
# Slurm wrapper that runs a command inside AMD's TheRock 7.14 JAX container, the only stack JAX-AITER supports
# (Ubuntu 24.04; the cluster's Rocky 9 glibc is too old for its libraries). Submit from the repository root.
# One-time setup builds the venv and JAX-AITER under $JAX_AITER_HOME:
#   sbatch -N1 -p mi3501x -t 90 -o logs/%x-%j.out experiments/amd/hpcfund/run_jax_aiter.sh \
#       bash experiments/amd/hpcfund/setup_jax_aiter.sh
# Then run Python entry points like run_gpu.sh does:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_jax_aiter.sh \
#       python experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --attention gpu_aiter_flash
cd "$SLURM_SUBMIT_DIR" || exit 1

JAX_AITER_HOME=${JAX_AITER_HOME:-$WORK/aiter}
SIF=$JAX_AITER_HOME/jax-dev-therock-7.14.sif

# Same XLA settings as run_gpu.sh; see the comments there.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= $XLA_FLAGS"
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=${JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES:-none}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export PYTHONUNBUFFERED=1
# JAX-AITER's MHA libraries (several GB) and ROCm's kernel caches default to ~/.cache, which has a small quota.
export XDG_CACHE_HOME=$JAX_AITER_HOME/cache
export JAX_AITER_HOME

# --cleanenv keeps the host's ROCm module environment out of the container; forward only the variables we use.
forward=()
while IFS='=' read -r name _; do
    forward+=(--env "$name=${!name}")
done < <(env | grep -E '^(XLA_|JAX_|JA_|RAGGED_DOT_|PYTHONUNBUFFERED|XDG_CACHE_HOME|UV_CACHE_DIR|HIP_|RCCL_|NCCL_)')

apptainer exec --cleanenv --rocm --bind /work1 "${forward[@]}" \
    --env "PATH=$JAX_AITER_HOME/venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    "$SIF" "$@" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
exit "${PIPESTATUS[0]}"
