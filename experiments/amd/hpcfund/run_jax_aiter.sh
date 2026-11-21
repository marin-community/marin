#!/bin/bash
# Slurm wrapper that runs a command inside AMD's TheRock 7.14 JAX container, the only stack JAX-AITER supports
# (Ubuntu 24.04; the HPC Fund cluster's Rocky 9 glibc is too old for its libraries). It also runs on other Slurm
# clusters with Apptainer, such as the Vultr MI355X nodes. Submit from the repository root.
# One-time setup pulls the image to $JAX_AITER_HOME, then builds the venv and JAX-AITER there:
#   apptainer pull $JAX_AITER_HOME/jax-dev-therock-7.14.sif docker://ghcr.io/rocm/jax-dev-ubu24.therock-7.14:7.14
# Unpacking the image needs xattrs, which NFS homes (Vultr) lack: there, set APPTAINER_TMPDIR and APPTAINER_CACHEDIR to
# node-local disk such as /tmp and pull inside a job. Vultr also needs `-A openathena-ihodes` on sbatch.
#   sbatch -N1 -p mi3501x -t 90 -o logs/%x-%j.out experiments/amd/hpcfund/run_jax_aiter.sh \
#       bash experiments/amd/hpcfund/setup_jax_aiter.sh
# Then run Python entry points like run_gpu.sh does:
#   sbatch -N1 -p mi3508x -t 45 -o logs/%x-%j.out experiments/amd/hpcfund/run_jax_aiter.sh \
#       python experiments/june_tpu_67b_a2b/moe/synthetic_benchmark.py --size full --attention gpu_pallas_triton_flash
# Across nodes, launch one wrapper per node with srun; it mounts the host's RDMA libraries so RCCL can use them.
cd "$SLURM_SUBMIT_DIR" || exit 1

JAX_AITER_HOME=${JAX_AITER_HOME:-${WORK:-$HOME}/aiter}
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
# APPTAINERENV_* passes values verbatim; --env splits a value with more than one '=' on commas, which breaks
# XLA_FLAGS such as --xla_gpu_enable_command_buffer=FUSION,CUBLAS.
while IFS='=' read -r name _; do
    export "APPTAINERENV_$name=${!name}"
done < <(env | grep -E '^(PYTHONPATH|SLURM_|HSA_|XLA_|JAX_|JA_|RAGGED_DOT_|PYTHONUNBUFFERED|XDG_CACHE_HOME|UV_CACHE_DIR|HIP_|RCCL_|NCCL_)')
export APPTAINERENV_PATH=$JAX_AITER_HOME/venv/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin

binds=(--bind "$JAX_AITER_HOME")
if [ -n "$UV_CACHE_DIR" ]; then
    binds+=(--bind "$UV_CACHE_DIR")
fi

# The image has no RDMA userspace, so across nodes RCCL would fall back to TCP sockets on one Ethernet port. When the
# host has libibverbs, mount it, librdmacm, the NIC providers and the libraries they link into the container. The host
# libibverbs looks for providers at its own compiled-in path, so the provider directory keeps its host path.
cmd=("$@")
host_verbs=$(ldconfig -p | awk '/libibverbs\.so\.1 .*x86-64/ {print $NF; exit}')
if [ -n "$host_verbs" ]; then
    rdma_lib=/opt/host-rdma/lib
    provider_dir=$(dirname "$(readlink -f "$host_verbs")")/libibverbs
    host_rdmacm=$(ldconfig -p | awk '/librdmacm\.so\.1 .*x86-64/ {print $NF; exit}')
    binds+=(--bind /etc/libibverbs.d:/etc/libibverbs.d:ro --bind "$provider_dir:$provider_dir:ro")
    # Some providers are symlinks out of that directory (Pensando's ionic points to ../libionic.so.1.*). Mount the
    # target at its host path for the symlink, and under its soname, which RCCL dlopens for vendor features.
    for provider in "$provider_dir"/*.so; do
        target=$(readlink -f "$provider")
        [ "$(dirname "$target")" != "$provider_dir" ] || continue
        binds+=(--bind "$target:$target:ro" --bind "$target:$rdma_lib/$(basename "$target" | sed -E 's/(\.so\.[0-9]+).*/\1/'):ro")
    done
    # glibc and the C++ runtime come from the image, which is newer than any host we run on.
    while read -r soname path; do
        binds+=(--bind "$(readlink -f "$path"):$rdma_lib/$soname:ro")
    done < <({
        echo "libibverbs.so.1 $host_verbs"
        [ -z "$host_rdmacm" ] || echo "librdmacm.so.1 $host_rdmacm"
        ldd "$host_verbs" $host_rdmacm "$provider_dir"/*.so | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}'
    } | grep -v -E '^(libc|libm|libdl|librt|libpthread|libgcc_s|libstdc\+\+|ld-linux)[.-]' | sort -u -k1,1)
    cmd=(bash -c "export LD_LIBRARY_PATH=$rdma_lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; exec \"\$@\"" _ "$@")
fi

apptainer exec --cleanenv --rocm "${binds[@]}" "$SIF" "${cmd[@]}" 2>&1 | grep --line-buffered -v -E "rocm_pcie_bandwidth|rocm_executor"
exit "${PIPESTATUS[0]}"
