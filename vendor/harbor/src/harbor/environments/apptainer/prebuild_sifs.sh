#!/bin/bash
#SBATCH --job-name=prebuild_sifs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --partition=batch
#SBATCH --account=projectnucleus
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/transfernetx/nezhurina1/sif_build_%j.out

# Pre-build Apptainer SIF images for all tasks in a Harbor dataset.
# Converts each task's Dockerfile → SIF and caches on scratch.
#
# Usage:
#   sbatch prebuild_sifs.sh <dataset_path> [sif_cache_dir]
#
# Example:
#   sbatch prebuild_sifs.sh /path/to/DCAgent/nl2bash-tasks-cleaned-oracle
#   sbatch prebuild_sifs.sh /path/to/dataset /p/scratch/transfernetx/nezhurina1/sif_cache

set -e

DATASET_PATH="${1:?Usage: prebuild_sifs.sh <dataset_path> [sif_cache_dir]}"
SIF_CACHE="${2:-/p/scratch/transfernetx/nezhurina1/sif_cache}"
MAX_PARALLEL="${3:-4}"

mkdir -p "$SIF_CACHE"

echo "============================================="
echo "Pre-building SIF images"
echo "  Dataset: $DATASET_PATH"
echo "  Cache: $SIF_CACHE"
echo "  Max parallel: $MAX_PARALLEL"
echo "============================================="

# Detect apptainer/singularity
if command -v apptainer &>/dev/null; then
    APPTAINER=apptainer
elif command -v singularity &>/dev/null; then
    APPTAINER=singularity
else
    echo "ERROR: Neither apptainer nor singularity found"
    exit 1
fi

echo "Using: $APPTAINER ($($APPTAINER --version))"

# Set up proxy if needed (for pulling base images)
NODE_HOST=$(hostname -s)
if [[ $NODE_HOST == jrc* ]] || [[ $NODE_HOST == jwb* ]] || [[ $NODE_HOST == jpb* ]]; then
    echo "JSC compute node detected, setting up proxy..."
    export SSH_KEY="${SSH_KEY:-$HOME/.ssh/docker}"
    CMD_PREFIX=$(bash /e/project1/jureap59/marianna/ot/dc-agent/train/hpc/sbatch/start_proxy_tunnel.sh 2>/dev/null || true)
    if [ -n "$CMD_PREFIX" ]; then
        PROXYCHAINS_DIR="/e/project1/jureap59/marianna/ot/proxychains-ng"
        CMD_PREFIX=$(echo "$CMD_PREFIX" | sed "s|proxychains4|${PROXYCHAINS_DIR}/proxychains4|")
        echo "Proxy: $CMD_PREFIX"
        # Set up apptainer to use proxy
        export SINGULARITY_HTTP_PROXY="socks5://${PROXYCHAINS_SOCKS5_HOST}:${PROXYCHAINS_SOCKS5_PORT}"
        export SINGULARITY_HTTPS_PROXY="socks5://${PROXYCHAINS_SOCKS5_HOST}:${PROXYCHAINS_SOCKS5_PORT}"
    fi
fi

# Find all unique Dockerfiles and build them
BUILT=0
CACHED=0
FAILED=0
TOTAL=0

build_sif() {
    local task_dir="$1"
    local task_name=$(basename "$task_dir")
    local dockerfile="$task_dir/environment/Dockerfile"

    if [ ! -f "$dockerfile" ]; then
        dockerfile="$task_dir/Dockerfile"
    fi
    if [ ! -f "$dockerfile" ]; then
        echo "  [$task_name] No Dockerfile, skipping"
        return 0
    fi

    # Hash the Dockerfile for cache key
    local df_hash=$(sha256sum "$dockerfile" | cut -c1-12)
    local sif_path="$SIF_CACHE/build_${task_name}-${df_hash}.sif"

    if [ -f "$sif_path" ]; then
        echo "  [$task_name] Cached: $sif_path"
        return 0
    fi

    echo "  [$task_name] Building..."

    # Extract FROM line
    local base_image=$(grep -i '^FROM' "$dockerfile" | head -1 | awk '{print $2}')

    # First try pulling the base image directly (faster if no RUN commands needed)
    local base_sif="$SIF_CACHE/base_$(echo $base_image | tr '/:' '_').sif"
    if [ ! -f "$base_sif" ]; then
        echo "    Pulling base: $base_image"
        $APPTAINER pull --force "$base_sif.tmp" "docker://$base_image" 2>/dev/null || true
        if [ -f "$base_sif.tmp" ]; then
            mv "$base_sif.tmp" "$base_sif"
        fi
    fi

    # Convert Dockerfile to .def and build
    local def_file=$(mktemp /tmp/singularity_XXXXXX.def)
    python3 -c "
import sys
dockerfile = '$dockerfile'
with open(dockerfile) as f:
    lines = f.readlines()

base = 'ubuntu:22.04'
runs, envs, workdir = [], {}, '/workspace'
current = ''
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'): continue
    if current:
        line = current + ' ' + line
        current = ''
    if line.endswith('\\\\'):
        current = line[:-1].strip()
        continue
    u = line.upper()
    if u.startswith('FROM '):
        base = line[5:].strip().split(' AS ')[0]
        if base.startswith('--platform'):
            base = base.split(None, 2)[-1]
    elif u.startswith('RUN '):
        runs.append(line[4:].strip())
    elif u.startswith('ENV '):
        ep = line[4:].strip()
        if '=' in ep:
            for p in ep.split():
                if '=' in p:
                    k,_,v = p.partition('=')
                    envs[k] = v.strip('\"\\\"')
        else:
            parts = ep.split(None, 1)
            if len(parts) == 2:
                envs[parts[0]] = parts[1].strip('\"\\\"')
    elif u.startswith('WORKDIR '):
        workdir = line[8:].strip()

with open('$def_file', 'w') as f:
    f.write(f'Bootstrap: docker\nFrom: {base}\n\n')
    if envs:
        f.write('%environment\n')
        for k,v in envs.items():
            f.write(f'    export {k}=\"{v}\"\n')
        f.write('\n')
    f.write('%post\n')
    f.write(f'    mkdir -p {workdir}\n')
    f.write('    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true\n')
    f.write('    pip install chardet 2>/dev/null || python3 -m pip install chardet 2>/dev/null || true\n')
    for k,v in envs.items():
        f.write(f'    export {k}=\"{v}\"\n')
    if runs:
        for cmd in runs:
            f.write(f'    {cmd}\n')
    f.write('\n')
    f.write(f'%runscript\n    cd {workdir}\n    exec \"\\\$@\"\n')
"

    # Build SIF
    local tmp_sif="$sif_path.tmp"
    if $APPTAINER build --fakeroot "$tmp_sif" "$def_file" 2>/dev/null; then
        mv "$tmp_sif" "$sif_path"
        echo "    [$task_name] Built: $sif_path"
        rm -f "$def_file"
        return 0
    fi

    # Retry without fakeroot
    if $APPTAINER build "$tmp_sif" "$def_file" 2>/dev/null; then
        mv "$tmp_sif" "$sif_path"
        echo "    [$task_name] Built (no fakeroot): $sif_path"
        rm -f "$def_file"
        return 0
    fi

    echo "    [$task_name] FAILED to build"
    rm -f "$tmp_sif" "$def_file"
    return 1
}

# Process tasks
for task_dir in "$DATASET_PATH"/*/; do
    TOTAL=$((TOTAL + 1))
    if build_sif "$task_dir"; then
        BUILT=$((BUILT + 1))
    else
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================="
echo "Done! Total: $TOTAL, Built/Cached: $BUILT, Failed: $FAILED"
echo "SIF cache: $SIF_CACHE"
ls -lh "$SIF_CACHE"/*.sif 2>/dev/null | wc -l
echo " SIF files"
du -sh "$SIF_CACHE" 2>/dev/null
echo "============================================="
