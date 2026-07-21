#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(pwd)"
repro_script="${repo_root}/scripts/debug/b200_nccl_fast_restart.py"
repeat_count="${MARIN_REPRO_REPEATS:-3}"
run_timeout="${MARIN_REPRO_TIMEOUT:-180}"
processes_per_task="${MARIN_REPRO_PROCESSES_PER_TASK:-4}"
python_bin="${MARIN_REPRO_PYTHON:-python}"
run_root="$(mktemp -d /tmp/b200-nccl-fast-restart.XXXXXX)"

mkdir -p "${run_root}/cache"
export JAX_COMPILATION_CACHE_DIR="${run_root}/cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_command_buffer=}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,BOOTSTRAP}"

nvidia-smi --query-gpu=name,driver_version --format=csv
echo "CONFIG processes_per_task=${processes_per_task} repeats=${repeat_count} timeout=${run_timeout}s"

status=0
pass_count=0
fail_count=0
for ((repeat = 1; repeat <= repeat_count; repeat++)); do
    export MARIN_REPRO_REPEAT="${repeat}"
    echo "REPEAT ${repeat} START $(date --iso-8601=seconds)"
    set +e
    timeout --signal=TERM --kill-after=30s "${run_timeout}s" \
        "${python_bin}" -m iris.cluster.hooks.multigpu_main \
        --nproc "${processes_per_task}" \
        -- "${python_bin}" "${repro_script}" \
        2>&1 | tee "${run_root}/repeat-${repeat}.log"
    repeat_status="${PIPESTATUS[0]}"
    set -e
    echo "REPEAT ${repeat} END status=${repeat_status} $(date --iso-8601=seconds)"
    if [[ "${repeat_status}" -ne 0 ]]; then
        status="${repeat_status}"
        ((fail_count += 1))
    else
        ((pass_count += 1))
    fi
done

grep -H -E \
    'REPRO_OK|compile-complete|first-execution|Init START|Init COMPLETE|leader.*deadlocked' \
    "${run_root}"/repeat-*.log || true
echo "RESULT pass=${pass_count} fail=${fail_count} logs=${run_root}"
exit "${status}"
