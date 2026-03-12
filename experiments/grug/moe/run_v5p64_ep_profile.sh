#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="experiments:lib/fray/src:lib/haliax/src:lib/iris/src:lib/levanter/src:lib/marin/src:lib/zephyr/src${PYTHONPATH:+:$PYTHONPATH}"
export FRAY_CLUSTER_SPEC="${FRAY_CLUSTER_SPEC:-ray}"

python experiments/grug/moe/launch_qwen3_32b_a4b_v5p64_ep_profile.py "$@"
