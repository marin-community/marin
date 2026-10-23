#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Set up the GPU environment of a Baby RSI Iris task, then run Python with the arguments, for example
# `job.sh -m experiments.post_training.baby_rsi.run baseline ...`.
#
# vLLM runs from its own virtual environment, because its Torch pin differs from the task's environment. Submit with
# `--sync-package marin-taskforge --container-profile container_profile_privileged` (bubblewrap needs namespaces).
set -euo pipefail

# A privileged task sees every GPU of its node, but owns only the GPUs that Kubernetes assigned to it. The device
# plugin names them in NVIDIA_VISIBLE_DEVICES, or, with volume mounts, by the file names in the directory that it
# names. vLLM reads CUDA_VISIBLE_DEVICES as indices, so the UUIDs become the indices of nvidia-smi, in PCI bus order.
assigned="${NVIDIA_VISIBLE_DEVICES:-}"
if [ -d "$assigned" ]; then
  assigned=$(ls "$assigned" | paste -sd, -)
fi
if [ -z "$assigned" ] || [ "$assigned" = all ]; then
  echo "Cannot tell which GPUs this task owns: NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
indices=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', ' -v wanted="$assigned" 'BEGIN { n = split(wanted, list, ","); for (i = 1; i <= n; i++) keep[list[i]] = 1 }
      $2 in keep { print $1 }' | paste -sd, -)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$indices"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

VLLM_VENV=/tmp/vllm-venv
# The installs run outside the repository, so that its uv index settings do not select Torch builds.
(
  cd /tmp
  uv venv --quiet --python 3.12 "$VLLM_VENV"
  uv pip install --quiet --python "$VLLM_VENV/bin/python" --torch-backend cu128 vllm==0.11.0 transformers==4.57.1
)
export BABY_RSI_VLLM="$VLLM_VENV/bin/vllm"
exec .venv/bin/python "$@"
