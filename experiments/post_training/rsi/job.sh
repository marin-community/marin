#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Set up the GPU environment of a Baby RSI Iris task, then run Python with the arguments, for example
# `job.sh -m experiments.post_training.rsi.evaluate evaluate ...`.
#
# Submit with `--sync-package marin-taskforge --sync-package marin-core` and the privileged container profile
# (bubblewrap needs namespaces). The vLLM backend of marin.inference installs the Marin vLLM fork in its own
# environment.
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

# The script graders of the pool tasks run in bubblewrap with this environment; without these imports every grade of
# their strata fails, so stop here instead.
.venv/bin/python -c "import numpy, pandas, reasoning_gym, verifyit"
exec .venv/bin/python "$@"
