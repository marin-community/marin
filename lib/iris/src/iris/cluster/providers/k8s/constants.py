# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes constants for Iris cluster components."""

# CoreWeave GPU nodes carry this taint. All pods targeting GPU nodes must
# tolerate it or they will remain Pending / be evicted.
CW_INTERRUPTABLE_TOLERATION: dict = {
    "key": "qos.coreweave.cloud/interruptable",
    "operator": "Exists",
    "effect": "NoExecute",
}

# NVIDIA GPU nodes commonly carry this taint. Pods requesting GPUs must
# tolerate it or they will remain Pending.
NVIDIA_GPU_TOLERATION: dict = {
    "key": "nvidia.com/gpu",
    "operator": "Exists",
    "effect": "NoSchedule",
}
