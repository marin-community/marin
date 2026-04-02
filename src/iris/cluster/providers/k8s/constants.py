# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes constants for Iris cluster components."""

# NVIDIA GPU nodes commonly carry this taint. Pods requesting GPUs must
# tolerate it or they will remain Pending.
NVIDIA_GPU_TOLERATION: dict = {
    "key": "nvidia.com/gpu",
    "operator": "Exists",
    "effect": "NoSchedule",
}
