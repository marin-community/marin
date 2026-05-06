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

NODE_ATTRIBUTE_LABEL_PREFIX = "iris"


def node_attribute_label(attribute_key: str) -> str:
    return f"{NODE_ATTRIBUTE_LABEL_PREFIX}.{attribute_key}"
