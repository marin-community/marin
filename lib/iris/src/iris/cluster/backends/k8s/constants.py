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

# CoreWeave taints nodes provisioned from interruptable capacity with
# qos.coreweave.cloud/interruptable:NoExecute. Iris tasks are retryable/preemptible,
# so we tolerate it to run on interruptable capacity (and so Kueue's TAS, which
# excludes nodes whose NoExecute taints the pod doesn't tolerate, can place the
# gang). Harmless on clusters without the taint (kind, CoreWeave reserved pools).
COREWEAVE_INTERRUPTABLE_TOLERATION: dict = {
    "key": "qos.coreweave.cloud/interruptable",
    "operator": "Exists",
    "effect": "NoExecute",
}
