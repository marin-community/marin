# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes constants for Iris cluster components."""

from rigging.secrets import AWS_CREDENTIALS_FILENAME

# CoreWeave's Helm repository. ``install_kueue.py`` and ``install_cw_network.py``
# both ``helm repo add`` this alias on the operator's machine, so the two must
# name the same URL or whichever ran last repoints the alias for the other.
CW_REPO_NAME = "coreweave"
CW_REPO_URL = "https://charts.core-services.ingress.coreweave.com"

DEFAULT_TASK_CACHE_DIR = "/cache"
CONTROLLER_TLS_NATIVE_PORT_OFFSET = 1
NVIDIA_GPU_RESOURCE = "nvidia.com/gpu"
RDMA_RESOURCE = "rdma/ib"

# NVIDIA GPU nodes commonly carry this taint. Pods requesting GPUs must
# tolerate it or they will remain Pending.
NVIDIA_GPU_TOLERATION: dict = {
    "key": NVIDIA_GPU_RESOURCE,
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

# A pod off the cluster network carries this label with its TaskNetwork value
# ("none" or "internet"). Controller start creates one NetworkPolicy per value,
# named EGRESS_NETWORK_POLICY_PREFIX + value, that selects those pods.
EGRESS_LABEL = "iris.egress"
EGRESS_NETWORK_POLICY_PREFIX = "iris-egress-"

# Secret volumes are readable by the pod's supplemental group.
SECRET_GROUP = 2000
SECRET_FILE_MODE = 0o440
CONTROLLER_SECRET_NAME = "iris-controller-secrets"
CONTROLLER_SECRETS_DIR = "/run/rigging/controller-secrets"
STORAGE_SECRET_NAME = "iris-storage-secrets"
STORAGE_SECRETS_DIR = "/run/rigging/storage-secrets"
STORAGE_CREDENTIALS_FILE = f"{STORAGE_SECRETS_DIR}/{AWS_CREDENTIALS_FILENAME}"
