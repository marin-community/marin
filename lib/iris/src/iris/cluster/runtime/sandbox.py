# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map a container profile and egress policy to what a task container receives.

:func:`task_isolation` returns a backend-neutral :class:`TaskIsolation`. The
profile decides the cluster resources: ``CONTAINER_PROFILE_SANDBOX`` withholds
every one and all other profiles receive all of them. The egress policy decides
the network, for every profile.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from iris.cluster.runtime.env import STANDARD_MOUNTS
from iris.cluster.runtime.types import MountKind, MountSpec
from iris.rpc import job_pb2

# A writable cache shared with other tasks would let a sandbox plant packages
# they later install. The cache env still names these paths; without a mount
# they land in the container's own layer.
_UNSHARED_MOUNTS: tuple[MountSpec, ...] = tuple(m for m in STANDARD_MOUNTS if m.kind is not MountKind.CACHE)


# Destinations an INTERNET task may not reach: private networks (pods, nodes,
# the controller, VPC peers), carrier-grade NAT, and link-local (the cloud
# metadata server). The Kubernetes NetworkPolicy and the Docker worker's host
# filter (worker bootstrap) both block these.
EGRESS_BLOCKED_CIDRS: tuple[str, ...] = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",
    "169.254.0.0/16",
)

SANDBOX_SERVICE_PREFIX = "service-"
SANDBOX_SERVICE_GROUP_CAPABILITY = "sandbox-service-groups"
_SERVICE_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_PINNED_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_CREDENTIAL_NAMES = frozenset(
    {
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CREDENTIALS",
        "GOOGLE_CLOUD_KEYFILE_JSON",
        "GCP_CREDENTIALS",
        "GCP_SERVICE_ACCOUNT_KEY",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "KUBECONFIG",
        "SSH_AUTH_SOCK",
        "FSSPEC_S3",
        "WANDB_API_KEY",
        "RIGGING_SECRETS_DIR",
        "MARIN_CLUSTER_TOKEN",
        "IRIS_TASK_TOKEN",
        "IRIS_CONTROLLER_ADDRESS",
        "IRIS_CONTROLLER_URL",
    }
)
_CREDENTIAL_PREFIXES = ("AWS_", "S3_", "CW_KEY_", "AZURE_", "CLOUDSDK_AUTH_")


def validate_sandbox_environment(environment: dict[str, str]) -> None:
    """Reject cloud and control-plane credential delivery into untrusted machines."""
    rejected = [
        name
        for name in environment
        if name.upper() in _CREDENTIAL_NAMES or name.upper().startswith(_CREDENTIAL_PREFIXES)
    ]
    if rejected:
        raise ValueError(f"Sandbox environments cannot receive cloud or controller credentials: {sorted(rejected)}")
    if any("${" in value for value in environment.values()):
        raise ValueError("Sandbox service environments require literal values")


def validate_sandbox_services(
    profile: int,
    egress_policy: int,
    environment: job_pb2.EnvironmentConfig,
    resources: job_pb2.ResourceSpecProto,
    task_image: str,
) -> None:
    """Validate a private sandbox group before admission and again before launch."""
    services = environment.sandbox_services
    if not services:
        return
    if profile != job_pb2.CONTAINER_PROFILE_SANDBOX or egress_policy != job_pb2.EGRESS_POLICY_NONE:
        raise ValueError("Sandbox service groups require the sandbox profile and NONE egress")
    if environment.secrets or environment.setup_scripts:
        raise ValueError("Sandbox service groups accept no secrets or environment setup scripts")
    if _PINNED_IMAGE.fullmatch(task_image) is None:
        raise ValueError("Sandbox service groups require a digest-pinned task image")
    validate_sandbox_environment(dict(environment.env_vars))
    names = [service.name for service in services]
    if len(names) > 8 or len(set(names)) != len(names):
        raise ValueError("Sandbox service groups require unique service names and at most eight services")
    for service in services:
        if len(service.name) > 55 or _SERVICE_NAME.fullmatch(service.name) is None or service.name == "task":
            raise ValueError(f"Invalid sandbox service name: {service.name!r}")
        if _PINNED_IMAGE.fullmatch(service.image) is None:
            raise ValueError(f"Sandbox service {service.name} requires a digest-pinned image")
        if not PurePosixPath(service.workdir).is_absolute():
            raise ValueError(f"Sandbox service {service.name} requires an absolute workdir")
        validate_sandbox_environment(dict(service.env_vars))
        if service.resources.device.WhichOneof("device") is not None:
            raise ValueError("Sandbox services cannot request accelerator devices")
        if service.resources.cpu_millicores <= 0 or service.resources.memory_bytes <= 0:
            raise ValueError("Sandbox services require explicit positive CPU and memory reservations")
        if service.resources.disk_bytes < 0:
            raise ValueError("Sandbox service disk reservations must be nonnegative")
        for field, override in (("entrypoint", service.entrypoint), ("arguments", service.arguments)):
            if service.HasField(field) and not override.argv:
                raise ValueError("Empty image command overrides are unsupported")
    if (
        sum(service.resources.cpu_millicores for service in services) >= resources.cpu_millicores
        or sum(service.resources.memory_bytes for service in services) >= resources.memory_bytes
        or sum(service.resources.disk_bytes for service in services) > resources.disk_bytes
    ):
        raise ValueError("Group resource totals must reserve resources for every service and the task")


class TaskNetwork(StrEnum):
    """What a task's network can reach (``job_pb2.EgressPolicy``, resolved).

    CLUSTER is the cluster network: the node's host network where configured,
    the controller, workers and other pods. INTERNET (public addresses only)
    and NONE both exclude the controller, workers, other pods and the metadata
    server.
    """

    CLUSTER = "cluster"
    INTERNET = "internet"
    NONE = "none"


_NETWORKS: dict[int, TaskNetwork] = {
    job_pb2.EGRESS_POLICY_CLUSTER: TaskNetwork.CLUSTER,
    job_pb2.EGRESS_POLICY_INTERNET: TaskNetwork.INTERNET,
    job_pb2.EGRESS_POLICY_NONE: TaskNetwork.NONE,
}


def resolve_egress_policy(profile: int, egress_policy: int) -> job_pb2.EgressPolicy:
    """Resolve UNSPECIFIED to the profile's default: INTERNET for SANDBOX, CLUSTER otherwise.

    Raises ValueError for an unknown policy or a SANDBOX job on the cluster network.
    """
    sandbox = profile == job_pb2.CONTAINER_PROFILE_SANDBOX
    if egress_policy == job_pb2.EGRESS_POLICY_UNSPECIFIED:
        return job_pb2.EGRESS_POLICY_INTERNET if sandbox else job_pb2.EGRESS_POLICY_CLUSTER
    if egress_policy not in _NETWORKS:
        raise ValueError(f"Unknown egress policy {egress_policy}")
    if sandbox and egress_policy == job_pb2.EGRESS_POLICY_CLUSTER:
        raise ValueError("Container profile sandbox cannot use egress policy cluster")
    return job_pb2.EgressPolicy.ValueType(egress_policy)


@dataclass(frozen=True)
class TaskIsolation:
    """Which cluster-provided resources a task container receives.

    Attributes:
        include_cluster_env: The operator's task_env and, on Kubernetes, the
            env Secret (object-store keys, injected credentials).
        include_controller_address: The controller address in the task env.
        include_task_token: The controller-minted task token, which lets the
            task's Iris client act as the job's owner.
        include_shared_caches: The node-shared download caches.
        include_service_account: On Kubernetes, the pod service account and its token.
        network: What the task's network reaches. Outside CLUSTER, a Docker
            worker runs the task on the filtered egress network (INTERNET) or
            with no network (NONE), and a Kubernetes pod gets no host network,
            carries the label its NetworkPolicy selects, and runs without the
            log-shipping and output-upload sidecars, which would share the
            task's network.
    """

    include_cluster_env: bool
    include_controller_address: bool
    include_task_token: bool
    include_shared_caches: bool
    include_service_account: bool
    network: TaskNetwork

    @property
    def mounts(self) -> tuple[MountSpec, ...]:
        return STANDARD_MOUNTS if self.include_shared_caches else _UNSHARED_MOUNTS


def task_isolation(profile: int, egress_policy: int) -> TaskIsolation:
    """Isolation for a ``job_pb2.ContainerProfile`` and its ``job_pb2.EgressPolicy``.

    SANDBOX withholds every cluster resource; every other profile receives all
    of them. The network follows the egress policy, resolved as
    :func:`resolve_egress_policy` does.
    """
    cluster_resources = profile != job_pb2.CONTAINER_PROFILE_SANDBOX
    return TaskIsolation(
        include_cluster_env=cluster_resources,
        include_controller_address=cluster_resources,
        include_task_token=cluster_resources,
        include_shared_caches=cluster_resources,
        include_service_account=cluster_resources,
        network=_NETWORKS[resolve_egress_policy(profile, egress_policy)],
    )
