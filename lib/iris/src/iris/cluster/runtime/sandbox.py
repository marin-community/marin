# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map a container profile and egress policy to what a task container receives.

:func:`task_isolation` returns a backend-neutral :class:`TaskIsolation`. The
profile decides the cluster resources: ``CONTAINER_PROFILE_SANDBOX`` withholds
every one and all other profiles receive all of them. The egress policy decides
the network, for every profile.
"""

from dataclasses import dataclass
from enum import StrEnum

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
