# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map a container profile to the cluster resources its task container receives.

:func:`task_isolation` returns a backend-neutral :class:`TaskIsolation`;
``CONTAINER_PROFILE_SANDBOX`` withholds every cluster resource and all other
profiles receive all of them.
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


class TaskNetwork(StrEnum):
    """What a task's network can reach.

    CLUSTER is the cluster network: the node's host network where configured,
    the controller, workers and other pods. A sandbox gets INTERNET (public
    addresses only) or NONE; both exclude the controller, workers, other pods
    and the metadata server.
    """

    CLUSTER = "cluster"
    INTERNET = "internet"
    NONE = "none"


_SANDBOX_NETWORKS = {
    job_pb2.SANDBOX_EGRESS_INTERNET: TaskNetwork.INTERNET,
    job_pb2.SANDBOX_EGRESS_NONE: TaskNetwork.NONE,
}


@dataclass(frozen=True)
class TaskIsolation:
    """Which cluster-provided resources a task container receives.

    Attributes:
        include_cluster_env: The operator's task_env and, on Kubernetes, the
            env Secret (object-store keys, injected credentials).
        include_controller_address: The controller address in the task env.
        include_shared_caches: The node-shared download caches.
        include_service_account: On Kubernetes, the pod service account and its token.
        network: What the task's network reaches. Outside CLUSTER, a Docker
            worker runs the task with no network (it rejects INTERNET), and a
            Kubernetes pod gets no host network, carries the label its
            NetworkPolicy selects, and runs without the log-shipping and
            output-upload sidecars, which would share the task's network.
    """

    include_cluster_env: bool
    include_controller_address: bool
    include_shared_caches: bool
    include_service_account: bool
    network: TaskNetwork

    @property
    def mounts(self) -> tuple[MountSpec, ...]:
        return STANDARD_MOUNTS if self.include_shared_caches else _UNSHARED_MOUNTS


_CLUSTER_TASK = TaskIsolation(
    include_cluster_env=True,
    include_controller_address=True,
    include_shared_caches=True,
    include_service_account=True,
    network=TaskNetwork.CLUSTER,
)


def task_isolation(profile: int, sandbox_egress: int) -> TaskIsolation:
    """Isolation for a ``job_pb2.ContainerProfile`` and its resolved ``job_pb2.SandboxEgress``.

    SANDBOX withholds every cluster resource and gets the network its egress
    names; every other profile receives everything.
    """
    if profile != job_pb2.CONTAINER_PROFILE_SANDBOX:
        return _CLUSTER_TASK
    network = _SANDBOX_NETWORKS.get(sandbox_egress)
    if network is None:
        raise ValueError(f"Sandbox task has unresolved egress {sandbox_egress}")
    return TaskIsolation(
        include_cluster_env=False,
        include_controller_address=False,
        include_shared_caches=False,
        include_service_account=False,
        network=network,
    )
