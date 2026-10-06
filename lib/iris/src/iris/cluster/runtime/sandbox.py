# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a task's container profile withholds from it, independent of backend.

The worker (Docker/process) and Kubernetes backends both consult
:func:`task_isolation` rather than checking for ``CONTAINER_PROFILE_SANDBOX``
themselves, so the sandbox rule lives in one place.
"""

from dataclasses import dataclass

from iris.cluster.runtime.env import STANDARD_MOUNTS
from iris.cluster.runtime.types import MountKind, MountSpec
from iris.rpc import job_pb2

# A writable cache shared with other tasks would let a sandbox plant packages
# they later install. The cache env still names these paths; without a mount
# they land in the container's own layer.
_UNSHARED_MOUNTS: tuple[MountSpec, ...] = tuple(m for m in STANDARD_MOUNTS if m.kind is not MountKind.CACHE)


@dataclass(frozen=True)
class TaskIsolation:
    """Which cluster-provided resources a task container receives.

    Attributes:
        include_cluster_env: The operator's task_env and, on Kubernetes, the
            env Secret (object-store keys, injected credentials).
        include_controller_address: The controller address in the task env.
        include_shared_caches: The node-shared download caches.
        include_service_account: On Kubernetes, the pod service account and its token.
        reach_cluster_network: Whether the task's network reaches cluster
            services (controller, workers, metadata server). On Docker workers
            a task without it runs with no network at all. On Kubernetes its pod
            gets no host network and carries the label the sandbox
            NetworkPolicy selects, and its log sidecar writes to finelog
            directly instead of resolving it through the controller.
    """

    include_cluster_env: bool
    include_controller_address: bool
    include_shared_caches: bool
    include_service_account: bool
    reach_cluster_network: bool

    @property
    def mounts(self) -> tuple[MountSpec, ...]:
        return STANDARD_MOUNTS if self.include_shared_caches else _UNSHARED_MOUNTS


_CLUSTER_TASK = TaskIsolation(
    include_cluster_env=True,
    include_controller_address=True,
    include_shared_caches=True,
    include_service_account=True,
    reach_cluster_network=True,
)

_SANDBOX_TASK = TaskIsolation(
    include_cluster_env=False,
    include_controller_address=False,
    include_shared_caches=False,
    include_service_account=False,
    reach_cluster_network=False,
)


def task_isolation(profile: int) -> TaskIsolation:
    """Isolation for a ``job_pb2.ContainerProfile`` value: SANDBOX withholds everything, others nothing."""
    if profile == job_pb2.CONTAINER_PROFILE_SANDBOX:
        return _SANDBOX_TASK
    return _CLUSTER_TASK
