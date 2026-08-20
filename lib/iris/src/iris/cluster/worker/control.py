# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native ports between the worker daemon and its hosted RPC process."""

from dataclasses import dataclass
from typing import Protocol

from iris.cluster.stats.tables import ProfileTrigger
from iris.cluster.worker.worker_types import TaskInfo
from iris.managed_thread import ThreadContainer
from iris.resources.attempt import AttemptLaunch
from iris.resources.endpoint import ExecResult, ProfileConfiguration
from iris.resources.worker import WorkerMetadata, WorkerReconcileRequest, WorkerReconcileResponse


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    """Native worker registration request."""

    address: str
    metadata: WorkerMetadata
    worker_id: str
    slice_id: str
    scale_group: str


@dataclass(frozen=True, slots=True)
class WorkerRegistrationResult:
    """Controller decision for one registration attempt."""

    accepted: bool
    worker_id: str


class WorkerController(Protocol):
    """Controller operations used by the worker lifecycle."""

    def register(self, request: WorkerRegistration) -> WorkerRegistrationResult: ...

    def resolve_endpoint(self, name: str) -> str: ...

    def close(self) -> None: ...


class WorkerTaskProvider(Protocol):
    """Native task operations hosted by the worker RPC adapter."""

    def submit_task(self, request: AttemptLaunch) -> str: ...

    def get_task(self, task_id: str, attempt_id: int = -1) -> TaskInfo | None: ...

    def list_tasks(self) -> list[TaskInfo]: ...

    def kill_task(self, task_id: str, term_timeout_ms: int = 5000) -> bool: ...

    def handle_reconcile(self, request: WorkerReconcileRequest) -> WorkerReconcileResponse: ...

    def capture_and_log_profile(
        self,
        *,
        target: str,
        duration: int,
        profile: ProfileConfiguration,
        trigger: ProfileTrigger,
    ) -> bytes: ...

    def exec_in_container(self, task_id: str, command: list[str], timeout_seconds: int = 60) -> ExecResult: ...


class WorkerServer(Protocol):
    """Hosted worker service lifecycle."""

    def start(self, provider: WorkerTaskProvider, threads: ThreadContainer) -> None: ...

    def stop(self) -> None: ...
