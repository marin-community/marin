# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BackendRelay: the per-remote-backend hand-off between the control tick and the poll RPC.

One relay sits between a :class:`RemoteTaskBackend` (driven by the control-tick
thread through ``reconcile``) and the :class:`RemoteAgentServer` (driven by RPC
threads as remote agents poll). The relay holds the desired attempt set the root
wants the agent to converge to, plus an inbox of the latest agent observations
the backend has not yet folded into the database. Every mutable field is guarded
by one :class:`threading.Lock` because both threads touch them.

Wire keys: the desired set and the observation inbox are both keyed by
``attempt_uid`` (``RunTaskRequest.attempt_uid``). ``key_to_uid`` resolves a
``running_tasks`` entry (which carries only ``(task_id, attempt_id)``, no uid)
back to its uid via the spec it was first dispatched with.
"""

import logging
import threading
from collections.abc import Iterable

from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.types import JobName
from iris.rpc import job_pb2, remote_agent_pb2

logger = logging.getLogger(__name__)


class BackendRelay:
    """Thread-safe desired-set + observation-inbox shared by reconcile and poll.

    The control tick calls :meth:`set_desired` and drains observations through
    :meth:`take_observations`; the poll RPC reads the desired set via
    :meth:`snapshot_desired` and feeds observations in through
    :meth:`ingest_observations`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # attempt_uid -> the RunTaskRequest the agent must converge to.
        self.desired: dict[str, job_pb2.RunTaskRequest] = {}
        # attempt_uid -> (task_id, attempt_id) for resolving observations.
        self.uid_to_task: dict[str, tuple[JobName, int]] = {}
        # (task_id_wire, attempt_id) -> attempt_uid, populated from every
        # tasks_to_run so a uid-less running task resolves to its cached spec.
        self.key_to_uid: dict[tuple[str, int], str] = {}
        # attempt_uid -> latest unconsumed observation (inbox).
        self.observations: dict[str, remote_agent_pb2.AgentObservation] = {}
        # Version of the desired set; bumped when the set or any spec changes.
        self.sync_id: int = 0

    def set_desired(
        self,
        tasks_to_run: list[job_pb2.RunTaskRequest],
        running_tasks: list[RunningTaskEntry],
    ) -> None:
        """Recompute the desired attempt set from this tick's dispatch drain.

        The new set is ``tasks_to_run`` plus every ``running_tasks`` entry
        resolved to its cached ``RunTaskRequest`` via ``key_to_uid``. A running
        task with no cached uid (or whose spec has been evicted) is skipped with
        a warning. ``sync_id`` bumps only when the uid set or any spec changed.
        """
        with self._lock:
            prev_desired = self.desired
            new_desired: dict[str, job_pb2.RunTaskRequest] = {}

            for req in tasks_to_run:
                uid = req.attempt_uid
                new_desired[uid] = req
                self.uid_to_task[uid] = (JobName.from_wire(req.task_id), req.attempt_id)
                self.key_to_uid[(req.task_id, req.attempt_id)] = uid

            for entry in running_tasks:
                key = (entry.task_id.to_wire(), entry.attempt_id)
                uid = self.key_to_uid.get(key)
                if uid is None:
                    logger.warning(
                        "remote relay: running task %s attempt %d has no cached uid; skipping",
                        entry.task_id.to_wire(),
                        entry.attempt_id,
                    )
                    continue
                if uid not in new_desired:
                    cached = prev_desired.get(uid)
                    if cached is None:
                        logger.warning(
                            "remote relay: running task %s attempt %d (uid %s) has no cached spec; skipping",
                            entry.task_id.to_wire(),
                            entry.attempt_id,
                            uid,
                        )
                        continue
                    new_desired[uid] = cached
                self.uid_to_task[uid] = (entry.task_id, entry.attempt_id)

            if self._desired_changed(prev_desired, new_desired):
                self.sync_id += 1
            self.desired = new_desired

    @staticmethod
    def _desired_changed(
        prev: dict[str, job_pb2.RunTaskRequest],
        new: dict[str, job_pb2.RunTaskRequest],
    ) -> bool:
        if prev.keys() != new.keys():
            return True
        return any(prev[uid] != new[uid] for uid in new)

    def take_observations(self) -> list[remote_agent_pb2.AgentObservation]:
        """Atomically drain and clear the observation inbox."""
        with self._lock:
            drained = list(self.observations.values())
            self.observations.clear()
            return drained

    def ingest_observations(self, observations: Iterable[remote_agent_pb2.AgentObservation]) -> None:
        """Store the latest observation per uid (last write wins)."""
        with self._lock:
            for obs in observations:
                self.observations[obs.attempt_uid] = obs

    def snapshot_desired(self) -> tuple[int, list[job_pb2.RunTaskRequest]]:
        """Return ``(sync_id, desired RunTaskRequests)`` under the lock."""
        with self._lock:
            return self.sync_id, list(self.desired.values())

    def resolve(self, attempt_uid: str) -> tuple[JobName, int] | None:
        """Resolve an attempt uid to its ``(task_id, attempt_id)``, or None."""
        with self._lock:
            return self.uid_to_task.get(attempt_uid)


class RelayRegistry:
    """Lookup of ``backend_id -> BackendRelay`` shared by the composer and server.

    The composer registers one relay per remote backend; the
    :class:`RemoteAgentServer` looks them up by the ``backend_id`` on each poll.
    """

    def __init__(self) -> None:
        self._relays: dict[str, BackendRelay] = {}

    def register(self, backend_id: str, relay: BackendRelay) -> None:
        self._relays[backend_id] = relay

    def get(self, backend_id: str) -> BackendRelay | None:
        return self._relays.get(backend_id)
