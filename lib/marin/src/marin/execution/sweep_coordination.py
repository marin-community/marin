# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-gang coordination for sweeps, via the Iris endpoint registry.

A multi-host (gang-scheduled) TPU job runs the sweep entrypoint on *every*
host. Only one host — the **leader** (Iris task 0) — may interact with a
target's ``step_lock``; all hosts must then run ``run_fn`` together as one
``jax.distributed`` gang. This mirrors ``iris.runtime.jax_init``: task 0
publishes a value through the job-scoped endpoint registry and tasks 1..N-1
poll the resolver until they see it.

Every sweep worker is therefore either a **leader** or a **follower**. A
single-host job (or one not running under Iris) is just a leader of a one-host
gang: it claims and runs targets alone, and its ``publish`` calls are inert
because there are no followers to hear them. The leader publishes an ordered
stream of *rounds*: round ``seq`` carries the ``target_id`` the gang should
train next, under the unique endpoint name ``sweep_round_{seq}``; a sentinel
marks "no more rounds — stop". The registry namespace is per Iris job, so a
gang's tasks share it while separate gang jobs stay isolated.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Protocol

from iris.runtime.endpoint_poll import poll_endpoint

logger = logging.getLogger(__name__)

# Endpoint name for round ``seq``. Append-only: each name is registered exactly
# once, so a follower polling ``sweep_round_{seq}`` reads an unambiguous value.
_ROUND_ENDPOINT = "sweep_round_{seq}"

# Registry address value standing in for "no more rounds". Target ids never
# contain whitespace, so this can never collide with a real target id.
_STOP = "__SWEEP_STOP__"


class GangRole(StrEnum):
    """This task's part in a sweep gang."""

    LEADER = "leader"
    """Task 0 (or any single-host worker): races ``step_lock`` and publishes rounds."""

    FOLLOWER = "follower"
    """Task 1..N-1 of a multi-host gang: blocks on the leader's rounds and mirrors them."""


class SweepCoordinator(Protocol):
    """Leader/follower channel for a sweep gang.

    The leader calls :meth:`publish`; followers call :meth:`receive`.
    """

    @property
    def role(self) -> GangRole: ...

    def publish(self, seq: int, target_id: str | None) -> None:
        """Announce round ``seq``. ``target_id is None`` signals the final stop."""
        ...

    def receive(self, seq: int) -> str | None:
        """Block until round ``seq`` is published; return its ``target_id`` or
        ``None`` for the stop sentinel."""
        ...


class GangCoordinator:
    """Registry-backed leader/follower channel for a sweep gang.

    When the job has a single task the coordinator is *inert*: it is always the
    leader, ``publish`` does nothing (no followers exist, and there may be no
    Iris context at all), and ``receive`` is never called. The Iris context is
    created lazily, so a single-host or non-Iris worker never touches it.

    Args:
        role: ``LEADER`` for task 0, ``FOLLOWER`` otherwise.
        num_tasks: Number of tasks in the job. Coordination is active only when
            ``> 1`` — that is the only case with followers to coordinate.
        poll_interval: Initial backoff for a follower's resolver polling.
        poll_timeout: Per-round wait budget for a follower, or ``None`` to wait
            without a deadline. The default is unbounded: a legitimate
            inter-round gap can be hours (the leader may be blocked on a peer
            gang's lock, or the previous round's training is long), and gang
            scheduling tears the whole job down if the leader dies — so a
            follower never needs to give up on its own.
    """

    def __init__(
        self,
        role: GangRole,
        *,
        num_tasks: int,
        poll_interval: float = 2.0,
        poll_timeout: float | None = None,
    ):
        self._role = role
        self._active = num_tasks > 1
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._ctx = None

    @property
    def role(self) -> GangRole:
        return self._role

    def _context(self):
        if self._ctx is None:
            from iris.client.client import iris_ctx  # noqa: PLC0415  # avoid requiring a live Iris ctx at import

            self._ctx = iris_ctx()
        return self._ctx

    def publish(self, seq: int, target_id: str | None) -> None:
        if not self._active:
            return
        value = _STOP if target_id is None else target_id
        # Best-effort cleanup is handled by the controller's cascade delete on
        # task teardown, as in jax_init — we do not unregister per round.
        self._context().registry.register(_ROUND_ENDPOINT.format(seq=seq), value)
        logger.info("Published sweep round %d: %s", seq, value)

    def receive(self, seq: int) -> str | None:
        value = poll_endpoint(
            self._context().resolver,
            _ROUND_ENDPOINT.format(seq=seq),
            poll_interval=self._poll_interval,
            poll_timeout=self._poll_timeout,
            waiting_log=f"Waiting for leader to publish sweep round {seq} ...",
        )
        return None if value == _STOP else value


def gang_coordinator() -> SweepCoordinator:
    """Build the coordinator for the current process from Iris job metadata.

    A process is a ``FOLLOWER`` only when it is task 1..N-1 of a multi-host
    gang; otherwise it is the ``LEADER`` (including every single-host or
    non-Iris worker, which leads a gang of one).
    """
    from iris.cluster.client.job_info import get_job_info  # noqa: PLC0415

    job_info = get_job_info()
    num_tasks = job_info.num_tasks if job_info is not None else 1
    task_index = job_info.task_index if job_info is not None else 0
    role = GangRole.FOLLOWER if (num_tasks > 1 and task_index != 0) else GangRole.LEADER
    return GangCoordinator(role, num_tasks=num_tasks)
