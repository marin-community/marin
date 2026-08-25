# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-gang coordination for sweeps, via an Iris actor on the leader.

A multi-host (gang-scheduled) TPU job runs the sweep entrypoint on *every*
host. Only one host — the **leader** (Iris task 0) — may interact with a
target's ``step_lock``; all hosts must then run ``run_fn`` together as one
``jax.distributed`` gang. So the leader hosts a :class:`SweepLeaderActor` and
*decides* which target the gang runs next; **followers** (tasks 1..N-1) connect
to it and call :meth:`SweepLeaderActor.next_round` for each round, running
``run_fn`` alongside the leader. ``run_fn`` (gang training) is the cross-host
barrier, so leader and followers move in lockstep: a follower asks for the next
round only after finishing the current one, exactly when the leader — also done
training — is claiming the next lock.

The endpoint registry is used only for discovery (mapping the actor name to the
leader's address); the rounds themselves travel as ``Round`` objects over the
actor RPC. A single-host (or non-Iris) worker is the leader of a gang of one and
runs the loop alone with no actor at all.

Multi-host sweep jobs must request an ``actor`` port (``ports=['actor']``) so the
leader's actor server is reachable by its followers.
"""

import logging
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info

logger = logging.getLogger(__name__)

# Actor name the leader registers and followers resolve.
_LEADER_ACTOR = "sweep_leader"
# Named port the multi-host sweep job must allocate for the leader's server.
_ACTOR_PORT = "actor"


@dataclass(frozen=True)
class Round:
    """One round the leader announces. ``target_id is None`` is the terminal stop."""

    target_id: str | None


class SweepLeaderActor:
    """Serves an ordered stream of rounds to followers.

    The leader publishes each claimed target locally via :meth:`publish`; a
    follower fetches round ``seq`` via :meth:`next_round`, which blocks until the
    leader has published it. :meth:`wait_for_followers` lets the leader hold the
    server open until every follower has consumed the final round, so none gets a
    connection error during teardown.
    """

    def __init__(self):
        self._rounds: dict[int, Round] = {}
        self._fetched: dict[int, int] = {}
        self._cond = threading.Condition()

    def publish(self, seq: int, target_id: str | None) -> None:
        with self._cond:
            self._rounds[seq] = Round(target_id)
            self._cond.notify_all()

    def next_round(self, seq: int) -> Round:
        with self._cond:
            self._cond.wait_for(lambda: seq in self._rounds)
            self._fetched[seq] = self._fetched.get(seq, 0) + 1
            self._cond.notify_all()
            return self._rounds[seq]

    def wait_for_followers(self, seq: int, count: int) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._fetched.get(seq, 0) >= count)


class GangRole(StrEnum):
    """This task's part in a sweep gang."""

    LEADER = "leader"
    """Task 0 (or any single-host worker): races ``step_lock`` and announces rounds."""

    FOLLOWER = "follower"
    """Task 1..N-1 of a multi-host gang: fetches the leader's rounds and mirrors them."""


class SweepCoordinator(Protocol):
    """Leader/follower channel for a sweep gang.

    The leader calls :meth:`publish` then :meth:`wait_for_followers`; followers
    call :meth:`receive`. :meth:`close` releases resources and must be safe to
    call after an error (so it never blocks on followers).
    """

    @property
    def role(self) -> GangRole: ...

    def publish(self, seq: int, target_id: str | None) -> None:
        """Announce round ``seq``. ``target_id is None`` is the terminal stop."""
        ...

    def receive(self, seq: int) -> str | None:
        """Fetch round ``seq``; return its ``target_id`` or ``None`` for stop."""
        ...

    def wait_for_followers(self, seq: int) -> None:
        """Block until every follower has consumed round ``seq`` (leader only)."""
        ...

    def close(self) -> None:
        """Release resources. Safe to call after an error."""
        ...


class SoloLeader:
    """Leader of a gang of one: no actor, announcements go nowhere."""

    @property
    def role(self) -> GangRole:
        return GangRole.LEADER

    def publish(self, seq: int, target_id: str | None) -> None:
        return None

    def receive(self, seq: int) -> str | None:
        raise AssertionError("SoloLeader.receive must never be called")

    def wait_for_followers(self, seq: int) -> None:
        return None

    def close(self) -> None:
        return None


class GangLeader:
    """Hosts the :class:`SweepLeaderActor` and announces rounds to followers."""

    def __init__(self, num_followers: int):
        ctx = iris_ctx()
        job_info = get_job_info()
        assert job_info is not None, "GangLeader requires an Iris job context"
        self._num_followers = num_followers
        self._registry = ctx.registry
        self._actor = SweepLeaderActor()
        self._server = ActorServer(host="0.0.0.0", port=ctx.get_port(_ACTOR_PORT))
        self._server.register(_LEADER_ACTOR, self._actor)
        actual_port = self._server.serve_background()
        address = f"http://{job_info.advertise_host}:{actual_port}"
        self._endpoint_id = ctx.registry.register(_LEADER_ACTOR, address)
        logger.info("Sweep leader actor serving at %s for %d follower(s)", address, num_followers)

    @property
    def role(self) -> GangRole:
        return GangRole.LEADER

    def publish(self, seq: int, target_id: str | None) -> None:
        self._actor.publish(seq, target_id)
        logger.info("Announced sweep round %d: %s", seq, target_id)

    def receive(self, seq: int) -> str | None:
        raise AssertionError("GangLeader.receive must never be called")

    def wait_for_followers(self, seq: int) -> None:
        self._actor.wait_for_followers(seq, self._num_followers)

    def close(self) -> None:
        # Stop the server first so the resource is released even if unregister
        # fails; the registry entry is best-effort and the controller's cascade
        # delete reclaims it on task teardown regardless.
        self._server.stop()
        self._registry.unregister(self._endpoint_id)


class GangFollower:
    """Connects to the leader's actor and mirrors its rounds."""

    def __init__(self):
        ctx = iris_ctx()
        self._client = ActorClient(ctx.resolver, _LEADER_ACTOR)

    @property
    def role(self) -> GangRole:
        return GangRole.FOLLOWER

    def publish(self, seq: int, target_id: str | None) -> None:
        raise AssertionError("GangFollower.publish must never be called")

    def receive(self, seq: int) -> str | None:
        return self._client.next_round(seq).target_id

    def wait_for_followers(self, seq: int) -> None:
        return None

    def close(self) -> None:
        return None


def gang_coordinator() -> SweepCoordinator:
    """Build the coordinator for the current process from Iris job metadata.

    A single-task job (or one not running under Iris) is a :class:`SoloLeader`.
    A multi-host gang gets a :class:`GangLeader` for task 0 and a
    :class:`GangFollower` for the rest.
    """
    job_info = get_job_info()
    num_tasks = job_info.num_tasks if job_info is not None else 1
    task_index = job_info.task_index if job_info is not None else 0
    if num_tasks <= 1:
        return SoloLeader()
    if task_index == 0:
        return GangLeader(num_followers=num_tasks - 1)
    return GangFollower()
