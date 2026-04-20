# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory worker health score with exponential decay.

Accumulates failure evidence across heterogeneous signals (heartbeat/ping RPC
failures, build failures) into a per-worker score that decays exponentially
with time. The reaper thread reads scores and terminates workers whose
aggregate score crosses the threshold.

``TASK_STATE_WORKER_FAILED`` is deliberately *not* a tracker signal: a worker
that actually died will also fail its next ping/heartbeat RPC, so observing
it via both paths would double-count the same failure.

Lives entirely in memory. A dying worker recurs within one signal cycle,
so losing accumulated evidence on controller restart doesn't meaningfully
delay termination.
"""

import enum
import logging
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from iris.cluster.types import WorkerId
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)


class HealthSignal(enum.StrEnum):
    """Failure signals that move the worker health score."""

    RPC_FAILURE = "rpc_failure"
    TASK_BUILD_FAILED = "task_build_failed"


SIGNAL_WEIGHT: Mapping[HealthSignal, float] = {
    HealthSignal.RPC_FAILURE: 1.0,
    HealthSignal.TASK_BUILD_FAILED: 0.5,
}
"""Per-signal additive weight.

RPC_FAILURE carries unit weight — this preserves today's 10-strike heartbeat
behavior (10 consecutive RPC failures trip the threshold).

TASK_BUILD_FAILED is a weaker signal (0.5): a user-visible FAILED originating
from a task still in BUILDING usually means the worker couldn't pull the image
or set up the environment — a soft hint at disk / network trouble, but it can
also be a broken Dockerfile or a user-side build command. Half weight means
~20 build failures on one worker cross the threshold by themselves, while
mixing with RPC failures still reaps promptly."""

HEALTH_SCORE_THRESHOLD = 10.0

HEALTH_SCORE_HALF_LIFE_S = 300.0


@dataclass(slots=True)
class _Entry:
    score: float
    updated_ms: int


def _now_ms() -> int:
    return Timestamp.now().epoch_ms()


class WorkerHealthTracker:
    """Tracks per-worker failure scores with exponential decay.

    Thread-safe: bumped from the ping/heartbeat and task-update threads,
    read from the reaper thread.
    """

    def __init__(
        self,
        *,
        half_life_s: float = HEALTH_SCORE_HALF_LIFE_S,
        threshold: float = HEALTH_SCORE_THRESHOLD,
        weights: Mapping[HealthSignal, float] = SIGNAL_WEIGHT,
        clock: Callable[[], int] = _now_ms,
    ) -> None:
        assert half_life_s > 0, "half_life_s must be positive"
        assert threshold > 0, "threshold must be positive"
        self._lam = math.log(2) / half_life_s
        self._threshold = threshold
        self._weights = dict(weights)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[WorkerId, _Entry] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    def bump(self, worker_id: WorkerId, signal: HealthSignal) -> float:
        """Record a failure signal for a worker. Returns the new score."""
        weight = self._weights[signal]
        now = self._clock()
        with self._lock:
            entry = self._entries.get(worker_id)
            decayed = self._decay(entry.score, now - entry.updated_ms) if entry else 0.0
            new_score = decayed + weight
            self._entries[worker_id] = _Entry(new_score, now)
        logger.debug(
            "Worker %s health bump: signal=%s weight=%.1f score=%.2f",
            worker_id,
            signal.value,
            weight,
            new_score,
        )
        return new_score

    def current_score(self, worker_id: WorkerId) -> float:
        now = self._clock()
        with self._lock:
            entry = self._entries.get(worker_id)
            if entry is None:
                return 0.0
            return self._decay(entry.score, now - entry.updated_ms)

    def workers_over_threshold(self) -> list[tuple[WorkerId, float]]:
        """Return (worker_id, score) pairs whose current score meets the threshold."""
        now = self._clock()
        out: list[tuple[WorkerId, float]] = []
        with self._lock:
            for wid, entry in self._entries.items():
                score = self._decay(entry.score, now - entry.updated_ms)
                if score >= self._threshold:
                    out.append((wid, score))
        return out

    def forget(self, worker_id: WorkerId) -> None:
        with self._lock:
            self._entries.pop(worker_id, None)

    def forget_many(self, worker_ids: Iterable[WorkerId]) -> None:
        with self._lock:
            for wid in worker_ids:
                self._entries.pop(wid, None)

    def snapshot(self) -> dict[WorkerId, float]:
        """Current decayed scores for every tracked worker (for diagnostics)."""
        now = self._clock()
        with self._lock:
            return {wid: self._decay(entry.score, now - entry.updated_ms) for wid, entry in self._entries.items()}

    def _decay(self, score: float, dt_ms: int) -> float:
        if dt_ms <= 0:
            return score
        return score * math.exp(-self._lam * dt_ms / 1000.0)
