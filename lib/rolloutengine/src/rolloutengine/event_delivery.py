# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deliver rollout events to an observer without letting the observer affect the rollout."""

import asyncio
import logging

from rolloutengine.contracts import RolloutEvent, RolloutEventPayload, RolloutObserver

logger = logging.getLogger(__name__)

EVENT_QUEUE_CAPACITY = 1024
"""Events waiting for one attempt's observer; a full queue drops new events."""


class _EventDelivery:
    """Queue one attempt's events and deliver them in order from a single task.

    ``emit`` wraps each payload in a ``RolloutEvent`` with the attempt's ``run_key`` and ``task_id``
    and never blocks. When the queue is full, the event is dropped and counted. An observer
    exception is logged and ends delivery for the attempt; it never reaches the rollout. ``close``
    delivers what is queued within a deadline and returns the counts of events the observer did not
    receive, keyed for ``RolloutData.metrics``.
    """

    def __init__(self, observer: RolloutObserver | None, run_key: str, task_id: str):
        self.run_key = run_key
        self.task_id = task_id
        self.queue: asyncio.Queue[RolloutEvent] = asyncio.Queue(EVENT_QUEUE_CAPACITY)
        self.emitted = 0
        self.delivered = 0
        self.dropped = 0
        self.failed = False
        self.task = None if observer is None else asyncio.create_task(self._deliver(observer))

    def emit(self, payload: RolloutEventPayload) -> None:
        if self.task is None:
            return
        self.emitted += 1
        # After a failure nothing delivers, so the event counts as undelivered without being queued.
        if self.failed:
            return
        try:
            self.queue.put_nowait(RolloutEvent(self.run_key, self.task_id, payload))
        except asyncio.QueueFull:
            self.dropped += 1

    async def _deliver(self, observer: RolloutObserver) -> None:
        while True:
            event = await self.queue.get()
            try:
                await observer.on_event(event)
                self.delivered += 1
            except Exception:
                # Telemetry must not change a rollout, so the failure is logged and counted, not raised.
                logger.exception("Rollout observer failed on %s; delivery stops for this attempt", event.payload.type)
                self.failed = True
                return
            finally:
                self.queue.task_done()

    async def close(self, timeout: float) -> dict[str, float]:
        if self.task is None:
            return {}
        drained = asyncio.create_task(self.queue.join())
        try:
            await asyncio.wait((drained, self.task), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.cancel()
            drained.cancel()
        counts = {
            "observer_dropped_events": float(self.dropped),
            # The event that failed counts as an error, not as undelivered.
            "observer_undelivered_events": float(self.emitted - self.delivered - self.dropped - self.failed),
            "observer_error_count": float(self.failed),
        }
        return {key: value for key, value in counts.items() if value}

    def cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()
