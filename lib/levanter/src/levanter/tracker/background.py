# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Background-thread wrapper around any :class:`~levanter.tracker.Tracker`.

Many trackers (notably W&B) talk to a remote service that can fail
intermittently — quota exhausted, network blips, transient 5xx responses.
Those failures should not crash a long-running training job, and the trainer
thread should not block on tracker I/O. :class:`BackgroundTracker` provides
both guarantees:

* All forwarded calls are pushed onto a bounded queue and executed serially
  on a daemon thread, so the trainer thread returns immediately.
* Exceptions raised by the wrapped tracker are caught and logged; the worker
  keeps running so subsequent updates still go through.
* If the queue fills up (e.g. the wrapped tracker is wedged), additional
  updates are dropped with a rate-limited warning rather than blocking.

Tracker initialization is *not* wrapped. If e.g. ``wandb.init()`` fails
because of bad auth, that's a fatal configuration problem and the run
should refuse to start.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import typing
from typing import Any, Optional

from levanter.tracker.tracker import Tracker


logger = logging.getLogger(__name__)


class _Shutdown:
    """Sentinel placed on the queue to stop the worker."""


_SHUTDOWN = _Shutdown()

# Number of times we log a warning when the queue is full before throttling.
_DROP_LOG_BURST = 5
_DROP_LOG_PERIOD = 1000


class BackgroundTracker(Tracker):
    """Run another tracker's calls on a background thread, swallowing failures.

    Args:
        wrapped: Tracker whose ``log_*``/``finish`` calls will be deferred.
        max_queue_size: Maximum number of pending updates. When the queue is
            full, additional updates are dropped (with a warning) rather than
            blocking the producer.
        finish_timeout: Maximum time in seconds to wait for the queue to drain
            and the wrapped tracker to finish during :meth:`finish`.
    """

    def __init__(
        self,
        wrapped: Tracker,
        *,
        max_queue_size: int = 10000,
        finish_timeout: float = 120.0,
    ):
        self.wrapped = wrapped
        # Mirror the wrapped tracker's name so get_tracker() lookups still work.
        self.name = getattr(wrapped, "name", "background")
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._finish_timeout = finish_timeout
        self._dropped = 0
        self._stopped = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"BackgroundTracker[{self.name}]",
            daemon=True,
        )
        self._thread.start()

    # ---- worker -------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                method, args, kwargs = item
                try:
                    method(*args, **kwargs)
                except Exception:
                    logger.exception(
                        "Background tracker '%s' raised while processing %s; dropping update and continuing.",
                        self.name,
                        method.__name__,
                    )
            finally:
                self._queue.task_done()

    def _enqueue(self, method, *args, **kwargs) -> None:
        if self._stopped:
            logger.debug("Background tracker '%s' already stopped; dropping %s", self.name, method.__name__)
            return
        try:
            self._queue.put_nowait((method, args, kwargs))
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            if dropped <= _DROP_LOG_BURST or dropped % _DROP_LOG_PERIOD == 0:
                logger.error(
                    "Background tracker '%s' queue full; dropped %d update(s) so far.",
                    self.name,
                    dropped,
                )

    # ---- Tracker API --------------------------------------------------------

    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        self._enqueue(self.wrapped.log_hyperparameters, hparams)

    def log(
        self,
        metrics: typing.Mapping[str, Any],
        *,
        step: Optional[int],
        commit: Optional[bool] = None,
    ) -> None:
        self._enqueue(self.wrapped.log, metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        self._enqueue(self.wrapped.log_summary, metrics)

    def log_artifact(
        self,
        artifact_path,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
    ) -> None:
        self._enqueue(self.wrapped.log_artifact, artifact_path, name=name, type=type)

    def finish(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        deadline = time.monotonic() + self._finish_timeout

        # Enqueue the wrapped finish() so it runs after any pending logs.
        try:
            self._queue.put_nowait((self.wrapped.finish, (), {}))
        except queue.Full:
            logger.warning(
                "Background tracker '%s' queue full at shutdown; finish() may "
                "not be called on the wrapped tracker.",
                self.name,
            )

        # Wait for the worker to drain everything before sending the sentinel.
        # This is a best-effort drain — if the wrapped tracker is wedged we
        # still want to bound how long we block.
        remaining = max(deadline - time.monotonic(), 1.0)
        try:
            self._queue.put(_SHUTDOWN, timeout=remaining)
        except queue.Full:
            logger.warning(
                "Background tracker '%s' did not accept shutdown sentinel within %.1fs.",
                self.name,
                remaining,
            )

        remaining = max(deadline - time.monotonic(), 0.0)
        self._thread.join(timeout=remaining)
        if self._thread.is_alive():
            logger.warning(
                "Background tracker '%s' did not exit within %.1fs; abandoning thread (some updates may be lost).",
                self.name,
                self._finish_timeout,
            )
        if self._dropped:
            logger.warning(
                "Background tracker '%s' dropped %d update(s) total during run.",
                self.name,
                self._dropped,
            )

    # ---- Helpers (mostly for tests) -----------------------------------------

    def _wait_until_idle(self, timeout: float = 5.0) -> bool:
        """Block until every queued item has been processed.

        Returns ``True`` if the queue drained within ``timeout``, else
        ``False``. Intended for tests; mirrors :meth:`queue.Queue.join` but
        with a deadline.
        """
        deadline = time.monotonic() + timeout
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(timeout=remaining)
            return True


def maybe_wrap_background(
    tracker: Tracker,
    *,
    enabled: bool,
    max_queue_size: int,
    finish_timeout: float,
) -> Tracker:
    """Return ``tracker`` wrapped in a :class:`BackgroundTracker` iff ``enabled``."""
    if not enabled:
        return tracker
    return BackgroundTracker(tracker, max_queue_size=max_queue_size, finish_timeout=finish_timeout)
