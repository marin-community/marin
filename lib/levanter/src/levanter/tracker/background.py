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
* Payloads are prepared on the calling thread via the wrapped tracker's
  ``_prepare_*`` hooks before they cross the queue, so the worker only does I/O.
  Keeping the ``jax.device_get`` there also matters for multi-host runs, where
  that transfer is a collective that must stay in program order on the main
  thread.
* Artifacts are staged (copied to a tracker-owned temp dir) on the calling
  thread before they cross the queue. Callers routinely build an artifact inside
  a ``tempfile.TemporaryDirectory`` and delete it as soon as ``log_artifact``
  returns; because the worker uploads asynchronously, the source would be gone by
  the time it runs. Staging captures the bytes synchronously; the worker uploads
  the staged copy and removes it.

Tracker initialization is *not* wrapped. If e.g. ``wandb.init()`` fails
because of bad auth, that's a fatal configuration problem and the run
should refuse to start.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import typing
from typing import Any, Optional

from levanter.tracker.tracker import Tracker


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _StagedArtifact:
    """A producer-thread copy of an artifact, living in its own temp dir.

    Callers routinely build an artifact inside a ``tempfile.TemporaryDirectory``
    and delete it as soon as ``log_artifact`` returns. Staging copies the bytes
    synchronously so the source can vanish immediately; the worker uploads
    :attr:`path`, then calls :meth:`cleanup` to remove the temp dir.
    """

    path: str

    @classmethod
    def stage(cls, source) -> Optional["_StagedArtifact"]:
        """Copy ``source`` into a fresh temp dir, or return ``None`` if it is missing.

        The basename is preserved so the artifact's contents and W&B's default
        artifact name are unchanged. Raises ``OSError`` if the copy itself fails.
        """
        src = os.fspath(source)
        if not os.path.exists(src):
            return None
        staging_dir = tempfile.mkdtemp(prefix="levanter-artifact-")
        dst = os.path.join(staging_dir, os.path.basename(src.rstrip("/\\")) or "artifact")
        try:
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        except OSError:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return cls(dst)

    def cleanup(self) -> None:
        shutil.rmtree(os.path.dirname(self.path), ignore_errors=True)


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

    def _enqueue(self, method, *args, **kwargs) -> bool:
        """Push a deferred call onto the worker queue. Returns whether it was accepted."""
        if self._stopped:
            logger.debug("Background tracker '%s' already stopped; dropping %s", self.name, method.__name__)
            return False
        try:
            self._queue.put_nowait((method, args, kwargs))
            return True
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
            return False

    def _defer(self, prepare, method, payload, **kwargs) -> None:
        """Prepare ``payload`` on the calling thread, then enqueue ``method``.

        ``prepare`` and ``method`` are bound methods of the wrapped tracker (e.g.
        ``_prepare_log`` and ``log``). A failure to prepare drops the update
        rather than crashing the producer.
        """
        if self._stopped:
            logger.debug("Background tracker '%s' already stopped; dropping %s", self.name, method.__name__)
            return
        try:
            payload = prepare(payload)
        except Exception:
            logger.exception(
                "Background tracker '%s' failed to prepare payload for %s; dropping update.",
                self.name,
                method.__name__,
            )
            return
        self._enqueue(method, payload, **kwargs)

    # ---- Tracker API --------------------------------------------------------

    def log_hyperparameters(self, hparams: dict[str, Any]) -> None:
        self._defer(self.wrapped._prepare_hyperparameters, self.wrapped.log_hyperparameters, hparams)

    def log(
        self,
        metrics: typing.Mapping[str, Any],
        *,
        step: Optional[int],
        commit: Optional[bool] = None,
    ) -> None:
        self._defer(self.wrapped._prepare_log, self.wrapped.log, metrics, step=step, commit=commit)

    def log_summary(self, metrics: dict[str, Any]) -> None:
        self._defer(self.wrapped._prepare_summary, self.wrapped.log_summary, metrics)

    def log_artifact(
        self,
        artifact_path,
        *,
        name: Optional[str] = None,
        type: Optional[str] = None,
    ) -> None:
        # Stage the bytes on the producer thread so the caller may delete the source
        # the moment this returns (callers often build it in a TemporaryDirectory).
        try:
            staged = _StagedArtifact.stage(artifact_path)
        except OSError:
            logger.exception(
                "Background tracker '%s': failed to stage artifact %s; dropping it.", self.name, artifact_path
            )
            return
        if staged is None:
            logger.warning(
                "Background tracker '%s': artifact %s does not exist; dropping it.", self.name, artifact_path
            )
            return
        if not self._enqueue(self._upload_staged_artifact, staged, name=name, type=type):
            staged.cleanup()

    def _upload_staged_artifact(self, staged: _StagedArtifact, *, name: Optional[str], type: Optional[str]) -> None:
        """Worker-side: upload a staged artifact, then remove its staging dir."""
        try:
            self.wrapped.log_artifact(staged.path, name=name, type=type)
        finally:
            staged.cleanup()

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
                "Background tracker '%s' queue full at shutdown; finish() may not be called on the wrapped tracker.",
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
