# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the BackgroundTracker wrapper.

These tests cover the behavior we want for robustness against W&B failures:

* Calls run on a background thread (don't block the producer)
* Exceptions from the wrapped tracker are caught and logged, never raised
* finish() drains pending updates
* Queue-full updates are dropped, not blocked
* Hardened CompositeTracker continues calling later trackers when an earlier
  one raises
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import pytest

from levanter.tracker import BackgroundTracker, CompositeTracker
from levanter.tracker.tracker import Tracker


class RecordingTracker(Tracker):
    """A tracker that records calls and optionally raises on demand."""

    name = "recording"

    def __init__(self, *, raise_on_log: BaseException | None = None) -> None:
        self.logs: list[tuple[dict[str, Any], int | None, bool | None]] = []
        self.summary: list[dict[str, Any]] = []
        self.hparams: list[dict[str, Any]] = []
        self.artifacts: list[tuple[Any, str | None, str | None]] = []
        self.finished = False
        self._raise_on_log = raise_on_log

    def log_hyperparameters(self, hparams):
        self.hparams.append(dict(hparams))

    def log(self, metrics, *, step, commit=None):
        if self._raise_on_log is not None:
            exc = self._raise_on_log
            self._raise_on_log = None
            raise exc
        self.logs.append((dict(metrics), step, commit))

    def log_summary(self, metrics):
        self.summary.append(dict(metrics))

    def log_artifact(self, artifact_path, *, name=None, type=None):
        self.artifacts.append((artifact_path, name, type))

    def finish(self):
        self.finished = True


class AlwaysRaisingTracker(Tracker):
    """A tracker that raises on every call. Used to verify exception isolation."""

    name = "always-raises"

    def __init__(self) -> None:
        self.call_count = 0

    def _boom(self):
        self.call_count += 1
        raise RuntimeError(f"boom #{self.call_count}")

    def log_hyperparameters(self, hparams):
        self._boom()

    def log(self, metrics, *, step, commit=None):
        self._boom()

    def log_summary(self, metrics):
        self._boom()

    def log_artifact(self, artifact_path, *, name=None, type=None):
        self._boom()

    def finish(self):
        self._boom()


def test_background_tracker_forwards_calls():
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    try:
        bt.log_hyperparameters({"lr": 0.1})
        bt.log({"loss": 1.0}, step=0)
        bt.log({"loss": 0.5}, step=1, commit=True)
        bt.log_summary({"final_loss": 0.5})
        bt.log_artifact("/tmp/x", name="x", type="model")
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    assert inner.hparams == [{"lr": 0.1}]
    assert inner.logs == [({"loss": 1.0}, 0, None), ({"loss": 0.5}, 1, True)]
    assert inner.summary == [{"final_loss": 0.5}]
    assert inner.artifacts == [("/tmp/x", "x", "model")]
    assert inner.finished is True


def test_background_tracker_swallows_exceptions(caplog):
    """Exceptions from the wrapped tracker must not crash the producer."""
    inner = RecordingTracker(raise_on_log=RuntimeError("wandb storage exceeded"))
    bt = BackgroundTracker(inner)
    try:
        with caplog.at_level(logging.ERROR, logger="levanter.tracker.background"):
            # First log raises in worker -- producer thread must not see it.
            bt.log({"loss": 1.0}, step=0)
            # Subsequent logs should still be processed.
            bt.log({"loss": 0.5}, step=1)
            assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    # The first call raised before recording; the second succeeded.
    assert inner.logs == [({"loss": 0.5}, 1, None)]
    # An ERROR log should have captured the wandb exception.
    assert any(
        "wandb storage exceeded" in r.getMessage() or "raised while processing" in r.getMessage()
        for r in caplog.records
    )


def test_background_tracker_runs_off_caller_thread():
    """The wrapped tracker must not run on the calling thread."""
    caller_thread = threading.get_ident()
    seen_threads: list[int] = []

    class ThreadRecording(Tracker):
        name = "threadrec"

        def log_hyperparameters(self, hparams):
            seen_threads.append(threading.get_ident())

        def log(self, metrics, *, step, commit=None):
            seen_threads.append(threading.get_ident())

        def log_summary(self, metrics):
            seen_threads.append(threading.get_ident())

        def log_artifact(self, artifact_path, *, name=None, type=None):
            seen_threads.append(threading.get_ident())

        def finish(self):
            seen_threads.append(threading.get_ident())

    bt = BackgroundTracker(ThreadRecording())
    try:
        bt.log({"loss": 1.0}, step=0)
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    assert seen_threads, "wrapped tracker was never called"
    for tid in seen_threads:
        assert tid != caller_thread, "wrapped tracker ran on caller thread"


def test_background_tracker_does_not_block_producer():
    """Producer thread must not block, even if wrapped tracker is slow."""
    block_event = threading.Event()
    release_event = threading.Event()

    class SlowTracker(Tracker):
        name = "slow"

        def log(self, metrics, *, step, commit=None):
            block_event.set()
            release_event.wait(timeout=10)

        def log_hyperparameters(self, hparams):
            pass

        def log_summary(self, metrics):
            pass

        def log_artifact(self, artifact_path, *, name=None, type=None):
            pass

        def finish(self):
            pass

    bt = BackgroundTracker(SlowTracker())
    try:
        # First log() blocks the worker indefinitely (until release_event).
        start = time.monotonic()
        bt.log({"loss": 1.0}, step=0)
        # While the worker is blocked, more logs should still return quickly.
        block_event.wait(timeout=5)
        for i in range(10):
            bt.log({"loss": float(i)}, step=i + 1)
        elapsed = time.monotonic() - start
        # Without queueing, this would have blocked ~10s. With queueing, it
        # must return effectively instantly.
        assert elapsed < 1.0, f"producer was blocked for {elapsed:.2f}s"
    finally:
        release_event.set()
        bt.finish()


def test_background_tracker_drops_when_queue_full(caplog):
    """When the queue is full, additional log calls are dropped, not blocked."""
    block_event = threading.Event()
    release_event = threading.Event()

    class GatedTracker(Tracker):
        name = "gated"

        def log(self, metrics, *, step, commit=None):
            block_event.set()
            release_event.wait(timeout=10)

        def log_hyperparameters(self, hparams):
            pass

        def log_summary(self, metrics):
            pass

        def log_artifact(self, artifact_path, *, name=None, type=None):
            pass

        def finish(self):
            pass

    bt = BackgroundTracker(GatedTracker(), max_queue_size=2)
    try:
        with caplog.at_level(logging.WARNING, logger="levanter.tracker.background"):
            # Saturate worker.
            bt.log({"i": 0}, step=0)
            block_event.wait(timeout=5)
            # Fill the queue (capacity 2).
            bt.log({"i": 1}, step=1)
            bt.log({"i": 2}, step=2)
            # These should be dropped, not blocked.
            start = time.monotonic()
            for i in range(100):
                bt.log({"i": i + 3}, step=i + 3)
            elapsed = time.monotonic() - start
            assert elapsed < 1.0
            assert bt._dropped >= 50
    finally:
        release_event.set()
        bt.finish()


def test_background_tracker_finish_calls_wrapped_finish():
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    bt.log({"x": 1}, step=0)
    bt.finish()
    assert inner.finished is True
    assert inner.logs == [({"x": 1}, 0, None)]


def test_background_tracker_finish_is_idempotent():
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    bt.finish()
    # second call should be a no-op, not raise.
    bt.finish()


def test_background_tracker_logs_after_finish_are_dropped():
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    bt.finish()
    bt.log({"x": 1}, step=0)
    # finish() already drained; the call after finish is a no-op.
    assert inner.logs == []


def test_composite_tracker_isolates_failures(caplog):
    """A failing member tracker must not prevent later members from being called."""
    bad = AlwaysRaisingTracker()
    good = RecordingTracker()
    composite = CompositeTracker([bad, good])

    with caplog.at_level(logging.ERROR, logger="levanter.tracker.tracker"):
        composite.log_hyperparameters({"lr": 0.1})
        composite.log({"loss": 1.0}, step=0)
        composite.log_summary({"final": 0.5})
        composite.log_artifact("/tmp/x", name="a", type="model")
        composite.finish()

    assert good.hparams == [{"lr": 0.1}]
    assert good.logs == [({"loss": 1.0}, 0, None)]
    assert good.summary == [{"final": 0.5}]
    assert good.artifacts == [("/tmp/x", "a", "model")]
    assert good.finished is True
    assert bad.call_count == 5  # one per method call
    assert any("continuing with remaining trackers" in r.getMessage() for r in caplog.records)


def test_composite_tracker_finish_does_not_raise():
    """Even if every member raises, finish() should not propagate."""
    bad1 = AlwaysRaisingTracker()
    bad2 = AlwaysRaisingTracker()
    composite = CompositeTracker([bad1, bad2])
    # Should log and swallow rather than raise.
    composite.finish()


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("network unreachable"),
        ConnectionError("dns resolve failed"),
        OSError("disk full"),
        ValueError("storage quota exceeded"),
    ],
)
def test_background_tracker_swallows_various_exceptions(exc, caplog):
    """Sample of error types we expect from W&B at runtime."""
    inner = RecordingTracker(raise_on_log=exc)
    bt = BackgroundTracker(inner)
    try:
        with caplog.at_level(logging.ERROR, logger="levanter.tracker.background"):
            bt.log({"loss": 1.0}, step=0)
            bt.log({"loss": 0.5}, step=1)
            assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()
    # Second call (after the raise) should still have been recorded.
    assert inner.logs == [({"loss": 0.5}, 1, None)]
