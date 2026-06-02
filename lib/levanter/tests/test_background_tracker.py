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
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
import wandb

from levanter.tracker import BackgroundTracker, CompositeTracker
from levanter.tracker.histogram import SummaryStats
from levanter.tracker.tracker import Tracker
from levanter.tracker.wandb import WandbTracker


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
    # The swallowed exception is surfaced as an ERROR rather than silently dropped.
    assert any(r.levelno == logging.ERROR for r in caplog.records)


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
    # Each swallowed member failure is surfaced as an ERROR.
    assert sum(r.levelno == logging.ERROR for r in caplog.records) >= 5


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


def test_background_tracker_materializes_jax_arrays_before_enqueue():
    """jax.Array metric values must reach the wrapped tracker as host data.

    Materializing a jax.Array triggers a device->host transfer, which for a
    sharded array is a cross-host collective. That must run on the caller
    (trainer) thread so multi-host launch IDs stay in lockstep; if it ran on
    the worker thread the TPU slice would desync and halt. Observable proxy:
    the wrapped tracker never sees a jax.Array.
    """
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    try:
        bt.log({"loss": jnp.asarray(1.5), "steps": jnp.arange(3)}, step=0)
        bt.log_summary({"final": jnp.asarray(0.25)})
        bt.log_hyperparameters({"lr": jnp.asarray(3e-4)})
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    logged = inner.logs[0][0]
    assert not isinstance(logged["loss"], jax.Array)
    assert not isinstance(logged["steps"], jax.Array)
    assert not isinstance(inner.summary[0]["final"], jax.Array)
    assert not isinstance(inner.hparams[0]["lr"], jax.Array)
    # Values must survive materialization unchanged.
    assert float(logged["loss"]) == 1.5
    assert list(logged["steps"]) == [0, 1, 2]


def test_background_tracker_materializes_summary_stats_leaves():
    """SummaryStats is a pytree; its jax.Array leaves must be materialized too.

    Grug MoE logs per-layer routing SummaryStats every step. Their scalar
    fields are sharded jax.Arrays, so leaving them for the worker thread to
    convert is exactly the multi-host hazard this materialization prevents.
    """
    inner = RecordingTracker()
    bt = BackgroundTracker(inner)
    try:
        bt.log({"grads/hist": SummaryStats.from_array(jnp.arange(100.0))}, step=0)
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    logged = inner.logs[0][0]["grads/hist"]
    assert isinstance(logged, SummaryStats)
    for leaf in jax.tree_util.tree_leaves(logged):
        assert not isinstance(leaf, jax.Array)


def test_prepare_hooks_run_on_producer_thread():
    """Payload preparation runs on the caller thread, and its output is what crosses the queue.

    This is the contract that keeps JAX work off the background worker (the
    H100 SIGSEGV in #6108): the wrapped tracker's ``_prepare_*`` hooks are
    invoked on the producer thread, and the worker only ever sees their output.
    """
    caller_thread = threading.get_ident()
    prep_threads: list[int] = []

    class PrepRecordingTracker(RecordingTracker):
        def _prepare_log(self, metrics):
            prep_threads.append(threading.get_ident())
            return {"prepared": True}

        def _prepare_summary(self, metrics):
            prep_threads.append(threading.get_ident())
            return metrics

        def _prepare_hyperparameters(self, hparams):
            prep_threads.append(threading.get_ident())
            return hparams

    inner = PrepRecordingTracker()
    bt = BackgroundTracker(inner)
    try:
        bt.log({"loss": jnp.asarray(1.0)}, step=0)
        bt.log_summary({"final": jnp.asarray(0.5)})
        bt.log_hyperparameters({"lr": 0.1})
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    assert len(prep_threads) == 3, "every log method must invoke its prepare hook"
    assert all(tid == caller_thread for tid in prep_threads), "preparation must happen on the caller thread"

    # The worker receives the hook's output, not the raw input.
    logged_metrics, _, _ = inner.logs[0]
    assert logged_metrics == {"prepared": True}


def test_wandb_tracker_prepare_log_flattens_summary_stats():
    """WandbTracker._prepare_log flattens SummaryStats into a device-free, wandb-ready dict.

    After preparation the dict must hold no nested SummaryStats and no jax.Array
    leaves — that is what lets the W&B background worker upload without issuing
    any JAX op mid-profiler-upload (the H100 SIGSEGV in #6108).
    """
    tracker = WandbTracker(run=MagicMock(), suppress_logging=True)

    metrics = {
        "loss": jnp.asarray(1.5),
        "grads/hist": SummaryStats.from_array(jnp.arange(100.0)),
    }

    prepped = tracker._prepare_log(metrics)

    # SummaryStats has been expanded into its component scalar keys.
    assert "grads/hist" not in prepped
    assert {"grads/hist/min", "grads/hist/mean", "grads/hist/variance", "grads/hist/rms"} <= prepped.keys()

    for k, v in prepped.items():
        if isinstance(v, wandb.Histogram):
            continue
        assert not isinstance(v, jax.Array), f"{k} is still a jax.Array: {v!r}"


def test_background_tracker_no_jax_dispatch_on_worker_for_summary_stats():
    """End-to-end: nothing handed to ``wandb.run.log`` on the worker is a jax.Array.

    Direct regression for the H100 SIGSEGV (#6108). Now that SummaryStats
    derives mean/variance/rms eagerly and WandbTracker flattens it in
    ``_prepare_log`` on the producer thread, the worker only does the upload —
    every value it passes to ``run.log`` is device-free.
    """
    captured: list[dict] = []
    run = MagicMock()
    run.step = 0

    def _capture(to_log, *, step, commit):
        captured.append(to_log)

    run.log = _capture

    wandb_tracker = WandbTracker(run=run)
    bt = BackgroundTracker(wandb_tracker)
    try:
        bt.log({"grads/hist": SummaryStats.from_array(jnp.arange(100.0))}, step=0)
        assert bt._wait_until_idle(timeout=5)
    finally:
        bt.finish()

    assert captured, "wandb.run.log was never called"
    logged = captured[0]
    for k, v in logged.items():
        if isinstance(v, wandb.Histogram):
            continue
        assert not isinstance(v, jax.Array), f"{k} reached wandb.log() as a jax.Array: {v!r}"
