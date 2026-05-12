# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import threading
import time

import jax.numpy as jnp
import pytest

from levanter.callbacks._core import StepInfo
from levanter.callbacks.state_adapter import CallbackStateView
from levanter.callbacks.watchdog import ForwardProgressWatchdog


def _make_step_info(step: int) -> StepInfo:
    # CallbackStateView.step is exposed as `state.step` and StepInfo.step subtracts 1.
    state = CallbackStateView(
        step=jnp.array(step + 1, dtype=jnp.int32),
        model="model",
        eval_model="eval_model",
        opt_state="opt_state",
    )
    return StepInfo(state=state, loss=0.0, step_duration=0.0)


def test_watchdog_does_not_fire_before_first_heartbeat():
    """If no step has ever completed (e.g., compilation is still running), the
    watchdog must not fire, even if the wall clock exceeds the timeout."""
    fired = threading.Event()

    cb = ForwardProgressWatchdog(
        timeout=0.05,
        check_interval=0.01,
        on_timeout=lambda elapsed, step: fired.set(),
    )
    try:
        # Wait several check intervals and a couple of timeouts. Since on_step
        # has never been called, fired must remain unset.
        time.sleep(0.25)
        assert not fired.is_set(), "watchdog fired before any heartbeat"
        assert not cb.triggered
    finally:
        cb.stop()


def test_watchdog_fires_after_timeout_with_no_progress():
    captured: list[tuple[float, int]] = []
    fired = threading.Event()

    def on_timeout(elapsed: float, last_step: int) -> None:
        captured.append((elapsed, last_step))
        fired.set()

    cb = ForwardProgressWatchdog(timeout=0.1, check_interval=0.01, on_timeout=on_timeout)
    try:
        cb.on_step(_make_step_info(step=42))
        # No further heartbeats: watchdog should fire within timeout + check_interval.
        assert fired.wait(timeout=2.0), "watchdog did not fire within expected window"
        assert cb.triggered
        assert len(captured) == 1
        elapsed, last_step = captured[0]
        assert last_step == 42
        assert elapsed >= 0.1
    finally:
        cb.stop()


def test_watchdog_does_not_fire_when_heartbeats_continue():
    fired = threading.Event()

    cb = ForwardProgressWatchdog(
        timeout=0.2,
        check_interval=0.01,
        on_timeout=lambda elapsed, step: fired.set(),
    )
    try:
        # Send heartbeats much faster than the timeout for longer than one timeout.
        deadline = time.monotonic() + 0.5
        step = 0
        while time.monotonic() < deadline:
            cb.on_step(_make_step_info(step=step))
            step += 1
            time.sleep(0.02)
        assert not fired.is_set(), "watchdog fired despite continuous heartbeats"
        assert not cb.triggered
    finally:
        cb.stop()


def test_watchdog_fires_only_once():
    call_count = 0
    fired = threading.Event()

    def on_timeout(elapsed: float, last_step: int) -> None:
        nonlocal call_count
        call_count += 1
        fired.set()

    cb = ForwardProgressWatchdog(timeout=0.05, check_interval=0.01, on_timeout=on_timeout)
    try:
        cb.on_step(_make_step_info(step=1))
        assert fired.wait(timeout=2.0)
        # Give the daemon plenty of opportunity to fire again; it should not.
        time.sleep(0.2)
        assert call_count == 1
    finally:
        cb.stop()


def test_watchdog_rejects_nonpositive_timeout():
    with pytest.raises(ValueError):
        ForwardProgressWatchdog(timeout=0)
    with pytest.raises(ValueError):
        ForwardProgressWatchdog(timeout=-1.0)


def test_watchdog_rejects_nonpositive_check_interval():
    with pytest.raises(ValueError):
        ForwardProgressWatchdog(timeout=1.0, check_interval=0)
    with pytest.raises(ValueError):
        ForwardProgressWatchdog(timeout=1.0, check_interval=-0.1)
