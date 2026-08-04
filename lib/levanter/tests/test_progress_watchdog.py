# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from threading import Event

import pytest

from levanter.callbacks import ProgressEvent, progress_watchdog as progress_watchdog_module
from levanter.callbacks.progress_watchdog import ProgressWatchdog, ProgressWatchdogConfig, STALLED_TRAINING_EXIT_CODE


def test_watchdog_ignores_first_compile_then_times_out_a_steady_state_step(monkeypatch):
    terminated = Event()
    exit_codes: list[int] = []

    def record_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        terminated.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        poll_interval=0.005,
    )

    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    assert not terminated.wait(timeout=0.05), "the first compiling step must remain unarmed"

    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert terminated.wait(timeout=1)
    watchdog.stop()
    assert exit_codes == [STALLED_TRAINING_EXIT_CODE]


def test_watchdog_uses_process_timeout_during_evaluation(monkeypatch):
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.EVALUATION_STARTED)

    current_time = 0.75
    assert not terminated.wait(timeout=0.03), "evaluation must not inherit the train-step deadline"
    watchdog.on_event(ProgressEvent.EVALUATION_FINISHED)
    current_time = 1.25
    assert not terminated.wait(timeout=0.03), "evaluation completion must reset process progress"
    watchdog.stop()


def test_watchdog_terminates_an_evaluation_that_stops_progress(monkeypatch):
    terminated = Event()
    exit_codes: list[int] = []

    def record_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        terminated.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(milliseconds=80),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.EVALUATION_STARTED)

    assert terminated.wait(timeout=1)
    watchdog.stop()
    assert exit_codes == [STALLED_TRAINING_EXIT_CODE]


def test_watchdog_runs_diagnostic_before_exit(monkeypatch):
    exited = Event()
    order: list[str] = []

    def diagnostic(_timeout) -> None:
        order.append("diagnostic")

    def record_exit(_exit_code: int) -> None:
        order.append("exit")
        exited.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        diagnostic=diagnostic,
        diagnostic_timeout=timedelta(milliseconds=100),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert exited.wait(timeout=1)
    watchdog.stop()
    assert order == ["diagnostic", "exit"]


def test_watchdog_diagnostic_cannot_postpone_exit(monkeypatch):
    exited = Event()
    never_returns = Event()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: exited.set())
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        diagnostic=lambda _timeout: never_returns.wait(),
        diagnostic_timeout=timedelta(milliseconds=30),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert exited.wait(timeout=1)
    watchdog.stop()


def test_watchdog_config_with_diagnostic_timeout_only_arms_process_zero() -> None:
    config = ProgressWatchdogConfig(
        step_timeout=timedelta(seconds=1),
        diagnostic_timeout=timedelta(seconds=1),
    )

    assert config.create(process_index=1, diagnostic=lambda _timeout: None) is None
    with pytest.raises(ValueError, match="diagnostic is required"):
        config.create(process_index=0)
    watchdog = config.create(process_index=0, diagnostic=lambda _timeout: None)
    assert watchdog is not None
    watchdog.stop()
