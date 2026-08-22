# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from threading import Event

import pytest

from levanter.callbacks import ProgressEvent, progress_watchdog as progress_watchdog_module
from levanter.callbacks.progress_watchdog import (
    ProgressState,
    ProgressWatchdog,
    ProgressWatchdogConfig,
    STALLED_TRAINING_EXIT_CODE,
)


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
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )

    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    assert not terminated.wait(timeout=0.05), "the first compiling step must remain unarmed"

    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert terminated.wait(timeout=1)
    watchdog.stop()
    assert exit_codes == [STALLED_TRAINING_EXIT_CODE]


def test_watchdog_waits_for_startup_grace_period_before_terminating(monkeypatch):
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    current_time = timedelta(hours=1).total_seconds() - 1
    assert not terminated.wait(timeout=0.03)

    current_time = timedelta(hours=1).total_seconds()
    assert terminated.wait(timeout=1)
    watchdog.stop()


def test_watchdog_uses_process_timeout_during_evaluation(monkeypatch):
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
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
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
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
        startup_grace_period=timedelta(0),
        diagnostic=diagnostic,
        diagnostic_timeout=timedelta(milliseconds=100),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
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
        startup_grace_period=timedelta(0),
        diagnostic=lambda _timeout: never_returns.wait(),
        diagnostic_timeout=timedelta(milliseconds=30),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert exited.wait(timeout=1)
    watchdog.stop()


def test_watchdog_config_arms_every_process_but_only_requires_rank_zero_diagnostics() -> None:
    config = ProgressWatchdogConfig(
        step_timeout=timedelta(seconds=1),
        diagnostic_timeout=timedelta(seconds=1),
    )

    worker_watchdog = config.create(process_index=1)
    assert worker_watchdog is not None
    worker_watchdog.stop()
    with pytest.raises(ValueError):
        config.create(process_index=0)
    watchdog = config.create(process_index=0, diagnostic=lambda _timeout: None)
    assert watchdog is not None
    watchdog.stop()


def test_watchdog_terminates_when_the_first_training_step_never_starts(monkeypatch):
    """Restore, cache construction and compile precede any progress event.

    The step and process deadlines only arm once a step has reported progress, so without a
    startup deadline a process that never reaches its first step waits forever.
    """
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        startup_timeout=timedelta(hours=2),
        poll_interval=0.005,
    )

    current_time = timedelta(hours=2).total_seconds() - 1
    assert not terminated.wait(timeout=0.03)

    current_time = timedelta(hours=2).total_seconds()
    assert terminated.wait(timeout=1)
    watchdog.stop()


def test_watchdog_startup_deadline_covers_a_first_step_that_never_finishes(monkeypatch):
    """The first step is exempt from the step deadline while it compiles, so if that compile hangs
    the startup deadline is the only thing left to catch it."""
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        startup_timeout=timedelta(hours=2),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    current_time = timedelta(hours=2).total_seconds()

    assert terminated.wait(timeout=1)
    watchdog.stop()


def test_watchdog_startup_deadline_lapses_once_a_step_completes(monkeypatch):
    """A run that is training must not be killed for outliving the startup deadline."""
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(hours=10),
        process_timeout=timedelta(hours=10),
        startup_timeout=timedelta(hours=2),
        startup_grace_period=timedelta(hours=1),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)

    current_time = timedelta(hours=3).total_seconds()

    assert not terminated.wait(timeout=0.05)
    watchdog.stop()


def test_health_tracks_the_deadline_governing_the_current_wait(monkeypatch) -> None:
    current_time = 0.0
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=600),
        process_timeout=timedelta(seconds=900),
        startup_timeout=timedelta(seconds=4800),
        startup_grace_period=timedelta(seconds=60),
        poll_interval=3600.0,
    )

    health = watchdog.health()
    assert health.state is ProgressState.STARTING
    assert (health.event, health.timeout) == (ProgressEvent.PROCESS_STARTED, 4800.0)

    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    current_time = 3000.0
    assert watchdog.health().state is ProgressState.STARTING, "a first step that still compiles is unarmed"

    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    current_time = 3599.0
    health = watchdog.health()
    assert health.state is ProgressState.PROGRESSING
    assert (health.event, health.elapsed, health.timeout) == (ProgressEvent.TRAIN_STEP_STARTED, 599.0, 600.0)

    current_time = 3600.0
    assert watchdog.health().state is ProgressState.STALLED

    watchdog.stop()
    assert watchdog.health().state is ProgressState.FINISHED


def test_health_reports_a_startup_that_outlives_its_deadline(monkeypatch) -> None:
    current_time = 0.0
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=None,
        process_timeout=None,
        startup_timeout=timedelta(seconds=4800),
        poll_interval=3600.0,
    )

    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    current_time = 4800.0

    health = watchdog.health()
    assert health.state is ProgressState.STALLED
    assert health.event is ProgressEvent.PROCESS_STARTED

    watchdog.stop()


def test_health_reports_the_process_deadline_between_steps(monkeypatch) -> None:
    current_time = 0.0
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=600),
        process_timeout=timedelta(seconds=900),
        startup_grace_period=timedelta(0),
        poll_interval=3600.0,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.CHECKPOINT_STARTED)

    current_time = 700.0
    health = watchdog.health()
    assert health.state is ProgressState.PROGRESSING, "a checkpoint must not inherit the train-step deadline"
    assert (health.event, health.timeout) == (ProgressEvent.CHECKPOINT_STARTED, 900.0)

    watchdog.stop()


def test_health_stays_stalled_while_the_diagnostic_runs(monkeypatch) -> None:
    """A terminating watchdog must not report FINISHED during its diagnostic budget."""
    exited = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: exited.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(seconds=600),
        process_timeout=timedelta(seconds=900),
        startup_grace_period=timedelta(0),
        diagnostic=lambda _timeout: None,
        diagnostic_timeout=timedelta(seconds=20),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    current_time = 600.0
    assert exited.wait(timeout=1)

    health = watchdog.health()
    assert health.state is ProgressState.STALLED
    assert health.event is ProgressEvent.TRAIN_STEP_STARTED
