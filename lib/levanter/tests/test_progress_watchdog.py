# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from threading import Event

import pytest

from levanter.callbacks import ProgressEvent, progress_watchdog as progress_watchdog_module
from levanter.callbacks._core import StepInfo
from levanter.callbacks.progress_watchdog import ProgressWatchdog, ProgressWatchdogConfig, STALLED_TRAINING_EXIT_CODE


@pytest.fixture
def watchdog():
    """Build watchdogs and stop them even when an assertion fails.

    A watchdog runs a daemon thread that calls `os._exit` on expiry. A test that leaks one past
    `monkeypatch` teardown fires it against the real `os._exit` and kills the test session, which
    hides the failure that leaked it.
    """
    built: list[ProgressWatchdog] = []

    def build(**kwargs) -> ProgressWatchdog:
        return track(ProgressWatchdog(**kwargs))

    def track(made: ProgressWatchdog | None) -> ProgressWatchdog | None:
        if made is not None:
            built.append(made)
        return made

    build.track = track  # type: ignore[attr-defined]
    yield build
    for made in built:
        made.stop()


def test_watchdog_ignores_first_compile_then_times_out_a_steady_state_step(monkeypatch, watchdog):
    terminated = Event()
    exit_codes: list[int] = []

    def record_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        terminated.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    made = watchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )

    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    assert not terminated.wait(timeout=0.05), "the first compiling step must remain unarmed"

    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert terminated.wait(timeout=1)
    assert exit_codes == [STALLED_TRAINING_EXIT_CODE]


def test_watchdog_waits_for_startup_grace_period_before_terminating(monkeypatch, watchdog):
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    made = watchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    current_time = timedelta(hours=1).total_seconds() - 1
    assert not terminated.wait(timeout=0.03)

    current_time = timedelta(hours=1).total_seconds()
    assert terminated.wait(timeout=1)


def test_watchdog_uses_process_timeout_during_evaluation(monkeypatch, watchdog):
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    made = watchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.EVALUATION_STARTED)

    current_time = 0.75
    assert not terminated.wait(timeout=0.03), "evaluation must not inherit the train-step deadline"
    made.on_event(ProgressEvent.EVALUATION_FINISHED)
    current_time = 1.25
    assert not terminated.wait(timeout=0.03), "evaluation completion must reset process progress"


def test_watchdog_terminates_an_evaluation_that_stops_progress(monkeypatch, watchdog):
    terminated = Event()
    exit_codes: list[int] = []

    def record_exit(exit_code: int) -> None:
        exit_codes.append(exit_code)
        terminated.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    made = watchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(milliseconds=80),
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.EVALUATION_STARTED)

    assert terminated.wait(timeout=1)
    assert exit_codes == [STALLED_TRAINING_EXIT_CODE]


def test_watchdog_runs_diagnostic_before_exit(monkeypatch, watchdog):
    exited = Event()
    order: list[str] = []

    def diagnostic(_timeout) -> None:
        order.append("diagnostic")

    def record_exit(_exit_code: int) -> None:
        order.append("exit")
        exited.set()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", record_exit)
    made = watchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        startup_grace_period=timedelta(0),
        diagnostic=diagnostic,
        diagnostic_timeout=timedelta(milliseconds=100),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert exited.wait(timeout=1)
    assert order == ["diagnostic", "exit"]


def test_watchdog_diagnostic_cannot_postpone_exit(monkeypatch, watchdog):
    exited = Event()
    never_returns = Event()

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: exited.set())
    made = watchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(seconds=1),
        startup_grace_period=timedelta(0),
        diagnostic=lambda _timeout: never_returns.wait(),
        diagnostic_timeout=timedelta(milliseconds=30),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    assert exited.wait(timeout=1)


def test_watchdog_arms_on_a_process_that_does_not_capture_diagnostics(watchdog) -> None:
    """Setting a diagnostic used to disarm every rank but zero.

    Diagnostics are captured once, but the deadlines have to hold everywhere: a rank stalled after
    a collective has to terminate itself rather than wait for process zero to block behind it.
    """
    config = ProgressWatchdogConfig(
        step_timeout=timedelta(seconds=10),
        diagnostic_timeout=timedelta(seconds=1),
    )

    made = watchdog.track(config.create(process_index=1))

    assert made is not None


def test_watchdog_config_still_requires_a_diagnostic_on_the_capturing_process() -> None:
    config = ProgressWatchdogConfig(step_timeout=timedelta(seconds=1), diagnostic_timeout=timedelta(seconds=1))

    with pytest.raises(ValueError):
        config.create(process_index=0)


def test_watchdog_terminates_when_the_first_training_step_never_starts(monkeypatch, watchdog):
    """Restore, cache construction and compile precede any progress event.

    The step and process deadlines only arm once a step has reported progress, so without a
    startup deadline a process that never reaches its first step waits forever.
    """
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    watchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        startup_timeout=timedelta(hours=2),
        poll_interval=0.005,
    )

    current_time = timedelta(hours=2).total_seconds() - 1
    assert not terminated.wait(timeout=0.03)

    current_time = timedelta(hours=2).total_seconds()
    assert terminated.wait(timeout=1)


def test_watchdog_startup_deadline_covers_a_first_step_that_never_finishes(monkeypatch, watchdog):
    """The first step is exempt from the step deadline while it compiles, so if that compile hangs
    the startup deadline is the only thing left to catch it."""
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    made = watchdog(
        step_timeout=timedelta(seconds=1),
        process_timeout=timedelta(seconds=1),
        startup_timeout=timedelta(hours=2),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)

    current_time = timedelta(hours=2).total_seconds()

    assert terminated.wait(timeout=1)


def test_watchdog_startup_deadline_lapses_once_a_step_completes(monkeypatch, watchdog):
    """A run that is training must not be killed for outliving the startup deadline."""
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    made = watchdog(
        step_timeout=timedelta(hours=10),
        process_timeout=timedelta(hours=10),
        startup_timeout=timedelta(hours=2),
        startup_grace_period=timedelta(hours=1),
        poll_interval=0.005,
    )
    made.on_event(ProgressEvent.TRAIN_STEP_STARTED)
    made.on_event(ProgressEvent.TRAIN_STEP_FINISHED)

    current_time = timedelta(hours=3).total_seconds()

    assert not terminated.wait(timeout=0.05)


def test_watchdog_bounds_a_finalization_that_never_ran_a_step(monkeypatch, watchdog):
    """A run restoring at its final step reaches the forced callback pass with no lifecycle event
    behind it. That pass has to leave a deadline armed, or a hung final checkpoint hangs forever."""
    terminated = Event()
    current_time = 0.0

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    monkeypatch.setattr(progress_watchdog_module, "monotonic", lambda: current_time)
    made = watchdog(
        step_timeout=timedelta(hours=1),
        process_timeout=timedelta(minutes=30),
        startup_timeout=timedelta(hours=2),
        startup_grace_period=timedelta(0),
        poll_interval=0.005,
    )
    made.on_step(StepInfo(state=None, loss=0.0, step_duration=0.0), force=True)

    current_time = timedelta(minutes=30).total_seconds()

    assert terminated.wait(timeout=1)
