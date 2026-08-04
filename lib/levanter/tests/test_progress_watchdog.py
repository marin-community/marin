# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from threading import Event

from levanter.callbacks import ProgressEvent, progress_watchdog as progress_watchdog_module
from levanter.callbacks.progress_watchdog import ProgressWatchdog, STALLED_TRAINING_EXIT_CODE


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

    monkeypatch.setattr(progress_watchdog_module.os, "_exit", lambda _exit_code: terminated.set())
    watchdog = ProgressWatchdog(
        step_timeout=timedelta(milliseconds=30),
        process_timeout=timedelta(milliseconds=500),
        poll_interval=0.005,
    )
    watchdog.on_event(ProgressEvent.TRAIN_STEP_FINISHED)
    watchdog.on_event(ProgressEvent.EVALUATION_STARTED)

    assert not terminated.wait(timeout=0.06), "evaluation must not inherit the train-step deadline"
    watchdog.on_event(ProgressEvent.EVALUATION_FINISHED)
    assert not terminated.wait(timeout=0.06), "evaluation completion must reset process progress"
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
