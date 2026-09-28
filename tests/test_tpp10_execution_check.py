import threading
import time

import pytest

from experiments.domain_phase_mix import tpp10_execution_check as execution_check


class FakeRunner:
    """Records step batches; the canary batch blocks until released and may fail."""

    def __init__(self, canary_fails: bool = False):
        self.calls: list[tuple[str, ...]] = []
        self.canary_fails = canary_fails
        self.canary_started = threading.Event()
        self.canary_may_finish = threading.Event()
        self.canary_finished = threading.Event()
        self.remaining_ran_while_canary_running = False

    def __call__(self, *steps: str, max_concurrent: int, force_run_failed: bool) -> None:
        self.calls.append(steps)
        if "canary" in steps:
            self.canary_started.set()
            self.canary_may_finish.wait(timeout=5)
            self.canary_finished.set()
            if self.canary_fails:
                raise RuntimeError("canary crashed")
            return
        self.remaining_ran_while_canary_running = self.canary_started.is_set() and not self.canary_finished.is_set()


def _release_canary_after_remaining(runner: FakeRunner, ready: threading.Event) -> threading.Thread:
    def drive() -> None:
        runner.canary_started.wait(timeout=5)
        ready.set()
        deadline = time.monotonic() + 5
        while len(runner.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        runner.canary_may_finish.set()

    thread = threading.Thread(target=drive)
    thread.start()
    return thread


def test_checkpoint_releases_the_remaining_points_while_the_canary_still_runs():
    runner = FakeRunner()
    ready = threading.Event()
    driver = _release_canary_after_remaining(runner, ready)
    execution_check.run_with_early_release(
        ("canary",), ("p5", "p10"), release_ready=ready.is_set, poll_seconds=0.01, runner=runner
    )
    driver.join()
    assert runner.calls == [("canary",), ("p5", "p10")]
    assert runner.remaining_ran_while_canary_running


def test_canary_failure_before_release_submits_nothing_else():
    runner = FakeRunner(canary_fails=True)
    runner.canary_may_finish.set()
    with pytest.raises(RuntimeError, match="before release"):
        execution_check.run_with_early_release(
            ("canary",), ("p5",), release_ready=lambda: False, poll_seconds=0.01, runner=runner
        )
    assert runner.calls == [("canary",)]


def test_canary_failure_after_release_is_raised_once_the_released_points_finish():
    runner = FakeRunner(canary_fails=True)
    ready = threading.Event()
    driver = _release_canary_after_remaining(runner, ready)
    with pytest.raises(RuntimeError, match="after release"):
        execution_check.run_with_early_release(
            ("canary",), ("p5",), release_ready=ready.is_set, poll_seconds=0.01, runner=runner
        )
    driver.join()
    assert runner.calls == [("canary",), ("p5",)]


def test_fast_canary_success_releases_without_a_checkpoint():
    runner = FakeRunner()
    runner.canary_may_finish.set()
    execution_check.run_with_early_release(
        ("canary",), ("p5",), release_ready=lambda: False, poll_seconds=0.01, runner=runner
    )
    assert runner.calls == [("canary",), ("p5",)]


def test_already_built_canary_releases_immediately():
    runner = FakeRunner()
    execution_check.run_with_early_release((), ("p5",), release_ready=lambda: False, runner=runner)
    assert runner.calls == [("p5",)]
