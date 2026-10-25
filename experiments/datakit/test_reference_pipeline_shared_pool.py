# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle of the shared Zephyr pool during a reference DAG run."""

import threading
import time

import pytest
from marin.execution.step_spec import StepSpec

from experiments.datakit import reference_pipeline
from experiments.datakit.reference_pipeline import DatakitSteps, run_reference_steps


class _RecordingPool:
    """Stands in for the shared ZephyrContext and records when it holds workers."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.stopped = threading.Event()

    def __enter__(self) -> "_RecordingPool":
        self.events.append("pool-start")
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if not self.stopped.is_set():
            self.events.append("pool-stop")
            self.stopped.set()


def test_shared_pool_is_held_only_between_normalize_and_its_last_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_pipeline, "SHARED_POOL_RELEASE_POLL_INTERVAL", 0.01)
    events: list[str] = []
    pool = _RecordingPool(events)

    def record(name: str):
        return lambda output_path: events.append(name)

    def store_after_pool_release(output_path: str) -> None:
        # A dedicated downstream stage must get the hosts while it still runs.
        assert pool.stopped.wait(timeout=10)
        events.append("store")

    prefix = str(tmp_path)
    normalize = StepSpec(name="normalize", output_path_prefix=prefix, fn=record("normalize"))
    tokenize = StepSpec(name="tokenize", output_path_prefix=prefix, deps=[normalize], fn=record("tokenize"))
    minhash = StepSpec(name="minhash", output_path_prefix=prefix, deps=[normalize], fn=record("minhash"))
    store = StepSpec(name="store", output_path_prefix=prefix, deps=[tokenize, minhash], fn=store_after_pool_release)
    steps = DatakitSteps(
        sources={"a": normalize},
        output_buckets=store,
        all_steps=[tokenize, minhash, store],
        shared_pool_steps=[tokenize, minhash],
    )

    run_reference_steps(steps, pool, max_concurrent=1)

    assert events == ["normalize", "pool-start", "tokenize", "minhash", "pool-stop", "store"]


def test_failed_pool_release_check_fails_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_pipeline, "SHARED_POOL_RELEASE_POLL_INTERVAL", 0.01)

    def unreadable_status(step: StepSpec) -> bool:
        raise OSError("status file read failed")

    monkeypatch.setattr(reference_pipeline, "step_is_built", unreadable_status)
    events: list[str] = []
    pool = _RecordingPool(events)

    def tokenize_after_check(output_path: str) -> None:
        # Let the watcher poll at least once before the DAG ends.
        time.sleep(0.1)
        events.append("tokenize")

    prefix = str(tmp_path)
    normalize = StepSpec(name="normalize", output_path_prefix=prefix, fn=lambda output_path: None)
    tokenize = StepSpec(name="tokenize", output_path_prefix=prefix, deps=[normalize], fn=tokenize_after_check)
    steps = DatakitSteps(
        sources={"a": normalize},
        output_buckets=tokenize,
        all_steps=[tokenize],
        shared_pool_steps=[tokenize],
    )

    with pytest.raises(OSError, match="status file read failed"):
        run_reference_steps(steps, pool, max_concurrent=1)
    assert events == ["pool-start", "tokenize", "pool-stop"]
