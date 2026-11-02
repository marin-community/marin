# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle of the shared Zephyr pool during a reference DAG run."""

import threading
import time

import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.execution.step_spec import StepSpec
from zephyr.coordinator import ZephyrCoordinator
from zephyr.dataset import Dataset

from experiments.datakit import reference_pipeline
from experiments.datakit.reference_pipeline import (
    DatakitSteps,
    PipelineScale,
    PoolConfig,
    run_reference_steps,
    shared_zephyr_context,
)


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


def test_shared_pool_accepts_coordinator_registration_after_fray_default(tmp_path, monkeypatch):
    client = LocalClient()
    create_group = client.create_actor_group

    def create_group_with_delayed_coordinator(actor_class, *args, **kwargs):
        group = create_group(actor_class, *args, **kwargs)
        if actor_class is ZephyrCoordinator:
            wait_ready = group.wait_ready

            def delayed_wait_ready(count=None, timeout=300.0):
                if timeout < 901:
                    raise TimeoutError("coordinator setup exceeded the actor readiness deadline")
                return wait_ready(count=count, timeout=timeout)

            monkeypatch.setattr(group, "wait_ready", delayed_wait_ready)
        return group

    monkeypatch.setattr(client, "create_actor_group", create_group_with_delayed_coordinator)
    worker = ResourceConfig(cpu=1, ram="1g")
    scale = PipelineScale(pool=PoolConfig(n_workers=1, worker=worker, task=worker))
    context = shared_zephyr_context(scale)
    context.client = client
    context.chunk_storage_prefix = str(tmp_path)

    with context:
        result = context.execute(Dataset.from_list([1]).map(lambda value: value + 1))

    assert result.results == [2]


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
