# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest

from levanter import megascale


@dataclass(frozen=True)
class _FakeJobInfo:
    task_index: int
    num_tasks: int
    advertise_host: str = "10.0.0.2"


@dataclass(frozen=True)
class _FakeResolveResult:
    is_empty: bool


class _FakeRegistry:
    def __init__(self, ready_tasks: set[int]):
        self.ready_tasks = ready_tasks
        self.registered: list[tuple[str, str]] = []

    def register(self, name: str, address: str) -> str:
        self.registered.append((name, address))
        if name.startswith(megascale.MEGASCALE_READY_ENDPOINT_PREFIX):
            self.ready_tasks.add(int(name.removeprefix(megascale.MEGASCALE_READY_ENDPOINT_PREFIX)))
        return f"endpoint-{name}"

    def unregister(self, endpoint_id: str) -> None:
        pass


class _FakeResolver:
    def __init__(self, ready_tasks: set[int]):
        self.ready_tasks = ready_tasks

    def resolve(self, name: str) -> _FakeResolveResult:
        if not name.startswith(megascale.MEGASCALE_READY_ENDPOINT_PREFIX):
            return _FakeResolveResult(is_empty=True)
        task_index = int(name.removeprefix(megascale.MEGASCALE_READY_ENDPOINT_PREFIX))
        return _FakeResolveResult(is_empty=task_index not in self.ready_tasks)


@dataclass(frozen=True)
class _FakeIrisContext:
    registry: _FakeRegistry
    resolver: _FakeResolver


def test_configure_megascale_skips_without_iris_slice_topology(monkeypatch):
    monkeypatch.delenv(megascale.IRIS_SLICE_COUNT, raising=False)

    assert megascale.configure_megascale_from_iris() == {}


def test_configure_megascale_maps_iris_slice_topology_to_megascale_env(monkeypatch):
    monkeypatch.setenv(megascale.IRIS_SLICE_COUNT, "2")
    monkeypatch.setenv(megascale.IRIS_TASKS_PER_SLICE, "4")
    monkeypatch.setattr(megascale, "get_job_info", lambda: _FakeJobInfo(task_index=5, num_tasks=8))
    monkeypatch.setattr(megascale, "_wait_for_all_tasks_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(megascale, "_coordinator_address", lambda *args, **kwargs: "10.0.0.1:8081")

    env = megascale.configure_megascale_from_iris()

    assert env == {
        "MEGASCALE_COORDINATOR_ADDRESS": "10.0.0.1:8081",
        "MEGASCALE_NUM_SLICES": "2",
        "MEGASCALE_PORT": "8081",
        "MEGASCALE_SLICE_ID": "1",
    }
    assert all(os.environ[key] == value for key, value in env.items())


def test_megascale_env_rejects_wrong_task_count(monkeypatch):
    monkeypatch.setattr(megascale, "get_job_info", lambda: _FakeJobInfo(task_index=0, num_tasks=3))

    with pytest.raises(ValueError, match="Megascale expects 4"):
        megascale.megascale_env_for_iris_task(slice_count=2, tasks_per_slice=2)


def test_wait_for_all_tasks_ready_registers_current_task_and_waits(monkeypatch):
    ready_tasks = {0}
    registry = _FakeRegistry(ready_tasks)
    resolver = _FakeResolver(ready_tasks)
    monkeypatch.setattr(megascale, "iris_ctx", lambda: _FakeIrisContext(registry=registry, resolver=resolver))

    megascale._wait_for_all_tasks_ready(
        _FakeJobInfo(task_index=1, num_tasks=2, advertise_host="10.0.0.3"),
        timeout=1.0,
        poll_interval=0.01,
    )

    assert registry.registered == [(f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}1", "10.0.0.3")]
