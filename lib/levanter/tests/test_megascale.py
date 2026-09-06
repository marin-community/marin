# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

import pytest

from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from levanter import megascale


@dataclass(frozen=True)
class _FakeResolveResult:
    is_empty: bool
    url: str | None = None

    def first(self):
        assert self.url is not None
        return self


class _FakeRegistry:
    def __init__(self):
        self.registered: list[tuple[str, str]] = []

    def register(self, name: str, address: str) -> str:
        self.registered.append((name, address))
        return f"endpoint-{name}"

    def unregister(self, endpoint_id: str) -> None:
        pass


class _FakeResolver:
    def __init__(self, ready_names: set[str], coordinators: dict[str, str]):
        self.ready_names = ready_names
        self.coordinators = coordinators
        self.resolved_names: list[str] = []

    def resolve(self, name: str) -> _FakeResolveResult:
        self.resolved_names.append(name)
        if name in self.coordinators:
            return _FakeResolveResult(is_empty=False, url=self.coordinators[name])
        return _FakeResolveResult(is_empty=name not in self.ready_names)


@dataclass(frozen=True)
class _FakeIrisContext:
    registry: _FakeRegistry
    resolver: _FakeResolver


def _make_job_info(
    *,
    job_id: str = "/testuser/testroot/train-a",
    task_index: int = 5,
    num_tasks: int = 8,
    attempt_id: int = 3,
) -> JobInfo:
    return JobInfo(
        task_id=JobName.from_string(f"{job_id}/{task_index}"),
        num_tasks=num_tasks,
        attempt_id=attempt_id,
        advertise_host="10.0.0.2",
    )


def test_configure_megascale_maps_iris_slice_topology_to_megascale_env(monkeypatch):
    info = _make_job_info()
    job_token = info.job_id.to_safe_token()
    sibling_token = JobName.from_string("/testuser/testroot/train-b").to_safe_token()
    scoped_ready_names = {
        f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}{task_index}-{job_token}-attempt-3" for task_index in range(8)
    }
    bare_ready_names = {f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}{task_index}" for task_index in range(8)}
    stale_ready_names = {
        f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}{task_index}-{job_token}-attempt-2" for task_index in range(8)
    }
    sibling_ready_names = {
        f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}{task_index}-{sibling_token}-attempt-3" for task_index in range(8)
    }
    scoped_coordinator = f"{megascale.MEGASCALE_COORDINATOR_ENDPOINT}-{job_token}-attempt-3"
    stale_coordinator = f"{megascale.MEGASCALE_COORDINATOR_ENDPOINT}-{job_token}-attempt-2"
    sibling_coordinator = f"{megascale.MEGASCALE_COORDINATOR_ENDPOINT}-{sibling_token}-attempt-3"
    registry = _FakeRegistry()
    resolver = _FakeResolver(
        ready_names=scoped_ready_names | bare_ready_names | stale_ready_names | sibling_ready_names,
        coordinators={
            scoped_coordinator: "10.0.0.1:8081",
            stale_coordinator: "10.0.0.8:8081",
            sibling_coordinator: "10.0.0.7:8081",
            megascale.MEGASCALE_COORDINATOR_ENDPOINT: "10.0.0.9:8081",
        },
    )

    for name in (
        megascale.MEGASCALE_COORDINATOR_ADDRESS,
        megascale.MEGASCALE_NUM_SLICES,
        megascale.MEGASCALE_PORT,
        megascale.MEGASCALE_SLICE_ID,
    ):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv(megascale.IRIS_SLICE_COUNT, "2")
    monkeypatch.setenv(megascale.IRIS_TASKS_PER_SLICE, "4")
    monkeypatch.setattr(megascale, "get_job_info", lambda: info)
    monkeypatch.setattr(megascale, "iris_ctx", lambda: _FakeIrisContext(registry=registry, resolver=resolver))

    env = megascale.configure_megascale_from_iris()

    assert env == {
        "MEGASCALE_COORDINATOR_ADDRESS": "10.0.0.1:8081",
        "MEGASCALE_NUM_SLICES": "2",
        "MEGASCALE_PORT": "8081",
        "MEGASCALE_SLICE_ID": "1",
    }
    assert all(os.environ[key] == value for key, value in env.items())
    assert registry.registered == [(f"{megascale.MEGASCALE_READY_ENDPOINT_PREFIX}5-{job_token}-attempt-3", "10.0.0.2")]
    assert set(resolver.resolved_names[:-1]) == scoped_ready_names
    assert len(resolver.resolved_names[:-1]) == len(scoped_ready_names)
    assert resolver.resolved_names[-1] == scoped_coordinator


def test_megascale_env_rejects_wrong_task_count(monkeypatch):
    monkeypatch.setattr(megascale, "get_job_info", lambda: _make_job_info(task_index=0, num_tasks=3))

    with pytest.raises(ValueError, match="Megascale expects 4"):
        megascale.megascale_env_for_iris_task(slice_count=2, tasks_per_slice=2)
