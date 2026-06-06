# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.grug import dispatch


@dataclass
class _Submitted:
    request: object | None = None


class _FakeJob:
    def __init__(self) -> None:
        self.wait_called = False

    def wait(self, *, raise_on_failure: bool) -> None:
        self.wait_called = raise_on_failure


class _FakeClient:
    def __init__(self, submitted: _Submitted, job: _FakeJob) -> None:
        self._submitted = submitted
        self._job = job

    def submit(self, request):
        self._submitted.request = request
        return self._job


def _entrypoint(config) -> None:
    del config


def test_grug_dispatch_uses_training_env(monkeypatch):
    submitted = _Submitted()
    job = _FakeJob()
    resources = ResourceConfig.with_tpu("v6e-4")
    expected_env = {
        "JAX_COMPILATION_CACHE_DIR": "gs://marin-eu-west4/tmp/ttl=30d/compilation-cache/test",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
    }

    monkeypatch.setattr(dispatch, "current_client", lambda: _FakeClient(submitted, job))
    monkeypatch.setattr(dispatch, "resolve_training_env", lambda *, base_env, resources: expected_env)

    dispatch.dispatch_grug_training_run(
        run_id="grug/test",
        config={"x": 1},
        local_entrypoint=_entrypoint,
        resources=resources,
    )

    assert submitted.request is not None
    for key, value in expected_env.items():
        assert submitted.request.environment.env_vars[key] == value
    assert submitted.request.environment.extras == ["tpu"]
    assert job.wait_called
