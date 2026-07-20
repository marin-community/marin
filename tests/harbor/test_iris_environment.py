# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IrisEnvironment resource cleanup on a failed start.

Regression coverage for a leaked controller endpoint (SSH tunnel / local cluster):
_start_sync used to wrap only the final wait-for-running step in its cleanup
try/except, so a failure constructing the IrisClient, RPC client, or submitting
the job left the endpoint opened by connect_controller() never closed.
"""

from pathlib import Path

import pytest
from iris.rpc.controller_connect import ControllerServiceClientSync

try:
    from harbor.models.task.config import EnvironmentConfig
    from harbor.models.trial.paths import TrialPaths
    from marin.harbor import iris_environment
    from marin.harbor.iris_environment import IrisEnvironment
    from upath import UPath
except ImportError:
    pytest.skip("harbor is not installed (marin[harbor] extra)", allow_module_level=True)


class _FakeEndpoint:
    """Stands in for ControllerEndpoint: records whether close() ran."""

    def __init__(self):
        self.url = "http://localhost:10000"
        self.credentials = None
        self.closed = False

    def close(self):
        self.closed = True


def _make_environment(tmp_path: Path) -> IrisEnvironment:
    return IrisEnvironment(
        environment_dir=tmp_path,
        environment_name="test",
        session_id="session-1",
        trial_paths=TrialPaths(trial_dir=UPath(tmp_path)),
        task_env_config=EnvironmentConfig(docker_image="ubuntu:24.04"),
        cluster="marin",
    )


def test_start_sync_closes_endpoint_when_client_construction_fails(tmp_path, monkeypatch):
    endpoint = _FakeEndpoint()
    monkeypatch.setattr(iris_environment, "connect_controller", lambda **_kwargs: endpoint)

    class _FailingIrisClient:
        @staticmethod
        def remote(*_args, **_kwargs):
            raise RuntimeError("controller unreachable")

    monkeypatch.setattr(iris_environment, "IrisClient", _FailingIrisClient)

    env = _make_environment(tmp_path)
    with pytest.raises(RuntimeError, match="controller unreachable"):
        env._start_sync()

    assert endpoint.closed


def test_start_sync_closes_endpoint_when_submit_fails(tmp_path, monkeypatch):
    endpoint = _FakeEndpoint()
    monkeypatch.setattr(iris_environment, "connect_controller", lambda **_kwargs: endpoint)

    class _FakeIrisClient:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("submit rejected")

        def shutdown(self):
            pass

    monkeypatch.setattr(iris_environment.IrisClient, "remote", staticmethod(lambda *a, **k: _FakeIrisClient()))
    monkeypatch.setattr(ControllerServiceClientSync, "__init__", lambda self, *a, **k: None)

    env = _make_environment(tmp_path)
    with pytest.raises(RuntimeError, match="submit rejected"):
        env._start_sync()

    assert endpoint.closed
