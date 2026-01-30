# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for current_client() and set_current_client()."""

import pytest

from fray.v2.client import current_client, set_current_client
from fray.v2.local import LocalClient


def test_default_returns_local_client():
    client = current_client()
    assert isinstance(client, LocalClient)


def test_set_current_client_context_manager():
    explicit = LocalClient(max_threads=2)
    with set_current_client(explicit) as c:
        assert c is explicit
        assert current_client() is explicit
    # After exiting, should return a fresh default
    assert current_client() is not explicit


def test_set_current_client_restores_on_exception():
    explicit = LocalClient(max_threads=2)
    with pytest.raises(RuntimeError):
        with set_current_client(explicit):
            raise RuntimeError("boom")
    assert current_client() is not explicit


def test_env_var_local(monkeypatch):
    monkeypatch.setenv("FRAY_CLIENT_SPEC", "local")
    client = current_client()
    assert isinstance(client, LocalClient)


def test_env_var_local_with_threads(monkeypatch):
    monkeypatch.setenv("FRAY_CLIENT_SPEC", "local?threads=4")
    client = current_client()
    assert isinstance(client, LocalClient)
    assert client._executor._max_workers == 4


def test_env_var_ray_returns_ray_client(monkeypatch):
    from fray.v2.ray.backend import RayClient

    monkeypatch.setenv("FRAY_CLIENT_SPEC", "ray")
    client = current_client()
    assert isinstance(client, RayClient)


def test_env_var_iris_creates_client(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("FRAY_CLIENT_SPEC", "iris://host:1234")
    with patch("fray.v2.iris_backend.IrisClientLib"):
        from fray.v2.iris_backend import FrayIrisClient

        client = current_client()
        assert isinstance(client, FrayIrisClient)


def test_explicit_client_overrides_env_var(monkeypatch):
    monkeypatch.setenv("FRAY_CLIENT_SPEC", "ray")
    explicit = LocalClient(max_threads=1)
    with set_current_client(explicit):
        # Should not raise NotImplementedError because explicit takes priority
        assert current_client() is explicit


def test_unknown_spec_raises(monkeypatch):
    monkeypatch.setenv("FRAY_CLIENT_SPEC", "unknown_backend")
    with pytest.raises(ValueError, match="Unknown FRAY_CLIENT_SPEC"):
        current_client()
