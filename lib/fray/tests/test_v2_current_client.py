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


def test_auto_detect_iris_environment(monkeypatch):
    """Test that IRIS_CONTROLLER_ADDRESS triggers auto-detection."""
    from unittest.mock import patch

    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://localhost:10000")
    with patch("fray.v2.iris_backend.IrisClientLib"):
        from fray.v2.iris_backend import FrayIrisClient

        client = current_client()
        assert isinstance(client, FrayIrisClient)


def test_auto_detect_iris_with_bundle(monkeypatch):
    """Test that IRIS_BUNDLE_GCS_PATH is passed through auto-detection."""
    from unittest.mock import patch

    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://localhost:10000")
    monkeypatch.setenv("IRIS_BUNDLE_GCS_PATH", "gs://bucket/bundle.zip")
    with patch("fray.v2.iris_backend.IrisClientLib") as mock_iris:
        from fray.v2.iris_backend import FrayIrisClient

        client = current_client()
        assert isinstance(client, FrayIrisClient)
        mock_iris.remote.assert_called_once_with(
            "http://localhost:10000", workspace=None, bundle_gcs_path="gs://bucket/bundle.zip"
        )


def test_auto_detect_ray_environment(monkeypatch):
    """Test that ray.is_initialized() triggers auto-detection."""
    from unittest.mock import patch

    with patch("fray.v2.client.ray") as mock_ray:
        mock_ray.is_initialized.return_value = True
        from fray.v2.ray.backend import RayClient

        client = current_client()
        assert isinstance(client, RayClient)


def test_fray_client_spec_overrides_auto_detection(monkeypatch):
    """Test that FRAY_CLIENT_SPEC takes precedence over auto-detection."""
    # Set both Iris env vars and FRAY_CLIENT_SPEC
    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://localhost:10000")
    monkeypatch.setenv("FRAY_CLIENT_SPEC", "local")

    client = current_client()
    # Should get LocalClient from FRAY_CLIENT_SPEC, not FrayIrisClient from auto-detect
    assert isinstance(client, LocalClient)


def test_explicit_client_overrides_auto_detection(monkeypatch):
    """Test that set_current_client() takes precedence over auto-detection."""
    from unittest.mock import patch

    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://localhost:10000")
    explicit = LocalClient(max_threads=1)

    with patch("fray.v2.iris_backend.IrisClientLib"):
        with set_current_client(explicit):
            # Should get explicit client, not auto-detected FrayIrisClient
            assert current_client() is explicit
