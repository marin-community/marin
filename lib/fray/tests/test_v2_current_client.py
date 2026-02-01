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

from unittest.mock import MagicMock, patch

import pytest

from fray.v2.client import current_client, set_current_client
from fray.v2.local import LocalClient


def test_default_returns_local_client():
    """When no context is set, should return LocalClient."""
    client = current_client()
    assert isinstance(client, LocalClient)


def test_set_current_client_context_manager():
    """Explicitly set client should take priority over auto-detection."""
    explicit = LocalClient(max_threads=2)
    with set_current_client(explicit) as c:
        assert c is explicit
        assert current_client() is explicit
    # After exiting, should return a fresh default (or auto-detect)
    assert current_client() is not explicit


def test_set_current_client_restores_on_exception():
    """Context manager should restore previous client even on exception."""
    explicit = LocalClient(max_threads=2)
    with pytest.raises(RuntimeError):
        with set_current_client(explicit):
            raise RuntimeError("boom")
    assert current_client() is not explicit


def test_iris_auto_detection_with_context():
    """Should auto-detect Iris when get_iris_ctx() returns a context."""
    mock_ctx = MagicMock()
    mock_ctx.client._controller_address = "http://controller:8080"

    with patch("fray.v2.client.get_iris_ctx", return_value=mock_ctx):
        with patch("fray.v2.client.FrayIrisClient") as mock_client_cls:
            mock_iris_client = MagicMock()
            mock_client_cls.return_value = mock_iris_client

            client = current_client()
            assert client is mock_iris_client
            mock_client_cls.assert_called_once_with("http://controller:8080", bundle_gcs_path=None)


def test_iris_auto_detection_with_bundle_path(monkeypatch):
    """Should pass bundle_gcs_path when IRIS_BUNDLE_GCS_PATH is set."""
    monkeypatch.setenv("IRIS_BUNDLE_GCS_PATH", "gs://bucket/bundle.tar.gz")

    mock_ctx = MagicMock()
    mock_ctx.client._controller_address = "http://controller:8080"

    with patch("fray.v2.client.get_iris_ctx", return_value=mock_ctx):
        with patch("fray.v2.client.FrayIrisClient") as mock_client_cls:
            mock_iris_client = MagicMock()
            mock_client_cls.return_value = mock_iris_client

            client = current_client()
            assert client is mock_iris_client
            mock_client_cls.assert_called_once_with(
                "http://controller:8080", bundle_gcs_path="gs://bucket/bundle.tar.gz"
            )


def test_iris_not_detected_when_no_context():
    """Should not detect Iris when get_iris_ctx() returns None."""
    with patch("fray.v2.client.get_iris_ctx", return_value=None):
        client = current_client()
        assert isinstance(client, LocalClient)


def test_ray_auto_detection():
    """Should auto-detect Ray when ray.is_initialized() is True."""
    with patch("fray.v2.client.get_iris_ctx", return_value=None):
        with patch("fray.v2.client.ray") as mock_ray:
            mock_ray.is_initialized.return_value = True
            with patch("fray.v2.client.RayClient") as mock_client_cls:
                mock_ray_client = MagicMock()
                mock_client_cls.return_value = mock_ray_client

                client = current_client()
                assert client is mock_ray_client
                mock_client_cls.assert_called_once_with()


def test_ray_not_detected_when_not_initialized():
    """Should not detect Ray when ray.is_initialized() is False."""
    with patch("fray.v2.client.get_iris_ctx", return_value=None):
        with patch("fray.v2.client.ray") as mock_ray:
            mock_ray.is_initialized.return_value = False
            client = current_client()
            assert isinstance(client, LocalClient)


def test_explicit_client_overrides_auto_detection():
    """Explicitly set client should override auto-detection."""
    mock_ctx = MagicMock()
    mock_ctx.client._controller_address = "http://controller:8080"

    explicit = LocalClient(max_threads=1)
    with patch("fray.v2.client.get_iris_ctx", return_value=mock_ctx):
        with set_current_client(explicit):
            # Should return explicit client, not auto-detected Iris client
            assert current_client() is explicit


def test_iris_takes_priority_over_ray():
    """Iris auto-detection should take priority over Ray."""
    mock_ctx = MagicMock()
    mock_ctx.client._controller_address = "http://controller:8080"

    with patch("fray.v2.client.get_iris_ctx", return_value=mock_ctx):
        with patch("fray.v2.client.ray") as mock_ray:
            mock_ray.is_initialized.return_value = True
            with patch("fray.v2.client.FrayIrisClient") as mock_iris_cls:
                mock_iris_client = MagicMock()
                mock_iris_cls.return_value = mock_iris_client

                client = current_client()
                # Should get Iris client, not Ray client
                assert client is mock_iris_client
