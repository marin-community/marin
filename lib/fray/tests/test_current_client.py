# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for current_client() and set_current_client()."""

from unittest.mock import MagicMock, patch

import pytest
from fray.client import current_client, set_current_client
from fray.local_backend import LocalClient


def test_default_returns_local_client():
    """When no context is set, should return LocalClient."""
    with patch("iris.client.client.get_iris_ctx", return_value=None):
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
    mock_iris_client_lib = MagicMock()
    mock_ctx.client = mock_iris_client_lib

    with patch("iris.client.client.get_iris_ctx", return_value=mock_ctx):
        with patch("fray.iris_backend.FrayIrisClient") as mock_client_cls:
            mock_fray_client = MagicMock()
            mock_client_cls.from_iris_client.return_value = mock_fray_client

            client = current_client()
            assert client is mock_fray_client
            mock_client_cls.from_iris_client.assert_called_once_with(mock_iris_client_lib)


def test_iris_auto_detection_reuses_client():
    """Should reuse the existing Iris client from the context."""
    mock_ctx = MagicMock()
    mock_iris_client_lib = MagicMock()
    mock_ctx.client = mock_iris_client_lib

    with patch("iris.client.client.get_iris_ctx", return_value=mock_ctx):
        with patch("fray.iris_backend.FrayIrisClient") as mock_client_cls:
            mock_fray_client = MagicMock()
            mock_client_cls.from_iris_client.return_value = mock_fray_client

            client = current_client()
            assert client is mock_fray_client
            mock_client_cls.from_iris_client.assert_called_once_with(mock_iris_client_lib)


def test_iris_not_detected_when_no_context():
    """Should not detect Iris when get_iris_ctx() returns None."""
    with patch("iris.client.client.get_iris_ctx", return_value=None):
        client = current_client()
        assert isinstance(client, LocalClient)


def test_explicit_client_overrides_auto_detection():
    """Explicitly set client should override auto-detection."""
    mock_ctx = MagicMock()
    mock_iris_client_lib = MagicMock()
    mock_ctx.client = mock_iris_client_lib

    explicit = LocalClient(max_threads=1)
    with patch("iris.client.client.get_iris_ctx", return_value=mock_ctx):
        with set_current_client(explicit):
            # Should return explicit client, not auto-detected Iris client
            assert current_client() is explicit
