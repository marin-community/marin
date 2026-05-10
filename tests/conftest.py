# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import fsspec.core
import pytest
from fray import LocalClient, set_current_client

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"

# URL schemes treated as "remote cloud storage" — tests must not silently reach these.
# (http/https are intentionally excluded; mock-server-based tests use them.)
_REMOTE_PROTOCOLS = ("gs://", "gcs://", "s3://", "hf://")


class _StubRemoteFS:
    """fsspec stub returned for remote URLs in tests.

    `.exists()` reports False so callers that probe for cache hits (e.g.
    ``StatusFile``) see "no remote state" and proceed locally. Any other
    operation raises so accidental real I/O is loud, not slow. Tests that
    legitimately need remote access should be marked with one of the
    integration markers and run in a lane where this stub is disabled.
    """

    protocol = "memory"

    def exists(self, *_args, **_kwargs):
        return False

    def isdir(self, *_args, **_kwargs):
        return False

    def isfile(self, *_args, **_kwargs):
        return False

    def __getattr__(self, name):
        def _raise(*_a, **_kw):
            raise RuntimeError(
                f"Test attempted real remote I/O via {name}(). "
                "Mark the test as data_integration/integration, or stub the call explicitly."
            )

        return _raise


@pytest.fixture(autouse=True)
def _stub_remote_filesystem(monkeypatch):
    """Block remote fsspec access in tests by stubbing ``fsspec.core.url_to_fs``.

    Patching at the fsspec layer covers every higher-level wrapper (including
    ``rigging.filesystem.url_to_fs`` and the many call sites that did
    ``from rigging.filesystem import url_to_fs``).
    """
    real_url_to_fs = fsspec.core.url_to_fs

    def stubbed_url_to_fs(url, **kwargs):
        if isinstance(url, str) and url.startswith(_REMOTE_PROTOCOLS):
            return _StubRemoteFS(), url
        return real_url_to_fs(url, **kwargs)

    monkeypatch.setattr(fsspec.core, "url_to_fs", stubbed_url_to_fs)


@pytest.fixture(autouse=True)
def fray_client():
    """Set up a v2 LocalClient for all tests."""
    with set_current_client(LocalClient()) as client:
        yield client


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Disable WANDB logging during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" in os.environ:
        yield
        return

    with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
        os.environ["MARIN_PREFIX"] = temp_dir
        yield
        del os.environ["MARIN_PREFIX"]
