# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import pytest
from fray import LocalClient, set_current_client
from marin.execution.artifact_registry import FilesystemArtifactRegistry, use_default_registry

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


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


@pytest.fixture(autouse=True)
def _isolate_artifact_registry(tmp_path_factory):
    """Keep tests off the real ``gs://marin-us-central1`` artifact registry.

    ``get_default_registry`` defaults to the production root, so without this an unconfigured test
    would read/write live GCS. Install a temp-dir registry as the context-local default for the
    duration of each test; ``use_default_registry`` restores the previous default on exit so the
    production default is never reachable and nothing leaks between tests.
    """
    temp_root = str(tmp_path_factory.mktemp("artifact_registry"))
    with use_default_registry(FilesystemArtifactRegistry(temp_root)):
        yield
