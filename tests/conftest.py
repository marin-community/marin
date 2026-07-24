# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


@pytest.fixture(autouse=True)
def fray_client(_configure_marin_prefix):
    """Set up a v2 LocalClient for all tests.

    Depends on ``_configure_marin_prefix`` so it tears down first: shutting
    down the client joins every local-backend worker thread before that
    fixture removes the MARIN_PREFIX temp directory. Without this ordering, a
    straggler task can still be writing under MARIN_PREFIX when the temp
    directory is removed, raising ``OSError: Directory not empty`` at
    teardown.
    """
    client = LocalClient()
    with set_current_client(client):
        yield client
    client.shutdown()


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
