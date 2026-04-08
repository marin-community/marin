# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import pytest
from fray.v2 import LocalClient, set_current_client

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
