# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile

import pytest
from fray.v1.cluster import create_cluster, set_current_cluster
from fray.v1.job.context import _job_context

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"


@pytest.fixture(autouse=True)
def reset_fray_context():
    """Reset fray context between tests for isolation."""
    _job_context.set(None)
    yield
    _job_context.set(None)


@pytest.fixture(autouse=True)
def fray_cluster():
    set_current_cluster(create_cluster("local"))
    yield
    set_current_cluster(None)


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Disable WANDB logging during tests."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" not in os.environ:
        with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
            os.environ["MARIN_PREFIX"] = temp_dir
            yield
            del os.environ["MARIN_PREFIX"]
