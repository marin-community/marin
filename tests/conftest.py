# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from pathlib import Path

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"

# A Cloud TPU exposes its chips through /dev/vfio (the signal the GrugMoE e2e checks too).
_TPU_DEVICE_DIR = Path("/dev/vfio")


def tpu_is_available() -> bool:
    """Probe for TPU existence."""
    return _TPU_DEVICE_DIR.is_dir() and any(_TPU_DEVICE_DIR.iterdir())


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tpu_ci tests when no TPU is present."""
    if item.get_closest_marker("tpu_ci") is not None and not tpu_is_available():
        pytest.skip("no TPU available")


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
