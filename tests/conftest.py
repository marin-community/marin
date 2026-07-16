# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys
import tempfile
from functools import cache

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient

DEFAULT_BUCKET_NAME = "marin-us-east5"
DEFAULT_DOCUMENT_PATH = "documents/test-document-path"

# A ``tpu_ci`` test needs a real TPU. Rather than gate it out of the default suite with a
# ``-m 'not tpu_ci'`` filter -- which also deselects it when you point pytest straight at the
# file on a TPU host -- the test self-skips when no TPU is reachable, mirroring the
# detect-the-resource-and-skip idiom the cluster smokes use for cluster credentials.
#
# The probe runs jax in a throwaway subprocess rather than calling ``jax.devices()`` in-process:
# initializing the accelerator here would hold ``/dev/vfio`` for the whole pytest session, which
# breaks tpu_ci tests that own the TPU through their own subprocesses (e.g. the GrugMoE e2e, whose
# preflight refuses to start when another process holds the device). The subprocess acquires and
# releases the TPU before any test fixture runs.
_TPU_PROBE_SCRIPT = (
    "import jax\n"
    "try:\n"
    "    has_tpu = bool(jax.devices('tpu'))\n"
    "except RuntimeError:\n"
    "    has_tpu = False\n"
    "raise SystemExit(0 if has_tpu else 1)\n"
)


@cache
def tpu_is_available() -> bool:
    """Whether a TPU backend is reachable, probed in an isolated subprocess."""
    probe = subprocess.run([sys.executable, "-c", _TPU_PROBE_SCRIPT], capture_output=True)
    return probe.returncode == 0


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Self-skip ``tpu_ci`` tests when no TPU is present, so the marker needs no ``-m`` gate."""
    if item.get_closest_marker("tpu_ci") is not None and not tpu_is_available():
        pytest.skip("requires a TPU; none is available")


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
