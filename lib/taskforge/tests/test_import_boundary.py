# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Importing Taskforge needs only the distributions its package declares, whatever else the environment holds."""

import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).parent / "import_boundary_probe.py"


def test_every_taskforge_module_imports_with_only_its_declared_dependencies() -> None:
    """Review 2026-11-21 (n11 follow-up 2, P1): the solve-review imported TaskCompendium's review pipeline, which loads
    Zephyr from an optional extra Taskforge does not install, so a clean install could not import the driver. The
    shared development environment, which has Zephyr, hid it; the probe refuses every undeclared distribution."""
    done = subprocess.run([sys.executable, str(PROBE)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-3000:]
