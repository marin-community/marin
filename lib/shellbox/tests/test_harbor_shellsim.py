# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the ShellSim adapter through a complete local Harbor trial."""

import asyncio
import importlib
from pathlib import Path

import pytest


def test_shellsim_harbor_trial(tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    pytest.importorskip("shellsim")
    smoke = importlib.import_module("harbor_smoke")

    asyncio.run(
        smoke.main(
            tmp_path / "jobs",
            Path(__file__).parent / "manual/task",
            ["echo 'hello from qemu' > /workspace/answer.txt"],
            "shellbox.backends.shellsim.environment:ShellSimEnvironment",
            {"network_policy": "deny"},
        )
    )
