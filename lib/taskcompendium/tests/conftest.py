# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bridge() -> str:
    crate = Path(__file__).parents[1] / "shellsim-bridge"
    subprocess.run(["cargo", "build", "--locked", "--manifest-path", str(crate / "Cargo.toml")], check=True)
    return str(crate / "target" / "debug" / "taskcompendium-shellsim")


@pytest.fixture(scope="session")
def runtime_image():
    dockerfile = Path(__file__).parents[1] / "src/taskcompendium/harbor/runtime.Dockerfile"
    subprocess.run(
        [
            "docker",
            "build",
            "-q",
            "-t",
            "taskcompendium-runtime:validation",
            "-f",
            str(dockerfile),
            str(dockerfile.parent),
        ],
        check=True,
    )
    return subprocess.check_output(
        ["docker", "image", "inspect", "taskcompendium-runtime:validation", "--format", "{{.Id}}"], text=True
    ).strip()
