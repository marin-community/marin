# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EnvironmentSpec resolves the user setup scripts onto the wire."""

import os
import subprocess

import pytest
from iris.cluster.setup_scripts import SetupPlan, default_setup_script
from iris.cluster.types import EnvironmentSpec


@pytest.mark.parametrize(
    "setup, expected",
    [
        # Default: iris builds one project-setup script. The iris runtime-deps
        # script is appended later, in build_runtime_entrypoint — not here.
        (None, None),
        # Custom scripts pass through verbatim, in order.
        (SetupPlan.custom(["echo a", "echo b"]), ["echo a", "echo b"]),
        # Whitespace-only entries are dropped.
        (SetupPlan.custom(["echo a", "   "]), ["echo a"]),
        # No setup at all.
        (SetupPlan.empty(), []),
    ],
)
def test_to_proto_resolves_user_setup_scripts(setup, expected):
    resolved = [layer.setup_script for layer in EnvironmentSpec(setup=setup).to_proto().setup_layers]

    if expected is None:
        assert len(resolved) == 1  # the generated default
    else:
        assert resolved == expected


def test_default_setup_supports_project_without_dependency_groups(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "pyproject.toml").write_text(
        """\
[project]
name = "setup-test"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.uv]
package = false
"""
    )
    venv = tmp_path / "venv"
    env = {
        **os.environ,
        "IRIS_VENV": str(venv),
        "IRIS_WORKDIR": str(workdir),
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_PROJECT_ENVIRONMENT": str(venv),
    }

    subprocess.run(
        ["bash", "-c", default_setup_script(python_version="3.12")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert (venv / "bin" / "python").is_file()
