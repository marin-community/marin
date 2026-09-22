# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for how EnvironmentSpec resolves the user setup scripts onto the wire."""

import os
import subprocess

import pytest
from iris.cluster.runtime.env import render_setup_steps
from iris.cluster.setup_scripts import default_setup_script
from iris.cluster.types import EnvironmentSpec


@pytest.mark.parametrize(
    "setup_scripts, expected",
    [
        # Default: iris builds one project-setup script. The iris runtime-deps
        # script is appended later, in build_runtime_entrypoint — not here.
        (None, None),
        # Custom scripts pass through verbatim, in order.
        (["echo a", "echo b"], ["echo a", "echo b"]),
        # Whitespace-only entries are dropped.
        (["echo a", "   "], ["echo a"]),
        # No setup at all.
        ([], []),
    ],
)
def test_to_proto_resolves_user_setup_scripts(setup_scripts, expected):
    resolved = list(EnvironmentSpec(setup_scripts=setup_scripts).to_proto().setup_scripts)

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


@pytest.mark.parametrize("shared_cache_fails", [False, True])
def test_setup_steps_switch_uv_installs_to_local_cache_after_shared_failure(tmp_path, shared_cache_fails):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "pyproject.toml").write_text("[tool.uv]\npackage = false\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        """\
#!/bin/sh
set -e
if [ "$UV_CACHE_DIR" = "$SHARED_UV_CACHE" ] && [ "$SHARED_CACHE_FAILS" = "1" ] && [ "$1 $2" = "pip install" ]; then
  exit 1
fi
mkdir -p "$UV_CACHE_DIR/wheels" "$IRIS_VENV"
touch "$UV_CACHE_DIR/wheels/package.whl"
ln -sf "$UV_CACHE_DIR/wheels/package.whl" "$IRIS_VENV/package.whl"
"""
    )
    uv.chmod(0o755)
    shared_cache = tmp_path / "shared-cache"
    venv = tmp_path / "venv"
    env = {
        **os.environ,
        "IRIS_VENV": str(venv),
        "IRIS_WORKDIR": str(workdir),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SHARED_CACHE_FAILS": str(int(shared_cache_fails)),
        "SHARED_UV_CACHE": str(shared_cache),
        "UV_CACHE_DIR": str(shared_cache),
        "UV_PROJECT_ENVIRONMENT": str(venv),
    }

    setup = "\n".join(
        [
            "set -e",
            *render_setup_steps(["uv pip install package", default_setup_script(python_version="3.12")]),
        ]
    )
    subprocess.run(
        ["bash", "-c", setup],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    expected_cache = workdir / ".uv-recovery-cache" if shared_cache_fails else shared_cache
    assert (venv / "package.whl").resolve() == expected_cache / "wheels/package.whl"
