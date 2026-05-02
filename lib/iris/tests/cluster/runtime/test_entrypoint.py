# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for entrypoint construction and bash script generation."""

import pytest
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint, runtime_entrypoint_to_bash_script
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def _make_env_config(
    extras: list[str] | None = None,
    pip_packages: list[str] | None = None,
    python_version: str = "",
) -> job_pb2.EnvironmentConfig:
    cfg = job_pb2.EnvironmentConfig()
    if extras:
        cfg.extras[:] = extras
    if pip_packages:
        cfg.pip_packages[:] = pip_packages
    if python_version:
        cfg.python_version = python_version
    return cfg


def _make_entrypoint(command: list[str]) -> Entrypoint:
    return Entrypoint(command=command, workdir_files={})


def test_build_runtime_entrypoint_includes_extras():
    ep = _make_entrypoint(["python", "train.py"])
    env = _make_env_config(extras=["gpu", "mypackage:data"])
    rt = build_runtime_entrypoint(ep, env)

    setup = "\n".join(rt.setup_commands)
    assert "--extra gpu" in setup
    assert "--extra data" in setup
    assert "--all-packages" in setup


def test_build_runtime_entrypoint_includes_pip_packages():
    ep = _make_entrypoint(["python", "train.py"])
    env = _make_env_config(pip_packages=["torch>=2.0", "numpy"])
    rt = build_runtime_entrypoint(ep, env)

    setup = "\n".join(rt.setup_commands)
    assert "torch" in setup
    assert "numpy" in setup
    # cloudpickle, py-spy, memray are always included
    assert "cloudpickle" in setup
    assert "py-spy" in setup
    assert "memray" in setup


def test_build_runtime_entrypoint_with_python_version():
    ep = _make_entrypoint(["python", "app.py"])
    env = _make_env_config(python_version="3.11")
    rt = build_runtime_entrypoint(ep, env)

    setup = "\n".join(rt.setup_commands)
    assert "--python 3.11" in setup


def test_build_runtime_entrypoint_no_python_version():
    ep = _make_entrypoint(["python", "app.py"])
    env = _make_env_config()
    rt = build_runtime_entrypoint(ep, env)

    setup = "\n".join(rt.setup_commands)
    assert "--python" not in setup


def test_runtime_entrypoint_to_bash_script_structure():
    rt = job_pb2.RuntimeEntrypoint()
    rt.setup_commands[:] = ["cd /app", "echo hello"]
    rt.run_command.argv[:] = ["python", "train.py", "--lr", "0.001"]

    script = runtime_entrypoint_to_bash_script(rt)

    assert script.startswith("#!/bin/bash\n")
    assert "set -e" in script
    assert "cd /app" in script
    assert "echo hello" in script
    assert "exec python train.py --lr 0.001" in script


@pytest.mark.parametrize(
    "extras,expected_fragments",
    [
        ([], []),
        (["train"], ["--extra train"]),
        (["pkg:feat", "other"], ["--extra feat", "--extra other"]),
    ],
)
def test_build_runtime_entrypoint_extras_parametrized(extras, expected_fragments):
    ep = _make_entrypoint(["python", "run.py"])
    env = _make_env_config(extras=extras)
    rt = build_runtime_entrypoint(ep, env)

    setup = "\n".join(rt.setup_commands)
    for fragment in expected_fragments:
        assert fragment in setup
