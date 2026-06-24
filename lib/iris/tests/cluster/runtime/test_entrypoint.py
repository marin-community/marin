# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeEntrypoint assembly from an Entrypoint + EnvironmentConfig."""

from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def _make_entrypoint(command: list[str]) -> Entrypoint:
    return Entrypoint(command=command, workdir_files={})


def test_default_env_carries_setup_script_in_setup_commands():
    ep = _make_entrypoint(["python", "train.py"])
    env = job_pb2.EnvironmentConfig(extras=["gpu"], pip_packages=["torch>=2.0"])
    rt = build_runtime_entrypoint(ep, env)

    # The resolved default script lands as a single setup_commands element.
    assert len(rt.setup_commands) == 1
    setup = rt.setup_commands[0]
    assert "uv sync" in setup
    assert "--all-packages" in setup
    assert "--extra gpu" in setup
    assert "torch>=2.0" in setup
    assert list(rt.run_command.argv) == ["python", "train.py"]


def test_custom_setup_script_passes_through():
    ep = _make_entrypoint(["python", "train.py"])
    env = job_pb2.EnvironmentConfig(
        setup_mode=job_pb2.SETUP_MODE_CUSTOM,
        setup_script="echo prep\n",
        extras=["gpu"],  # ignored in custom mode
    )
    rt = build_runtime_entrypoint(ep, env)

    assert list(rt.setup_commands) == ["echo prep\n"]
    assert "uv sync" not in rt.setup_commands[0]


def test_no_setup_leaves_setup_commands_empty():
    ep = _make_entrypoint(["python", "train.py"])
    env = job_pb2.EnvironmentConfig(setup_mode=job_pb2.SETUP_MODE_CUSTOM, setup_script="")
    rt = build_runtime_entrypoint(ep, env)

    # Empty script => no build phase; the command runs in the image as-is.
    assert list(rt.setup_commands) == []
    assert list(rt.run_command.argv) == ["python", "train.py"]


def test_workdir_files_and_refs_propagate():
    refs = {"_callable.pkl": "sha256abc", "weights.bin": "sha256def"}
    ep = Entrypoint(command=["python", "run.py"], workdir_files={"small.txt": b"hi"}, workdir_file_refs=refs)
    rt = build_runtime_entrypoint(ep, job_pb2.EnvironmentConfig())

    assert dict(rt.workdir_files) == {"small.txt": b"hi"}
    assert dict(rt.workdir_file_refs) == refs
