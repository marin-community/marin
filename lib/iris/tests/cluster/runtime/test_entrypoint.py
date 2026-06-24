# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeEntrypoint assembly from an Entrypoint + EnvironmentConfig."""

from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


def _make_entrypoint(command: list[str]) -> Entrypoint:
    return Entrypoint(command=command, workdir_files={})


def test_resolved_setup_script_lands_in_setup_commands():
    ep = _make_entrypoint(["python", "train.py"])
    env = job_pb2.EnvironmentConfig(setup_script="uv sync --all-packages\n")
    rt = build_runtime_entrypoint(ep, env)

    assert list(rt.setup_commands) == ["uv sync --all-packages\n"]
    assert list(rt.run_command.argv) == ["python", "train.py"]


def test_empty_setup_script_leaves_setup_commands_empty():
    ep = _make_entrypoint(["python", "train.py"])
    env = job_pb2.EnvironmentConfig(setup_script="")
    rt = build_runtime_entrypoint(ep, env)

    # No setup => no build phase; the command runs in the image as-is.
    assert list(rt.setup_commands) == []
    assert list(rt.run_command.argv) == ["python", "train.py"]


def test_workdir_files_and_refs_propagate():
    refs = {"_callable.pkl": "sha256abc", "weights.bin": "sha256def"}
    ep = Entrypoint(command=["python", "run.py"], workdir_files={"small.txt": b"hi"}, workdir_file_refs=refs)
    rt = build_runtime_entrypoint(ep, job_pb2.EnvironmentConfig(setup_script="uv sync\n"))

    assert dict(rt.workdir_files) == {"small.txt": b"hi"}
    assert dict(rt.workdir_file_refs) == refs
