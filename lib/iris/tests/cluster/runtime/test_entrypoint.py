# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RuntimeEntrypoint assembly from an Entrypoint + EnvironmentConfig."""

import pytest
from iris.cluster.runtime.entrypoint import build_runtime_entrypoint
from iris.cluster.setup_scripts import iris_runtime_setup_script
from iris.cluster.types import Entrypoint
from iris.rpc import job_pb2


@pytest.mark.parametrize(
    "setup_scripts, expected_user_commands",
    [
        (["a", "b"], ["a", "b"]),  # user scripts pass through in order
        (["a", "  ", ""], ["a"]),  # whitespace-only scripts are dropped
        ([], []),  # no setup => no build phase at all (not even the iris script)
    ],
)
def test_setup_scripts_become_setup_commands(setup_scripts, expected_user_commands):
    ep = Entrypoint(command=["python", "train.py"], workdir_files={})
    rt = build_runtime_entrypoint(ep, job_pb2.EnvironmentConfig(setup_scripts=setup_scripts))

    # iris appends its own runtime-deps script as the final step whenever any user
    # setup runs, so its features work regardless of what the user setup does.
    expected = [*expected_user_commands, iris_runtime_setup_script()] if expected_user_commands else []
    assert list(rt.setup_commands) == expected
    assert list(rt.run_command.argv) == ["python", "train.py"]


def test_workdir_files_and_refs_propagate():
    refs = {"_callable.pkl": "sha256abc", "weights.bin": "sha256def"}
    ep = Entrypoint(command=["python", "run.py"], workdir_files={"small.txt": b"hi"}, workdir_file_refs=refs)
    rt = build_runtime_entrypoint(ep, job_pb2.EnvironmentConfig(setup_scripts=["uv sync\n"]))

    assert dict(rt.workdir_files) == {"small.txt": b"hi"}
    assert dict(rt.workdir_file_refs) == refs
