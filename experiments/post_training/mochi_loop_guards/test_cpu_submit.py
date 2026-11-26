# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one generic in-region CPU submit path."""

import subprocess

import pytest
from marin.rl.skyrl import skyrl_cpu_command

from experiments.post_training.mochi_loop_guards import cpu_submit
from experiments.post_training.mochi_loop_guards.cpu_submit import submit_command

PROGRAM = ["python", "-m", "experiments.post_training.mochi_loop_guards", "audit", "--run-output", "s3://run"]


def envelope(arguments, **overrides):
    values = {
        "job_name": "mochi-loop-guards-audit",
        "cpu": 16,
        "memory": "64GB",
        "disk": "128GB",
        "arguments": arguments,
    }
    values.update(overrides)
    return submit_command(**values)


def test_submit_command_wraps_the_program_in_the_standard_cpu_envelope():
    command = envelope(PROGRAM)
    assert command[:8] == ["uv", "run", "--no-sync", "iris", "--cluster", "cw-us-east-08a", "job", "run"]
    assert command[-len(PROGRAM) :] == PROGRAM
    for flag, value in (
        ("--priority", "production"),
        ("--cpu", "16"),
        ("--memory", "64GB"),
        ("--disk", "128GB"),
        ("--timeout", "3600"),
        ("--sync-package", "marin-core"),
        ("--job-name", "mochi-loop-guards-audit"),
    ):
        assert command[command.index(flag) + 1] == value, (flag, command)
    for flag in ("--no-wait", "--enable-extra-resources"):
        assert flag in command
    separator = command.index("--")
    assert command[separator + 1 : separator + 5] == ["uv", "run", "--package", "marin-core"]


def test_submit_command_translates_env_and_secret_flags():
    command = envelope(PROGRAM, env=["MARIN_PREFIX=s3://marin-us-east-08a/marin"], secrets=["DAYTONA_RL_API_KEY"])
    assert ["-e", "MARIN_PREFIX", "s3://marin-us-east-08a/marin"] == command[
        command.index("-e") : command.index("-e") + 3
    ]
    assert command[command.index("--secret") + 1 : command.index("--secret") + 3] == [
        "DAYTONA_RL_API_KEY",
        "env:DAYTONA_RL_API_KEY",
    ]


def test_cpu_submit_cli_forwards_secret_flags_and_submission_exit_code(monkeypatch):
    submitted = []

    def run(command):
        submitted.extend(command)
        return subprocess.CompletedProcess(command, 19)

    monkeypatch.setattr(cpu_submit.subprocess, "run", run)
    with pytest.raises(SystemExit) as outcome:
        cpu_submit.main(
            [
                "--job-name",
                "mochi-loop-guards-conserve",
                "--cpu",
                "16",
                "--memory",
                "64GB",
                "--disk",
                "128GB",
                "--secret",
                "CW_KEY_ID",
                "--secret",
                "CW_KEY_SECRET",
                "--",
                "fsutil",
                "verified-copy",
                "s3://source/checkpoint",
                "s3://permanent/checkpoint",
            ]
        )
    assert outcome.value.code == 19
    first_secret = submitted.index("--secret")
    assert submitted[first_secret : first_secret + 6] == [
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
    ]
    assert submitted[-4:] == ["fsutil", "verified-copy", "s3://source/checkpoint", "s3://permanent/checkpoint"]


def test_submit_command_skyrl_flag_swaps_the_inner_environment():
    plain = envelope(PROGRAM)
    skyrl = envelope(PROGRAM, skyrl=True)
    assert skyrl[: skyrl.index("--")] == plain[: plain.index("--")]
    assert skyrl[skyrl.index("--") + 1 :] == skyrl_cpu_command(*PROGRAM)


def test_submit_command_strips_the_argument_separator():
    assert envelope(["--", *PROGRAM]) == envelope(PROGRAM)


def test_submit_command_rejects_empty_program():
    with pytest.raises(ValueError):
        envelope([])
    with pytest.raises(ValueError):
        envelope(["--"])


def test_submit_command_rejects_malformed_env():
    with pytest.raises(ValueError, match="KEY=VALUE"):
        envelope(PROGRAM, env=["MARIN_PREFIX"])
