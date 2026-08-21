# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the shared deployment CLI."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin_deploy.cli import cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pulumi_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "pulumi"
    executable.write_text(
        """#!/bin/sh
{
    printf '%s\\n' "$PWD"
    printf '%s\\n' "$@"
} > "$PULUMI_RECORD"
exit "${PULUMI_EXIT:-0}"
"""
    )
    executable.chmod(0o755)
    record = tmp_path / "pulumi-record"
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PULUMI_RECORD", str(record))
    return record


@pytest.mark.parametrize(
    ("service", "stack"),
    [
        ("ducky", "ducky-marin"),
        ("echo", "marin-echo"),
        ("evaldash", "marin-evaldash"),
        ("grafana", "marin-grafana"),
        ("loom", "marin-loom"),
        ("xprof", "xprof-marin"),
    ],
)
def test_service_rollout_applies_stack_from_project_directory(pulumi_record: Path, service: str, stack: str) -> None:
    result = CliRunner().invoke(
        cli,
        [service, "rollout", "--yes", "--config", "first=value", "--config", "second=other"],
    )

    assert result.exit_code == 0, result.output
    assert pulumi_record.read_text().splitlines() == [
        str(REPOSITORY_ROOT / "infra" / service),
        "up",
        "--stack",
        stack,
        "--yes",
        "--config",
        "first=value",
        "--config",
        "second=other",
    ]


def test_service_rollout_preserves_pulumi_failure_exit_code(
    pulumi_record: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PULUMI_EXIT", "42")

    result = CliRunner().invoke(cli, ["echo", "rollout"])

    assert result.exit_code == 42
