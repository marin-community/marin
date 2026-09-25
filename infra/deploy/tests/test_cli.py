# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the shared deployment CLI."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from click.testing import CliRunner
from marin_deploy.cli import cli

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DeploymentRecord:
    pulumi: Path
    config: Path
    environment: Path
    gcloud: Path


@pytest.fixture
def deployment_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DeploymentRecord:
    pulumi = tmp_path / "pulumi"
    pulumi.write_text(
        """#!/bin/sh
{
    printf '%s\\n' "$PWD"
    printf '%s\\n' "$@"
} > "$PULUMI_RECORD"
previous=""
for argument in "$@"; do
    if [ "$previous" = "--config-file" ]; then
        cp "$argument" "$PULUMI_CONFIG_RECORD"
    fi
    previous="$argument"
done
printf '%s' "${CLOUDFLARE_API_TOKEN:-}" > "$PULUMI_ENVIRONMENT_RECORD"
exit "${PULUMI_EXIT:-0}"
"""
    )
    pulumi.chmod(0o755)

    gcloud = tmp_path / "gcloud"
    gcloud.write_text(
        """#!/bin/sh
printf '%s\\n' "$@" > "$GCLOUD_RECORD"
printf '%s' cloudflare-token
"""
    )
    gcloud.chmod(0o755)

    record = DeploymentRecord(
        pulumi=tmp_path / "pulumi-record",
        config=tmp_path / "pulumi-config-record",
        environment=tmp_path / "pulumi-environment-record",
        gcloud=tmp_path / "gcloud-record",
    )
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("PULUMI_RECORD", str(record.pulumi))
    monkeypatch.setenv("PULUMI_CONFIG_RECORD", str(record.config))
    monkeypatch.setenv("PULUMI_ENVIRONMENT_RECORD", str(record.environment))
    monkeypatch.setenv("GCLOUD_RECORD", str(record.gcloud))
    return record


@pytest.mark.parametrize(
    ("service", "stack"),
    [
        ("ducky", "ducky-marin"),
        ("grafana", "marin-grafana"),
        ("loom", "marin-loom"),
        ("xprof", "xprof-marin"),
    ],
)
def test_service_rollout_applies_stack_from_project_directory(
    deployment_record: DeploymentRecord, service: str, stack: str
) -> None:
    result = CliRunner().invoke(
        cli,
        [service, "rollout", "--yes", "--config", "first=value", "--config", "second=other"],
    )

    assert result.exit_code == 0, result.output
    arguments = deployment_record.pulumi.read_text().splitlines()
    assert arguments[:6] == [
        str(REPOSITORY_ROOT / "infra" / service),
        "up",
        "--stack",
        stack,
        "--yes",
        "--config-file",
    ]
    config_file = Path(arguments[6])
    assert config_file.name == f"Pulumi.{stack}.yaml"
    assert arguments[7:] == [
        "--config",
        "first=value",
        "--config",
        "second=other",
    ]
    assert deployment_record.config.read_text() == (REPOSITORY_ROOT / "infra" / service / config_file.name).read_text()
    assert not config_file.exists()


def test_service_rollout_loads_secret_environment(deployment_record: DeploymentRecord) -> None:
    result = CliRunner().invoke(cli, ["marina", "rollout"])

    assert result.exit_code == 0, result.output
    assert deployment_record.gcloud.read_text().splitlines() == [
        "secrets",
        "versions",
        "access",
        "latest",
        "--secret=cloudflare-oa-dns-token",
        "--project=hai-gcp-models",
    ]
    assert deployment_record.environment.read_text() == "cloudflare-token"


def test_service_rollout_preserves_pulumi_failure_exit_code(
    deployment_record: DeploymentRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PULUMI_EXIT", "42")

    result = CliRunner().invoke(cli, ["marina", "rollout"])

    assert result.exit_code == 42
