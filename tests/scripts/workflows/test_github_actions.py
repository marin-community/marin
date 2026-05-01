# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/workflows/github_actions.py."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from scripts.workflows.github_actions import (
    cli,
    policy_failures,
    required_status_contexts,
    workflow_records,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_workflow(tmp_path: Path, filename: str, content: str) -> Path:
    """Write a workflow YAML file and return its path."""
    p = tmp_path / filename
    p.write_text(content)
    return p


MINIMAL_GOOD_WORKFLOW = """\
name: Iris - Unit Tests
on: [push]
jobs:
  run-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
"""

WORKFLOW_WITH_MATRIX = """\
name: Marin - Release
on: [push]
jobs:
  build-matrix:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v5
"""

WORKFLOW_WITH_THIRD_PARTY_SHA = """\
name: Zephyr - Lint
on: [push]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: peaceiris/actions-gh-pages@4d5e6fa134f4617b7b8c3f0ca7f9f6e9c7b5a3d2
"""

WORKFLOW_WITH_THIRD_PARTY_TAG = """\
name: Marin - Lint
on: [push]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: peaceiris/actions-gh-pages@v4
"""

WORKFLOW_BAD_JOB_ID_CAMEL = """\
name: Iris - Unit Tests
on: [push]
jobs:
  someBadName:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""

WORKFLOW_BAD_JOB_ID_UPPERCASE = """\
name: Iris - Unit Tests
on: [push]
jobs:
  Job_Id:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""

WORKFLOW_BAD_DOMAIN = """\
name: UnknownDomain - Unit Tests
on: [push]
jobs:
  run-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""


# ---------------------------------------------------------------------------
# Test: basic parsing
# ---------------------------------------------------------------------------


def test_parses_workflow_name_jobs_and_actions(tmp_path: Path) -> None:
    write_workflow(tmp_path, "iris-tests.yaml", MINIMAL_GOOD_WORKFLOW)
    records = workflow_records(tmp_path)

    assert len(records) == 1
    rec = records[0]
    assert rec.workflow_name == "Iris - Unit Tests"
    assert len(rec.jobs) == 1
    job = rec.jobs[0]
    assert job.job_id == "run-tests"
    assert job.job_name is None
    # Both checkout and setup-python are trusted, still appear in third_party_actions
    assert "actions/checkout@v5" in rec.third_party_actions
    assert "actions/setup-python@v6" in rec.third_party_actions


def test_uses_path_stem_as_name_when_name_missing(tmp_path: Path) -> None:
    content = """\
on: [push]
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""
    write_workflow(tmp_path, "iris-tests.yaml", content)
    records = workflow_records(tmp_path)
    assert records[0].workflow_name == "iris-tests"


# ---------------------------------------------------------------------------
# Test: file extension check
# ---------------------------------------------------------------------------


def test_audit_fails_on_yml_extension(tmp_path: Path) -> None:
    write_workflow(tmp_path, "iris-tests.yml", MINIMAL_GOOD_WORKFLOW)
    records = workflow_records(tmp_path)
    failures = policy_failures(records)
    assert any("extension" in f and ".yml" in f for f in failures)


def test_audit_passes_on_yaml_extension(tmp_path: Path) -> None:
    write_workflow(tmp_path, "iris-tests.yaml", MINIMAL_GOOD_WORKFLOW)
    records = workflow_records(tmp_path)
    ext_failures = [f for f in policy_failures(records) if "extension" in f]
    assert ext_failures == []


# ---------------------------------------------------------------------------
# Test: workflow name domain check
# ---------------------------------------------------------------------------


def test_audit_fails_when_name_lacks_allowed_domain(tmp_path: Path) -> None:
    write_workflow(tmp_path, "bad-domain.yaml", WORKFLOW_BAD_DOMAIN)
    records = workflow_records(tmp_path)
    failures = policy_failures(records)
    assert any("UnknownDomain" in f for f in failures)


@pytest.mark.parametrize(
    "domain",
    ["Iris", "Zephyr", "Marin", "Levanter", "Haliax", "Fray", "Dupekit", "Ops"],
)
def test_audit_passes_for_all_allowed_domains(tmp_path: Path, domain: str) -> None:
    content = f"""\
name: {domain} - Unit Tests
on: [push]
jobs:
  run-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""
    write_workflow(tmp_path, "workflow.yaml", content)
    records = workflow_records(tmp_path)
    name_failures = [f for f in policy_failures(records) if "domain" in f or "name" in f]
    assert name_failures == []


# ---------------------------------------------------------------------------
# Test: job id kebab-case check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["someBadName", "Job_Id"])
def test_audit_fails_on_bad_job_id(tmp_path: Path, bad_id: str) -> None:
    content = f"""\
name: Iris - Unit Tests
on: [push]
jobs:
  {bad_id}:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""
    write_workflow(tmp_path, "workflow.yaml", content)
    records = workflow_records(tmp_path)
    failures = policy_failures(records)
    assert any(bad_id in f for f in failures)


@pytest.mark.parametrize("good_id", ["run-tests", "build", "iris-e2e-smoke", "a1b2"])
def test_audit_passes_on_valid_job_id(tmp_path: Path, good_id: str) -> None:
    content = f"""\
name: Iris - Unit Tests
on: [push]
jobs:
  {good_id}:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
"""
    write_workflow(tmp_path, "workflow.yaml", content)
    records = workflow_records(tmp_path)
    job_id_failures = [f for f in policy_failures(records) if "kebab" in f or "job id" in f]
    assert job_id_failures == []


# ---------------------------------------------------------------------------
# Test: action pinning checks
# ---------------------------------------------------------------------------


def test_audit_fails_on_non_trusted_tag_pinned_action(tmp_path: Path) -> None:
    write_workflow(tmp_path, "workflow.yaml", WORKFLOW_WITH_THIRD_PARTY_TAG)
    records = workflow_records(tmp_path)
    failures = policy_failures(records)
    assert any("peaceiris/actions-gh-pages@v4" in f for f in failures)


def test_audit_passes_on_sha_pinned_non_trusted_action(tmp_path: Path) -> None:
    write_workflow(tmp_path, "workflow.yaml", WORKFLOW_WITH_THIRD_PARTY_SHA)
    records = workflow_records(tmp_path)
    # Should have no pinning failures for peaceiris (it's SHA-pinned)
    pinning_failures = [f for f in policy_failures(records) if "peaceiris" in f]
    assert pinning_failures == []


def test_audit_passes_on_trusted_tag_pinned_action(tmp_path: Path) -> None:
    # actions/checkout@v5 is in the allowlist — tag pinning is fine
    write_workflow(tmp_path, "workflow.yaml", MINIMAL_GOOD_WORKFLOW)
    records = workflow_records(tmp_path)
    pinning_failures = [f for f in policy_failures(records) if "actions/checkout" in f]
    assert pinning_failures == []


# ---------------------------------------------------------------------------
# Test: matrix context rendering
# ---------------------------------------------------------------------------


def test_matrix_context_renders_key_list(tmp_path: Path) -> None:
    write_workflow(tmp_path, "workflow.yaml", WORKFLOW_WITH_MATRIX)
    records = workflow_records(tmp_path)
    assert len(records) == 1
    job = records[0].jobs[0]
    assert job.matrix_context == "os, python-version"


def test_matrix_context_is_none_without_matrix(tmp_path: Path) -> None:
    write_workflow(tmp_path, "workflow.yaml", MINIMAL_GOOD_WORKFLOW)
    records = workflow_records(tmp_path)
    assert records[0].jobs[0].matrix_context is None


# ---------------------------------------------------------------------------
# Test: required_status_contexts parsing (both API shapes)
# ---------------------------------------------------------------------------


def _make_subprocess_result(stdout: str) -> MagicMock:
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.stdout = stdout
    return result


def test_required_status_contexts_new_checks_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "required_status_checks": {
            "checks": [
                {"context": "iris-tests", "app_id": 1},
                {"context": "marin-lint", "app_id": 1},
            ]
        }
    }
    monkeypatch.setattr(
        "scripts.workflows.github_actions.subprocess.run",
        lambda *args, **kwargs: _make_subprocess_result(json.dumps(payload)),
    )
    contexts = required_status_contexts("marin-community/marin", "main")
    assert contexts == ("iris-tests", "marin-lint")


def test_required_status_contexts_old_contexts_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "required_status_checks": {
            "contexts": ["iris-tests", "marin-lint"],
        }
    }
    monkeypatch.setattr(
        "scripts.workflows.github_actions.subprocess.run",
        lambda *args, **kwargs: _make_subprocess_result(json.dumps(payload)),
    )
    contexts = required_status_contexts("marin-community/marin", "main")
    assert contexts == ("iris-tests", "marin-lint")


# ---------------------------------------------------------------------------
# Test: end-to-end CLI behavior
# ---------------------------------------------------------------------------


def test_audit_cli_exits_nonzero_with_failures_on_stderr(tmp_path: Path) -> None:
    # One good, one bad (yml extension + bad domain + bad job id)
    write_workflow(tmp_path, "good.yaml", MINIMAL_GOOD_WORKFLOW)
    write_workflow(
        tmp_path,
        "bad.yml",
        """\
name: BadDomain - Tests
on: [push]
jobs:
  BadJobId:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
""",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["audit", "--workflows-dir", str(tmp_path)])

    assert result.exit_code == 1
    # Click 8.3 CliRunner mixes stdout+stderr into result.output by default.
    assert "FAIL:" in result.output
    assert "Audited" in result.output


def test_audit_cli_exits_zero_on_clean_workflows(tmp_path: Path) -> None:
    write_workflow(tmp_path, "iris-tests.yaml", MINIMAL_GOOD_WORKFLOW)

    runner = CliRunner()
    result = runner.invoke(cli, ["audit", "--workflows-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "0 failure(s)" in result.output


def test_audit_cli_checks_required_contexts_when_repo_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The workflow defines job 'run-tests'; branch protection requires 'missing-job'.
    write_workflow(tmp_path, "iris-tests.yaml", MINIMAL_GOOD_WORKFLOW)

    payload = {"required_status_checks": {"contexts": ["missing-job"]}}
    monkeypatch.setattr(
        "scripts.workflows.github_actions.subprocess.run",
        lambda *args, **kwargs: _make_subprocess_result(json.dumps(payload)),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["audit", "--workflows-dir", str(tmp_path), "--repo", "marin-community/marin"],
    )

    assert result.exit_code == 1
    assert "missing-job" in result.output


def test_required_contexts_command_prints_one_per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"required_status_checks": {"contexts": ["ctx-a", "ctx-b"]}}
    monkeypatch.setattr(
        "scripts.workflows.github_actions.subprocess.run",
        lambda *args, **kwargs: _make_subprocess_result(json.dumps(payload)),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["required-contexts", "--repo", "marin-community/marin", "--branch", "main"])

    assert result.exit_code == 0
    assert result.output.strip().splitlines() == ["ctx-a", "ctx-b"]
