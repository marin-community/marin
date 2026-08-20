# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts for post-merge Pulumi deployment checks."""

import json
import subprocess
import sys
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(".github/workflows/ops-iac-preview.yaml")
SUMMARY_SCRIPT = Path(".github/actions/pulumi-preview/summarize_deployment.py")


def _workflow() -> dict:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _triggers(workflow: dict) -> dict:
    # PyYAML 1.1 parses the unquoted workflow key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def test_iac_preview_checks_merged_pull_requests() -> None:
    workflow = _workflow()
    triggers = _triggers(workflow)

    assert "closed" in triggers["pull_request"]["types"]
    dispatch_inputs = triggers["workflow_dispatch"]["inputs"]
    assert "deployment_check_attempt" in dispatch_inputs
    assert "deployment_check_ref" not in dispatch_inputs
    prepare = workflow["jobs"]["prepare"]
    assert prepare["outputs"]["deployment_check"]
    assert all(step.get("uses") != "actions/checkout@v5" for step in prepare["steps"])
    preview = workflow["jobs"]["preview"]
    assert preview["needs"] == "prepare"
    assert preview["strategy"]["matrix"] == "${{ fromJSON(needs.prepare.outputs.preview_matrix) }}"
    checkout = next(step for step in preview["steps"] if step["name"] == "Checkout code")
    assert checkout["with"]["ref"] == "${{ needs.prepare.outputs.ref }}"

    comment = workflow["jobs"]["comment"]
    assert "actions" not in comment["permissions"]
    retry = workflow["jobs"]["retry-deployment-check"]
    assert retry["permissions"]["actions"] == "write"
    assert "needs.comment.outputs.next_attempt != ''" in retry["if"]
    delays = json.loads(workflow["env"]["CHECK_DELAYS_MINUTES"])
    assert delays[0] == 30
    assert delays[1:] == [2 * delay for delay in delays[:-1]]


def _write_preview(root: Path, stack: str, severity: str) -> None:
    artifact = root / stack
    artifact.mkdir(parents=True)
    (artifact / "meta.json").write_text(
        json.dumps({"stack": stack, "severity": severity}),
        encoding="utf-8",
    )


def _summarize(tmp_path: Path, *, attempt: int) -> tuple[str, dict[str, str]]:
    previews = tmp_path / "previews"
    comment = tmp_path / "comment.md"
    github_output = tmp_path / "github-output"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--previews-dir",
            str(previews),
            "--preview-matrix",
            '{"include":[{"stack":"marin"},{"stack":"cw-rno2a"}]}',
            "--check-delays-minutes",
            "[30,60,120]",
            "--attempt",
            str(attempt),
            "--merger",
            "operator",
            "--run-url",
            "https://github.test/runs/1",
            "--out",
            str(comment),
            "--github-output",
            str(github_output),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    outputs = dict(line.split("=", 1) for line in github_output.read_text().splitlines())
    return comment.read_text(), outputs


def test_deployment_summary_requests_retry_for_changes_and_missing_stacks(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "change")

    comment, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "true", "next_attempt": "2", "should_comment": "true"}
    assert "@operator" in comment
    assert "`marin`" in comment
    assert "`cw-rno2a`" in comment


def test_deployment_summary_clears_existing_reminder_after_retry(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "none")
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=2)

    assert outputs == {"needs_retry": "false", "next_attempt": "", "should_comment": "true"}


def test_deployment_summary_does_not_comment_on_initial_clean_check(tmp_path: Path) -> None:
    _write_preview(tmp_path / "previews", "marin", "none")
    _write_preview(tmp_path / "previews", "cw-rno2a", "none")

    _, outputs = _summarize(tmp_path, attempt=1)

    assert outputs == {"needs_retry": "false", "next_attempt": "", "should_comment": "false"}
