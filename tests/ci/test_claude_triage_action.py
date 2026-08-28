# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TRIAGE_ACTION = Path(".github/actions/claude-triage/action.yaml")
CANARY_WORKFLOWS = tuple(sorted(Path(".github/workflows").glob("marin-canary-*.yaml")))
TRIAGE_OUTPUT_VALIDATOR = Path("scripts/ci/validate_canary_triage_output.py")
REQUIRED_REPORTING_TOOLS = frozenset(
    {
        "Bash(gh:*)",
        "Bash(mktemp:*)",
        "Bash(rm:*)",
        "Read",
        "Write",
    }
)


def _document(path: Path) -> dict:
    document = yaml.safe_load(path.read_text())
    assert isinstance(document, dict)
    return document


def _tool_names(value: str) -> set[str]:
    return set(value.split(","))


def test_canary_triage_preserves_diagnostics_and_requires_slack_summary() -> None:
    action = _document(TRIAGE_ACTION)

    assert REQUIRED_REPORTING_TOOLS <= _tool_names(action["inputs"]["allowed-tools"]["default"])

    steps = {step["name"]: step for step in action["runs"]["steps"]}
    trace_upload = steps["Upload Claude triage log"]
    assert trace_upload["with"]["path"] == "${{ runner.temp }}/claude-execution-output.json"
    assert trace_upload["with"]["if-no-files-found"] == "warn"

    summary_check = steps["Require Slack summary"]
    assert summary_check["if"] == "always() && steps.claude.outputs.rate_limited != 'true'"


def test_canary_triage_callers_do_not_replace_reporting_tools() -> None:
    for workflow_path in CANARY_WORKFLOWS:
        workflow_text = workflow_path.read_text()
        if "uses: ./.github/actions/claude-triage" not in workflow_text:
            continue

        workflow = _document(workflow_path)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses") != "./.github/actions/claude-triage":
                    continue
                assert "allowed-tools" not in step.get("with", {}), workflow_path


@pytest.mark.parametrize("contents", [None, "", " \n"])
def test_canary_triage_output_validator_rejects_missing_summary(tmp_path: Path, contents: str | None) -> None:
    summary = tmp_path / "slack_message.md"
    if contents is not None:
        summary.write_text(contents)

    result = subprocess.run([sys.executable, TRIAGE_OUTPUT_VALIDATOR, summary], check=False, capture_output=True)

    assert result.returncode != 0


def test_canary_triage_output_validator_accepts_summary(tmp_path: Path) -> None:
    summary = tmp_path / "slack_message.md"
    summary.write_text(":red_circle: *TPU Canary failed* — training crash\n")

    result = subprocess.run([sys.executable, TRIAGE_OUTPUT_VALIDATOR, summary], check=False, capture_output=True)

    assert result.returncode == 0
