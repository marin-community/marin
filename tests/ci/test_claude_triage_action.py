# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml


TRIAGE_ACTION = Path(".github/actions/claude-triage/action.yaml")
CANARY_WORKFLOWS = tuple(sorted(Path(".github/workflows").glob("marin-canary-*.yaml")))
REQUIRED_REPORTING_TOOLS = {
    "Bash(gh:*)",
    "Bash(mktemp:*)",
    "Bash(rm:*)",
    "Read",
    "Write",
}


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
    assert "-s slack_message.md" in summary_check["run"]


def test_canary_triage_tool_overrides_keep_reporting_tools() -> None:
    for workflow_path in CANARY_WORKFLOWS:
        workflow_text = workflow_path.read_text()
        if "uses: ./.github/actions/claude-triage" not in workflow_text:
            continue

        workflow = _document(workflow_path)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses") != "./.github/actions/claude-triage":
                    continue
                allowed_tools = step.get("with", {}).get("allowed-tools")
                if allowed_tools is not None:
                    assert REQUIRED_REPORTING_TOOLS <= _tool_names(allowed_tools), workflow_path
