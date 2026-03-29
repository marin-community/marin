# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent adapter interface and implementations for Claude Code and Codex."""

import json
import logging
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from marin.dispatch.schema import TickEvent

logger = logging.getLogger(__name__)

LOGBOOK_START = "<<<LOGBOOK_ENTRY>>>"
LOGBOOK_END = "<<<END_LOGBOOK_ENTRY>>>"
COMMENT_START = "<<<ISSUE_COMMENT>>>"
COMMENT_END = "<<<END_ISSUE_COMMENT>>>"
ESCALATE_MARKER = "<<<ESCALATE>>>"


@dataclass(frozen=True)
class AgentResult:
    success: bool
    logbook_entry: str
    issue_comment: str | None = None
    error: str | None = None
    escalate: bool = False


class AgentSession(Protocol):
    def launch(self, event: TickEvent, worktree_path: Path) -> AgentResult: ...


def build_agent_prompt(event: TickEvent) -> str:
    run = event.run_pointer
    if run.ray is not None:
        run_desc = f"Ray job: {run.ray.job_id}\n" f"Cluster: {run.ray.cluster}\n" f"Experiment: {run.ray.experiment}"
    elif run.iris is not None:
        run_desc = (
            f"Iris job: {run.iris.job_id}\n" f"Config: {run.iris.config}\n" f"Resubmit: {run.iris.resubmit_command}"
        )
    else:
        run_desc = "(no run config)"

    return textwrap.dedent(
        f"""\
        You are a monitoring agent for the Marin project.

        ## Your Role
        You are dispatched by the monitoring system to check on a research run.
        Your job is to:
        1. Query the run's status (logs, metrics, W&B if available).
        2. Diagnose any failures or anomalies.
        3. If possible, take corrective action (e.g. resubmit a failed job, adjust config).
        4. Write a concise logbook entry summarizing what you found and did.
        5. If there is a meaningful update (status change, failure, milestone), post an issue comment.
        6. If the problem is beyond automated recovery, escalate to the operator.

        ## Operator Instructions
        {event.prompt}

        ## Trigger
        Event: {event.kind}
        Timestamp: {event.timestamp}

        ## Run Details
        {run_desc}

        ## Context
        Collection: {event.collection_name}
        Branch: {event.branch}
        Issue: #{event.issue}
        Logbook: {event.logbook}

        ## Output Format
        Wrap your logbook entry between {LOGBOOK_START} and {LOGBOOK_END} markers.
        If you have a meaningful update for the GitHub issue, wrap it between {COMMENT_START} and {COMMENT_END}.
        If the situation requires human escalation, include {ESCALATE_MARKER} on its own line.
    """
    )


def build_conflict_prompt(conflicted_files: list[str], branch: str) -> str:
    files_list = ", ".join(conflicted_files)
    return f"Merge conflict on `{branch}`. Resolve conflict markers in: {files_list}"


def parse_agent_output(output: str) -> AgentResult:
    logbook_entry = _extract_between(output, LOGBOOK_START, LOGBOOK_END)
    issue_comment = _extract_between(output, COMMENT_START, COMMENT_END)
    escalate = ESCALATE_MARKER in output

    if not logbook_entry:
        return AgentResult(
            success=False,
            logbook_entry="",
            error="Agent produced no logbook entry",
            escalate=escalate,
        )

    return AgentResult(
        success=True,
        logbook_entry=logbook_entry,
        issue_comment=issue_comment or None,
        escalate=escalate,
    )


def _extract_between(text: str, start: str, end: str) -> str:
    si = text.find(start)
    if si == -1:
        return ""
    si += len(start)
    ei = text.find(end, si)
    if ei == -1:
        return text[si:].strip()
    return text[si:ei].strip()


def _run_agent_cli(
    cmd: list[str],
    prompt: str,
    worktree_path: Path,
    timeout_seconds: int,
    agent_name: str,
) -> AgentResult | str:
    """Run an agent CLI subprocess and parse the output."""
    logger.info("Launching %s in %s", agent_name, worktree_path)
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            cwd=str(worktree_path),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return AgentResult(success=False, logbook_entry="", error=f"{agent_name} timed out", escalate=True)

    if result.returncode != 0:
        error_msg = result.stderr[:500] if result.stderr else f"exit code {result.returncode}"
        return AgentResult(success=False, logbook_entry="", error=f"{agent_name} failed: {error_msg}")

    return result.stdout


class ClaudeCodeAdapter:
    def __init__(self, model: str = "opus", timeout_seconds: int = 1800):
        self.model = model
        self.timeout_seconds = timeout_seconds

    def launch(self, event: TickEvent, worktree_path: Path) -> AgentResult:
        prompt = build_agent_prompt(event)
        cmd = ["claude", "--model", self.model, "--print", "--dangerously-skip-permissions", "--output-format", "text"]
        result = _run_agent_cli(cmd, prompt, worktree_path, self.timeout_seconds, "Claude Code")
        if isinstance(result, AgentResult):
            return result
        return parse_agent_output(result)


class CodexAdapter:
    def __init__(self, timeout_seconds: int = 1800):
        self.timeout_seconds = timeout_seconds

    def launch(self, event: TickEvent, worktree_path: Path) -> AgentResult:
        prompt = build_agent_prompt(event)
        cmd = ["codex", "--quiet", "--json"]
        result = _run_agent_cli(cmd, prompt, worktree_path, self.timeout_seconds, "Codex")
        if isinstance(result, AgentResult):
            return result
        # Codex --json outputs JSON; extract the text content.
        try:
            data = json.loads(result)
            text = data.get("output", result)
        except json.JSONDecodeError:
            text = result
        return parse_agent_output(text)
