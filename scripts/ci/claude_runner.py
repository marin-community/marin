# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Code and distinguish quota exhaustion from agent failures."""

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ClaudeRunStatus(StrEnum):
    SUCCESS = "success"
    QUOTA_EXHAUSTED = "quota_exhausted"
    FAILED = "failed"


@dataclass(frozen=True)
class ClaudeRunResult:
    status: ClaudeRunStatus
    output: str


def _result_messages(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [message for item in value for message in _result_messages(item)]
    if not isinstance(value, dict):
        return []
    messages = [value] if value.get("type") == "result" else []
    return messages + [message for item in value.values() for message in _result_messages(item)]


def classify_claude_result(value: object) -> ClaudeRunStatus:
    """Classify a CLI envelope or claude-code-action execution trace."""
    messages = _result_messages(value)
    if any(message.get("is_error") is True and message.get("api_error_status") == 429 for message in messages):
        return ClaudeRunStatus.QUOTA_EXHAUSTED
    if any(message.get("is_error") is True for message in messages):
        return ClaudeRunStatus.FAILED
    return ClaudeRunStatus.SUCCESS


def run_claude(
    prompt: str,
    args: Sequence[str],
    *,
    executable: Path = Path("claude"),
    cwd: Path | None = None,
    timeout: float | None = None,
) -> ClaudeRunResult:
    """Run the Claude CLI in JSON mode, raising on non-quota failures."""
    command = [str(executable), "--print", "--output-format", "json", *args, "--", prompt]
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError:
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode, command, output=completed.stdout, stderr=completed.stderr
            ) from None
        raise ValueError("Claude CLI returned invalid JSON") from None

    status = classify_claude_result(envelope)
    output = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(output, str):
        raise ValueError("Claude CLI result is missing its text output")
    if status == ClaudeRunStatus.FAILED or (completed.returncode != 0 and status != ClaudeRunStatus.QUOTA_EXHAUSTED):
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout, stderr=completed.stderr
        )
    return ClaudeRunResult(status=status, output=output)


def report_quota_exhaustion() -> None:
    print("::warning title=Claude quota exhausted::Skipping Claude agent because the account returned HTTP 429.")


def _write_github_output(quota_exhausted: bool) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a") as output:
            output.write(f"quota_exhausted={str(quota_exhausted).lower()}\n")


def classify_action(outcome: str, execution_file: Path | None) -> None:
    """Fail an action invocation unless it succeeded or exhausted quota."""
    if outcome == "success":
        _write_github_output(quota_exhausted=False)
        return
    if execution_file is None or not execution_file.is_file():
        raise ValueError("Claude action failed without an execution file")

    execution = json.loads(execution_file.read_text())
    if classify_claude_result(execution) != ClaudeRunStatus.QUOTA_EXHAUSTED:
        raise RuntimeError("Claude action failed for a reason other than quota exhaustion")

    _write_github_output(quota_exhausted=True)
    report_quota_exhaustion()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outcome", choices=("success", "failure", "cancelled", "skipped"))
    parser.add_argument("execution_file", type=Path, nargs="?")
    args = parser.parse_args()
    classify_action(args.outcome, args.execution_file)


if __name__ == "__main__":
    main()
