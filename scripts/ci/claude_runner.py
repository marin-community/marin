# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Claude Code and distinguish rate limiting from agent failures."""

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

RATE_LIMITED_OUTPUT = "rate_limited"
WEEKLY_LIMIT_MESSAGE = "You've hit your weekly limit"


class ClaudeRunStatus(StrEnum):
    SUCCESS = "success"
    RATE_LIMITED = RATE_LIMITED_OUTPUT
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
    if any(
        message.get("is_error") is True
        and (message.get("api_error_status") == 429 or WEEKLY_LIMIT_MESSAGE in str(message.get("result", "")))
        for message in messages
    ):
        return ClaudeRunStatus.RATE_LIMITED
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
    if status == ClaudeRunStatus.FAILED or (completed.returncode != 0 and status != ClaudeRunStatus.RATE_LIMITED):
        raise subprocess.CalledProcessError(
            completed.returncode, command, output=completed.stdout, stderr=completed.stderr
        )
    return ClaudeRunResult(status=status, output=output)


def report_rate_limit() -> None:
    print("::warning title=Claude rate limited::Skipping Claude agent because the account returned HTTP 429.")


def _write_github_output(output_path: Path, rate_limited: bool) -> None:
    with output_path.open("a") as output:
        output.write(f"{RATE_LIMITED_OUTPUT}={str(rate_limited).lower()}\n")


def classify_action(outcome: str, execution_file: Path | None, github_output: Path) -> None:
    """Fail an action invocation unless it succeeded or was rate limited."""
    if outcome == "success":
        _write_github_output(github_output, rate_limited=False)
        return
    if execution_file is None or not execution_file.is_file():
        raise ValueError("Claude action failed without an execution file")

    execution = json.loads(execution_file.read_text())
    if classify_claude_result(execution) != ClaudeRunStatus.RATE_LIMITED:
        raise RuntimeError("Claude action failed for a reason other than rate limiting")

    _write_github_output(github_output, rate_limited=True)
    report_rate_limit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outcome", choices=("success", "failure", "cancelled", "skipped"))
    parser.add_argument("execution_file", type=Path, nargs="?")
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()
    classify_action(args.outcome, args.execution_file, args.github_output)


if __name__ == "__main__":
    main()
