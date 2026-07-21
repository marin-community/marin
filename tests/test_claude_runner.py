# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci.claude_runner import ClaudeRunStatus, classify_action, run_claude


def error_payload(api_status: int) -> dict[str, object]:
    return {
        "type": "result",
        "is_error": True,
        "api_error_status": api_status,
        "result": "weekly limit reached" if api_status == 429 else "model not found",
    }


def write_fake_claude(path: Path, payload: object, returncode: int) -> Path:
    executable = path / "claude"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"print({json.dumps(json.dumps(payload))})\n"
        f"sys.exit({returncode})\n"
    )
    executable.chmod(0o755)
    return executable


def test_run_claude_success_returns_model_output(tmp_path: Path) -> None:
    executable = write_fake_claude(
        tmp_path,
        {"type": "result", "is_error": False, "result": "finished"},
        returncode=0,
    )

    result = run_claude("prompt", [], executable=executable)

    assert result.status == ClaudeRunStatus.SUCCESS
    assert result.output == "finished"


def test_run_claude_quota_exhaustion_returns_quota_status(tmp_path: Path) -> None:
    executable = write_fake_claude(
        tmp_path,
        error_payload(429),
        returncode=1,
    )

    result = run_claude("prompt", [], executable=executable)

    assert result.status == ClaudeRunStatus.QUOTA_EXHAUSTED
    assert result.output == "weekly limit reached"


def test_run_claude_non_quota_error_raises(tmp_path: Path) -> None:
    executable = write_fake_claude(
        tmp_path,
        error_payload(404),
        returncode=1,
    )

    with pytest.raises(subprocess.CalledProcessError):
        run_claude("prompt", [], executable=executable)


def test_classify_action_quota_exhaustion_writes_soft_failure_output(tmp_path: Path) -> None:
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(json.dumps([error_payload(429)]))
    github_output = tmp_path / "github-output"

    classify_action("failure", execution_file, github_output)

    assert github_output.read_text() == "quota_exhausted=true\n"


def test_classify_action_non_quota_error_raises(tmp_path: Path) -> None:
    execution_file = tmp_path / "execution.json"
    execution_file.write_text(json.dumps([error_payload(404)]))

    with pytest.raises(RuntimeError):
        classify_action("failure", execution_file, tmp_path / "github-output")
