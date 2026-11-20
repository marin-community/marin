# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infra import linter


@pytest.fixture
def review_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    sample = tmp_path / "sample.py"
    sample.write_text("value = 1\n")
    subprocess.run(["git", "add", "sample.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "--quiet", "-m", "Base"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=tmp_path, check=True)
    sample.write_text("value = 2\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = shutil.which("git")
    assert git is not None
    (bin_dir / "git").symlink_to(git)
    agents = tmp_path / "agents"
    agents.mkdir()
    call_path = tmp_path / "call.json"
    for agent in linter.LintReviewAgent:
        executable = agents / agent.value
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "sys.stdin.read()\n"
            "pathlib.Path(os.environ['REVIEW_TEST_CALL']).write_text(json.dumps(\n"
            "    {'agent': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:]}))\n"
        )
        executable.chmod(0o755)
    # Exclude real agent CLIs and telemetry; the explicit-path cases have only git on PATH.
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("REVIEW_TEST_CALL", str(call_path))
    monkeypatch.setattr(linter, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(linter, "LINT_REVIEW_LOG_ROOT", tmp_path / "logs")
    return agents, call_path


@pytest.mark.parametrize("calling_agent", list(linter.LintReviewAgent))
@pytest.mark.parametrize("installation", ["on-path", "absolute"])
def test_review_uses_callers_agent_with_readonly_permissions(
    calling_agent: linter.LintReviewAgent,
    installation: str,
    review_workspace: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
):
    agents, call_path = review_workspace
    executable = agents / calling_agent.value if installation == "absolute" else None
    if installation == "on-path":
        monkeypatch.setenv("PATH", f"{agents}{os.pathsep}{os.environ['PATH']}")
    assert linter.run_lint_review(calling_agent, lane_names=["cruft"], agent_executable=executable) == 0
    call = json.loads(call_path.read_text())
    assert call["agent"] == calling_agent.value

    # The agent CLIs are the permission boundary; parse their security options without calling a model service.
    parser = argparse.ArgumentParser()
    if calling_agent == linter.LintReviewAgent.CLAUDE:
        parser.add_argument("-p", action="store_true")
        parser.add_argument("--tools")
        parser.add_argument("--allowedTools")
        parser.add_argument("--disallowedTools")
        parser.add_argument("--strict-mcp-config", action="store_true")
        parser.add_argument("--mcp-config", type=json.loads)
        permissions = parser.parse_args(call["args"])
        assert set(permissions.tools.split(",")) == {"Bash", "Read", "Grep", "Glob"}
        assert {"Bash(git commit:*)", "Bash(git push:*)", "Bash(gh:*)"} <= set(permissions.disallowedTools.split(","))
        assert permissions.strict_mcp_config
        assert permissions.mcp_config == {"mcpServers": {}}
    else:
        parser.add_argument("command")
        parser.add_argument("--ephemeral", action="store_true")
        parser.add_argument("--sandbox")
        parser.add_argument("--ignore-user-config", action="store_true")
        permissions = parser.parse_args(call["args"])
        assert permissions.command == "exec"
        assert permissions.ephemeral
        assert permissions.sandbox == "read-only"
        assert permissions.ignore_user_config


def test_review_reports_failure_when_agent_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert linter.run_lint_review(linter.LintReviewAgent.CODEX) == 1
