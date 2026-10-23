# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from infra import linter


@pytest.fixture
def review_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "sample.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "sample.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "--quiet", "-m", "Base"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=repo, check=True)
    (repo / "sample.py").write_text("value = 2\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = shutil.which("git")
    assert git is not None
    (bin_dir / "git").symlink_to(git)
    calls = tmp_path / "calls"
    calls.mkdir()
    for agent in linter.LintReviewAgent:
        executable = bin_dir / agent.value
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, subprocess, sys, uuid\n"
            "sys.stdin.read()\n"
            "diff = subprocess.check_output(['git', 'diff', 'origin/main'], text=True)\n"
            "record = {'agent': pathlib.Path(sys.argv[0]).name, 'args': sys.argv[1:], 'diff': diff}\n"
            "path = pathlib.Path(os.environ['REVIEW_TEST_CALLS']) / (uuid.uuid4().hex + '.json')\n"
            "path.write_text(json.dumps(record))\n"
            "print('sample.py:1: ml-cruft-dead-branch (0.95) Remove the dead branch.')\n"
        )
        executable.chmod(0o755)
    # Only the fake reviewers and real git are visible; no telemetry process can start.
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("REVIEW_TEST_CALLS", str(calls))
    monkeypatch.setattr(linter, "ROOT_DIR", repo)
    monkeypatch.setattr(linter, "LINT_REVIEW_LOG_ROOT", tmp_path / "logs")
    return calls


@pytest.mark.parametrize("calling_agent", list(linter.LintReviewAgent))
def test_review_uses_callers_agent_with_readonly_permissions(
    calling_agent: linter.LintReviewAgent, review_workspace: Path, capsys: pytest.CaptureFixture[str]
):
    assert linter.run_lint_review(calling_agent, lane_names=["cruft", "interfaces"]) == 0
    assert "sample.py:1: ml-cruft-dead-branch (0.95) Remove the dead branch." in capsys.readouterr().out

    calls = [json.loads(path.read_text()) for path in review_workspace.glob("*.json")]
    assert len(calls) == 3  # Two lanes and their composer use the same reviewer.
    for call in calls:
        assert call["agent"] == calling_agent.value
        assert "+value = 2" in call["diff"]
        parser = argparse.ArgumentParser()
        if calling_agent == linter.LintReviewAgent.CLAUDE:
            parser.add_argument("-p", action="store_true")
            parser.add_argument("--tools")
            parser.add_argument("--allowedTools")
            parser.add_argument("--disallowedTools")
            permissions = parser.parse_args(call["args"])
            assert set(permissions.tools.split(",")) == {"Bash", "Read", "Grep", "Glob"}
            assert "Bash(git diff:*)" in permissions.allowedTools.split(",")
            assert {"Bash(git commit:*)", "Bash(git push:*)", "Bash(gh:*)"} <= set(
                permissions.disallowedTools.split(",")
            )
        else:
            parser.add_argument("command")
            parser.add_argument("--ephemeral", action="store_true")
            parser.add_argument("--sandbox")
            permissions = parser.parse_args(call["args"])
            assert permissions.command == "exec"
            assert permissions.ephemeral
            assert permissions.sandbox == "read-only"
