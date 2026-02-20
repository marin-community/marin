# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def _load_gdnctl_module():
    gdnctl_path = Path(__file__).resolve().parents[1] / "gdnctl.py"
    spec = importlib.util.spec_from_file_location("gdnctl", gdnctl_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gdnctl = _load_gdnctl_module()


def _run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")
    (repo / "data.txt").write_text("v1\n", encoding="utf-8")
    _run_git(repo, "add", "data.txt")
    _run_git(repo, "commit", "-m", "baseline")
    return repo, _run_git(repo, "rev-parse", "HEAD")


def _commit_change(repo: Path, text: str, message: str) -> str:
    (repo / "data.txt").write_text(text, encoding="utf-8")
    _run_git(repo, "add", "data.txt")
    _run_git(repo, "commit", "-m", message)
    return _run_git(repo, "rev-parse", "HEAD")


def _perf_args(*, regression_policy: str = "revert-count-failure") -> argparse.Namespace:
    return argparse.Namespace(
        perf_mode="required",
        perf_metric="throughput/mfu",
        perf_min_improvement_pct=0.25,
        perf_max_regression_pct=1.0,
        perf_regression_policy=regression_policy,
    )


def _read_state(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_perf_policy_promotes_better_candidate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args()

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {"throughput/mfu": 4.0}, "warnings": []},
    )
    assert ok
    assert not count_failure
    assert rc == 0

    better_commit = _commit_change(repo, "v2\n", "improve")
    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=better_commit,
        validation_info={"metrics": {"throughput/mfu": 4.2}, "warnings": []},
    )
    assert ok
    assert not count_failure
    assert rc == 0

    state = _read_state(state_path)
    champion = state["champion"]
    assert isinstance(champion, dict)
    assert champion["commit"] == better_commit


def test_perf_policy_reverts_regression_commit(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {"throughput/mfu": 4.0}, "warnings": []},
    )
    assert ok
    assert not count_failure
    assert rc == 0

    regression_commit = _commit_change(repo, "regression\n", "regress")
    head_before = _run_git(repo, "rev-parse", "HEAD")
    assert head_before == regression_commit

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=regression_commit,
        validation_info={"metrics": {"throughput/mfu": 3.0}, "warnings": []},
    )
    assert ok
    assert count_failure
    assert rc == 0

    head_after = _run_git(repo, "rev-parse", "HEAD")
    assert head_after != regression_commit
    assert (repo / "data.txt").read_text(encoding="utf-8") == "v1\n"

    state = _read_state(state_path)
    champion = state["champion"]
    assert isinstance(champion, dict)
    assert champion["commit"] == baseline_commit


def test_perf_policy_requires_metric(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args()

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {}, "warnings": []},
    )
    assert not ok
    assert count_failure
    assert rc == 1
