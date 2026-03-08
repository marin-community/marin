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
        perf_max_while_increase_ms=5.0,
        perf_new_conditional_ms=5.0,
        perf_max_train_path_budget_increase_ms=5.0,
        perf_control_gate_override_pct=5.0,
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
    args = _perf_args(regression_policy="fail")

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


def test_perf_policy_missing_metric_reverts_candidate_under_revert_policy(tmp_path: Path) -> None:
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

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    assert _run_git(repo, "rev-parse", "HEAD") == candidate_commit

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={"metrics": {}, "warnings": []},
    )
    assert ok
    assert count_failure
    assert rc == 0
    assert _run_git(repo, "rev-parse", "HEAD") != candidate_commit
    assert (repo / "data.txt").read_text(encoding="utf-8") == "v1\n"


def test_latest_iteration_hotspot_context_parses_structured_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 10 - Baseline",
                "- Forward closed-call `shard_map/pallas_call`: `41.000 ms -> 20.000 ms`",
                "- Backward closed-call `shard_map/pallas_call`: `26.000 ms -> 13.000 ms`",
                "- `while`: `0.000 ms -> 0.000 ms`",
                "- `conditional`: `0.000 ms -> 0.000 ms`",
                "",
                "### Iteration 11 - Candidate",
                "- Forward closed-call `shard_map/pallas_call`: `41.000 ms -> 19.000 ms`",
                "- Backward closed-call `shard_map/pallas_call`: `26.000 ms -> 12.000 ms`",
                "- `while`: `0.000 ms -> 31.500 ms`",
                "- `conditional`: `0.000 ms -> 0.000 ms`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context = gdnctl._latest_iteration_hotspot_context(log_path)
    current = context["hotspot_metrics"]
    baseline = context["hotspot_baseline_metrics"]

    assert current["forward_closed_call_ms"] == 19.0
    assert current["backward_closed_call_ms"] == 12.0
    assert current["kernel_budget_ms"] == 31.0
    assert current["control_budget_ms"] == 31.5
    assert current["train_path_budget_ms"] == 62.5
    assert baseline["train_path_budget_ms"] == 33.0


def test_perf_policy_reverts_candidate_when_while_growth_outweighs_small_metric_gain(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "forward_closed_call_ms": 41.0,
        "backward_closed_call_ms": 26.0,
        "while_ms": 0.0,
        "conditional_ms": 0.0,
        "kernel_budget_ms": 67.0,
        "control_budget_ms": 0.0,
        "train_path_budget_ms": 67.0,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {"throughput/mfu": 4.0}, "warnings": [], "hotspot_metrics": baseline_hotspots},
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "forward_closed_call_ms": 20.0,
        "backward_closed_call_ms": 13.0,
        "while_ms": 31.5,
        "conditional_ms": 0.0,
        "kernel_budget_ms": 33.0,
        "control_budget_ms": 31.5,
        "train_path_budget_ms": 64.5,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 4.04},
            "warnings": [],
            "hotspot_metrics": candidate_hotspots,
            "hotspot_baseline_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert count_failure
    assert rc == 0
    assert _run_git(repo, "rev-parse", "HEAD") != candidate_commit

    state = _read_state(state_path)
    last = state["history"][-1]
    assert last["decision"] == "control_flow_regression"


def test_perf_policy_allows_large_metric_win_to_override_control_gate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "forward_closed_call_ms": 41.0,
        "backward_closed_call_ms": 26.0,
        "while_ms": 0.0,
        "conditional_ms": 0.0,
        "kernel_budget_ms": 67.0,
        "control_budget_ms": 0.0,
        "train_path_budget_ms": 67.0,
    }

    gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {"throughput/mfu": 4.0}, "warnings": [], "hotspot_metrics": baseline_hotspots},
    )

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "forward_closed_call_ms": 20.0,
        "backward_closed_call_ms": 13.0,
        "while_ms": 31.5,
        "conditional_ms": 0.0,
        "kernel_budget_ms": 33.0,
        "control_budget_ms": 31.5,
        "train_path_budget_ms": 64.5,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 4.3},
            "warnings": [],
            "hotspot_metrics": candidate_hotspots,
            "hotspot_baseline_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    state = _read_state(state_path)
    champion = state["champion"]
    assert isinstance(champion, dict)
    assert champion["commit"] == candidate_commit


def test_perf_policy_reverts_new_conditional_bucket(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "forward_closed_call_ms": 41.0,
        "backward_closed_call_ms": 26.0,
        "while_ms": 0.0,
        "conditional_ms": 0.0,
        "kernel_budget_ms": 67.0,
        "control_budget_ms": 0.0,
        "train_path_budget_ms": 67.0,
    }
    gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={"metrics": {"throughput/mfu": 4.0}, "warnings": [], "hotspot_metrics": baseline_hotspots},
    )

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = dict(baseline_hotspots)
    candidate_hotspots["conditional_ms"] = 12.0
    candidate_hotspots["control_budget_ms"] = 12.0
    candidate_hotspots["train_path_budget_ms"] = 79.0

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 4.02},
            "warnings": [],
            "hotspot_metrics": candidate_hotspots,
            "hotspot_baseline_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert count_failure
    assert rc == 0
    assert _run_git(repo, "rev-parse", "HEAD") != candidate_commit


def test_extract_wandb_run_url_supports_slug_with_underscore() -> None:
    text = (
        "wandb: View run at: "
        "https://wandb.ai/marin-community/marin/runs/gdn_trisolve_i6_dev_130m_ch128_seg16_20steps-1f9dea"
    )
    url = gdnctl._extract_last_wandb_run_url(text)
    assert url == "https://wandb.ai/marin-community/marin/runs/gdn_trisolve_i6_dev_130m_ch128_seg16_20steps-1f9dea"


def test_stamp_last_log_commit_placeholder(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 1 - Example",
                "- Commit: abc123",
                "",
                "### Iteration 2 - Example",
                "- Commit: this commit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    changed = gdnctl._stamp_last_log_commit_placeholder(log_path, commit_sha="deadbeef")
    assert changed
    text = log_path.read_text(encoding="utf-8")
    assert "- Commit: deadbeef" in text
    assert "- Commit: this commit" not in text


def test_lint_log_rejects_this_commit_by_default(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 9 - Example",
                "- Commit: this commit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = argparse.Namespace(
        log_file=str(log_path),
        scope="last-entry",
        allow_pending=False,
        allow_this_commit=False,
    )
    rc = gdnctl.cmd_lint_log(args)
    assert rc == 1
