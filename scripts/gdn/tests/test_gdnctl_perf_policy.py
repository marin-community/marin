# Copyright The Marin Authors
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
        perf_min_train_path_drop_ms=1.0,
        perf_step_duration_improvement_margin_ms=0.25,
        perf_max_remainder_budget_increase_ms=2.0,
        perf_max_dispatch_shard_shell_increase_ms=2.0,
        perf_max_ad_wrapper_shell_increase_ms=1.0,
        perf_max_xprof_idle_increase_ms=2.0,
        perf_min_dispatch_shard_shell_drop_ms=1.0,
        perf_max_decoder_layer_shell_budget_increase_ms=2.0,
        perf_min_decoder_layer_shell_drop_ms=1.0,
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


def test_latest_iteration_hotspot_context_parses_ce_hotspot_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 12 - Candidate",
                "- `CE-attributed while`: `31.600 ms -> 28.500 ms`",
                "- `CE control budget`: `31.600 ms -> 28.510 ms`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context = gdnctl._latest_iteration_hotspot_context(log_path)
    current = context["hotspot_metrics"]
    assert current["ce_attributed_while_ms"] == 28.5
    assert current["ce_control_budget_ms"] == 28.51


def test_collect_profile_metrics_parses_ce_backend_selection() -> None:
    args = argparse.Namespace(
        perf_metric="throughput/mfu",
        perf_aggregation="summary",
        perf_history_step_start=10,
        perf_history_step_end=18,
        perf_history_aggregation="median",
        perf_history_min_points=5,
        perf_wandb_entity="marin-community",
        perf_wandb_project="marin",
        validation_profile_wandb_mode="offline",
        ce_implementation="pallas_tpu",
        ce_bwd_mode="xla_streaming",
    )
    info = gdnctl._collect_profile_metrics(
        args,
        output_text=(
            "INFO Fused cross-entropy selected implementation: pallas_tpu\n"
            "throughput/mfu=5.1\n"
            "throughput/duration=0.167\n"
        ),
        profile_prefix="demo",
    )
    assert info["ce_backend_selected"] == "pallas_tpu"
    assert info["ce_requested_implementation"] == "pallas_tpu"
    assert info["ce_bwd_mode"] == "xla_streaming"
    metrics = info["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["step_duration_ms"] == 167.0


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


def test_latest_iteration_hotspot_context_parses_step_and_remainder_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 13 - Candidate",
                "- `Kernel budget`: `33.789 ms -> 29.078 ms`",
                "- `Control budget`: `10.176 ms -> 10.095 ms`",
                "- `Train-path budget`: `43.965 ms -> 39.173 ms`",
                "- `Step duration`: `167.098 ms -> 168.073 ms`",
                "- `Remainder budget`: `123.133 ms -> 128.900 ms`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context = gdnctl._latest_iteration_hotspot_context(log_path)
    current = context["hotspot_metrics"]
    assert current["step_duration_ms"] == 168.073
    assert current["remainder_budget_ms"] == 128.9


def test_latest_iteration_hotspot_context_parses_decoder_shell_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "hillclimb.md"
    log_path.write_text(
        "\n".join(
            [
                "### Iteration 14 - Candidate",
                "- `Decoder-layer shell budget`: `61.200 ms -> 55.000 ms`",
                "- `Hybrid generic shell delta budget`: `17.000 ms -> 14.250 ms`",
                "- `Dispatch/shard shell delta budget`: `9.000 ms -> 7.750 ms`",
                "- `AD/wrapper shell delta budget`: `4.500 ms -> 3.250 ms`",
                "- `AD shell budget`: `18.000 ms -> 16.500 ms`",
                "- `Sharding shell budget`: `24.000 ms -> 20.250 ms`",
                "- `Layout shell budget`: `7.000 ms -> 5.750 ms`",
                "- `Interaction remainder`: `54.000 ms -> 50.500 ms`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    context = gdnctl._latest_iteration_hotspot_context(log_path)
    current = context["hotspot_metrics"]
    assert current["decoder_layer_shell_budget_ms"] == 55.0
    assert current["hybrid_generic_shell_delta_budget_ms"] == 14.25
    assert current["dispatch_shard_shell_delta_ms"] == 7.75
    assert current["ad_wrapper_shell_delta_ms"] == 3.25
    assert current["ad_shell_budget_ms"] == 16.5
    assert current["sharding_shell_budget_ms"] == 20.25
    assert current["layout_shell_budget_ms"] == 5.75
    assert current["interaction_remainder_ms"] == 50.5


def test_perf_policy_reverts_off_critical_path_candidate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "forward_closed_call_ms": 20.663,
        "backward_closed_call_ms": 13.126,
        "while_ms": 10.150,
        "conditional_ms": 0.026,
        "kernel_budget_ms": 33.789,
        "control_budget_ms": 10.176,
        "train_path_budget_ms": 43.965,
        "step_duration_ms": 167.098,
        "remainder_budget_ms": 123.133,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.061863, "throughput/duration": 0.167098},
            "warnings": [],
            "hotspot_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "forward_closed_call_ms": 15.952,
        "backward_closed_call_ms": 13.126,
        "while_ms": 10.081,
        "conditional_ms": 0.014,
        "kernel_budget_ms": 29.078,
        "control_budget_ms": 10.095,
        "train_path_budget_ms": 39.173,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.026688, "throughput/duration": 0.168073},
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
    reasons = last["control_gate_reasons"]
    assert isinstance(reasons, list)
    assert any("off-critical-path" in reason for reason in reasons)


def test_perf_policy_reverts_wrong_boundary_progress_candidate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "train_path_budget_ms": 42.720,
        "decoder_layer_shell_budget_ms": 64.000,
        "step_duration_ms": 166.300,
        "remainder_budget_ms": 123.580,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.09, "throughput/duration": 0.1663},
            "warnings": [],
            "hotspot_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "train_path_budget_ms": 40.900,
        "decoder_layer_shell_budget_ms": 63.700,
        "step_duration_ms": 166.420,
        "remainder_budget_ms": 125.520,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.08, "throughput/duration": 0.16642},
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
    reasons = last["control_gate_reasons"]
    assert isinstance(reasons, list)
    assert any("off-critical-path or overlap-loss" in reason for reason in reasons)


def test_perf_policy_reverts_namespace_only_progress_candidate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "train_path_budget_ms": 42.700,
        "decoder_layer_shell_budget_ms": 20.400,
        "hybrid_generic_shell_delta_budget_ms": 14.200,
        "interaction_remainder_ms": 53.100,
        "step_duration_ms": 166.300,
        "remainder_budget_ms": 123.600,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.09, "throughput/duration": 0.1663},
            "warnings": [],
            "hotspot_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "train_path_budget_ms": 40.900,
        "decoder_layer_shell_budget_ms": 20.000,
        "hybrid_generic_shell_delta_budget_ms": 14.150,
        "interaction_remainder_ms": 55.500,
        "step_duration_ms": 166.420,
        "remainder_budget_ms": 125.520,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.08, "throughput/duration": 0.16642},
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
    reasons = last["control_gate_reasons"]
    assert isinstance(reasons, list)
    assert any("namespace-only / renamed-bucket progress" in reason for reason in reasons)


def test_perf_policy_rejects_dispatch_shard_shell_regression(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "dispatch_shard_shell_delta_ms": 9.8,
        "ad_wrapper_shell_delta_ms": 6.1,
        "interaction_remainder_ms": 47.9,
        "step_duration_ms": 167.7,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.03, "throughput/duration": 0.1677},
            "warnings": [],
            "hotspot_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "dispatch_shard_shell_delta_ms": 12.4,
        "ad_wrapper_shell_delta_ms": 6.0,
        "interaction_remainder_ms": 47.8,
        "step_duration_ms": 167.6,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.04, "throughput/duration": 0.1676},
            "warnings": [],
            "hotspot_metrics": candidate_hotspots,
            "hotspot_baseline_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert count_failure
    assert rc == 0
    state = _read_state(state_path)
    reasons = state["history"][-1]["control_gate_reasons"]
    assert any("dispatch_shard_shell_delta_ms" in reason for reason in reasons)


def test_perf_policy_rejects_waiting_dominant_candidate(tmp_path: Path) -> None:
    repo, baseline_commit = _init_repo(tmp_path)
    state_path = tmp_path / "perf_state.json"
    args = _perf_args(regression_policy="revert-count-failure")

    baseline_hotspots = {
        "train_path_budget_ms": 42.7,
        "dispatch_shard_shell_delta_ms": 9.8,
        "ad_wrapper_shell_delta_ms": 6.1,
        "interaction_remainder_ms": 47.9,
        "xprof_idle_attributed_ms": 38.1,
        "step_duration_ms": 167.7,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=1,
        commit_sha=baseline_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.03, "throughput/duration": 0.1677},
            "warnings": [],
            "hotspot_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert not count_failure
    assert rc == 0

    candidate_commit = _commit_change(repo, "candidate\n", "candidate")
    candidate_hotspots = {
        "train_path_budget_ms": 40.5,
        "dispatch_shard_shell_delta_ms": 9.5,
        "ad_wrapper_shell_delta_ms": 6.0,
        "interaction_remainder_ms": 47.8,
        "xprof_idle_attributed_ms": 38.5,
        "step_duration_ms": 167.8,
    }

    ok, count_failure, rc = gdnctl._apply_performance_policy(
        args,
        workdir=repo,
        perf_state_path=state_path,
        iteration=2,
        commit_sha=candidate_commit,
        validation_info={
            "metrics": {"throughput/mfu": 6.02, "throughput/duration": 0.1678},
            "warnings": [],
            "hotspot_metrics": candidate_hotspots,
            "hotspot_baseline_metrics": baseline_hotspots,
        },
    )
    assert ok
    assert count_failure
    assert rc == 0
    state = _read_state(state_path)
    reasons = state["history"][-1]["control_gate_reasons"]
    assert any("waiting/serialization still dominant" in reason for reason in reasons)


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
