# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import pytest

from scripts.gdn import gdnctl


def test_parse_profile_env_parses_key_value_pairs() -> None:
    parsed = gdnctl._parse_profile_env(
        [
            "GDN_TRIANGULAR_SOLVE_PROBE=identity",
            "WANDB_MODE=offline",
            "EMPTY_VALUE=",
        ]
    )
    assert parsed == [
        ("GDN_TRIANGULAR_SOLVE_PROBE", "identity"),
        ("WANDB_MODE", "offline"),
        ("EMPTY_VALUE", ""),
    ]


def test_append_profile_env_includes_ce_override_only_when_requested() -> None:
    base_args = argparse.Namespace(
        wandb_mode="online",
        size="130m",
        num_steps=20,
        profile_start_step=2,
        profile_num_steps=6,
        run_name_prefix="demo",
        tpu="v5p-8",
        batch_size=8,
        chunk_size=None,
        segment_size=None,
        profile_env=[],
        ce_bwd_mode="pallas",
    )

    cmd = ["execute"]
    args = argparse.Namespace(**vars(base_args), ce_implementation="pallas_tpu")
    gdnctl._append_profile_env(cmd, args)
    assert f"{gdnctl.FUSED_CE_IMPLEMENTATION_ENV}=pallas_tpu" in cmd
    assert f"{gdnctl.PALLAS_TPU_CE_BWD_STREAMING_ENV}=0" in cmd

    cmd_auto = ["execute"]
    args_auto = argparse.Namespace(**vars(base_args), ce_implementation="auto")
    gdnctl._append_profile_env(cmd_auto, args_auto)
    assert not any(item.startswith(f"{gdnctl.FUSED_CE_IMPLEMENTATION_ENV}=") for item in cmd_auto)
    assert f"{gdnctl.PALLAS_TPU_CE_BWD_STREAMING_ENV}=0" in cmd_auto


def test_append_profile_env_supports_xla_streaming_ce_backward_mode() -> None:
    args = argparse.Namespace(
        wandb_mode="offline",
        size="130m",
        num_steps=20,
        profile_start_step=2,
        profile_num_steps=6,
        run_name_prefix="demo",
        tpu="v5p-8",
        batch_size=8,
        chunk_size=None,
        segment_size=None,
        profile_env=[],
        ce_implementation="pallas_tpu",
        ce_bwd_mode="xla_streaming",
    )

    cmd = ["execute"]
    gdnctl._append_profile_env(cmd, args)
    assert f"{gdnctl.FUSED_CE_IMPLEMENTATION_ENV}=pallas_tpu" in cmd
    assert f"{gdnctl.PALLAS_TPU_CE_BWD_STREAMING_ENV}=1" in cmd


def test_append_profile_env_supports_model_boundary_overrides() -> None:
    args = argparse.Namespace(
        wandb_mode="offline",
        size="130m",
        num_steps=20,
        profile_start_step=2,
        profile_num_steps=6,
        run_name_prefix="demo",
        tpu="v5p-8",
        batch_size=8,
        chunk_size=128,
        segment_size=16,
        gdn_layers_per_block=1,
        gdn_block_size=4,
        all_transformer=False,
        profile_env=[],
        ce_implementation="pallas_tpu",
        ce_bwd_mode="pallas",
    )

    cmd = ["execute"]
    gdnctl._append_profile_env(cmd, args)
    assert "GDN_PROFILE_GDN_LAYERS_PER_BLOCK=1" in cmd
    assert "GDN_PROFILE_GDN_BLOCK_SIZE=4" in cmd
    assert "GDN_PROFILE_ALL_TRANSFORMER=1" not in cmd


def test_append_profile_env_supports_attention_only_flag() -> None:
    args = argparse.Namespace(
        wandb_mode="offline",
        size="130m",
        num_steps=20,
        profile_start_step=2,
        profile_num_steps=6,
        run_name_prefix="demo",
        tpu="v5p-8",
        batch_size=8,
        chunk_size=None,
        segment_size=None,
        gdn_layers_per_block=None,
        gdn_block_size=None,
        all_transformer=True,
        profile_env=[],
        ce_implementation="pallas_tpu",
        ce_bwd_mode="pallas",
    )

    cmd = ["execute"]
    gdnctl._append_profile_env(cmd, args)
    assert "GDN_PROFILE_ALL_TRANSFORMER=1" in cmd


@pytest.mark.parametrize(
    ("cluster", "expected"),
    [
        ("us-east5-a", "gs://marin-us-east5"),
        ("us-east5", "gs://marin-us-east5"),
        ("marin-us-central1", "gs://marin-us-central1"),
    ],
)
def test_default_marin_prefix_uses_bucket_mapping(cluster: str, expected: str) -> None:
    assert gdnctl._default_marin_prefix(cluster) == expected


@pytest.mark.parametrize(
    "item",
    [
        "MISSING_EQUALS",
        "1BAD=value",
        "=value",
    ],
)
def test_parse_profile_env_rejects_invalid_entries(item: str) -> None:
    with pytest.raises(SystemExit):
        gdnctl._parse_profile_env([item])


def test_validation_mode_profile_only_skips_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"tests": False, "profile": False}

    def _fake_tests(_: argparse.Namespace) -> tuple[int, bool]:
        called["tests"] = True
        return 0, True

    def _fake_profile(_: argparse.Namespace, *, iteration: int) -> tuple[int, bool, dict[str, object]]:
        called["profile"] = True
        return 0, True, {"profile_prefix": f"iter-{iteration:03d}", "metrics": {"throughput/mfu": 1.23}}

    monkeypatch.setattr(gdnctl, "_run_validation_tests_once", _fake_tests)
    monkeypatch.setattr(gdnctl, "_run_validation_profile_once", _fake_profile)

    args = argparse.Namespace(
        validation_mode="profile-only",
        validation_max_attempts=1,
        validation_retry_sleep=0.0,
    )
    ok, rc, info = gdnctl._run_validation_gate_for_iteration(args, iteration=7)
    assert ok
    assert rc == 0
    assert info["profile_prefix"] == "iter-007"
    assert called["profile"]
    assert not called["tests"]


def test_validation_gate_records_full_parity_test_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_tests(_: argparse.Namespace) -> tuple[int, bool]:
        return 0, True

    def _fake_profile(_: argparse.Namespace, *, iteration: int) -> tuple[int, bool, dict[str, object]]:
        return 0, True, {"profile_prefix": f"iter-{iteration:03d}", "metrics": {"throughput/mfu": 1.23}}

    monkeypatch.setattr(gdnctl, "_run_validation_tests_once", _fake_tests)
    monkeypatch.setattr(gdnctl, "_run_validation_profile_once", _fake_profile)

    args = argparse.Namespace(
        validation_mode="required",
        validation_max_attempts=1,
        validation_retry_sleep=0.0,
        validation_tests="both",
    )
    ok, rc, info = gdnctl._run_validation_gate_for_iteration(args, iteration=9)
    assert ok
    assert rc == 0
    assert info["validation_tests_full_parity"] is True
    assert info["validation_test_dependencies"] == ["torch", "transformers"]
    assert info["validation_test_targets"] == [gdnctl.GDN_KERNEL_TEST, gdnctl.GDN_LAYER_TEST]
    assert info["profile_prefix"] == "iter-009"


def test_validation_tests_reacquire_dev_tpu_before_ray_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_reacquire(args: argparse.Namespace, *, reason: str) -> bool:
        calls.append(reason)
        args.active_dev_tpu_cluster = "us-east5-a"
        return True

    def _fake_dev_test(args: argparse.Namespace) -> int:
        assert args.cluster == "us-east5-a"
        return 0

    def _unexpected_ray(*_args: object, **_kwargs: object) -> tuple[int, bool]:
        raise AssertionError("Ray fallback should not run when dev TPU re-acquire succeeds.")

    monkeypatch.setattr(gdnctl, "_try_reacquire_managed_dev_tpu", _fake_reacquire)
    monkeypatch.setattr(gdnctl, "cmd_dev_tpu_test", _fake_dev_test)
    monkeypatch.setattr(gdnctl, "_run_validation_ray_test_once", _unexpected_ray)

    args = argparse.Namespace(
        dev_tpu_name="calvinxu-gdn",
        hold_dev_tpu=True,
        validation_tests="both",
        validation_pytest_args=None,
        validation_dev_no_sync=True,
    )
    rc, retryable = gdnctl._run_validation_tests_once(args)
    assert rc == 0
    assert retryable is True
    assert calls == ["validation tests starting without an active held dev TPU"]


def test_validation_tests_cycle_back_to_dev_tpu_after_ray_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    reacquire_calls: list[str] = []

    def _fake_reacquire(args: argparse.Namespace, *, reason: str) -> bool:
        reacquire_calls.append(reason)
        if "starting without" in reason:
            return False
        args.active_dev_tpu_cluster = "us-central1"
        return True

    def _fake_ray_clusters(_args: argparse.Namespace, *, purpose: str) -> list[str]:
        assert purpose == "test"
        return ["us-east5-a"]

    def _fake_ray_test(_args: argparse.Namespace, *, cluster: str) -> tuple[int, bool]:
        assert cluster == "us-east5-a"
        return 1, True

    def _fake_dev_test(args: argparse.Namespace) -> int:
        assert args.cluster == "us-central1"
        return 0

    monkeypatch.setattr(gdnctl, "_try_reacquire_managed_dev_tpu", _fake_reacquire)
    monkeypatch.setattr(gdnctl, "_validation_ray_clusters", _fake_ray_clusters)
    monkeypatch.setattr(gdnctl, "_run_validation_ray_test_once", _fake_ray_test)
    monkeypatch.setattr(gdnctl, "cmd_dev_tpu_test", _fake_dev_test)

    args = argparse.Namespace(
        dev_tpu_name="calvinxu-gdn",
        hold_dev_tpu=True,
        validation_tests="both",
        validation_pytest_args=None,
        validation_dev_no_sync=False,
    )
    rc, retryable = gdnctl._run_validation_tests_once(args)
    assert rc == 0
    assert retryable is True
    assert reacquire_calls == [
        "validation tests starting without an active held dev TPU",
        "validation tests Ray fallback failed",
    ]


def test_validation_ray_profile_stops_job_after_wait_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[tuple[str, str]] = []

    monkeypatch.setattr(
        gdnctl,
        "_submit_ray_job",
        lambda _cmd: (0, "Job submitted with ID: ray-run-demo\n", "ray-run-demo"),
    )
    monkeypatch.setattr(
        gdnctl,
        "_wait_for_ray_job",
        lambda *, cluster, job_id, timeout_seconds, tail: (
            124,
            f"timed out waiting for {job_id} on {cluster} timeout={timeout_seconds} tail={tail}",
        ),
    )
    monkeypatch.setattr(
        gdnctl,
        "_stop_ray_job_best_effort",
        lambda *, cluster, job_id: stopped.append((cluster, job_id)),
    )

    args = argparse.Namespace(
        validation_profile_tpu="v5p-8",
        validation_profile_wandb_mode="online",
        validation_profile_size="130m",
        validation_profile_num_steps=20,
        validation_profile_start_step=2,
        validation_profile_num_steps_window=6,
        validation_profile_batch_size=8,
        validation_profile_chunk_size=None,
        validation_profile_segment_size=None,
        validation_profile_gdn_layers_per_block=None,
        validation_profile_gdn_block_size=None,
        validation_profile_all_transformer=False,
        validation_profile_env=[],
        validation_profile_dry_run=False,
        validation_ray_profile_timeout=123.0,
        validation_ray_log_tail=77,
    )
    rc, retryable, output = gdnctl._run_validation_ray_profile_once(
        args,
        cluster="us-east5-a",
        run_name_prefix="demo",
        ce_implementation="pallas_tpu",
        ce_bwd_mode="pallas",
    )
    assert rc == 124
    assert retryable is True
    assert "ray-run-demo" in output
    assert stopped == [("us-east5-a", "ray-run-demo")]


def test_collect_profile_metrics_records_profile_architecture_metadata() -> None:
    args = argparse.Namespace(
        perf_metric="throughput/mfu",
        perf_aggregation="summary",
        perf_history_step_start=10,
        perf_history_step_end=18,
        perf_history_aggregation="median",
        perf_history_min_points=5,
        perf_wandb_entity="marin-community",
        perf_wandb_project="marin",
        validation_profile_wandb_mode="disabled",
        ce_implementation="pallas_tpu",
        ce_bwd_mode="pallas",
        gdn_layers_per_block=1,
        gdn_block_size=4,
        all_transformer=False,
    )

    info = gdnctl._collect_profile_metrics(
        args,
        output_text=(
            "throughput/mfu=6.0\nthroughput/duration=0.1\nFused cross-entropy selected implementation: pallas_tpu\n"
        ),
        profile_prefix="demo",
    )
    assert info["profile_architecture"] == "hybrid"
    assert info["gdn_layers_per_block"] == 1
    assert info["gdn_block_size"] == 4
    assert info["gdn_layer_fraction"] == pytest.approx(0.25)
    assert info["metrics"]["step_duration_ms"] == pytest.approx(100.0)


def test_collect_profile_metrics_prefers_reported_model_boundary_metadata() -> None:
    args = argparse.Namespace(
        perf_metric="throughput/mfu",
        perf_aggregation="summary",
        perf_history_step_start=10,
        perf_history_step_end=18,
        perf_history_aggregation="median",
        perf_history_min_points=5,
        perf_wandb_entity="marin-community",
        perf_wandb_project="marin",
        validation_profile_wandb_mode="disabled",
        ce_implementation="pallas_tpu",
        ce_bwd_mode="pallas",
        gdn_layers_per_block=None,
        gdn_block_size=None,
        all_transformer=False,
    )

    output_text = (
        "[gdnctl] GDN profile model: all_transformer=1 gdn_layers_per_block=0 "
        "gdn_block_size=4 gdn_layer_fraction=0.000000\n"
        "throughput/mfu=21.0\nthroughput/duration=0.05\n"
    )
    info = gdnctl._collect_profile_metrics(args, output_text=output_text, profile_prefix="demo")
    assert info["profile_architecture"] == "attn_only"
    assert info["gdn_layers_per_block"] == 0
    assert info["gdn_block_size"] == 4
    assert info["gdn_layer_fraction"] == 0.0


def test_with_step_and_remainder_metrics_adds_upper_bound_gap() -> None:
    hotspot_metrics = {
        "train_path_budget_ms": 42.7,
        "step_duration_ms": 166.307,
    }
    augmented = gdnctl._with_step_and_remainder_metrics(
        hotspot_metrics,
        profile_metrics={},
        upper_bound_step_ms=57.8605,
    )
    assert augmented["remainder_budget_ms"] == pytest.approx(123.607)
    assert augmented["upper_bound_gap_ms"] == pytest.approx(108.4465)
    assert augmented["gap_explained_by_train_path"] == pytest.approx(42.7 / 108.4465)
