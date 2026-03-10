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
