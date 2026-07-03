# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.source_push_inbox_profiles import (
    SOURCE_PUSH_PROFILE_STABLE_216,
    SOURCE_PUSH_PROFILES,
    source_push_profile_defaults,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_inbox.py"
REPRO_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bench" / "repro_source_push_inbox_queue.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_source_push_inbox", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
source_push_cli = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(source_push_cli)

DIAGNOSTIC_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "bench" / "bench_source_push_inbox_diagnostics.py"
)
DIAGNOSTIC_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "bench_source_push_inbox_diagnostics",
    DIAGNOSTIC_SCRIPT_PATH,
)
assert DIAGNOSTIC_SCRIPT_SPEC is not None
source_push_diagnostic_cli = importlib.util.module_from_spec(DIAGNOSTIC_SCRIPT_SPEC)
assert DIAGNOSTIC_SCRIPT_SPEC.loader is not None
DIAGNOSTIC_SCRIPT_SPEC.loader.exec_module(source_push_diagnostic_cli)


@pytest.mark.parametrize(
    ("profile", "expected_routing", "expected_send_pipeline_depth"),
    [
        (SOURCE_PUSH_PROFILE_STABLE_216, "roughly_balanced", 1),
    ],
)
def test_source_push_profile_applies_current_best_candidate_defaults(
    profile, expected_routing, expected_send_pipeline_depth
):
    args = source_push_cli.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            profile,
        ]
    )

    config, settings = source_push_inbox.source_push_inbox_profile(profile)
    config.validate()
    assert args.routing == config.routing == expected_routing
    assert args.send_pipeline_depth == config.send_pipeline_depth == expected_send_pipeline_depth
    assert args.n_groups_per_job == config.n_groups_per_job == 2
    assert args.send_worker_programs_per_peer == config.send_worker_programs_per_peer == 2
    assert args.worker_programs_per_peer == config.worker_programs_per_peer == 32
    assert args.repeat_runs == settings.repeat_runs == 48


def test_source_push_profile_allows_explicit_overrides():
    args = source_push_cli.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            SOURCE_PUSH_PROFILE_STABLE_216,
            "--routing",
            "uniform",
            "--send-pipeline-depth",
            "2",
            "--send-worker-programs-per-peer",
            "1",
            "--worker-programs-per-peer",
            "16",
            "--repeat-runs",
            "3",
        ]
    )

    assert args.routing == "uniform"
    assert args.send_pipeline_depth == 2
    assert args.send_worker_programs_per_peer == 1
    assert args.worker_programs_per_peer == 16
    assert args.repeat_runs == 3
    assert args.n_groups_per_job == 2


def test_source_push_profile_defaults_are_copied():
    defaults = source_push_profile_defaults(SOURCE_PUSH_PROFILE_STABLE_216)
    defaults["routing"] = "uniform"

    fresh_defaults = source_push_profile_defaults(SOURCE_PUSH_PROFILE_STABLE_216)

    assert fresh_defaults["routing"] == "roughly_balanced"


def test_source_push_profile_returns_typed_config_and_run_settings():
    config, settings = source_push_inbox.source_push_inbox_profile(SOURCE_PUSH_PROFILE_STABLE_216)

    config.validate()
    assert config.routing == "roughly_balanced"
    assert config.n_groups_per_job == 2
    assert config.send_pipeline_depth == 1
    assert config.send_worker_programs_per_peer == 2
    assert config.worker_programs_per_peer == 32
    assert settings.warmup == 2
    assert settings.steps == 7
    assert settings.repeat_runs == 48
    assert not settings.check
    assert settings.separate_compile
    assert settings.progress_events


def test_source_push_profile_exposes_single_stable_candidate():
    assert SOURCE_PUSH_PROFILES == ("none", SOURCE_PUSH_PROFILE_STABLE_216)


def test_disabled_modes_are_not_public_cli_choices():
    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--receiver-schedule", "ready_scan"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--inbox-storage", "alias"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--implementation", "send_only"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--hidden-output-mode", "full"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--hidden-compute-mode", "store_zero"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--num-send-sms", "2"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--num-sms", "32"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--diagnostic-variant", "semaphore_only"])

    with pytest.raises(SystemExit):
        source_push_cli.parse_source_push_inbox_args(["--variants", "semaphore_only"])


def test_removed_experimental_modes_are_not_config_fields():
    for kwargs in (
        {"metadata_mode": "remote_slot"},
        {"receiver_schedule": "slot_group"},
        {"direct_self_compute": True},
        {"lowering_semantics": "warpgroup"},
        {"output_mode": "debug"},
        {"implementation": "send_only"},
        {"hidden_output_mode": "full"},
        {"hidden_compute_mode": "store_zero"},
        {"num_send_sms": 2},
        {"num_sms": 32},
    ):
        with pytest.raises(TypeError):
            source_push_inbox.PushInboxConfig(**kwargs)


def test_removed_send_pipeline_depths_are_rejected_by_config_validation():
    for kwargs in ({"send_pipeline_depth": 3},):
        with pytest.raises(ValueError):
            source_push_inbox.PushInboxConfig(**kwargs).validate()


def test_compact_routing_inputs_match_synthetic_queue_metadata():
    config = source_push_inbox.PushInboxConfig(
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=4,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        send_worker_programs_per_peer=1,
        worker_programs_per_peer=4,
        routing="balanced",
        tokens_per_rank=8,
        topk=2,
    )

    synthetic_inputs = source_push_inbox._make_routing_inputs(config)
    compact_inputs = source_push_inbox._make_compact_routing_inputs(config)

    assert np.array_equal(compact_inputs.send_meta, synthetic_inputs.send_meta)
    assert np.array_equal(compact_inputs.recv_meta, synthetic_inputs.recv_meta)
    assert compact_inputs.queue_stats["input_mode"] == "compact_routing"
    assert compact_inputs.queue_stats["dropped_entries_total"] == 0
    assert (
        compact_inputs.queue_stats["compact_pack_rows_total"] == config.ep_size * config.tokens_per_rank * config.topk
    )
    assert not np.all(compact_inputs.x[0, 0, 0, 0, :] == compact_inputs.x[0, 0, 0, 0, 0])


def test_source_push_package_private_runner_returns_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_inbox.run_source_push_inbox(
        config,
        warmup=0,
        steps=1,
        repeat_runs=1,
        check=False,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_inbox"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_repro_wrapper_imports_active_bench_cli():
    result = subprocess.run(
        [sys.executable, str(REPRO_SCRIPT_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_push_diagnostic_runner_tags_structured_validation_errors():
    config = source_push_inbox.PushInboxConfig(ep_size=1)

    rows = source_push_inbox.run_source_push_inbox_diagnostic(
        config,
        diagnostic_variant="semaphore_only",
        warmup=0,
        steps=1,
        repeat_runs=1,
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_inbox_diagnostic"
    assert rows[0]["implementation"] == "source_push_inbox_diagnostic:semaphore_only"
    assert rows[0]["diagnostic_variant"] == "semaphore_only"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_cli_runs_every_send_pipeline_depth_sweep_value(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_inbox(config, **kwargs):
        calls.append((config.send_pipeline_depth, kwargs["repeat_runs"]))
        return [{"send_pipeline_depth": config.send_pipeline_depth, "repeat_runs": kwargs["repeat_runs"]}]

    monkeypatch.setattr(source_push_cli, "run_source_push_inbox", fake_run_source_push_inbox)

    source_push_cli.main(
        [
            "--sweep-send-pipeline-depth",
            "1,2",
            "--repeat-runs",
            "1",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(1, 1), (2, 1)]
    assert [row["send_pipeline_depth"] for row in rows] == [1, 2]


def test_source_push_diagnostic_cli_runs_requested_variants(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_inbox_diagnostic(config, **kwargs):
        calls.append((kwargs["diagnostic_variant"], kwargs["repeat_runs"], kwargs["compact_routing"]))
        return [
            {
                "kernel": "source_push_inbox_diagnostic",
                "implementation": f"source_push_inbox_diagnostic:{kwargs['diagnostic_variant']}",
                "diagnostic_variant": kwargs["diagnostic_variant"],
                "config": {"entries_per_rank": config.entries_per_rank},
                "queue_stats": {},
                "repeat_run": 0,
                "repeat_runs": kwargs["repeat_runs"],
                "steady_state_time": 1.0,
                "w13_tflops_per_rank": 2.0,
                "send_gbps_per_rank": 3.0,
                "compile_time": 4.0,
                "lower_compile_time": 5.0,
                "first_run_time": 6.0,
                "error_type": None,
                "error": None,
            }
        ]

    monkeypatch.setattr(
        source_push_diagnostic_cli,
        "run_source_push_inbox_diagnostic",
        fake_run_source_push_inbox_diagnostic,
    )

    source_push_diagnostic_cli.main(
        [
            "--variants",
            "semaphore_only,copy_release_only",
            "--repeat-runs",
            "3",
            "--compact-routing",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [("semaphore_only", 3, True), ("copy_release_only", 3, True)]
    assert [row["row_type"] for row in rows] == ["repeat", "summary", "repeat", "summary"]
    assert [row["diagnostic_variant"] for row in rows] == [
        "semaphore_only",
        "semaphore_only",
        "copy_release_only",
        "copy_release_only",
    ]
