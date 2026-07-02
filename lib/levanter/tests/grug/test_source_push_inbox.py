# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import importlib.util
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
SCRIPT_SPEC = importlib.util.spec_from_file_location("bench_source_push_inbox", SCRIPT_PATH)
assert SCRIPT_SPEC is not None
source_push_cli = importlib.util.module_from_spec(SCRIPT_SPEC)
assert SCRIPT_SPEC.loader is not None
SCRIPT_SPEC.loader.exec_module(source_push_cli)


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
    assert args.implementation == config.implementation == "m_n_slots"
    assert args.hidden_output_mode == config.hidden_output_mode == "queue"
    assert args.n_groups_per_job == config.n_groups_per_job == 2
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
            "--repeat-runs",
            "3",
        ]
    )

    assert args.routing == "uniform"
    assert args.send_pipeline_depth == 2
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
    assert config.hidden_output_mode == "queue"
    assert config.n_groups_per_job == 2
    assert config.send_pipeline_depth == 1
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


def test_removed_experimental_modes_are_not_config_fields():
    for kwargs in (
        {"metadata_mode": "remote_slot"},
        {"receiver_schedule": "slot_group"},
        {"direct_self_compute": True},
        {"lowering_semantics": "warpgroup"},
        {"output_mode": "debug"},
    ):
        with pytest.raises(TypeError):
            source_push_inbox.PushInboxConfig(**kwargs)


def test_removed_send_pipeline_depths_are_rejected_by_config_validation():
    for kwargs in (
        {"implementation": "m_owner"},
        {"send_pipeline_depth": 3},
    ):
        with pytest.raises(ValueError):
            source_push_inbox.PushInboxConfig(**kwargs).validate()


def test_compact_routing_inputs_match_synthetic_queue_metadata():
    config = source_push_inbox.PushInboxConfig(
        implementation="m_n_slots",
        ep_size=2,
        entries_per_rank=2,
        inbox_slots=2,
        hidden_dim=8,
        intermediate_dim=8,
        block_m=4,
        block_k=4,
        block_n=4,
        experts_per_rank=2,
        num_send_sms=1,
        num_sms=4,
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
