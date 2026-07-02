# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.source_push_inbox_profiles import source_push_profile_defaults


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (
            "hopper_queue_ngroups2_210",
            {
                "routing": "uniform",
                "send_pipeline_depth": 2,
                "repeat_runs": 48,
            },
        ),
        (
            "hopper_queue_roughly_balanced_ngroups2_210",
            {
                "routing": "roughly_balanced",
                "send_pipeline_depth": 1,
                "repeat_runs": 48,
            },
        ),
    ],
)
def test_source_push_profile_applies_current_best_candidate_defaults(profile, expected):
    args = source_push_inbox.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            profile,
        ]
    )

    common_expected = {
        "implementation": "m_n_slots",
        "queue_mode": "routing",
        "traffic_pattern": "all_to_all",
        "peer_loop": "grid_switch",
        "lowering_semantics": "lane",
        "metadata_mode": "static_recv",
        "output_mode": "perf",
        "hidden_output_mode": "queue",
        "tokens_per_rank": 32768,
        "hidden_dim": 2560,
        "intermediate_dim": 1280,
        "experts_per_rank": 32,
        "topk": 4,
        "entries_per_rank": 288,
        "inbox_slots": 12,
        "num_send_sms": 2,
        "num_sms": 32,
        "block_m": 64,
        "block_k": 128,
        "block_n": 128,
        "n_group": 1,
        "n_groups_per_job": 2,
        "receiver_schedule": "fixed_wait",
        "slot_order": "current",
        "warmup": 2,
        "steps": 7,
        "check": False,
        "separate_compile": True,
        "progress_events": True,
    }
    common_expected.update(expected)
    for field, value in common_expected.items():
        assert getattr(args, field) == value


def test_source_push_profile_allows_explicit_overrides():
    args = source_push_inbox.parse_source_push_inbox_args(
        [
            "--source-push-profile",
            "hopper_queue_roughly_balanced_ngroups2_210",
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
    assert args.output_mode == "perf"


def test_source_push_profile_defaults_are_copied():
    defaults = source_push_profile_defaults("hopper_queue_ngroups2_210")
    defaults["routing"] = "roughly_balanced"

    fresh_defaults = source_push_profile_defaults("hopper_queue_ngroups2_210")

    assert fresh_defaults["routing"] == "uniform"


def test_source_push_profile_returns_typed_config_and_run_settings():
    config, settings = source_push_inbox.source_push_inbox_profile("hopper_queue_roughly_balanced_ngroups2_210")

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
    assert rows[0]["error"] == "ValueError: ep_size must be greater than 1, got 1"
    assert rows[0]["kernel"] == "source_push_inbox"
    assert rows[0]["repeat_runs"] == 1
