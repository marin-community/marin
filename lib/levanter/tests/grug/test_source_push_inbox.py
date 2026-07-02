# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.source_push_inbox_profiles import source_push_profile_defaults


@pytest.mark.parametrize(
    ("profile", "expected_routing", "expected_send_pipeline_depth"),
    [
        ("hopper_queue_ngroups2_210", "uniform", 2),
        ("hopper_queue_roughly_balanced_ngroups2_210", "roughly_balanced", 1),
    ],
)
def test_source_push_profile_applies_current_best_candidate_defaults(
    profile, expected_routing, expected_send_pipeline_depth
):
    args = source_push_inbox.parse_source_push_inbox_args(
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
    assert args.metadata_mode == config.metadata_mode == "static_recv"
    assert args.hidden_output_mode == config.hidden_output_mode == "queue"
    assert args.n_groups_per_job == config.n_groups_per_job == 2
    assert args.repeat_runs == settings.repeat_runs == 48


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
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["kernel"] == "source_push_inbox"
    assert rows[0]["repeat_runs"] == 1


def test_source_push_cli_runs_every_workers_per_slot_sweep_value(monkeypatch, capsys):
    calls = []

    def fake_run_source_push_inbox(config, **kwargs):
        calls.append((config.workers_per_slot, kwargs["repeat_runs"]))
        return [{"workers_per_slot": config.workers_per_slot, "repeat_runs": kwargs["repeat_runs"]}]

    monkeypatch.setattr(source_push_inbox, "run_source_push_inbox", fake_run_source_push_inbox)

    source_push_inbox.main(
        [
            "--sweep-workers-per-slot",
            "1,2",
            "--repeat-runs",
            "1",
        ]
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert calls == [(1, 1), (2, 1)]
    assert [row["workers_per_slot"] for row in rows] == [1, 2]
