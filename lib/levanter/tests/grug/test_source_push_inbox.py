# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import levanter.grug._moe.source_push_inbox as source_push_inbox
from levanter.grug._moe.source_push_inbox_profiles import source_push_profile_defaults


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
        source_push_inbox.SourcePushInboxRunSettings(
            warmup=0,
            steps=1,
            repeat_runs=1,
            check=False,
        ),
    )

    assert len(rows) == 1
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["steady_state_time"] is None
