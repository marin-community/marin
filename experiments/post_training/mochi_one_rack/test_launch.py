# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.mochi_one_rack.launch import build_smoke


def test_smoke_restores_the_first_stage_and_keeps_both_context_caps() -> None:
    smoke = build_smoke("2026.11.11.9263")
    launches = []
    for step in (smoke.four_k, smoke.thirty_two_k):
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        launches.append(yaml.safe_load(config.launch_config_yaml))
    first, resumed = launches
    assert first["iris"]["allocation"]["num_nodes"] == resumed["iris"]["allocation"]["num_nodes"] == 15
    assert first["skyrl"]["context_budget"]["request_window_tokens"] == 4096
    assert resumed["skyrl"]["context_budget"]["request_window_tokens"] == 32768
    assert first["skyrl"]["trainer"]["max_steps"] == 5
    assert resumed["skyrl"]["trainer"]["max_steps"] == 10
    assert resumed["skyrl"]["trainer"]["resume_mode"] == "latest"
    assert "mochi-one-rack/4k@2026.11.11.9263" in resumed["artifacts"]["checkpoint_root"]
    assert smoke.four_k in smoke.thirty_two_k.deps
    assert not first["run"]["export_hf"] and not resumed["run"]["export_hf"]
