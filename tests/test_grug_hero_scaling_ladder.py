# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run


@pytest.mark.parametrize(
    ("size", "num_steps", "expected_simulated_epoching"),
    [("d2048", None, True), ("d6144", 1, True), ("d6144", None, False)],
)
def test_scaling_ladder_disables_simulated_epoching_above_flop_limit(size, num_steps, expected_simulated_epoching):
    step = build_ladder_run(run_id=f"test-{size}", size=size, num_steps=num_steps, version="2026.08.18")
    ctx = StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps)

    data = step.build_config(ctx).data

    assert (data.target_budget is not None) is expected_simulated_epoching
    assert (data.experiment_budget is not None) is expected_simulated_epoching


def test_scaling_ladder_searches_data_local_temp_during_cluster_temp_migration(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("MARIN_CLUSTER_TEMP_PREFIX", "s3://hero-checkpoints")
    step = build_ladder_run(run_id="test-d6144", size="d6144", num_steps=1, version="2026.08.18")
    output_path = "s3://marin-us-east-02a/marin/grug/test-d6144/v"
    ctx = dataclasses.replace(
        StepContext.for_fingerprint(runtime_arg_keys=step.runtime_args, deps=step.deps),
        output_path=output_path,
    )

    trainer = step.build_config(ctx).trainer.trainer
    assert trainer.checkpoint_search_paths("test-d6144") == [
        f"{output_path}/checkpoints",
        "s3://hero-checkpoints/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
        "s3://marin-us-east-02a/tmp/ttl=14d/checkpoints-temp/marin-us-east-02a/marin/grug/test-d6144/v/checkpoints",
    ]
