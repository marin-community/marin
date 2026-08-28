# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run


def test_diagnostic_run_matches_the_d6144_rack_local_recipe():
    diagnostic = build_diagnostic_run(
        run_id="test-diagnostic",
        dp_racks=1,
        num_steps=1,
        schedule_steps=390_251,
        version="dev",
    )
    ladder = build_ladder_run(run_id="test-ladder", size="d6144", version="dev")
    diagnostic_config = diagnostic.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=diagnostic.runtime_args, deps=diagnostic.deps)
    )
    ladder_config = ladder.build_config(
        StepContext.for_fingerprint(runtime_arg_keys=ladder.runtime_args, deps=ladder.deps)
    )

    assert diagnostic_config.model == ladder_config.model
    assert diagnostic_config.processes_per_task == ladder_config.processes_per_task
    assert diagnostic_config.tensorstore_cache_bytes == ladder_config.tensorstore_cache_bytes
    assert diagnostic_config.trainer == dataclasses.replace(
        ladder_config.trainer,
        trainer=diagnostic_config.trainer.trainer,
        replica_axis_size=1,
        save_checkpoints=False,
    )
    assert diagnostic_config.trainer.trainer == dataclasses.replace(
        ladder_config.trainer.trainer,
        id=diagnostic_config.trainer.trainer.id,
        train_batch_size=diagnostic_config.trainer.trainer.train_batch_size,
        profiler=diagnostic_config.trainer.trainer.profiler,
        tracker=diagnostic_config.trainer.trainer.tracker,
        progress_watchdog=diagnostic_config.trainer.trainer.progress_watchdog,
        checkpointer=diagnostic_config.trainer.trainer.checkpointer,
        load_checkpoint_path=diagnostic_config.trainer.trainer.load_checkpoint_path,
    )
    assert diagnostic_config.data.target_budget is ladder_config.data.target_budget is None
    assert diagnostic_config.data.experiment_budget is ladder_config.data.experiment_budget is None
    assert diagnostic_config.data.train_weights == [
        (step, {name: weight for name, weight in weights.items() if weight > 0})
        for step, weights in ladder_config.data.train_weights
    ]


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


def test_scaling_ladder_searches_cluster_and_data_local_temp_roots(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    monkeypatch.setenv("MARIN_TEMP_PREFIX", "s3://hero-checkpoints")
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
