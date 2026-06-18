# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig
from marin.execution.types import ExecutorStep
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig

import experiments.domain_phase_mix.launch_single_phase_average_qsplit240_300m_6b as single_phase_qsplit240_300m
import experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b as qsplit240_300m
import experiments.domain_phase_mix.qsplit240_replay as qsplit240_replay


def test_single_phase_qsplit240_300m_builds_exposure_average_manifest():
    run_specs = single_phase_qsplit240_300m.build_run_specs(panel=qsplit240_replay.REPRESENTATIVE12_PANEL)
    source_specs = qsplit240_300m.build_run_specs(panel=qsplit240_replay.REPRESENTATIVE12_PANEL)

    assert len(run_specs) == len(source_specs) == 12
    assert [spec.source_run_name for spec in run_specs] == [spec.run_name for spec in source_specs]
    assert [spec.run_name for spec in run_specs] == [f"singleavg_{spec.run_name}" for spec in source_specs]
    assert {spec.cohort for spec in run_specs} == {"single_phase_exposure_average_qsplit240_300m"}
    assert {spec.model_family for spec in run_specs} == {qsplit240_300m.MODEL_FAMILY}
    assert {spec.experiment_budget for spec in run_specs} == {qsplit240_300m.EXPERIMENT_BUDGET}
    assert {spec.num_train_steps for spec in run_specs} == {qsplit240_300m.NUM_TRAIN_STEPS}
    assert {spec.single_phase_strategy for spec in run_specs} == {"exposure_average_80_20"}
    assert {spec.source_two_phase_experiment for spec in run_specs} == {qsplit240_300m.NAME}
    assert all(spec.data_seed == spec.source_run_id for spec in run_specs)

    for run_spec, source_spec in zip(run_specs, source_specs, strict=True):
        phase_0 = run_spec.phase_weights["phase_0"]
        phase_1 = run_spec.phase_weights["phase_1"]
        assert phase_0 == phase_1

        expected = {}
        for domain_name in phase_0:
            expected[domain_name] = (
                0.8 * source_spec.phase_weights["phase_0"][domain_name]
                + 0.2 * source_spec.phase_weights["phase_1"][domain_name]
            )
        expected_sum = sum(expected.values())
        expected = {domain_name: value / expected_sum for domain_name, value in expected.items()}
        assert phase_0 == pytest.approx(expected)
        assert run_spec.phase_tv == pytest.approx(
            0.5
            * sum(
                abs(source_spec.phase_weights["phase_1"][domain] - source_spec.phase_weights["phase_0"][domain])
                for domain in phase_0
            )
        )


def test_single_phase_qsplit240_300m_launch_graph_is_east5_perplexity_only_by_default(monkeypatch):
    class DummyExperiment:
        def create_training_step(self, *, name_prefix, run_name, **_):
            config = TrainLmOnPodConfig(
                train_config=SimpleNamespace(
                    trainer=SimpleNamespace(num_train_steps=qsplit240_300m.NUM_TRAIN_STEPS)
                ),
                resources=ResourceConfig(cpu=1, ram="1g"),
                env_vars={},
            )
            return ExecutorStep(
                name=f"{name_prefix}/{run_name}",
                fn=lambda _: None,
                config=config,
            )

    monkeypatch.setattr(
        single_phase_qsplit240_300m,
        "create_qsplit240_replay_experiment",
        lambda **_: DummyExperiment(),
    )

    artifacts = single_phase_qsplit240_300m.build_launch_artifacts(
        name_prefix=single_phase_qsplit240_300m.NAME,
        tpu_type="v5p-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
        panel=qsplit240_replay.BASELINES3_PANEL,
        limit=None,
        shard_count=1,
        shard_index=0,
        include_eval_harness=False,
        eval_datasets_cache_path=None,
    )

    assert len(artifacts.run_specs) == 3
    assert artifacts.cache_eval_datasets_step is None
    assert len(artifacts.steps) == 1 + len(artifacts.training_steps) + 2

    for training_step, run_spec in zip(artifacts.training_steps, artifacts.run_specs, strict=True):
        assert isinstance(training_step.config, TrainLmOnPodConfig)
        assert training_step.name == f"{single_phase_qsplit240_300m.NAME}/{run_spec.run_name}"
        assert training_step.config.env_vars["MARIN_PREFIX"] == marin_prefix_for_region("us-east5")
        assert training_step.config.env_vars[qsplit240_replay.SKIP_EVAL_HARNESS_ENV_VAR] == "1"
        assert training_step.config.train_config.trainer.num_train_steps == qsplit240_300m.NUM_TRAIN_STEPS
        assert qsplit240_replay.EVAL_DATASETS_CACHE_DEP_ENV_VAR not in training_step.config.env_vars
