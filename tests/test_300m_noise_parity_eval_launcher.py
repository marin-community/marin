# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep

from experiments.domain_phase_mix import launch_300m_noise_parity_evals as launcher


def _noise_parity_row() -> launcher.NoiseParityEvalSpec:
    return launcher.NoiseParityEvalSpec(
        eval_key="noiseparity300m_test",
        panel="proportional_controllability_300m_6b",
        run_name="pctrl_test",
        registry_key="proportional_controllability_300m_6b:pctrl_test",
        source_experiment="source",
        cohort="proportional_controllability_300m",
        checkpoint_root="gs://marin-us-east5/checkpoints/pctrl_test",
        expected_checkpoint_step=22887,
        hf_checkpoint_count=1,
        hf_checkpoint_latest="gs://marin-us-east5/checkpoints/pctrl_test/hf/step-22887",
        hf_checkpoint_latest_step=22887,
        has_exact_hf_checkpoint=True,
        existing_tasks="",
        missing_task_count=1,
        missing_tasks="arc_easy",
        task_aliases="arc_easy",
        launch_tpu_type="v5p-8",
        launch_tpu_region="us-east5",
        launch_tpu_zone="us-east5-a",
        eligible=True,
        launch_decision="launch",
        step_name="evaluation/lm_evaluation_harness_levanter/noise_parity_noiseparity300m_test",
        result_path="executor_output:noiseparity300m_test",
    )


def test_noise_parity_eval_steps_can_force_non_preemptible_child_tpus(monkeypatch) -> None:
    captured_resource_configs: list[ResourceConfig] = []

    def fake_eval_step(
        model_name: str,
        model_path: str,
        evals: list[Any],
        resource_config: ResourceConfig,
        **kwargs: Any,
    ) -> ExecutorStep:
        del model_path, evals, kwargs
        captured_resource_configs.append(resource_config)
        fn: Callable[[object], None] = lambda _config: None
        return ExecutorStep(name=f"fake_eval/{model_name}", fn=fn, config={})

    monkeypatch.setattr(
        "experiments.evals.evals.evaluate_levanter_lm_evaluation_harness",
        fake_eval_step,
    )

    launcher.build_eval_steps(
        name_prefix="pinlin/test_noise_parity",
        state_rows=[_noise_parity_row()],
        eval_datasets_cache_path="gs://marin-us-east5/raw/eval-datasets/noise-parity-test",
        child_preemptible=False,
    )

    assert [config.preemptible for config in captured_resource_configs] == [False]
