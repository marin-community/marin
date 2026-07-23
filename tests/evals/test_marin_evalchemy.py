# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composable evalchemy launcher behavior that does not require an Iris cluster."""

import json

from marin.evaluation.eval_result import EvalchemyResult
from marin.execution.lazy import materialized_config

from experiments.evals.evalchemy.marin_evalchemy import EvalSpec, _build_config, evalchemy_step
from experiments.evals.evalchemy.serve_and_eval import ServeBackend, ServeSpec


def test_aime_seed_suite_preserves_each_seed_and_uses_chat_generation():
    config = _build_config(EvalSpec(run_name="test", model="model", seeds=(42, 43)))

    math500, first_seed, second_seed = config.tasks
    assert math500.name == "MATH500"
    assert math500.generation is True
    assert (first_seed.task_alias, first_seed.task_kwargs) == ("AIME24_seed42", {"seed": 42})
    assert (second_seed.task_alias, second_seed.task_kwargs) == ("AIME24_seed43", {"seed": 43})


def test_step_records_eval_parameters_and_returns_typed_result_handle():
    step = evalchemy_step(
        EvalSpec(
            run_name="test",
            model="model",
            stage="base",
            max_gen_toks=4096,
            extra_gen_kwargs={"repetition_penalty": "1.1"},
            serve=ServeSpec(
                backend=ServeBackend.VLLM,
                tpu_type=None,
                gpu_type="H100",
                gpu_count=8,
                vllm_extra_args=("--revision", "abc123"),
            ),
        )
    )
    config = materialized_config(step, "gs://marin-test")

    assert step.artifact_type is EvalchemyResult
    assert config["max_gen_toks"] == 4096
    assert config["extra_gen_kwargs"] == {"repetition_penalty": "1.1"}
    assert config["serve"]["vllm_extra_args"] == ["--revision", "abc123"]
    assert json.loads(step.fingerprint_payload())["stage"] == "base"
