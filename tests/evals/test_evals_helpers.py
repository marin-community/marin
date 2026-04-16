# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""`experiments/evals/evals.py` helpers build the right typed config.

Catches helper-level regressions that the transport/encoding tests don't see.
"""

from __future__ import annotations

import pytest

from experiments.evals.evals import (
    _EvalchemyStepConfig,
    _HarborStepConfig,
    _LmEvalStepConfig,
    _append_step_suffix,
    evaluate_evalchemy,
    evaluate_harbor,
    evaluate_lm_evaluation_harness,
)
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig


def _task() -> EvalTaskConfig:
    return EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)


def test_evaluate_lm_evaluation_harness_routes_engine_kwargs_to_deployment():
    step = evaluate_lm_evaluation_harness(
        model_name="my-model",
        model_path="gs://bucket/path/hf",
        evals=[_task()],
        deployment_kwargs={"max_model_len": 4096, "gpu_memory_utilization": 0.9},
        extra_model_args=("max_gen_toks=4096",),
        resource_config=ResourceConfig.with_cpu(cpu=1),
        apply_chat_template=False,
    )
    config = step.config
    assert isinstance(config, _LmEvalStepConfig)
    assert config.deployment.path == "gs://bucket/path/hf"
    assert config.deployment.engine_kwargs == {"max_model_len": 4096, "gpu_memory_utilization": 0.9}
    assert config.run.apply_chat_template is False
    assert config.run.extra_model_args == ("max_gen_toks=4096",)
    assert config.run.base_eval_run_name == "my-model"
    assert step.name == "evaluation/lm_evaluation_harness/my-model"


def test_evaluate_evalchemy_forwards_generation_params_and_batch_size():
    step = evaluate_evalchemy(
        model_name="my-model",
        model_path="gs://bucket/path/hf",
        evals=[EvalTaskConfig(name="AIME25", num_fewshot=0)],
        deployment_kwargs={"tensor_parallel_size": 4, "max_num_seqs": 256},
        extra_model_args=(),
        generation_params={"temperature": 0.7, "max_gen_toks": 32768, "top_p": 0.95, "seed": 42},
        batch_size="256",
        apply_chat_template=True,
        base_eval_run_name="myrun",
    )
    config = step.config
    assert isinstance(config, _EvalchemyStepConfig)
    assert config.deployment.engine_kwargs == {"tensor_parallel_size": 4, "max_num_seqs": 256}
    assert config.run.apply_chat_template is True
    assert config.run.batch_size == "256"
    assert config.run.generation_params == {"temperature": 0.7, "max_gen_toks": 32768, "top_p": 0.95, "seed": 42}
    assert config.run.base_eval_run_name == "myrun"
    # Seed is baked into the step name so different seeds produce distinct output dirs.
    assert "_seed42" in step.name


def test_evaluate_evalchemy_omits_seed_suffix_when_no_seed():
    step = evaluate_evalchemy(
        model_name="my-model",
        model_path="gs://bucket/path/hf",
        evals=[EvalTaskConfig(name="AIME25", num_fewshot=0)],
        generation_params={"temperature": 0.7},
    )
    assert "_seed" not in step.name


def test_evaluate_harbor_local_vllm_mode_builds_deployment():
    step = evaluate_harbor(
        model_name="my-vllm-served-model",
        model_path="gs://bucket/path/hf",
        dataset="aime",
        version="1.0",
        resource_config=ResourceConfig.with_tpu("v5p-8"),
    )
    config = step.config
    assert isinstance(config, _HarborStepConfig)
    assert config.deployment is not None
    assert config.deployment.path == "gs://bucket/path/hf"
    # The `--served-model-name` value is the sanitized canonical name; we check
    # it's populated and valid for Harbor's hosted_vllm pattern.
    assert config.harbor_model_name
    assert "/" not in config.harbor_model_name  # Harbor pattern forbids '/' in canonical
    assert step.name == "evaluation/harbor/my-vllm-served-model-aime-1.0"


def test_evaluate_harbor_external_api_mode_has_no_deployment():
    step = evaluate_harbor(
        model_name="claude-opus-4",
        model_path=None,
        dataset="aime",
        version="1.0",
    )
    config = step.config
    assert isinstance(config, _HarborStepConfig)
    assert config.deployment is None
    # External mode: Harbor model name is the raw name (LiteLLM routes by it).
    assert config.harbor_model_name == "claude-opus-4"


def test_evaluate_harbor_carries_typed_run_fields():
    step = evaluate_harbor(
        model_name="claude-opus-4",
        model_path=None,
        dataset="terminal-bench",
        version="2.0",
        agent="claude-code",
        n_concurrent=8,
        env="daytona",
        agent_kwargs={"api_key": "sk-xxx"},
        max_eval_instances=10,
    )
    run = step.config.run
    assert run.dataset == "terminal-bench"
    assert run.version == "2.0"
    assert run.agent == "claude-code"
    assert run.n_concurrent == 8
    assert run.env == "daytona"
    assert run.agent_kwargs == {"api_key": "sk-xxx"}
    assert run.max_eval_instances == 10


def test_evaluate_harbor_forwards_deployment_kwargs_only_when_local_vllm():
    step_local = evaluate_harbor(
        model_name="my-model",
        model_path="gs://bucket/path/hf",
        dataset="aime",
        deployment_kwargs={"tensor_parallel_size": 4},
    )
    assert step_local.config.deployment is not None
    assert step_local.config.deployment.engine_kwargs == {"tensor_parallel_size": 4}

    # External API: deployment_kwargs is silently dropped (no server to launch).
    step_external = evaluate_harbor(
        model_name="claude-opus-4",
        model_path=None,
        dataset="aime",
        deployment_kwargs={"tensor_parallel_size": 4},  # ignored
    )
    assert step_external.config.deployment is None


@pytest.mark.parametrize(
    "base, path, expected",
    [
        ("myrun", "gs://bucket/run/hf/step-7022", "myrun-step7022"),
        ("myrun", "gs://bucket/run/hf", "myrun"),
        ("myrun", "hf/step-1", "myrun-step1"),
        ("myrun", "", "myrun"),
    ],
)
def test_append_step_suffix_extracts_step_number(base, path, expected):
    assert _append_step_suffix(base, path) == expected
