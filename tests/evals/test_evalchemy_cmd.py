# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier A: evalchemy CLI command encoding (no vLLM, no repo clone)."""

from __future__ import annotations

import pytest

from marin.evaluation.api import LmEvalRun
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evalchemy_evaluator import EvalchemyEvaluator
from marin.inference.model_launcher import OpenAIEndpoint, RunningModel


def _make_model() -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(url="http://host:8000/v1", model="served-model"),
        tokenizer_ref="hf/tok",
    )


def _make_run(**overrides) -> LmEvalRun:
    defaults = dict(
        evals=[EvalTaskConfig(name="AIME25", num_fewshot=0)],
        output_path="/tmp/out",
        apply_chat_template=False,
        generation_params={},
    )
    defaults.update(overrides)
    return LmEvalRun(**defaults)


def test_cmd_uses_local_completions_kind_by_default():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="model=x,base_url=http://h/v1/completions",
        eval_task=task,
        result_dir="/tmp/result",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={},
    )
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "local-completions"
    assert "vllm" not in cmd  # critical: we must not fall back to the in-proc vLLM backend


def test_cmd_uses_local_chat_completions_when_chat_template():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-chat-completions",
        model_args="model=x,base_url=http://h/v1/chat/completions",
        eval_task=task,
        result_dir="/tmp/result",
        max_eval_instances=None,
        apply_chat_template=True,
        generation_params={},
    )
    assert cmd[cmd.index("--model") + 1] == "local-chat-completions"
    assert "--apply_chat_template" in cmd


def test_cmd_includes_limit_when_max_eval_instances_set():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=5,
        apply_chat_template=False,
        generation_params={},
    )
    assert "--limit" in cmd
    assert cmd[cmd.index("--limit") + 1] == "5"


def test_cmd_omits_limit_when_no_max_eval_instances():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={},
    )
    assert "--limit" not in cmd


def test_cmd_adds_num_fewshot_when_positive():
    task = EvalTaskConfig(name="AIME25", num_fewshot=3)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={},
    )
    assert cmd[cmd.index("--num_fewshot") + 1] == "3"


def test_cmd_omits_num_fewshot_when_zero():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={},
    )
    assert "--num_fewshot" not in cmd


def test_cmd_passes_generation_params_via_gen_kwargs():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={"temperature": 0.7, "max_gen_toks": 32768, "top_p": 0.95, "seed": 42},
    )
    gen_kwargs_idx = cmd.index("--gen_kwargs")
    gen_kwargs_value = cmd[gen_kwargs_idx + 1]
    assert "temperature=0.7" in gen_kwargs_value
    assert "max_gen_toks=32768" in gen_kwargs_value
    assert "top_p=0.95" in gen_kwargs_value
    # seed goes into the generation_params but not into the CLI --gen_kwargs
    # (seed is per-request, used by evalchemy's sampler layer). We don't forward it.
    assert "seed=" not in gen_kwargs_value


def test_cmd_omits_gen_kwargs_when_all_generation_params_empty():
    task = EvalTaskConfig(name="AIME25", num_fewshot=0)
    cmd = EvalchemyEvaluator._build_cmd(
        lm_eval_kind="local-completions",
        model_args="m",
        eval_task=task,
        result_dir="/tmp/r",
        max_eval_instances=None,
        apply_chat_template=False,
        generation_params={},
    )
    assert "--gen_kwargs" not in cmd


@pytest.mark.parametrize(
    "base_eval_run_name, task_name, expected",
    [
        (None, "AIME25", "evalchemy-served-model-AIME25"),
        ("myrun-step7022", "AIME25", "evalchemy-myrun-step7022-AIME25"),
        ("myrun", "", "evalchemy-myrun"),
    ],
)
def test_wandb_run_name_formation(base_eval_run_name, task_name, expected):
    name = EvalchemyEvaluator._build_wandb_run_name(
        model=_make_model(), task_name=task_name, base_eval_run_name=base_eval_run_name
    )
    assert name == expected
