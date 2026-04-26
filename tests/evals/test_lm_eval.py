# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig, convert_to_levanter_task_config
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.run import evaluate
from marin.execution.remote import RemoteCallable

from experiments.evals.task_configs import (
    BASE_GENERATION_TASKS,
    HUMANEVAL_GSM8K_TASKS,
    MMLU_SL_VERB_5_SHOT,
    MMLU_SL_VERB_DOC_TO_CHOICE,
)
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_TASKS,
    add_olmo_base_easy_overlap_metrics,
)
from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness, evaluate_lm_evaluation_harness


@pytest.fixture
def current_date_time():
    # Get the current local time and format as MM-DD-YYYY-HH-MM-SS
    formatted_time = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())

    return formatted_time


@pytest.fixture
def model_config():
    config = ModelConfig(
        name="test-llama-200m",
        path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
        engine_kwargs={"enforce_eager": True, "max_model_len": 1024},
        generation_params={"max_tokens": 16},
    )
    return config


def test_convert_to_levanter_task_config_preserves_doc_to_choice_override():
    [task] = convert_to_levanter_task_config([MMLU_SL_VERB_5_SHOT])

    assert task.task == "mmlu"
    assert task.task_alias == "mmlu_sl_verb_5shot"
    assert task.num_fewshot == 5
    assert task.doc_to_choice == MMLU_SL_VERB_DOC_TO_CHOICE


def test_convert_to_levanter_task_config_rejects_unsupported_task_kwargs():
    with pytest.raises(ValueError, match="Unsupported Levanter task kwargs"):
        convert_to_levanter_task_config(
            [EvalTaskConfig("mmlu", 5, task_alias="bad_task", task_kwargs={"unsupported_key": "value"})]
        )


def test_olmo_base_easy_overlap_suite_uses_expected_aliases_and_shots():
    assert [task.task_alias for task in OLMO_BASE_EASY_OVERLAP_TASKS] == [
        "mmlu_5shot",
        "arc_easy_5shot",
        "arc_challenge_5shot",
        "csqa_5shot",
        "hellaswag_5shot",
        "winogrande_5shot",
        "socialiqa_5shot",
        "piqa_5shot",
        "sciq_5shot",
        "lambada_0shot",
        "medmcqa_5shot",
    ]
    assert [task.num_fewshot for task in OLMO_BASE_EASY_OVERLAP_TASKS] == [5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 5]


def test_base_generation_suite_includes_gsm8k_and_humaneval_hard_metrics():
    task_aliases = [task.task_alias for task in BASE_GENERATION_TASKS]

    assert [task.task_alias for task in HUMANEVAL_GSM8K_TASKS] == ["gsm8k_5shot", "humaneval_10shot"]
    assert "gsm8k_5shot" in task_aliases
    assert "humaneval_10shot" in task_aliases


def test_evaluate_lm_evaluation_harness_sets_code_eval_and_tpu_vllm_env():
    resource_config = ResourceConfig.with_tpu("v5p-8")
    step = evaluate_lm_evaluation_harness(
        model_name="unit-test-model",
        model_path="gs://unit-test/checkpoint",
        evals=list(HUMANEVAL_GSM8K_TASKS),
        resource_config=resource_config,
        env_vars={"EXTRA_ENV": "1"},
    )

    assert isinstance(step.fn, RemoteCallable)
    assert step.fn.resources == resource_config
    assert step.fn.env_vars["HF_ALLOW_CODE_EVAL"] == "1"
    assert step.fn.env_vars["MARIN_VLLM_MODE"] == "native"
    assert step.fn.env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert step.fn.env_vars["EXTRA_ENV"] == "1"


def test_add_olmo_base_easy_overlap_metrics_derives_mmlu_category_bpbs_and_macro():
    flat_metrics = {
        "lm_eval/mmlu_abstract_algebra_5shot/bpb": 1.1,
        "lm_eval/mmlu_astronomy_5shot/bpb": 1.3,
        "lm_eval/mmlu_formal_logic_5shot/bpb": 1.5,
        "lm_eval/mmlu_high_school_world_history_5shot/bpb": 1.7,
        "lm_eval/mmlu_econometrics_5shot/bpb": 1.9,
        "lm_eval/mmlu_public_relations_5shot/bpb": 2.1,
        "lm_eval/mmlu_business_ethics_5shot/bpb": 2.3,
        "lm_eval/mmlu_miscellaneous_5shot/bpb": 2.5,
        "lm_eval/arc_easy_5shot/bpb": 0.9,
        "lm_eval/arc_challenge_5shot/bpb": 1.0,
        "lm_eval/csqa_5shot/bpb": 1.1,
        "lm_eval/hellaswag_5shot/bpb": 1.2,
        "lm_eval/winogrande_5shot/bpb": 1.3,
        "lm_eval/socialiqa_5shot/bpb": 1.4,
        "lm_eval/piqa_5shot/bpb": 1.5,
        "lm_eval/sciq_5shot/bpb": 1.6,
        "lm_eval/lambada_0shot/bpb": 1.7,
        "lm_eval/medmcqa_5shot/bpb": 1.8,
    }

    derived = add_olmo_base_easy_overlap_metrics(flat_metrics)

    assert derived["lm_eval/mmlu_stem_5shot/bpb"] == pytest.approx(1.2)
    assert derived["lm_eval/mmlu_humanities_5shot/bpb"] == pytest.approx(1.6)
    assert derived["lm_eval/mmlu_social_sciences_5shot/bpb"] == pytest.approx(2.0)
    assert derived["lm_eval/mmlu_other_5shot/bpb"] == pytest.approx(2.4)
    assert derived["lm_eval/olmo_base_easy_overlap/task_count"] == pytest.approx(14.0)
    assert derived["lm_eval/olmo_base_easy_overlap/macro_bpb"] == pytest.approx(1.4785714285714284)


def test_evaluate_levanter_lm_evaluation_harness_dispatches_remotely_and_preserves_cache_dependency():
    cache_dependency = "gs://unit-test/cache/.eval_datasets_manifest.json"
    resource_config = ResourceConfig.with_tpu("v5p-8")
    step = evaluate_levanter_lm_evaluation_harness(
        model_name="unit-test-model",
        model_path="gs://unit-test/checkpoint",
        evals=[EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot")],
        resource_config=resource_config,
        eval_datasets_cache_path="gs://unit-test/cache",
        eval_datasets_cache_dependency=cache_dependency,
    )

    assert isinstance(step.fn, RemoteCallable)
    assert step.fn.resources == resource_config
    assert step.fn.pip_dependency_groups == ["eval", "tpu"]
    assert step.config.evaluator == "levanter_lm_evaluation_harness"
    assert step.config.eval_datasets_cache_path.value == "gs://unit-test/cache"
    assert step.config.eval_datasets_cache_dependency == cache_dependency


@pytest.mark.tpu_ci
def test_lm_eval_harness_levanter(current_date_time, model_config):
    mmlu_config = EvalTaskConfig("mmlu", 0, task_alias="mmlu_0shot")
    config = EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[mmlu_config],
        max_eval_instances=5,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        engine_kwargs=model_config.engine_kwargs,
    )
    evaluate(config=config)


@pytest.mark.tpu_ci
def test_lm_eval_harness(current_date_time, model_config):
    gsm8k_config = EvalTaskConfig(name="gsm8k_cot", num_fewshot=8)
    config = EvaluationConfig(
        evaluator="lm_evaluation_harness",
        model_name=model_config.name,
        model_path=model_config.path,
        evaluation_path=f"gs://marin-us-east5/evaluation/lm_eval/{model_config.name}-{current_date_time}",
        evals=[gsm8k_config],
        max_eval_instances=1,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        engine_kwargs=model_config.engine_kwargs,
    )
    evaluate(config=config)
