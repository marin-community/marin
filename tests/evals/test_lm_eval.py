# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import time

import pytest
from fray.cluster import ResourceConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig, convert_to_levanter_task_config
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import (
    _resolve_levanter_eval_tasks,
    add_sample_smooth_metrics,
    drop_sample_payloads,
)
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import (
    _lm_eval_task_spec,
    _patch_lm_eval_none_alias_compat,
)
from marin.evaluation.run import evaluate
from marin.execution.remote import RemoteCallable

from experiments.evals.evals import evaluate_levanter_lm_evaluation_harness, evaluate_lm_evaluation_harness
from experiments.evals.olmo_base_easy_overlap import (
    OLMO_BASE_EASY_OVERLAP_SUPPORTED_BPB_KEYS,
    OLMO_BASE_EASY_OVERLAP_TASKS,
    add_olmo_base_easy_overlap_metrics,
)
from experiments.evals.task_configs import (
    BASE_GENERATION_TASKS,
    HUMANEVAL_GSM8K_TASKS,
    MMLU_SL_VERB_5_SHOT,
    MMLU_SL_VERB_DOC_TO_CHOICE,
)


def _repo_relative_data_file() -> str:
    return "experiments/scaling_law_sweeps/dclm_core/custom_tasks/winograd/wsc273.jsonl"


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


def test_lm_eval_task_spec_preserves_alias_and_inline_task_fields():
    spec = _lm_eval_task_spec(
        EvalTaskConfig(
            "winograd",
            0,
            task_alias="winograd_0shot",
            task_kwargs={
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": "/tmp/wsc273.jsonl"},
                "output_type": "multiple_choice",
            },
        )
    )

    assert spec == {
        "task": "winograd",
        "num_fewshot": 0,
        "task_alias": "winograd_0shot",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": "/tmp/wsc273.jsonl"},
        "output_type": "multiple_choice",
    }


def test_lm_eval_task_spec_resolves_repo_relative_data_files(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    data_file = repo_root / _repo_relative_data_file()
    data_file.parent.mkdir(parents=True)
    data_file.write_text("{}\n")
    monkeypatch.setattr(
        "marin.evaluation.evaluators.lm_evaluation_harness_evaluator._repo_root",
        lambda: repo_root,
    )

    spec = _lm_eval_task_spec(
        EvalTaskConfig(
            "winograd",
            0,
            task_alias="winograd_0shot",
            task_kwargs={
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": _repo_relative_data_file()},
                "output_type": "multiple_choice",
            },
        )
    )

    assert spec["dataset_kwargs"]["data_files"] == str(data_file)


def test_levanter_eval_tasks_resolve_repo_relative_data_files(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    data_file = repo_root / _repo_relative_data_file()
    data_file.parent.mkdir(parents=True)
    data_file.write_text("{}\n")
    monkeypatch.setattr(
        "marin.evaluation.evaluators.lm_evaluation_harness_evaluator._repo_root",
        lambda: repo_root,
    )

    [task] = _resolve_levanter_eval_tasks(
        [
            EvalTaskConfig(
                "winograd",
                0,
                task_alias="winograd_0shot",
                task_kwargs={
                    "dataset_path": "json",
                    "dataset_kwargs": {"data_files": _repo_relative_data_file()},
                    "output_type": "multiple_choice",
                },
            )
        ]
    )

    assert task.task_kwargs["dataset_kwargs"]["data_files"] == str(data_file)
    assert task.task_alias == "winograd_0shot"


def test_lm_eval_none_alias_patch_uses_requested_alias_fallback():
    class FakeTask:
        VERSION = 1

        def higher_is_better(self):
            return {}

    _patch_lm_eval_none_alias_compat({"squad_completion": "squad_10shot"})

    from lm_eval.evaluator import consolidate_results
    from lm_eval.evaluator_utils import TaskOutput

    task_output = TaskOutput(
        task=FakeTask(),
        task_name="squad_completion",
        task_config={"task": "squad_completion", "task_alias": None},
        version=1,
        n_shot=10,
    )
    results, *_ = consolidate_results([task_output])

    assert results["squad_completion"]["alias"] == "squad_10shot"


def test_lm_eval_task_name_alias_patch_uses_requested_alias_fallback():
    class FakeTask:
        VERSION = 1

        def higher_is_better(self):
            return {}

    _patch_lm_eval_none_alias_compat({"squad_completion": "squad_10shot"})

    from lm_eval import evaluator
    from lm_eval.evaluator_utils import TaskOutput

    task_output = TaskOutput(
        task=FakeTask(),
        task_name="squad_completion",
        task_config={"task": "squad_completion"},
        version=1,
        n_shot=10,
    )
    results, *_ = evaluator.consolidate_results([task_output])

    assert results["squad_completion"]["alias"] == "squad_10shot"


def test_lm_eval_none_alias_patch_sanitizes_prepare_print_tasks():
    _patch_lm_eval_none_alias_compat({"jeopardy": "jeopardy_10shot"})

    from lm_eval import evaluator

    prepare_print_tasks = evaluator.evaluate.__wrapped__.__globals__["prepare_print_tasks"]
    task_agg, _group_agg = prepare_print_tasks({"jeopardy": object()}, {"jeopardy": {"alias": None}})

    assert task_agg["jeopardy"]["alias"] == "jeopardy_10shot"


def test_lm_eval_none_alias_patch_sanitizes_none_task_dict_key():
    _patch_lm_eval_none_alias_compat({"jeopardy": "jeopardy_10shot"})

    from lm_eval import evaluator

    task = type("FakeConfigurableTask", (), {"task_name": None})()
    prepare_print_tasks = evaluator.evaluate.__wrapped__.__globals__["prepare_print_tasks"]
    task_agg, _group_agg = prepare_print_tasks({None: task}, {"jeopardy": {"alias": None}})

    assert task_agg["jeopardy"]["alias"] == "jeopardy_10shot"


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
        generation_params={"max_gen_toks": 64},
        env_vars={"EXTRA_ENV": "1"},
    )

    assert isinstance(step.fn, RemoteCallable)
    assert step.config.generation_params == {"max_gen_toks": 64}
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
    assert derived["lm_eval/olmo_base_easy_overlap/task_count"] == pytest.approx(
        float(len(OLMO_BASE_EASY_OVERLAP_SUPPORTED_BPB_KEYS))
    )
    assert derived["lm_eval/olmo_base_easy_overlap/macro_bpb"] == pytest.approx(1.42)


def test_add_olmo_base_easy_overlap_metrics_omits_macro_when_supported_bpb_coverage_is_partial():
    flat_metrics = {
        "lm_eval/mmlu_stem_5shot/bpb": 1.2,
        "lm_eval/mmlu_humanities_5shot/bpb": 1.6,
        "lm_eval/mmlu_social_sciences_5shot/bpb": 2.0,
        "lm_eval/mmlu_other_5shot/bpb": 2.4,
        "lm_eval/arc_easy_5shot/bpb": 0.9,
        "lm_eval/arc_challenge_5shot/bpb": 1.0,
        "lm_eval/csqa_5shot/bpb": 1.1,
        "lm_eval/hellaswag_5shot/bpb": 1.2,
        "lm_eval/winogrande_5shot/bpb": 1.3,
    }

    derived = add_olmo_base_easy_overlap_metrics(flat_metrics)

    assert derived["lm_eval/olmo_base_easy_overlap/task_count"] == pytest.approx(9.0)
    assert "lm_eval/olmo_base_easy_overlap/macro_bpb" not in derived


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


def test_evaluate_levanter_lm_evaluation_harness_preserves_sample_logging_options():
    resource_config = ResourceConfig.with_tpu("v5p-8")
    step = evaluate_levanter_lm_evaluation_harness(
        model_name="unit-test-model",
        model_path="gs://unit-test/checkpoint",
        evals=[EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot")],
        resource_config=resource_config,
        log_samples=True,
        sample_log_all=True,
        max_logged_samples_per_task=20,
        sample_smooth_metrics=True,
        drop_samples_after_metrics=True,
    )

    assert step.config.log_samples is True
    assert step.config.sample_log_all is True
    assert step.config.max_logged_samples_per_task.value == 20
    assert step.config.sample_smooth_metrics is True
    assert step.config.drop_samples_after_metrics is True


def test_evaluate_levanter_lm_evaluation_harness_can_disable_wandb_tracker():
    step = evaluate_levanter_lm_evaluation_harness(
        model_name="unit-test-model",
        model_path="gs://unit-test/checkpoint",
        evals=[EvalTaskConfig("hellaswag", 10, task_alias="hellaswag_10shot")],
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        use_wandb_tracker=False,
    )

    assert step.config.use_wandb_tracker is False


def test_add_sample_smooth_metrics_derives_mcq_aggregate_metrics_and_can_drop_payloads():
    results = {
        "results": {"toy_mcq": {"acc,none": 0.5, "outputs": [{"prompt": "large"}]}},
        "samples": {
            "toy_mcq": [
                {
                    "target": [1],
                    "arguments": [["prompt", " A"], ["prompt", " BC"], ["prompt", " D"]],
                    "filtered_resps": [[-3.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]],
                },
                {
                    "target": [0],
                    "arguments": [["prompt", " A"], ["prompt", " B"], ["prompt", " C"]],
                    "filtered_resps": [[-4.0, 0.0], [-2.0, 0.0], [-3.0, 0.0]],
                },
            ]
        },
    }

    add_sample_smooth_metrics(results)
    metrics = results["results"]["toy_mcq"]

    assert metrics["native_sample_count,none"] == 2.0
    assert metrics["native_gold_logprob,none"] == pytest.approx(-2.5)
    assert metrics["native_margin,none"] == pytest.approx(-0.5)
    assert metrics["native_predicted_correct,none"] == pytest.approx(0.5)
    first_prob = 1.0 / (math.exp(-2.0) + 1.0 + math.exp(-1.0))
    second_prob = math.exp(-2.0) / (math.exp(-2.0) + 1.0 + math.exp(-1.0))
    assert metrics["native_choice_prob,none"] == pytest.approx((first_prob + second_prob) / 2.0)
    assert metrics["native_gold_bpb,none"] > 0.0
    assert metrics["native_gold_logprob_stderr,none"] > 0.0

    drop_sample_payloads(results)

    assert "samples" not in results
    assert "outputs" not in results["results"]["toy_mcq"]


def test_add_sample_smooth_metrics_handles_single_continuation_string_targets():
    results = {
        "results": {"lambada_0shot": {"acc,none": 0.0}},
        "samples": {
            "lambada_0shot": [
                {
                    "target": " signs",
                    "arguments": [["In my palm is a clear stone", " signs"]],
                    "filtered_resps": [[-9.0, 0.0]],
                }
            ],
            "ambiguous_string_choice": [
                {
                    "target": "same ending",
                    "arguments": [["prompt A", " same ending"], ["prompt B", " same ending"]],
                    "filtered_resps": [[-2.0, 0.0], [-1.0, 0.0]],
                }
            ],
        },
    }

    add_sample_smooth_metrics(results)

    assert results["results"]["lambada_0shot"]["native_gold_logprob,none"] == -9.0
    assert "ambiguous_string_choice" not in results["results"]


def test_add_sample_smooth_metrics_infers_string_choice_targets_from_arguments_and_doc():
    results = {
        "results": {
            "commonsense_qa_10shot": {"acc,none": 0.0},
            "copa_0shot": {"acc,none": 0.0},
            "winogrande_0shot": {"acc,none": 0.0},
        },
        "samples": {
            "commonsense_qa_10shot": [
                {
                    "target": "A",
                    "arguments": [
                        ["Question: ...\nAnswer:", " A"],
                        ["Question: ...\nAnswer:", " B"],
                        ["Question: ...\nAnswer:", " C"],
                    ],
                    "filtered_resps": [[-1.0, 0.0], [-3.0, 0.0], [-4.0, 0.0]],
                }
            ],
            "copa_0shot": [
                {
                    "target": " water flowed from the spout.",
                    "arguments": [
                        ["The man turned on the faucet therefore", "  the toilet filled with water."],
                        ["The man turned on the faucet therefore", "  water flowed from the spout."],
                    ],
                    "filtered_resps": [[-31.0, 0.0], [-40.0, 0.0]],
                    "doc": {"label": 1},
                }
            ],
            "winogrande_0shot": [
                {
                    "target": "always got the easier cases.",
                    "arguments": [
                        ["Sarah was a better surgeon than Maria so Sarah", " always got the easier cases."],
                        ["Sarah was a better surgeon than Maria so Maria", " always got the easier cases."],
                    ],
                    "filtered_resps": [[-28.0, 0.0], [-27.0, 0.0]],
                    "doc": {"answer": "2"},
                }
            ],
        },
    }

    add_sample_smooth_metrics(results)

    assert results["results"]["commonsense_qa_10shot"]["native_gold_logprob,none"] == -1.0
    assert results["results"]["copa_0shot"]["native_gold_logprob,none"] == -40.0
    assert results["results"]["winogrande_0shot"]["native_gold_logprob,none"] == -27.0


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
