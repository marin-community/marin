# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.evaluation_config import EvalTaskConfig

from marin.evaluation.benchmarks.long_context import (
    build_long_context_task,
    evaluate_long_context_tasks,
)


def test_passkey_task_is_deterministic():
    config = EvalTaskConfig(
        name="passkey_4k",
        num_fewshot=0,
        task_kwargs={
            "family": "passkey",
            "context_len": 128,
            "num_examples": 2,
            "seed": 7,
        },
    )

    first = build_long_context_task(config)
    second = build_long_context_task(config)

    assert [example.prompt for example in first.examples] == [example.prompt for example in second.examples]
    assert [example.gold_answer for example in first.examples] == [example.gold_answer for example in second.examples]
    assert first.metric_names == ("exact_match",)


def test_evaluate_long_context_scores_exact_match_per_example():
    config = EvalTaskConfig(
        name="kv_retrieval_4k",
        num_fewshot=0,
        task_kwargs={
            "family": "kv",
            "context_len": 128,
            "num_examples": 2,
            "seed": 3,
        },
    )

    task = build_long_context_task(config)
    predictions_by_prompt = {
        task.examples[0].prompt: task.examples[0].gold_answer,
        task.examples[1].prompt: "wrong-value",
    }

    results = evaluate_long_context_tasks(
        [config],
        completion_fn=lambda prompts, max_gen_toks: [predictions_by_prompt[prompt] for prompt in prompts],
    )
    task_result = results[0]

    assert task_result["metrics"]["exact_match"] == 0.5
    assert [example["correct"] for example in task_result["examples"]] == [True, False]
