# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from marin.evaluation.evaluation_config import EvalTaskConfig

from marin.evaluation.benchmarks.long_context import (
    build_long_context_task,
    evaluate_long_context_tasks,
    write_long_context_results,
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


def test_finepdf_task_loads_manifest_and_honors_limit(tmp_path):
    manifest_path = tmp_path / "finepdf_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "doc-1",
                        "context_len": 4096,
                        "context": "Alpha section. The answer is orion.",
                        "question": "What is the answer?",
                        "answer": "orion",
                    }
                ),
                json.dumps(
                    {
                        "id": "doc-2",
                        "context_len": 4096,
                        "context": "Beta section. The answer is vega.",
                        "question": "What is the answer?",
                        "answer": "vega",
                    }
                ),
            ]
        )
    )

    config = EvalTaskConfig(
        name="finepdf_extractive_qa_4k",
        num_fewshot=0,
        task_kwargs={
            "family": "finepdf_qa",
            "context_len": 4096,
            "manifest_path": str(manifest_path),
        },
    )

    task = build_long_context_task(config, max_eval_instances=1)

    assert len(task.examples) == 1
    assert task.examples[0].example_id == "doc-1"
    assert task.metric_names == ("exact_match", "token_f1")


def test_finepdf_task_requires_explicit_context_len(tmp_path):
    manifest_path = tmp_path / "finepdf_manifest_missing_context_len.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "context": "Alpha section. The answer is orion.",
                "question": "What is the answer?",
                "answer": "orion",
            }
        )
    )

    config = EvalTaskConfig(
        name="finepdf_extractive_qa_4k",
        num_fewshot=0,
        task_kwargs={
            "family": "finepdf_qa",
            "context_len": 4096,
            "manifest_path": str(manifest_path),
        },
    )

    with pytest.raises(ValueError, match="missing required 'context_len'"):
        build_long_context_task(config)


def test_evaluate_and_write_long_context_results(tmp_path):
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
    gold_by_prompt = {example.prompt: example.gold_answer for example in task.examples}

    results = evaluate_long_context_tasks(
        [config],
        completion_fn=lambda prompts, max_gen_toks: [gold_by_prompt[prompt] for prompt in prompts],
    )

    output_dir = tmp_path / "results"
    write_long_context_results(output_dir, results)

    summary = json.loads((output_dir / "results.json").read_text())
    task_result = json.loads((output_dir / "kv_retrieval_4k.json").read_text())

    assert summary["tasks"][0]["task_name"] == "kv_retrieval_4k"
    assert task_result["metrics"]["exact_match"] == 1.0
    assert all(example["correct"] for example in task_result["examples"])
