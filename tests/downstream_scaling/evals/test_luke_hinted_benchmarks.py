# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from thalas.execution.executor import instantiate_config
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.evals.tasks import luke_hinted_benchmarks as tasks


@pytest.fixture
def benchmark_inputs(tmp_path):
    problems = [
        {"problem_id": "second", "question": "  Question two?\n", "answer": 42, "source": "fixture"},
        {"problem_id": "first", "question": "\nQuestion one?  ", "answer": "a word", "source": "fixture"},
    ]
    hints = [
        {
            "problem_id": problem_id,
            "rollout_id": max(0, index - 1),
            "full_hint": f" {problem_id} hint {chr(65 + index)}\n",
        }
        for problem_id in ("first", "second")
        for index in (9, 1, 7, 5, 3, 2, 8, 6, 4, 0)
    ]
    problems_path = tmp_path / "problems.jsonl"
    hints_path = tmp_path / "hints.jsonl"
    problems_path.write_text("".join(json.dumps(row) + "\n" for row in problems))
    hints_path.write_text("".join(json.dumps(row) + "\n" for row in hints))
    return problems, problems_path, hints_path


@pytest.fixture
def tokenizers(tmp_path):
    paths = {}
    for side in ("student", "advisor"):
        tokenizer = Tokenizer(WordLevel({"<unk>": 0, "<bos>": 1}, unk_token="<unk>"))
        tokenizer.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", unk_token="<unk>")
        tokenizer.chat_template = (
            "{{ bos_token }}"
            f"<{side}>"
            "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}</{{ message['role'] }}>"
            "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
        )
        path = tmp_path / side
        tokenizer.save_pretrained(path)
        (path / "config.json").write_text('{"model_type": "qwen2"}')
        paths[side] = (path, tokenizer)
    return paths


@pytest.mark.parametrize(
    "benchmark,student_task,advisor_task,student_text,advisor_text",
    [
        (
            "aime",
            tasks.AIMETask,
            tasks.AIMEAdvisorTask,
            "Put your final answer within \\boxed{{}}.\n"
            "The answer is an integer between 0 and 999 inclusive.\n\nProblem:\n{question}",
            "You will be given a problem and a hint to the problem.\n"
            "Put your final answer within \\boxed{{}}.\n"
            "The answer is an integer between 0 and 999 inclusive.\n\n"
            "Problem:\n{question}\n\nHint:\n{hint}",
        ),
        (
            "hle",
            tasks.HLETask,
            tasks.HLEAdvisorTask,
            "Problem:\n{question}\n\nGive your final answer in the following format:\nAnswer: {{your answer}}",
            "You will be given a problem and a hint to the problem. Use the hint if helpful.\n\n"
            "Problem:\n{question}\n\nHint:\n{hint}\n\n"
            "Give your final answer in the following format:\nAnswer: {{your answer}}",
        ),
    ],
)
def test_tasks_write_paired_prompts_with_sorted_hints_and_each_tokenizer(
    tmp_path, benchmark_inputs, tokenizers, benchmark, student_task, advisor_task, student_text, advisor_text
):
    problems, problems_path, hints_path = benchmark_inputs
    rows_by_side = {}
    for side, task_type, expected_text in (
        ("student", student_task, student_text),
        ("advisor", advisor_task, advisor_text),
    ):
        tokenizer_path, tokenizer = tokenizers[side]
        step = task_type(tasks.HintedBenchmarkTaskConfig(tokenizer_path=str(tokenizer_path))).make_prompts_step()
        output_path = tmp_path / f"{side}_prompts"
        config = instantiate_config(step.config, output_path=str(output_path), output_paths={}, prefix=str(tmp_path))
        config = replace(config, problems_path=str(problems_path), hints_path=str(hints_path))
        tasks.write_hinted_benchmark_prompts(config)
        rows = list(read_prompt_rows(str(output_path / "prompts.jsonl.gz")))
        rows_by_side[side] = rows

        expected_rows = []
        for problem in problems:
            problem_id = problem["problem_id"]
            for sample_id in range(10):
                raw = expected_text.format(
                    question=problem["question"].strip(), hint=f"{problem_id} hint {chr(65 + sample_id)}"
                )
                expected_rows.append(
                    {
                        "id": f"{benchmark}/shared_teacher/{problem_id}/sample{sample_id}",
                        "prompt": f"<{side}><user>{raw}</user><assistant>",
                        "ground_truth": str(problem["answer"]),
                        "metadata": {
                            "problem": problem["question"],
                            "problem_id": problem_id,
                            "sample_id": sample_id,
                            "rollout_id": max(0, sample_id - 1),
                            "source": "fixture",
                        },
                    }
                )
        assert rows == expected_rows
        for row in rows:
            input_ids = tokenizer.encode(row["prompt"])
            assert input_ids[0] == tokenizer.bos_token_id
            assert input_ids.count(tokenizer.bos_token_id) == 1

    assert [row["id"] for row in rows_by_side["student"]] == [row["id"] for row in rows_by_side["advisor"]]
