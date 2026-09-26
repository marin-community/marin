# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from transformers.utils.chat_template_utils import render_jinja_template

from experiments.post_training.curriculum_sft.generation import (
    GenerateProblemsConfig,
    SolveProblemsConfig,
    parse_problem_batch,
    parse_solution_batch,
)

PACKET = {
    "capability_id": "d00.example",
    "sampling_facets": [{"id": "f1", "description": "first"}, {"id": "f2", "description": "second"}],
}


def _problem_response(index: int, *, problem: str, answer: str, finish_reason: str = "tool_calls") -> dict:
    arguments = json.dumps({"problem": problem, "answer": answer})
    return {
        "custom_id": f"problem-{index:05d}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {"tool_calls": [{"function": {"name": "submit_problem", "arguments": arguments}}]},
                    }
                ]
            },
        },
    }


def _solution_response(request_id: str, *, content: str, reasoning: str) -> dict:
    return {
        "custom_id": request_id,
        "response": {
            "status_code": 200,
            "body": {"choices": [{"finish_reason": "stop", "message": {"content": content, "reasoning": reasoning}}]},
        },
    }


def _jsonl(responses: list[dict]) -> str:
    return "\n".join(json.dumps(response) for response in responses)


def test_parse_problem_batch_accounts_for_rejections_and_cycles_facets():
    config = GenerateProblemsConfig(
        catalog_path="unused",
        output_path="unused",
        capability_id="d00.example",
        requested_problems=4,
        seed=17,
        max_completion_tokens=1024,
        task_specification="unused",
        relay_job="unused",
    )
    responses = [
        _problem_response(0, problem="Find x if 2x = 14.", answer="7"),
        _problem_response(1, problem="find X if 2x  = 14.", answer="7"),
        _problem_response(2, problem="Find the answer.", answer="???"),
        _problem_response(3, problem="Find y.", answer="3", finish_reason="length"),
    ]

    records = parse_problem_batch(_jsonl(responses), config, PACKET)

    assert [record["rejection_reason"] for record in records] == [
        None,
        "duplicate_problem",
        "unparsable_answer",
        "truncated",
    ]
    assert [record["facet_id"] for record in records] == ["f1", "f2", "f1", "f2"]
    assert records[0]["difficulty"] != records[2]["difficulty"]


def test_parse_solution_batch_keeps_first_verified_solution_with_reasoning():
    config = SolveProblemsConfig(
        problems_path="unused",
        output_path="unused",
        capability_id="d00.example",
        samples_per_problem=5,
        solutions_per_problem=1,
        max_solution_chars=200,
        seed=17,
        max_completion_tokens=1024,
        relay_job="unused",
    )
    problem = {"request_id": "problem-00000", "problem": "Compute 1/2.", "answer": "\\frac{1}{2}"}
    responses = [
        _solution_response("problem-00000-s00", content="So \\boxed{2}.", reasoning="Guess."),
        _solution_response("problem-00000-s01", content="The value is \\boxed{0.5}.", reasoning=""),
        _solution_response("problem-00000-s02", content="So \\boxed{1/2}.", reasoning="x" * 300),
        _solution_response("problem-00000-s03", content="Halve 1: \\boxed{0.5}.", reasoning="One half."),
        _solution_response("problem-00000-s04", content="So \\boxed{\\tfrac12}.", reasoning="Half."),
    ]

    solutions, chat_rows = parse_solution_batch(_jsonl(responses), config, [problem])

    assert [record["rejection_reason"] for record in solutions] == [
        "wrong_answer",
        "no_reasoning",
        "too_long",
        None,
        None,
    ]
    assert [record["selected"] for record in solutions] == [False, False, False, True, False]
    assert chat_rows == [
        {
            "id": "problem-00000-s03",
            "messages": [
                {"role": "user", "content": "Compute 1/2.", "reasoning_content": None},
                {"role": "assistant", "content": "Halve 1: \\boxed{0.5}.", "reasoning_content": "One half."},
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    ]

    rendered = render_jinja_template(
        [chat_rows[0]["messages"]],
        chat_template=MARIN_CHAT_TEMPLATE,
        bos_token="<|begin_of_text|>",
        **chat_rows[0]["chat_template_kwargs"],
    )[0][0]
    assert rendered.index("Reasoning: /think") < rendered.index("<|start_think|>One half.<|end_think|>Halve 1")
