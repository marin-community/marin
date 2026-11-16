# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.curriculum_sft.code_tasks import (
    ExecutionStatus,
    GenerateCodeTasksConfig,
    TaskKind,
    execute_python,
    literals_equal,
    parse_code_task_batch,
    run_tests,
)

PACKET = {
    "capability_id": "d02.example",
    "sampling_facets": [{"id": "f1", "description": "first"}],
    "includes": [],
}

REVERSE_WORDS_SOLUTION = "def reverse_words(text):\n    return ' '.join(reversed(text.split()))\n"
REVERSE_WORDS_TESTS = [
    "assert reverse_words('a b c') == 'c b a'",
    "assert reverse_words('') == ''",
    "assert reverse_words('one') == 'one'",
    "assert reverse_words('  x   y ') == 'y x'",
    "assert reverse_words('Hi there') == 'there Hi'",
    "assert reverse_words('1 2') == '2 1'",
]

TRACE_CODE = """def f(items):
    total = 0
    for index, item in enumerate(items):
        if index % 2:
            total -= item
        else:
            total += item * 2
    return [total, len(items)]"""


def _response(index: int, arguments: dict) -> dict:
    return {
        "custom_id": f"task-{index:05d}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [{"function": {"name": "submit_task", "arguments": json.dumps(arguments)}}]
                        },
                    }
                ]
            },
        },
    }


def _config(kind: TaskKind, requested: int) -> GenerateCodeTasksConfig:
    return GenerateCodeTasksConfig(
        catalog_path="unused",
        output_path="unused",
        capability_id="d02.example",
        kind=kind,
        requested=requested,
        seed=17,
        max_completion_tokens=1024,
        relay_job="unused",
    )


def _jsonl(responses: list[dict]) -> str:
    return "\n".join(json.dumps(response) for response in responses)


def _implement(specification: str, solution: str, tests: list[str]) -> dict:
    return {"function_name": "reverse_words", "specification": specification, "solution": solution, "tests": tests}


def test_implement_tasks_keep_passing_reference_and_reject_failing_or_vacuous_tests():
    spec = "def reverse_words(text: str) -> str:\n    '''Reverse word order.'''"
    buggy = "def reverse_words(text):\n    return ' '.join(reversed(text.split(' ')))\n"
    vacuous_tests = [f"assert callable(reverse_words) or {n}" for n in range(6)]
    responses = [
        _response(0, _implement(spec, REVERSE_WORDS_SOLUTION, REVERSE_WORDS_TESTS)),
        _response(1, _implement(spec.replace("Reverse", "Invert"), buggy, REVERSE_WORDS_TESTS)),
        _response(2, _implement(spec.replace("word", "the word"), REVERSE_WORDS_SOLUTION, vacuous_tests)),
    ]

    records = parse_code_task_batch(_jsonl(responses), _config(TaskKind.IMPLEMENT, 3), PACKET)

    assert [record["rejection_reason"] for record in records] == [None, "reference_failed", "vacuous_tests"]
    assert json.loads(records[0]["answer"]) == REVERSE_WORDS_TESTS
    assert records[0]["accepted"] and records[0]["kind"] == "implement"


def test_trace_answer_comes_from_execution_not_the_generator_claim():
    responses = [
        _response(0, {"code": TRACE_CODE, "call": "f([3, 1, 4])", "output": "[13, 3]"}),
        _response(1, {"code": TRACE_CODE, "call": "f([5, 2])", "output": "[3, 2]"}),
        _response(2, {"code": TRACE_CODE, "call": "f(None)", "output": "[0, 0]"}),
    ]

    records = parse_code_task_batch(_jsonl(responses), _config(TaskKind.TRACE, 3), PACKET)

    assert [record["rejection_reason"] for record in records] == [None, None, "trace_failed"]
    assert [record["answer"] for record in records] == ["[13, 3]", "[8, 2]", None]
    assert records[1]["claimed_output"] == "[3, 2]"
    assert "f([5, 2])" in records[1]["problem"]


def test_executor_stops_an_infinite_loop_at_its_timeout():
    result = execute_python("while True:\n    pass\n", "", timeout=1.0)

    assert result.status is ExecutionStatus.TIMEOUT
    assert run_tests("def f():\n    while True:\n        pass\n", ["assert f() is None"], timeout=1.0) == [False]


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ("[1, 2]", "[1,2]", True),
        ("{'a': (1, 'x')}", " {'a':(1,'x')} ", True),
        ("[1, 2]", "[2, 1]", False),
        ("1", "True", False),
        ("'abc'", "abc", False),
    ],
)
def test_literals_equal_ignores_formatting_but_not_value_or_type(reference, candidate, expected):
    assert literals_equal(reference, candidate) is expected
