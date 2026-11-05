"""Span selection in the CruxEval grader.

The responses are shortened from Snowball v0.1 CruxEval generations. Served without a reasoning
parser, its reasoning is graded content, and it often restates the prompt's instruction to use
"[ANSWER] and [/ANSWER] tags" before giving the real answer span.
"""

import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.programs import crux

DOC = {"code": "def f(x):\n    return x + 1", "input": "16", "output": "17"}


def _grade(direction: str, response: str) -> float:
    spec = EvalSpec(
        task="CruxEval", source=SourceRef("local"), source_id="sample_0", ordinal=0, requests=(),
        grader=GraderSpec("crux", {"direction": direction}), doc=DOC,
    )
    return crux(spec, [response]).metrics["accuracy"]


RESTATED_INPUT = (
    'The prompt says: "You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your '
    'answer as a passing assertion containing the input and the given output."\n\n'
    "Thus we need to output something like:\n\n[ANSWER]\nassert f(16) == 17\n[/ANSWER]"
)
RESTATED_INPUT_WITH_EXAMPLE = (
    "We must put the answer in [ANSWER] and [/ANSWER].\n\nSo:\n\n[ANSWER]\nassert f(16) == 17\n[/ANSWER]\n\n"
    "But wait, the first example had:\n\n[ANSWER]\nassert f(1) == 2\n[/ANSWER]"
)
RESTATED_OUTPUT = (
    "Provide the full assertion in [ANSWER] and [/ANSWER] tags, following the examples.\n\n"
    "Executing f(16) returns 17.\n\n[ANSWER]\nassert f(16) == 17\n[/ANSWER]"
)


@pytest.mark.parametrize(
    ("direction", "response"),
    [("input", RESTATED_INPUT), ("input", RESTATED_INPUT_WITH_EXAMPLE), ("output", RESTATED_OUTPUT)],
    ids=["input", "input-later-example", "output"],
)
def test_skips_restated_answer_tag_instruction(direction, response):
    assert _grade(direction, response) == 1.0


@pytest.mark.parametrize(
    ("direction", "response", "expected"),
    [
        ("input", "[ANSWER]\nassert f(16) == 17\n[/ANSWER]", 1.0),
        ("input", "[ANSWER]\nassert f(15) == 17\n[/ANSWER]", 0.0),
        ("output", "[ANSWER]\nassert f(16) == 17\n[/ANSWER]", 1.0),
        ("output", "[ANSWER]\nassert f(16) == 18\n[/ANSWER]", 0.0),
    ],
    ids=["input-correct", "input-wrong", "output-correct", "output-wrong"],
)
def test_single_answer_span_grades_its_assertion(direction, response, expected):
    assert _grade(direction, response) == expected
