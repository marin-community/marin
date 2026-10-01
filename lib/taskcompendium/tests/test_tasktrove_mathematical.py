# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean mathematical import and direct-chat Harbor coverage."""

import hashlib
import io
import json
import tarfile

import pytest
from tasktrove_verify.spec import MathType

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.mathematical import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationToolCall,
    ConversationTrace,
    TextMessage,
    VerifierKind,
)
from taskcompendium.submission import (
    AnswerCall,
    GradingAttempt,
    JsonAnswer,
    PlainText,
    render_instruction,
)
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial

SOURCE = "laion__nemotron-gym-math-v5"
PATH = "nemotron-math-05952d2a.tar.gz"
RELEASE_URI = "https://huggingface.co/datasets/open-athena/task-trove"
RELEASE_REVISION = "9065fa568394f286dab0081e43dc76fc87c48984"


def _archive_bytes(expected: str = r"\frac{1}{3}", math_type: str = "scalar") -> bytes:
    task_toml = f"""version = "1.0"

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 600.0

[metadata]
family = "math-answer"
tasktrove_source = "{SOURCE}"
tasktrove_path = "{PATH}"
converter = "nemotron_math"
template_id = "math-template-1"
mode = "math"
tags = ["math", "nemotron", "source tag"]
"""
    verifier_toml = f'mode = "math"\nexpected = {json.dumps(expected)}\nmath_type = "{math_type}"\n'
    files = {
        "task.toml": task_toml.encode(),
        "instruction.md": b"Compute the exact value. Preserve the requested fraction form.\n",
        "environment/Dockerfile": b"FROM python:3.11-slim\n",
        "tests/test.sh": b"#!/bin/bash\nexec tasktrove-verify /tests/verifier.toml\n",
        "tests/verifier.toml": verifier_toml.encode(),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def _archive(expected: str = r"\frac{1}{3}", math_type: str = "scalar"):
    return read_archive(_archive_bytes(expected, math_type), SOURCE, PATH, RELEASE_URI, RELEASE_REVISION)


def test_import_math_preserves_gold_tags_and_source_identity():
    archive_bytes = _archive_bytes()
    archive = read_archive(archive_bytes, SOURCE, PATH, RELEASE_URI, RELEASE_REVISION)
    result = import_task(archive)

    assert result.specification.answer_type is AnswerType.NUMBER
    assert result.specification.verifier.kind is VerifierKind.MATHEMATICAL_ANSWER
    assert json.loads(result.specification.verifier.parameters_json) == {
        "expected": r"\frac{1}{3}",
        "math_type": "scalar",
    }
    assert result.tags == ("math", "nemotron", "source tag")
    assert result.specification.tags == result.tags
    assert result.source_evidence.source == SOURCE
    assert result.source_evidence.path == PATH
    assert result.source_evidence.archive_sha256 == hashlib.sha256(archive_bytes).hexdigest()
    assert result.specification.source.revision == RELEASE_REVISION


@pytest.mark.parametrize(
    ("expected", "math_type"),
    [("(1, 2)", "scalar"), ("[1, 2]", "list"), ("\\{1,2\\}", "set"), ("x=2", "equation")],
)
def test_import_math_non_scalar_or_non_finite_answers_use_text(expected: str, math_type: str):
    result = import_task(_archive(expected, math_type))

    assert result.specification.answer_type is AnswerType.TEXT
    parameters = json.loads(result.specification.verifier.parameters_json)
    assert parameters["expected"] == expected
    assert parameters["math_type"] == MathType(math_type)


def test_import_math_removes_only_file_submission_scaffolding():
    archive = _archive()
    archive.files["instruction.md"] = (
        b"Compute $\\sqrt{2}$ subject to x > 0.\n"
        b"Write your final answer to `/app/answer.txt`.\n"
        b"Return the exact radical.\n"
    )

    prompt = import_task(archive).specification.context.events[0].content

    assert prompt == "Compute $\\sqrt{2}$ subject to x > 0.\nReturn the exact radical."


def test_import_openmath_removes_transport_formatting_without_changing_the_problem():
    archive = _archive()
    archive.files["task.toml"] = archive.files["task.toml"].replace(b"math-template-1", b"5ee94cf985a9")
    archive.files["instruction.md"] = (
        b"You are a careful mathematical reasoner. Read the problem below and show your work, "
        b"then write your final answer at the path `/app/answer.txt`.\n"
        b"Format requirement: the answer file should contain `\\boxed{<answer>}` on its\n"
        b"last meaningful line. The verifier extracts the last `\\boxed{...}` it finds.\n"
        b"Evaluate one third. Put the answer (and only the answer) inside \\boxed{}.\n"
        b"Write the final answer to `/app/answer.txt`. Put the final expression in `\\boxed{...}` when practical."
    )

    result = import_task(archive)
    prompt = result.specification.context.events[0].content

    assert prompt.startswith("You are a careful mathematical reasoner. Read the problem below and show your work.")
    assert "Evaluate one third." in prompt
    assert "answer file" not in prompt
    assert "/app/" not in prompt and "answer.txt" not in prompt
    assert "\\boxed" not in prompt
    assert "verifier" not in prompt.lower() and "grader" not in prompt.lower()
    assert result.specification.verifier.kind is VerifierKind.MATHEMATICAL_ANSWER


def test_import_sankalp_file_named_instruction_preserves_the_math_question():
    archive = _archive()
    archive.files["task.toml"] = archive.files["task.toml"].replace(b"math-template-1", b"5ee94cf985a9")
    archive.files["instruction.md"] = (
        b"Please place your final answer in a file named `/app/answer.txt`.\n"
        b"Evaluate the exact value of one third.\n"
        b"Write the final answer to `/app/answer.txt`. Put the final expression in `\\boxed{...}` when practical."
    )

    result = import_task(archive)
    prompt = result.specification.context.events[0].content

    assert prompt.startswith("Evaluate the exact value of one third.")
    assert "/app/" not in prompt and "answer.txt" not in prompt
    assert "\\boxed" not in prompt
    assert result.tags == ("math", "nemotron", "source tag")
    assert result.specification.verifier.kind is VerifierKind.MATHEMATICAL_ANSWER


def test_import_nemotron_v5_removes_terminal_submission_block():
    archive = _archive()
    archive.files["task.toml"] = archive.files["task.toml"].replace(b"math-template-1", b"334fa3ab07c7")
    archive.files["instruction.md"] = "\n".join(
        (
            "You are a careful mathematical reasoner. Read the problem below and show your work, "
            "then write your final answer at the path `/app/answer.txt`.",
            "",
            "Format requirement: the answer file should contain `\\boxed{<answer>}` on its",
            "last meaningful line. The verifier extracts the last `\\boxed{...}` it finds.",
            "",
            "Solve the stated problem. Remember to put your final answer inside \\boxed{}.",
            "",
            "## Submitting your answer (IMPORTANT)",
            "You are a terminal agent. Your chat reply is NOT graded — the grader only reads the file "
            "`/app/answer.txt` inside the sandbox. You MUST write your solution, ending with the final answer "
            "in \\boxed{...} to `/app/answer.txt` by RUNNING A SHELL COMMAND, e.g. a heredoc:",
            "",
            "    cat > /app/answer.txt <<'EOF'",
            "    <your solution, ending with the final answer in \\boxed{...} here>",
            "    EOF",
            "",
            "Then confirm it with `cat /app/answer.txt`. An empty or missing `/app/answer.txt` scores 0 "
            "regardless of what you wrote in your reply.",
            "",
            "Reliability note: write a provisional `/app/answer.txt` as soon as you have a",
            "candidate answer, then refine it if needed. This preserves a scoreable answer if",
            "later computation consumes the remaining agent time.",
        )
    ).encode()

    result = import_task(archive)
    prompt = result.specification.context.events[0].content

    assert "Solve the stated problem." in prompt
    assert "/app/" not in prompt and "answer.txt" not in prompt
    assert "RUNNING A SHELL COMMAND" not in prompt
    assert "graded" not in prompt.lower() and "grader" not in prompt.lower()
    assert "\\boxed" not in prompt


def test_import_all_puzzles_preserves_math_answer_shape_without_transport_instruction():
    archive = _archive(expected="(-5.167, 4.693)")
    archive.files["task.toml"] = archive.files["task.toml"].replace(b"nemotron_math", b"all_puzzles")
    archive.files["instruction.md"] = (
        b"# Geometry puzzle\n\n"
        b"<!-- laion v2 puzzles deliverable: answer.txt -->\n\n"
        b"## Deliverable (REQUIRED)\n\n"
        b"Write ONLY your final answer to **`/app/answer.txt`** (a single line, no\n"
        b"explanation). The verifier reads that file and compares it to the gold answer\n"
        b"using the format described in the problem statement:\n"
        b"- coordinates as (x, y) rounded to 3 decimals where applicable\n\n"
        b"## Problem Statement\nFind the orthocenter. Return only the coordinates.\n"
    )

    specification = import_task(archive).specification
    prompt = specification.context.events[0].content

    assert "answer.txt" not in prompt
    assert "coordinates as (x, y) rounded to 3 decimals" in prompt
    assert "Find the orthocenter." in prompt
    assert "Return only the coordinates." not in prompt


def test_import_preserves_boxed_notation_when_it_is_part_of_the_math_problem():
    archive = _archive()
    archive.files["instruction.md"] = b"Simplify the expression \\boxed{x} + 1."

    prompt = import_task(archive).specification.context.events[0].content

    assert prompt == r"Simplify the expression \boxed{x} + 1."


@pytest.mark.parametrize(
    ("expected", "math_type", "answer_type", "correct", "incorrect"),
    [
        (r"\frac{1}{3}", "scalar", AnswerType.NUMBER, "0.3333333333333333", "2/3"),
        ("[1/2, x+1]", "list", AnswerType.TEXT, "0.5, 1+x", "0.5, 1-x"),
    ],
    ids=("finite-scalar", "symbolic-list"),
)
@pytest.mark.parametrize("answer_format", ["plain", "json", "answer-call"])
async def test_imported_math_uses_selected_submission_convention(
    expected: str,
    math_type: str,
    answer_type: AnswerType,
    correct: str,
    incorrect: str,
    answer_format: str,
):
    archive = _archive(expected, math_type)
    archive.files["instruction.md"] = b"Synthetic task: simplify the given expression."
    specification = import_task(archive).specification
    convention = {
        "plain": PlainText(id="plain"),
        "json": JsonAnswer(id="json"),
        "answer-call": AnswerCall(id="answer-call"),
    }[answer_format]
    prompt = render_instruction(specification, convention)

    assert specification.answer_type is answer_type
    assert "Synthetic task: simplify the given expression." in prompt
    assert "/app/" not in prompt and "answer.txt" not in prompt
    assert "grader" not in prompt.lower() and "verifier" not in prompt.lower()
    assert r"\boxed" not in prompt

    def attempt(candidate: str) -> GradingAttempt:
        if isinstance(convention, AnswerCall):
            final = AssistantToolCalls(
                calls=(ConversationToolCall(call_id="answer", name="submit_answer", arguments={"answer": candidate}),)
            )
        elif isinstance(convention, JsonAnswer):
            final = TextMessage(role="assistant", content=json.dumps({"answer": candidate}))
        else:
            final = TextMessage(role="assistant", content=candidate)
        return GradingAttempt(ConversationTrace(events=(*specification.context.events, final)), object())

    positive = await grade_answer(specification, convention, attempt(correct))
    negative = await grade_answer(specification, convention, attempt(incorrect))

    assert (positive.status, positive.reward) == (Outcome.GRADED, 1.0)
    assert (negative.status, negative.reward) == (Outcome.GRADED, 0.0)


@pytest.mark.parametrize(
    "instruction",
    [
        "Use Python to calculate the answer.",
        "Read the provided data file before solving.",
        "Create /app/answer.txt with the result.",
    ],
)
def test_import_math_rejects_tool_requirements_and_unknown_file_templates(instruction: str):
    archive = _archive()
    archive.files["instruction.md"] = instruction.encode()

    with pytest.raises(ValueError, match="Invalid TaskTrove mathematical archive"):
        import_task(archive)


def test_import_math_rejects_unrecognized_task_resources():
    archive = _archive()
    archive.files["data/input.csv"] = b"secret,resource\n"

    with pytest.raises(ValueError, match="unsupported task resources"):
        import_task(archive)


async def test_imported_math_grades_correct_and_incorrect_chat_answers():
    specification = import_task(_archive()).specification
    convention = PlainText(id="plain")

    async def grade(response: str):
        return await grade_answer(
            specification,
            convention,
            GradingAttempt(
                ConversationTrace(
                    events=(*specification.context.events, TextMessage(role="assistant", content=response))
                ),
                object(),
            ),
        )

    correct, incorrect = await grade(r"\boxed{0.3333333333333333}"), await grade(r"\boxed{\frac{2}{3}}")

    assert (correct.status, correct.reward) == (Outcome.GRADED, 1.0)
    assert (incorrect.status, incorrect.reward) == (Outcome.GRADED, 0.0)


async def test_imported_math_runs_through_direct_chat_harbor(tmp_path):
    specification = import_task(_archive()).specification
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )

    correct = await run_replay_trial(
        task,
        {"role": "assistant", "content": r"\boxed{0.3333333333333333}"},
        tmp_path / "trials",
        "math-correct",
    )
    incorrect = await run_replay_trial(
        task,
        {"role": "assistant", "content": r"\boxed{\frac{2}{3}}"},
        tmp_path / "trials",
        "math-incorrect",
    )

    correct_outcome = json.loads((tmp_path / "trials/math-correct/verifier/taskcompendium-result.json").read_text())
    incorrect_outcome = json.loads((tmp_path / "trials/math-incorrect/verifier/taskcompendium-result.json").read_text())
    assert correct.exception_info is None, correct.exception_info
    assert incorrect.exception_info is None, incorrect.exception_info
    assert correct_outcome == {"status": "graded", "reward": 1.0, "error": None}
    assert incorrect_outcome == {"status": "graded", "reward": 0.0, "error": None}
