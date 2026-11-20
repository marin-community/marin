# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LEXam multiple-choice chat prompts in the authors' ``###X###`` answer format.

The prompt is ``MCQ_PROMPT["letters"]`` from ``litellm_eval.py`` in LEXam-Benchmark/LEXam
(Apache-2.0) at 64f2e548a4a84b6154be3498dd5fed8966990bca, including the leading and trailing spaces
around its answer-format example. Choices are appended to the question as ``\\n{letter}. {choice}``.
"""

import ast
import string
from collections.abc import Sequence
from typing import Any

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

LEXAM_MCQ_PROMPT = (
    "You are an expert in {course_name} and address legal issues in a structured, exam-style manner.\n"
    "You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.\n"
    "Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, "
    "consider legal frameworks beyond Swiss law as well.\n"
    "\n"
    "Please reason through the question step by step, using a chain-of-thought approach:\n"
    "- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.\n"
    "- Issue Identification: What legal issue(s) arise from the facts?\n"
    "- Rule Explanation: What legal rules or principles are relevant, and what are their sources "
    "(e.g., statutes, case law, doctrine)?\n"
    "- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, "
    "exceptions, or competing interpretations.\n"
    "- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.\n"
    "- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification "
    "for why it best fits the legal analysis.\n"
    "\n"
    "Format your final answer as follows:\n"
    " Correct Answer: ###C### \n"
    "\n"
    "Question:\n"
    " {question}\n"
    "\n"
    "Answer:"
)


def lexam_mcq_prompt(course: str, question: str, choices: Sequence[str]) -> str:
    lines = "".join(f"\n{string.ascii_uppercase[index]}. {choice}" for index, choice in enumerate(choices))
    return LEXAM_MCQ_PROMPT.format(course_name=course, question=question + lines)


def convert_lexam_mcq(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Convert a LEXam MCQ row, whose ``choices`` is a Python list literal and ``gold`` a 0-based index into it."""
    choices = ast.literal_eval(row.data["choices"])
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=lexam_mcq_prompt(row.data["course"], row.data["question"], choices)),),
        package=source_package(
            "lexam_mcq",
            {"expected": string.ascii_uppercase[row.data["gold"]]},
            environment=required_grader_environment(context),
        ),
        evidence={**row.data, "choices": choices},
    )
    return task.model_copy(update={"tags": ("eval", "source:lexam", "benchmark:LEXam")})


def is_english(row: dict[str, Any], _context: ConversionContext) -> bool:
    return row["language"] == "en"


def is_german(row: dict[str, Any], _context: ConversionContext) -> bool:
    return row["language"] == "de"


def lexam_mcq_item(task: TaskSpec) -> EvaluationItem:
    """Grade only the final answer."""
    return chat_item(task, response_selection="final_only")
