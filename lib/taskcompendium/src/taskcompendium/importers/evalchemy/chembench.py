# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ChemBench zero-shot chat prompts from the authors' instruction-tuned prompter.

Templates, option lettering and markup stripping follow ``INSTRUCT_TEMPLATE_1``/``_2``,
``_base_multiple_choice_prompt`` and ``post_process_prompts`` in lamalab-org/chembench (MIT) at
45f8bad062fe552810c52be3a328d5da8597ed30, with chain of thought and option permutation off, as in
its defaults. A row is multiple choice when its metrics include ``multiple_choice_grade``.
"""

import json
import re

from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

MULTIPLE_CHOICE_TEMPLATE = (
    "The following is a multiple choice question about chemistry.\n"
    "Please answer by responding with the letter of the correct answer.{cot}\n\n"
    "Question: {question}\n\n"
    "Options:\n{answers}\n\n"
    "You MUST include the letter(s) of the correct answer (separated by comma if there are many) within the "
    "following tags: [ANSWER] and [/ANSWER].\n"
    "For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is comma- or space-separated list of the correct "
    "letters. Always answer in exactly this format of comma-separated letters between the two tags, even if you "
    "are unsure. We require this because we use automatic parsing."
)
NUMERIC_TEMPLATE = (
    "The following is a question about chemistry.{cot}\n\n"
    "Question: {question}\n\n"
    "You MUST include the final answer within the following tags: [ANSWER] and [/ANSWER].\n"
    "For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is only one number. Always answer in exactly this "
    "format, with ONE NUMBER between the two tags, even if you are unsure. Use dots as decimal separator. We "
    "require this because we use automatic parsing."
)
# Applied in the source order: chemical equations, math, SMILES, reaction SMILES, physical units.
MARKUP_PATTERNS = (
    re.compile(r"\\ce\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
    re.compile(r"\$([^$]+)\$"),
    re.compile(r"\[START_SMILES\](.*?)\[END_SMILES\]"),
    re.compile(r"\[START_RXNSMILES\](.*?)\[END_RXNSMILES\]"),
    re.compile(r"\\pu\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"),
)
MULTIPLE_CHOICE_METRIC = "multiple_choice_grade"


def strip_markup(prompt: str) -> str:
    """Replace ChemBench's LaTeX and SMILES wrappers with their contents."""
    for pattern in MARKUP_PATTERNS:
        prompt = pattern.sub(lambda match: match.group(1), prompt)
    return prompt


def option_letters(count: int) -> list[str]:
    return [chr(ord("A") + index) for index in range(count)]


def multiple_choice_prompt(question: str, options: list[str]) -> str:
    letters = option_letters(len(options))
    answers = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options, strict=True))
    return strip_markup(MULTIPLE_CHOICE_TEMPLATE.format(question=question, answers=answers, cot=""))


def numeric_prompt(question: str) -> str:
    return strip_markup(NUMERIC_TEMPLATE.format(question=question, cot=""))


def chembench_topic(task: TaskSpec) -> str:
    """The topic is the source file's directory, such as ``organic_chemistry``."""
    return task.source.row.split("/", 1)[0]


def convert_chembench(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Convert a single-example ChemBench row into a multiple-choice or numeric question."""
    [example] = row.data["examples"]
    question = example["input"]
    if MULTIPLE_CHOICE_METRIC in row.data["metrics"]:
        target_scores = json.loads(example["target_scores"])
        options = list(target_scores)
        letters = option_letters(len(options))
        correct = [letter for letter, score in zip(letters, target_scores.values(), strict=True) if score == 1]
        if not correct:
            raise ValueError(f"ChemBench row {row.source.row} has no correct option")
        prompt = multiple_choice_prompt(question, options)
        config = {"kind": "multiple_choice", "letters": letters, "correct": correct}
    else:
        if example["target_scores"] is not None:
            raise ValueError(f"ChemBench row {row.source.row} has options but no multiple-choice metric")
        prompt = numeric_prompt(question)
        config = {"kind": "numeric", "target": float(example["target"])}
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=prompt),),
        package=source_package("chembench", config, environment=required_grader_environment(context)),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:chembench", "benchmark:ChemBench")})


def chembench_item(task: TaskSpec) -> EvaluationItem:
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"topic": chembench_topic(task)}}
    )
