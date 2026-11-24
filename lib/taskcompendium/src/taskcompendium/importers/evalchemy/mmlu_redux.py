# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Redux zero-shot chat prompts in ZeroEval's JSON answer format.

The prompt is ZeroEval's ``MCQA`` template (``src/templates/MCQA.py`` in WildEval/ZeroEval,
Apache-2.0, at 8c1485edf12c6efb5f69135a562927c5ad484059), filled with ``str.replace`` in the same
order as its ``apply_mc_template``, including the trailing spaces on four of its lines.
"""

import string
from collections.abc import Sequence

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

ZEROEVAL_MCQA_PROMPT = (
    "\n## Question: \n\n{question}\n\n## Choices:\n\n{choices}\n\n## Instruction \n\n"
    "Please answer this question by first reasoning and then selecting the correct choice. \n"
    "Present your reasoning and solution in the following json format. \n"
    'Please show your choice in the `answer` field with only the choice letter, e.g.,`"answer": "C"`.\n\n'
    '```json\n{\n    "reasoning": "___",\n    "answer": "___"\n}\n```\n'
)


def zeroeval_mcqa_prompt(question: str, choices: Sequence[str]) -> str:
    choice_lines = "".join(f"- ({string.ascii_uppercase[index]}) {choice}\n" for index, choice in enumerate(choices))
    return ZEROEVAL_MCQA_PROMPT.replace("{question}", question).replace("{choices}", choice_lines)


def convert_mmlu_redux(row: RawRow, context: ConversionContext) -> TaskSpec:
    """Convert a ZeroEval ``mmlu-redux`` row, whose gold is the text of one of its choices."""
    choices = list(row.data["choices"])
    expected = string.ascii_uppercase[choices.index(row.data["correct_answer"])]
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=zeroeval_mcqa_prompt(row.data["question"], choices)),),
        package=source_package(
            "zeroeval_mcqa", {"expected": expected}, environment=required_grader_environment(context)
        ),
        evidence={**row.data, "choices": choices},
    )
    return task.model_copy(update={"tags": ("eval", "source:zeroeval", "benchmark:MMLU-Redux")})


def mmlu_redux_item(task: TaskSpec) -> EvaluationItem:
    return chat_item(task).model_copy(update={"groups": {"subject": source_evidence(task)["config"]}})
