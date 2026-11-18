# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA zero-shot chat prompts.

The prompt is the authors' ``config/prompt/zero-shot.yaml`` filled as in ``infer/data_loader.py``
(SuperGPQA/SuperGPQA, ODC-By, at e31e8bf3e7089aef808be5ddecafd4538dee6e41). It always lists A to J as
allowed letters, including for questions with fewer than ten options. The grader reads only the final
answer, as the runner does for API reasoning models.
"""

from collections.abc import Sequence

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.importers.evalchemy.scorers.supergpqa import MAX_SEARCH_TIME
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

SUPERGPQA_PROMPT = (
    "Answer the following multiple choice question. There is only one correct answer. The last line of your "
    "response should be in the format 'Answer: $LETTER' (without quotes), where LETTER is one of A, B, C, D, E, "
    "F, G, H, I, or J.\n\n{}\n"
)


def supergpqa_prompt(question: str, options: Sequence[str]) -> str:
    choices = "\n".join(f"{chr(65 + index)}) {option}" for index, option in enumerate(options))
    return SUPERGPQA_PROMPT.format(question + "\n" + choices)


def convert_supergpqa(row: RawRow, context: ConversionContext) -> TaskSpec:
    options = list(row.data["options"])
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=supergpqa_prompt(row.data["question"], options)),),
        package=source_package(
            "supergpqa",
            {"expected": row.data["answer_letter"], "options": options},
            environment=required_grader_environment(context),
            timeout=60 + MAX_SEARCH_TIME,
        ),
        evidence={**row.data, "options": options},
    )
    return task.model_copy(update={"tags": ("eval", "source:supergpqa", "benchmark:SuperGPQA")})


def supergpqa_item(task: TaskSpec) -> EvaluationItem:
    evidence = source_evidence(task)
    return chat_item(task, response_selection="final_only").model_copy(
        update={"groups": {"discipline": evidence["discipline"], "difficulty": evidence["difficulty"]}}
    )
