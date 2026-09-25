# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron multichallenge-advanced: a checklist judge over the final turn of a conversation.

The old grader (``rewardkit`` driven by ``tests/judge.toml``) showed a judge model the conversation
transcript plus the candidate's final response and asked one yes/no question per criterion,
scoring the mean. That is ``judge`` mode with the ``checklist`` rubric: the questions come from the
``Requirement:`` block of each criterion description and the transcript ships as the judge's context.
"""

import re
import tomllib

from tasktrove_verify.spec import RUBRIC_CHECKLIST, JudgeSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.task_format import OLD_GRADER_LINE, RESPONSE_OUTPUT, drop_dockerfile_lines
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

JUDGE_TOML = "tests/judge.toml"
CONVERSATION = "conversation.txt"
_REQUIREMENT = re.compile(r"Requirement:\s*(.+)\Z", re.DOTALL)
_NEGATED = "Pass when the candidate clearly does not satisfy the condition queried by this requirement."


def _criteria(judge_toml: str) -> list[str]:
    config = tomllib.loads(judge_toml)
    questions = []
    for criterion in config.get("criterion", []):
        description = str(criterion.get("description", ""))
        match = _REQUIREMENT.search(description)
        question = match.group(1).strip() if match else ""
        if _NEGATED in description:
            question = f"The candidate must answer no to this question: {question}"
        questions.append(question)
    return questions


def convert_nemotron_multichallenge(task: TaskFiles) -> ConvertedTask | Rejected:
    judge_toml = task.get_text(JUDGE_TOML)
    transcript = task.get_text(f"tests/{CONVERSATION}")
    if judge_toml is None or transcript is None:
        return Rejected(ConvertStatus.NULL_GRADER, "judge.toml or conversation.txt missing")
    criteria = _criteria(judge_toml)
    if not criteria or any(not question for question in criteria):
        return Rejected(ConvertStatus.NULL_GRADER, f"{len(criteria)} criteria, some without a requirement")
    spec = JudgeSpec(
        rubric=RUBRIC_CHECKLIST,
        criteria=tuple(criteria),
        context=CONVERSATION,
        exact_gate=False,
        output=RESPONSE_OUTPUT,
    )
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=spec,
        dockerfile=drop_dockerfile_lines(task.text(DOCKERFILE), OLD_GRADER_LINE),
        tags=("instruction-following", "multi-turn", "judge", "checklist", "nemotron"),
        data_files={f"tests/{CONVERSATION}": transcript.encode()},
    )


CONVERTER = Converter(
    name="nemotron_multichallenge",
    keys=(ConverterKey("llm-judge-freeform", frozenset({"tests/test.sh", "tests/sitecustomize.py"})),),
    convert=convert_nemotron_multichallenge,
)
