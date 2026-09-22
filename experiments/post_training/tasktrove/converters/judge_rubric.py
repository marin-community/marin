# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rubric-only freeform tasks: a checklist judge with no reference answer.

The old grader (``rewardkit`` driven by ``tests/judge.toml``) showed a judge model the user's
request, the response and a short rubric, and asked for one score. ``tests/verifier_data.json``
carries that rubric in one of two shapes: a ``rubric`` list of ``{id, criteria}`` entries (the
StackExchange, Glaive and WizardLM sandboxes) or a numbered ``principle`` string (the Nemotron
safety set). Either becomes ``judge`` mode with the ``checklist`` rubric, one yes/no question per
entry, so the reward is the fraction of criteria the judge accepts.

There is no gold answer anywhere in these tasks: the reward is the judge's reading of the rubric.
Every converted task carries the ``no-reference`` tag so a mix can select or exclude them as a
group.
"""

import re

from tasktrove_verify.spec import RUBRIC_CHECKLIST, JudgeSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    Converter,
    ConverterKey,
    ConvertStatus,
    Rejected,
)
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.task_format import OLD_GRADER_LINE, RESPONSE_OUTPUT, drop_dockerfile_lines
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

NO_REFERENCE_TAG = "no-reference"
_NUMBERED = re.compile(r"^\s*\d+[.)]\s*")
# The judge prompt names the source community; that name becomes the task's domain tag.
DOMAIN_TAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Stack Overflow", ("stackexchange", "stackoverflow", "code")),
    ("Super User", ("stackexchange", "superuser")),
    ("Unix & Linux", ("stackexchange", "unix", "shell")),
    ("Tezos", ("stackexchange", "tezos")),
    ("Code Review", ("stackexchange", "codereview", "code")),
    ("Glaive", ("code-assistant", "code")),
    ("general instruction-following", ("general-assistant",)),
)


def _criteria(data: dict) -> list[str]:
    rubric = data.get("rubric")
    if isinstance(rubric, list):
        return [str(entry.get("criteria", "")).strip() for entry in rubric if isinstance(entry, dict)]
    principle = data.get("principle")
    if isinstance(principle, str):
        return [_NUMBERED.sub("", line).strip() for line in principle.splitlines() if line.strip()]
    return []


def _domain_tags(data: dict) -> tuple[str, ...]:
    if isinstance(data.get("principle"), str):
        return ("safety",)
    prompt = str(data.get("judge_system_prompt", ""))
    for needle, tags in DOMAIN_TAGS:
        if needle in prompt:
            return tags
    return ("freeform",)


def convert_judge_rubric(task: TaskFiles) -> ConvertedTask | Rejected:
    data = verifier_data(task)
    criteria = _criteria(data)
    if not criteria or any(not criterion for criterion in criteria):
        return Rejected(ConvertStatus.NULL_GRADER, f"{len(criteria)} rubric criteria, some empty")
    question = str(data.get("instruction", "")).strip()
    if not question:
        return Rejected(ConvertStatus.NULL_GRADER, "no instruction in verifier_data.json")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=JudgeSpec(
            rubric=RUBRIC_CHECKLIST,
            criteria=tuple(criteria),
            question=question,
            exact_gate=False,
            output=RESPONSE_OUTPUT,
        ),
        dockerfile=drop_dockerfile_lines(task.text(DOCKERFILE), OLD_GRADER_LINE),
        tags=("judge", "rubric", NO_REFERENCE_TAG, *_domain_tags(data)),
    )


CONVERTER = Converter(
    name="judge_rubric",
    keys=(ConverterKey("llm-judge-freeform", frozenset({"tests/test.sh"})),),
    convert=convert_judge_rubric,
)
