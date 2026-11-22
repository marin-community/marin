# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SciRecipe prompts from InternScience/Thoth a155710079da47bb26a021646fa84d5544aa180e."""

import random
from collections.abc import Mapping
from typing import Any

from taskcompendium.convert.answers import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import ChatGeneration, EvaluationItem
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import RawRow

SYSTEM_PROMPT = (
    "You are a bio-expert scientific assistant. Produce a single output containing "
    "four blocks in order: <think>, <key>, <orc>, <note>. "
    "<think> contains reasoning, <key> contains step-wise JSON actions with atomic "
    "action/objects/parameters, <orc> is a readable summary, and <note> contains "
    "safety comments. Output must begin with <think>."
)


def scirecipe_prompt(document: Mapping[str, Any]) -> str:
    actions_new = document.get("action_new")
    actions = actions_new if isinstance(actions_new, list) else document["action"]
    if not isinstance(actions, list) or not document["exp_goal"].strip():
        raise ValueError("SciRecipe requires a question and an action list")
    actions = [action for action in actions if isinstance(action, str) and action.strip()]
    if not isinstance(actions_new, list):
        random.Random(42).shuffle(actions)
    quoted = '", "'.join(action.strip().lower() for action in actions)
    return (
        document["exp_goal"].strip()
        + " Please provide the experimental protocol to solve this problem. "
        + f'Please note that you can only use the following actions: "{quoted}".'
    )


def convert_scirecipe(row: RawRow, context: ConversionContext) -> TaskSpec:
    task = conversation_task(
        row,
        events=(
            TextMessage(role="system", content=SYSTEM_PROMPT),
            TextMessage(role="user", content=scirecipe_prompt(row.data)),
        ),
        package=source_package(
            "scirecipe",
            {"key": row.data["key"].strip(), "orc": row.data["orc"].strip()},
            environment=required_grader_environment(context),
            timeout=300,
        ),
        evidence=row.data,
    )
    return task.model_copy(update={"tags": ("eval", "source:scirecipe", "benchmark:scirecipe-eval")})


def scirecipe_item(task: TaskSpec) -> EvaluationItem:
    evidence = source_evidence(task)
    return chat_item(task).model_copy(
        update={
            "operation": ChatGeneration(response_selection="final_only"),
            "groups": {"type": str(evidence["type"]), "level": str(evidence["level"])},
        }
    )
