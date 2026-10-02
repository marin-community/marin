# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build mathematical tasks from explicit source-owned field extraction."""

import json
from collections.abc import Callable
from dataclasses import replace

from pydantic import JsonValue

from taskcompendium.models import ConversationInput, ResourceVisibility, TaskSpec, TextMessage, task_resource
from taskcompendium.pipeline.datasets.hf_math import math_controls, normalize_math
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)


def math_task(
    row: RawRow, events: tuple[TextMessage, ...], expected: str, evidence: dict[str, JsonValue]
) -> TaskSpec | ImportRejection:
    """Bind an extracted reference and private evidence to public source messages."""
    problem = "\n\n".join(event.content for event in events)
    task = normalize_math(replace(row, data={"problem": problem, "answer": expected}), "problem", "answer")
    if isinstance(task, ImportRejection):
        return task
    resource = task_resource(
        "/reference/source-evidence.json", json.dumps(evidence, ensure_ascii=False).encode(), ResourceVisibility.VERIFIER
    )
    return task.model_copy(update={"context": ConversationInput(events=events), "resources": (resource,)})


def field_math_task(
    row: RawRow, problem_key: str, answer_key: str, evidence_keys: tuple[str, ...]
) -> TaskSpec | ImportRejection:
    """Extract a direct problem/reference pair while retaining source evidence."""
    problem, expected = row.data.get(problem_key), row.data.get(answer_key)
    if not isinstance(problem, str) or not isinstance(expected, str):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail=f"{problem_key} and {answer_key} strings are required"
        )
    evidence = {key: row.data[key] for key in evidence_keys}
    return math_task(row, (TextMessage(role="user", content=problem),), expected, evidence)


def math_recipe(
    name: str,
    source: HFSource,
    normalize: Callable[[RawRow], TaskSpec | ImportRejection],
    intended_use: IntendedUse,
    rubric: ReviewRubric,
) -> DatasetRecipe:
    """Bind a typed math source to the common comparator controls."""
    return DatasetRecipe(
        name=name,
        version=f"{name}-v1",
        source=source,
        normalize=normalize,
        intended_use=intended_use,
        rubric=rubric,
        check_suite=CheckSuite(
            id=f"{name}-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )
