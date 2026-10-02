# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build mathematical tasks from explicit source-owned field extraction."""

import json
from dataclasses import replace

from pydantic import JsonValue

from taskcompendium.models import ConversationInput, ResourceVisibility, TaskSpec, TextMessage, task_resource
from taskcompendium.pipeline.datasets.hf_math import normalize_math
from taskcompendium.pipeline.models import ImportRejection, RawRow


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
