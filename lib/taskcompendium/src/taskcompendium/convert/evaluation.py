# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small evaluation converters that preserve benchmark prompts and private source evidence."""

import json
from typing import Any

from taskcompendium.convert.answers import EVIDENCE_PATH, source_defect
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.importers.evalchemy.grader import source_package
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext, required_grader_environment
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes


def source_evidence(task: TaskSpec) -> dict[str, Any]:
    resource = next(resource for resource in task.resources.verifier if resource.path == EVIDENCE_PATH)
    return json.loads(resource_bytes(resource))


def boxed_math_task(
    row: RawRow,
    context: ConversionContext,
    *,
    prompt_template: str,
    answer_field: str,
    benchmark: str,
) -> TaskSpec | ImportRejection:
    """Render an exact source prompt and retain the boxed-math grading protocol."""
    answer = row.data.get(answer_field)
    if answer is None or not str(answer).strip():
        return source_defect("missing_reference", f"The source requires {answer_field}")
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=prompt_template.format_map(row.data)),),
        package=source_package(
            "boxed_math", {"expected": str(answer)}, environment=required_grader_environment(context)
        ),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{benchmark}")})
