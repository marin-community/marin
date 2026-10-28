# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small evaluation converters that preserve benchmark prompts and private source evidence."""

import json
from typing import Any

from verifyit.spec import MathProfile, MathSpec

from taskcompendium.convert.answers import EVIDENCE_PATH, answer_task, source_defect
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes


def source_evidence(task: TaskSpec) -> dict[str, Any]:
    resource = next(resource for resource in task.resources.verifier if resource.path == EVIDENCE_PATH)
    return json.loads(resource_bytes(resource))


def boxed_math_task(
    row: RawRow,
    _context: ConversionContext,
    *,
    prompt_template: str,
    answer_field: str,
    benchmark: str,
) -> TaskSpec | ImportRejection:
    """Render an exact source prompt and retain the boxed-math grading protocol."""
    answer = row.data.get(answer_field)
    if answer is None or not str(answer).strip():
        return source_defect("missing_reference", f"The source requires {answer_field}")
    task = answer_task(
        row,
        prompt=prompt_template.format_map(row.data),
        spec=MathSpec(expected=str(answer), profile=MathProfile.EVALCHEMY),
        evidence=dict(row.data),
    )
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{benchmark}")})
