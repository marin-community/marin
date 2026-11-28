# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect one row's task format from its payload structure; declared must equal detected."""

import json
from dataclasses import dataclass
from typing import Any

from finestore.rl.curation import TaskFormat
from pydantic import ValidationError

from taskcompendium.harbor.native_package import validate_native_package
from taskcompendium.models import TaskSpec

REQUEST_FIELD = "responses_create_params"
AGENT_REF_FIELD = "agent_ref"
TASK_SPEC_FIELD = "task_spec"
LOWERED_TASK_SPEC_FIELD = "lowered_task_spec"


class TaskFormatError(ValueError):
    """A row or source failed task-format detection or enforcement."""


class UndetectableTaskFormatError(TaskFormatError):
    """No format predicate matched the row's payload."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Row payload matches no task format: {detail}")


class AmbiguousTaskFormatError(TaskFormatError):
    """More than one format predicate matched the row's payload; detection never guesses."""

    def __init__(self, formats: list[TaskFormat], detail: str) -> None:
        names = ", ".join(sorted(format_.value for format_ in formats))
        super().__init__(f"Row payload matches several task formats ({names}): {detail}")


class MixedTaskFormatSourceError(TaskFormatError):
    """One source's rows span several task formats; the author must split the source."""

    def __init__(self, counts: dict[TaskFormat, int]) -> None:
        breakdown = ", ".join(f"{format_.value}={count}" for format_, count in sorted(counts.items()))
        super().__init__(
            "Source rows span several task formats " f"({breakdown}); split the source into one source per task format"
        )


class TaskFormatMismatchError(TaskFormatError):
    """The declared task format differs from the detected payload structure."""

    def __init__(self, declared: TaskFormat, detected: TaskFormat) -> None:
        super().__init__(f"Declared task format {declared.value} does not match detected {detected.value}")


@dataclass(frozen=True)
class FormatRow:
    """One ingested row at detection time: its original record and its parsed TaskSpec payload.

    ``raw`` is the record as the format's own loader reads it: a MarinSkyRL blend record, a
    cohort-carrier row (``task_spec``/``lowered_task_spec`` — the frozen population encoding),
    or a converted census record. ``task`` is the parsed TaskSpec, or None when the record
    carries no parseable one. The carrier is transport, never a format signal: a row's format
    comes from its TaskSpec's own layout, or from the blend-record schema.
    """

    raw: dict[str, Any]
    task: TaskSpec | None = None


def _parsed_task(row: FormatRow) -> TaskSpec | None:
    if row.task is not None:
        return row.task
    record = row.raw
    try:
        if TASK_SPEC_FIELD in record:
            return TaskSpec.model_validate_json(record[TASK_SPEC_FIELD])
        if LOWERED_TASK_SPEC_FIELD in record:
            lowered = record[LOWERED_TASK_SPEC_FIELD]
            document = json.loads(lowered) if isinstance(lowered, str) else lowered
            task_value = document.get("task") if isinstance(document, dict) else document
            return TaskSpec.model_validate_json(task_value if isinstance(task_value, str) else json.dumps(task_value))
    except (ValidationError, ValueError, TypeError, KeyError):
        return None
    return None


def _is_blend_record(record: dict[str, Any]) -> bool:
    """The MarinSkyRL blend-record schema the existing blend reader decodes.

    Blend rows carry a request with its input events and one component identity:
    a ``dataset`` key or an ``agent_ref`` with a name (the two shapes the
    existing blend-component reader accepts).
    """
    if TASK_SPEC_FIELD in record or LOWERED_TASK_SPEC_FIELD in record:
        return False
    request = record.get(REQUEST_FIELD)
    if not isinstance(request, dict) or not isinstance(request.get("input"), list):
        return False
    if isinstance(record.get("dataset"), str):
        return True
    agent = record.get(AGENT_REF_FIELD)
    return isinstance(agent, dict) and isinstance(agent.get("name"), str)


def _is_harbor_row(row: FormatRow) -> bool:
    task = _parsed_task(row)
    if task is None:
        return False
    try:
        validate_native_package(task)
    except ValueError:
        return False
    return True


def _is_taskspec_row(row: FormatRow) -> bool:
    """A TaskSpec row that is not a Harbor package."""
    # Conversion supplies a parsed TaskSpec for blend inputs too. That output
    # must not turn the original blend schema into a second format match.
    return not _is_blend_record(row.raw) and _parsed_task(row) is not None and not _is_harbor_row(row)


def _is_marinskyrl_row(row: FormatRow) -> bool:
    return _is_blend_record(row.raw)


# One predicate per shape. The surfaces are disjoint by construction: the blend
# record splits the raw-record level, and a parseable TaskSpec is exactly one of
# {Harbor package, plain TaskSpec}. detect() still counts every match, so a row
# matching none of them or several of them raises instead of guessing.
PREDICATES: tuple[tuple[TaskFormat, Any], ...] = (
    (TaskFormat.MARINSKYRL, _is_marinskyrl_row),
    (TaskFormat.HARBOR, _is_harbor_row),
    (TaskFormat.TASKSPEC, _is_taskspec_row),
)


def detect_task_format(row: FormatRow) -> TaskFormat:
    """Classify one row's payload from its structure; 0 or >1 predicate matches raise."""
    matches = [format_ for format_, predicate in PREDICATES if predicate(row)]
    if not matches:
        raise UndetectableTaskFormatError(row.raw.get("path") or row.raw.get("task_id") or str(row.raw)[:120])
    if len(matches) > 1:
        raise AmbiguousTaskFormatError(matches, row.raw.get("path") or str(row.raw)[:120])
    return matches[0]
