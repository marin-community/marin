# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive conversion results and their translation into TaskSpec resources."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from verifyit.spec import Spec

from taskcompendium.convert.tasktrove import SOLUTION_DIR, TaskFiles, archive_files
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection


class ConvertStatus(StrEnum):
    CONVERTED = "converted"
    NO_CONVERTER = "no_converter"
    CONVERTER_ERROR = "converter_error"
    DROPPED_SOURCE = "dropped_source"
    REVIEWED_DEFECT = "reviewed_defect"
    # Typed rejections a converter returns for a task its template cannot grade soundly.
    NULL_GRADER = "null_grader"
    TOO_FEW_CASES = "too_few_cases"
    GOLD_IN_INSTRUCTION = "gold_in_instruction"
    UNSUPPORTED_VARIANT = "unsupported_variant"


@dataclass(frozen=True)
class ConvertedTask:
    instruction: str
    spec: Spec
    dockerfile: str
    """The source Dockerfile after conversion, before runtime environment assembly."""
    tags: tuple[str, ...]
    language: str = ""
    data_files: dict[str, bytes] = field(default_factory=dict)
    """Files the spec references, keyed by path under the task root (``tests/cases/...``)."""
    solution_files: dict[str, bytes] = field(default_factory=dict)
    """Oracle solution; stored beside the task, never inside the binary the agent sees."""
    metadata: dict = field(default_factory=dict)
    """Source-specific metadata retained in the converted task."""
    agent_timeout: float = 900.0
    verifier_timeout: float = 600.0
    verifier_extras: tuple[str, ...] = ()
    """Optional verifier dependencies needed by a source-specific grading composition."""


@dataclass(frozen=True)
class Rejected:
    status: ConvertStatus
    detail: str


ConvertFn = Callable[[TaskFiles], ConvertedTask | Rejected]


REJECTION_KINDS = {
    ConvertStatus.NO_CONVERTER: ImportFailureKind.UNSUPPORTED,
    ConvertStatus.CONVERTER_ERROR: ImportFailureKind.CONVERTER_ERROR,
    ConvertStatus.DROPPED_SOURCE: ImportFailureKind.UNSUPPORTED,
    ConvertStatus.REVIEWED_DEFECT: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.NULL_GRADER: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.TOO_FEW_CASES: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.GOLD_IN_INSTRUCTION: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.UNSUPPORTED_VARIANT: ImportFailureKind.UNSUPPORTED,
}
"""The curation failure kind each converter rejection represents."""


def archive_conversion(data: Mapping[str, Any], convert: ConvertFn) -> ConvertedTask | ImportRejection:
    """Run ``convert`` on a row unpacked by ``unpack_task_binary``.

    A converter rejection becomes the matching import rejection, and a converter that cannot read
    its archive is a converter error. A converted task without its own oracle keeps the archive's
    ``solution/`` files as its oracle.
    """
    task = archive_files(data)
    try:
        converted = convert(task)
    except (ValueError, KeyError) as error:
        return ImportRejection(kind=ImportFailureKind.CONVERTER_ERROR, reason="converter_error", detail=str(error))
    if isinstance(converted, Rejected):
        return ImportRejection(
            kind=REJECTION_KINDS[converted.status], reason=converted.status.value, detail=converted.detail
        )
    return replace(converted, solution_files=converted.solution_files or task.under(SOLUTION_DIR))
