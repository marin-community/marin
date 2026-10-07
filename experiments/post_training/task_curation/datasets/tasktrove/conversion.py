# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize TaskTrove executable rows with each declaration's converter."""

import base64
import hashlib
from collections.abc import Callable, Mapping
from typing import Any

from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection
from verifyit.spec import PytestSpec, spec_to_table

from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, ConvertStatus, Rejected
from experiments.post_training.tasktrove.taskbinary import TaskFiles

CONVERSION_FAILURE_KINDS = {
    ConvertStatus.NO_CONVERTER: ImportFailureKind.UNSUPPORTED,
    ConvertStatus.CONVERTER_ERROR: ImportFailureKind.CONVERTER_ERROR,
    ConvertStatus.DROPPED_SOURCE: ImportFailureKind.UNSUPPORTED,
    ConvertStatus.REVIEWED_DEFECT: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.NULL_GRADER: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.TOO_FEW_CASES: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.GOLD_IN_INSTRUCTION: ImportFailureKind.SOURCE_DEFECT,
    ConvertStatus.UNSUPPORTED_VARIANT: ImportFailureKind.UNSUPPORTED,
}


def converted_row(
    data: Mapping[str, Any],
    *,
    converter: Callable[[TaskFiles], ConvertedTask | Rejected],
) -> dict[str, Any]:
    """Retain raw input and add one converter's result or typed rejection."""
    files = {path: base64.b64decode(encoded, validate=True) for path, encoded in data["files"].items()}
    row = dict(data)
    try:
        converted = converter(TaskFiles(files))
    except (ValueError, KeyError) as error:
        row["conversion_rejection"] = ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR, reason="converter_error", detail=str(error)
        ).model_dump(mode="json")
        return row
    if isinstance(converted, Rejected):
        row["conversion_rejection"] = ImportRejection(
            kind=CONVERSION_FAILURE_KINDS[converted.status], reason=converted.status.value, detail=converted.detail
        ).model_dump(mode="json")
        return row
    grader_spec = spec_to_table(converted.spec)
    if isinstance(converted.spec, PytestSpec):
        # The selected image owns its Python/report dependencies, rather than
        # assuming the source Dockerfile's private venv exists in this runtime.
        grader_spec["python"] = "python3"
    controls = converted.solution_files or {p: b for p, b in files.items() if p.startswith("solution/")}
    changes = []
    if converted.instruction != data["instruction"]:
        changes.append(
            {
                "field": "instruction",
                "reason": "Existing source converter corrected delivery boilerplate",
                "original": data["instruction"],
                "replacement": converted.instruction,
            }
        )
    row["converted"] = {
        "instruction": converted.instruction,
        "grader_spec": grader_spec,
        "data_files": {path: base64.b64encode(content).decode() for path, content in converted.data_files.items()},
        "control_files": {p: base64.b64encode(b).decode() for p, b in controls.items()},
        "source_dockerfile_sha256": hashlib.sha256(converted.dockerfile.encode()).hexdigest(),
        "tags": list(converted.tags),
        "normalization_changes": changes,
    }
    return row
