# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned source contracts for TaskTrove and direct dataset families."""

import base64
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, hub_inputs
from taskcompendium.pipeline.models import ReviewRubric

TASKTROVE_DATASET = "open-thoughts/TaskTrove"


def unpack_task_binary(row: dict[str, Any], _staged_root: StoragePath) -> dict[str, Any]:
    """Expose Harbor task files in the form consumed by TaskTrove converters."""
    blob = row["task_binary"]
    if not isinstance(blob, bytes):
        raise ValueError("Task binary must contain archived bytes")
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as archive:
        for member in archive:
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is not None:
                    files[member.name.removeprefix("./")] = handle.read()
    prepared = {
        "path": row["path"],
        "instruction": files["instruction.md"].decode(),
        "files": {name: base64.b64encode(data).decode() for name, data in files.items()},
        "archive_sha256": hashlib.sha256(blob).hexdigest(),
    }
    if "tests/verifier_data.json" in files:
        prepared["verifier_data"] = json.loads(files["tests/verifier_data.json"])
    return prepared


def tasktrove_files(config: str) -> SourceFiles:
    """Select and decode one TaskTrove component's archived tasks."""
    return SourceFiles((f"{config}/tasks.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary)


@dataclass(frozen=True)
class SourceDefinition:
    dataset: str
    revision: str
    config: str
    split: str
    rubric: ReviewRubric
    files: SourceFiles


def tasktrove_source(config: str, revision: str, rubric: ReviewRubric) -> SourceDefinition:
    """Describe one pinned TaskTrove component and its archived task records."""
    return SourceDefinition(
        dataset=TASKTROVE_DATASET,
        revision=revision,
        config=config,
        split="train",
        rubric=rubric,
        files=tasktrove_files(config),
    )


def tasktrove_inputs(config: str, revision: str) -> RecipeInputs:
    """Declare the pinned archive consumed by a TaskTrove recipe."""
    return hub_inputs(
        TASKTROVE_DATASET,
        revision,
        tasktrove_files(config),
    )
