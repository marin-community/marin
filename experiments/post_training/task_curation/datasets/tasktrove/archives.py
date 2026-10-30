# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove sources: archived Harbor tasks from the original ``open-thoughts/TaskTrove`` release."""

from dataclasses import dataclass, replace

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import Converter, ImportFailureKind, ImportRejection, NormalizedTask, RawRow

from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    TASKS_FILE,
    TASKTROVE_REPO,
    archive_resource,
    unpack_task_binary,
)
from experiments.post_training.task_curation.datasets.tasktrove.source_defects import SOURCE_DEFECTS
from experiments.post_training.task_curation.pipeline import HfSource

TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
ANSWER_FILE_DELIVERY = (
    ("write your final answer to `/app/answer.txt`", "return your final answer in the assistant response"),
    ("Write ONLY your final answer to **`/app/answer.txt`**", "Return ONLY your final answer in the assistant response"),
    ("The verifier reads that file", "The verifier reads the assistant response"),
)
"""The answer-file wording of the Nemotron Gym puzzle archives, and its reply-based replacement."""


@dataclass(frozen=True)
class TaskTroveConverter:
    """Retain reviewed source defects as explicit conversion outcomes."""

    config: str
    convert: Converter

    def __call__(self, row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
        reason = SOURCE_DEFECTS.get((self.config, row.data["path"]))
        if reason is not None:
            return ImportRejection(kind=ImportFailureKind.SOURCE_DEFECT, reason="reviewed_defect", detail=reason)
        converted = self.convert(row, context)
        if isinstance(converted, ImportRejection):
            return converted
        task = converted.task if isinstance(converted, NormalizedTask) else converted
        existing = {resource.path for resource in (*task.resources.all, *task.resources.oracle)}
        # Oracle controls may upload these files through runtimes without timestamp support.
        oracle = task.resources.oracle + tuple(
            archive_resource(row.data, path).model_copy(update={"mtime_ns": None})
            for path in row.data["files"]
            if path not in existing and not path.startswith(("tests/", "setup_files/"))
        )
        task = task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": oracle})})
        return replace(converted, task=task) if isinstance(converted, NormalizedTask) else task


def tasktrove_source(config: str) -> HfSource:
    """One TaskTrove config's task archives, unpacked into instruction, files and verifier data."""
    return HfSource(
        TASKTROVE_REPO, TASKTROVE_REVISION, (f"{config}/{TASKS_FILE}",), SourceFormat.PARQUET, decode=unpack_task_binary
    )
