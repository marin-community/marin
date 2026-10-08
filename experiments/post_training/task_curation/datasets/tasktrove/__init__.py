# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove sources: archived Harbor tasks from the original ``open-thoughts/TaskTrove`` release."""

from taskcompendium.convert.tasktrove import TASKS_FILE, unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.pipeline import HfSource

TASKTROVE_REPO = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"


def tasktrove_source(config: str) -> HfSource:
    """One TaskTrove config's task archives, unpacked into instruction, files and verifier data."""
    return HfSource(
        TASKTROVE_REPO, TASKTROVE_REVISION, (f"{config}/{TASKS_FILE}",), SourceFormat.PARQUET, decode=unpack_task_binary
    )
