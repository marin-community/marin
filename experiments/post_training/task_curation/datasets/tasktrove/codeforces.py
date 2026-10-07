# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned codeforces source declaration."""

from functools import partial

from taskcompendium.datasets import executable_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.codeforces import convert_codeforces


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:laion__codeforces-v3",
        name="tasktrove-codeforces",
        version="tasktrove-codeforces-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__codeforces-v3",
        split="train",
        files=tasktrove_files("laion__codeforces-v3"),
        adapter=ExecutionAdapter(
            policy=partial(executable_tasks.policy, "codeforces"),
            converter=partial(converted_row, converter=convert_codeforces),
            converter_revision="codeforces-v2",
        ),
        intended_use=IntendedUse.TRAIN,
    )
