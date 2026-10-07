# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nl2bash source declaration."""

from functools import partial

from taskcompendium.datasets import executable_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.nl2bash import convert_nl2bash


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        name="tasktrove-nl2bash",
        version="tasktrove-nl2bash-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
        split="train",
        files=tasktrove_files("DCAgent2__nl2bash-tasks-cleaned-oracle-v2"),
        adapter=ExecutionAdapter(
            policy=partial(executable_tasks.policy, "nl2bash"),
            converter=partial(converted_row, converter=convert_nl2bash),
            converter_revision="nl2bash-v2",
            output_paths=("/output/command_capture.txt",),
        ),
        intended_use=IntendedUse.TRAIN,
    )
