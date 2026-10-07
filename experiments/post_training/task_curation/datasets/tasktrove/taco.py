# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned taco source declaration."""

from functools import partial

from taskcompendium.datasets import executable_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.taco import convert_taco


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:laion__exp_rpt_taco-v2",
        name="tasktrove-taco",
        version="tasktrove-taco-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__exp_rpt_taco-v2",
        split="train",
        files=tasktrove_files("laion__exp_rpt_taco-v2"),
        adapter=ExecutionAdapter(
            policy=partial(executable_tasks.policy, "taco"),
            converter=partial(converted_row, converter=convert_taco),
            converter_revision="taco-v1",
        ),
        intended_use=IntendedUse.TRAIN,
    )
