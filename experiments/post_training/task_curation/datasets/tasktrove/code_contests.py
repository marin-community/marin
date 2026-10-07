# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned code contests source declaration."""

from functools import partial

from taskcompendium.datasets import atlas_code
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.code_contests import convert_code_contests


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:DCAgent__code-contests-noblock",
        name="tasktrove-code_contests",
        version="tasktrove-code_contests-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="DCAgent__code-contests-noblock",
        split="train",
        files=tasktrove_files("DCAgent__code-contests-noblock"),
        adapter=ExecutionAdapter(
            policy=partial(atlas_code.policy, "code_contests"),
            converter=partial(converted_row, converter=convert_code_contests),
            converter_revision="code_contests-v1",
        ),
        intended_use=IntendedUse.TRAIN,
    )
