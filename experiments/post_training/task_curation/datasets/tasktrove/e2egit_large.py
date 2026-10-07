# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned e2egit large source declaration."""

from functools import partial

from taskcompendium.datasets import python_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_python


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:DCAgent__exp_rpt_e2egit-large",
        name="tasktrove-e2egit_large",
        version="tasktrove-e2egit_large-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="DCAgent__exp_rpt_e2egit-large",
        split="train",
        files=tasktrove_files("DCAgent__exp_rpt_e2egit-large"),
        adapter=ExecutionAdapter(
            policy=partial(python_tasks.policy, rubric=python_tasks.RUBRICS["e2egit_large"]),
            converter=partial(converted_row, converter=convert_python),
            converter_revision="python-unit-tests-v3",
        ),
        intended_use=IntendedUse.TRAIN,
    )
