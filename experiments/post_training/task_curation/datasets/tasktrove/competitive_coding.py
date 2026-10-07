# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned competitive coding source declaration."""

from functools import partial

from taskcompendium.datasets import competitive_coding
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import ExecutionAdapter
from taskcompendium.pipeline.models import IntendedUse
from verifyit.spec import Compare, StdioSpec

from experiments.post_training.task_curation.datasets.shared import executable_pipeline
from experiments.post_training.task_curation.datasets.tasktrove.conversion import converted_row
from experiments.post_training.task_curation.pipeline import RlDataPipeline
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, ConvertStatus, Rejected
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.converters.stdio_cases import SOLUTION_COMMAND, case_files
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles


def convert_competitive_coding(task: TaskFiles) -> ConvertedTask | Rejected:
    """Bind the source input/output pairs to its solution command and exact comparator."""
    data = verifier_data(task)
    inputs, outputs = data.get("inputs"), data.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs) or not inputs:
        return Rejected(ConvertStatus.NULL_GRADER, "At least one aligned input/output case is required")
    if not all(isinstance(value, str) for value in [*inputs, *outputs]):
        return Rejected(ConvertStatus.NULL_GRADER, "Inputs and outputs must be strings")
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.EXACT),
        dockerfile=task.text(DOCKERFILE),
        tags=("code", "competitive-programming", "stdio", "nemotron"),
        language="python",
        data_files=case_files(inputs, outputs),
    )


def pipeline() -> RlDataPipeline:
    return executable_pipeline(
        source_key="Task Trove:laion__nemotron-gym-competitive-coding-v2",
        name="tasktrove-competitive_coding",
        version="tasktrove-competitive_coding-v1-raw-conversion-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__nemotron-gym-competitive-coding-v2",
        split="train",
        files=tasktrove_files("laion__nemotron-gym-competitive-coding-v2"),
        adapter=ExecutionAdapter(
            policy=competitive_coding.policy,
            converter=partial(converted_row, converter=convert_competitive_coding),
            converter_revision="competitive-coding-v1",
        ),
        intended_use=IntendedUse.TRAIN,
    )
