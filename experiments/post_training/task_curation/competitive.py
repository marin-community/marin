# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert archived competitive problems to stdin/stdout grading cases."""

from tasktrove_verify.spec import Compare, StdioSpec

from experiments.post_training.tasktrove.converters.converted_task import (
    ConvertedTask,
    ConvertStatus,
    Rejected,
)
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
