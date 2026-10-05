# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind CodeContests and CodeNet stdin/stdout converters for curation."""

from collections.abc import Callable
from dataclasses import replace

from verifyit.spec import Compare, StdioSpec

from experiments.post_training.tasktrove.converters.code_contests import convert_code_contests
from experiments.post_training.tasktrove.converters.codeforces import convert_codeforces
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, ConvertStatus, Rejected
from experiments.post_training.tasktrove.converters.stdio_cases import SOLUTION_COMMAND
from experiments.post_training.tasktrove.taskbinary import TaskFiles


def convert_codenet(task: TaskFiles) -> ConvertedTask | Rejected:
    """Extract CodeNet cases and bind whitespace-token output comparison."""
    converted = convert_codeforces(task)
    if isinstance(converted, Rejected):
        return converted
    cases = sum(path.startswith("tests/cases/input_") for path in converted.data_files)
    if cases < 2:
        return Rejected(ConvertStatus.NULL_GRADER, "CodeNet source requires at least two input/output pairs")
    return replace(
        converted,
        spec=StdioSpec(command=SOLUTION_COMMAND, compare=Compare.TOKENS, per_case_timeout=30.0, min_cases=2),
        tags=("code", "competitive-programming", "stdio", "codenet"),
    )


CONVERTERS: dict[str, Callable[[TaskFiles], ConvertedTask | Rejected]] = {
    "code_contests": convert_code_contests,
    "codenet": convert_codenet,
}
