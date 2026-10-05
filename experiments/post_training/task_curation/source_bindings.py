# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind recipe families to experiment-owned execution adapters."""

import re
from collections.abc import Callable
from dataclasses import replace
from functools import partial

from taskcompendium.pipeline.models import DatasetRecipe
from verifyit.spec import Compare, StdioSpec

from experiments.post_training.task_curation import archive_sources
from experiments.post_training.task_curation.direct_sources import RECIPES as DIRECT_RECIPES
from experiments.post_training.task_curation.executable import converted_row
from experiments.post_training.task_curation.nemotron import RECIPES as NEMOTRON_RECIPES
from experiments.post_training.tasktrove.converters.code_contests import convert_code_contests
from experiments.post_training.tasktrove.converters.codeforces import convert_codeforces
from experiments.post_training.tasktrove.converters.converted_task import ConvertedTask, ConvertStatus, Rejected
from experiments.post_training.tasktrove.converters.nemotron_data import verifier_data
from experiments.post_training.tasktrove.converters.nemotron_structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_python
from experiments.post_training.tasktrove.converters.stdio_cases import SOLUTION_COMMAND, case_files
from experiments.post_training.tasktrove.taskbinary import DOCKERFILE, INSTRUCTION, TaskFiles

SANDBOX_TIMEOUT = 120.0
SANDBOX_MEMORY_MB = 512


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


RECIPES = DIRECT_RECIPES | archive_sources.RECIPES | NEMOTRON_RECIPES
SOURCE_NAMES = (*RECIPES, *archive_sources.EXECUTABLE_SOURCES, "structured_outputs")


CONVERTERS: dict[str, Callable[[TaskFiles], ConvertedTask | Rejected]] = {
    "code_contests": convert_code_contests,
    "codenet": convert_codenet,
    "competitive_coding": convert_competitive_coding,
    **dict.fromkeys(
        (
            "curriculum_easy",
            "curriculum_medium",
            "e2egit",
            "e2egit_large",
            "multifile",
            "pymethods",
            "pymethods_large",
            "stack_pytest",
            "unitsyn_large",
        ),
        convert_python,
    ),
}


def source_recipe(name: str, image: str | None) -> DatasetRecipe:
    """Bind an experiment selection; legacy converters execute inside audit workers."""
    if name in RECIPES:
        return RECIPES[name]
    if name == "structured_outputs":
        return archive_sources.structured_outputs_recipe(
            converter=partial(converted_row, name=name, converter=convert_nemotron_structured_outputs),
            converter_revision="structured-outputs-v1",
        )
    if name not in archive_sources.EXECUTABLE_SOURCES:
        raise ValueError(f"Unknown curation source: {name}")
    if image is None or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-fA-F]{64}", image) is None:
        raise ValueError(f"Executable source {name} requires an immutable grader image")
    adapter = CONVERTERS.get(name)
    converter = partial(converted_row, name=name, converter=adapter) if adapter else partial(converted_row, name=name)
    if adapter is convert_python:
        converter_revision = "python-unit-tests-v1"
    elif name == "competitive_coding":
        converter_revision = "competitive-coding-v1"
    else:
        converter_revision = f"{name}-v1"
    return archive_sources.executable_recipe(
        name,
        image,
        converter=converter,
        converter_revision=converter_revision,
        timeout=SANDBOX_TIMEOUT,
        memory_mb=SANDBOX_MEMORY_MB,
    )
