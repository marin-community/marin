# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind recipe families to experiment-owned execution adapters."""

import re
from functools import partial

from taskcompendium.pipeline.datasets import (
    atlas_arc_injection,
    atlas_code,
    atlas_math_qa,
    calendar_tasks,
    code_contracts,
    competitive_coding,
    executable_tasks,
    gretel_text_to_sql,
    if_calendar,
    instruction_tasks,
    math_answers,
    multichallenge,
    nemo_actions,
    openscience,
    preference_tasks,
    python_tasks,
    qa_tasks,
    reasoning_gym_generated,
    reasoning_tasks,
    repository_tasks,
    rubric_tasks,
    tasktrove_math,
)
from taskcompendium.pipeline.datasets.nemotron import structured_outputs
from taskcompendium.pipeline.datasets.nemotron_ultra.catalog import NEMOTRON_SOURCES
from taskcompendium.pipeline.datasets.nemotron_ultra.source import recipe_for_source
from taskcompendium.pipeline.models import DatasetRecipe

from experiments.post_training.task_curation.competitive import convert_competitive_coding
from experiments.post_training.task_curation.executable import converted_row
from experiments.post_training.task_curation.next_code import CONVERTERS as NEXT_CODE_CONVERTERS
from experiments.post_training.tasktrove.converters.nemotron_structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_python

SANDBOX_TIMEOUT = 120.0
SANDBOX_MEMORY_MB = 512

RECIPES = (
    math_answers.RECIPES | instruction_tasks.RECIPES | code_contracts.RECIPES | {"nemo_actions": nemo_actions.recipe}
)
FAMILY_SOURCES = {
    name: family
    for family in (
        atlas_arc_injection,
        atlas_math_qa,
        preference_tasks,
        repository_tasks,
        rubric_tasks,
        tasktrove_math,
    )
    for name in family.SOURCES
}
SOURCE_FACTORIES = {
    "gretel_text_to_sql": gretel_text_to_sql.recipe,
    "openscience": openscience.recipe,
    "reasoning_gym_generated": reasoning_gym_generated.recipe,
    "if_calendar": if_calendar.recipe,
    "multichallenge": multichallenge.recipe,
    "calendar": calendar_tasks.recipe,
    "reasoning_gym": reasoning_tasks.reasoning_recipe,
    "all_puzzles": reasoning_tasks.puzzle_recipe,
    "knowledge_openqa": qa_tasks.knowledge_recipe,
    "science_openqa": qa_tasks.science_recipe,
}
SOURCE_NAMES = (
    *RECIPES,
    *SOURCE_FACTORIES,
    *FAMILY_SOURCES,
    *executable_tasks.CONFIGS,
    *atlas_code.CONFIGS,
    *python_tasks.SOURCES,
    "structured_outputs",
    "competitive_coding",
    *NEMOTRON_SOURCES,
)


def source_recipe(name: str, image: str | None) -> DatasetRecipe:
    """Bind a pinned recipe; legacy conversion executes inside audit workers."""
    if name in RECIPES:
        return RECIPES[name]
    if name in NEMOTRON_SOURCES:
        return recipe_for_source(NEMOTRON_SOURCES[name])
    if name in FAMILY_SOURCES:
        return FAMILY_SOURCES[name].recipe_for_source(name)
    if name in SOURCE_FACTORIES:
        return SOURCE_FACTORIES[name]()
    if name == "structured_outputs":
        return structured_outputs.recipe(
            converter=partial(converted_row, name=name, converter=convert_nemotron_structured_outputs),
            converter_revision="structured-outputs-v1",
        )
    if name not in SOURCE_NAMES:
        raise ValueError(f"Unknown curation source: {name}")
    if image is None or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-fA-F]{64}", image) is None:
        raise ValueError(f"Executable source {name} requires an immutable grader image")
    if name == "competitive_coding":
        return competitive_coding.recipe(
            image,
            converter=partial(converted_row, name=name, converter=convert_competitive_coding),
            converter_revision="competitive-coding-v1",
            timeout=SANDBOX_TIMEOUT,
            memory_mb=SANDBOX_MEMORY_MB,
        )
    if name in python_tasks.SOURCES:
        return python_tasks.recipe_for_source(
            name,
            image,
            converter=partial(converted_row, name=name, converter=convert_python),
            converter_revision="python-unit-tests-v1",
            timeout=SANDBOX_TIMEOUT,
            memory_mb=SANDBOX_MEMORY_MB,
        )
    if name in atlas_code.CONFIGS:
        return atlas_code.recipe_for_source(
            name,
            image,
            converter=partial(converted_row, name=name, converter=NEXT_CODE_CONVERTERS[name]),
            converter_revision=f"{name}-v1",
            timeout=SANDBOX_TIMEOUT,
            memory_mb=SANDBOX_MEMORY_MB,
        )
    return executable_tasks.recipe(
        name,
        image,
        converter=partial(converted_row, name=name),
        converter_revision=f"{name}-v1",
        timeout=SANDBOX_TIMEOUT,
        memory_mb=SANDBOX_MEMORY_MB,
    )
