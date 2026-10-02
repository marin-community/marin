# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire pinned Atlas source recipes without reading or converting their rows."""

import re
from dataclasses import replace
from importlib import import_module
from typing import Protocol, cast

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import (
    atlas_arc_injection,
    atlas_code,
    atlas_math_qa,
    calendar_tasks,
    competitive_coding,
    deepscaler,
    executable_tasks,
    hardmath,
    hendrycks_math,
    if_calendar,
    multichallenge,
    nemo_actions,
    preference_tasks,
    python_tasks,
    qa_tasks,
    reasoning_tasks,
    repository_tasks,
    rubric_tasks,
    tasktrove_math,
)
from taskcompendium.pipeline.datasets.nemotron import instruction_following, structured_outputs
from taskcompendium.pipeline.datasets.nemotron_ultra.catalog import NEMOTRON_SOURCES
from taskcompendium.pipeline.datasets.nemotron_ultra.source import recipe_for_source
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

from experiments.post_training.task_curation.competitive import convert_competitive_coding
from experiments.post_training.task_curation.executable import converted_row
from experiments.post_training.task_curation.next_code import CONVERTERS as NEXT_CODE_CONVERTERS
from experiments.post_training.tasktrove.converters.nemotron_structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_python


class RecipeModule(Protocol):
    def recipe(self) -> DatasetRecipe: ...


SANDBOX_TIMEOUT = 120.0
SANDBOX_MEMORY_MB = 512

ADDITIONAL_SOURCE_NAMES = (
    *NEMOTRON_SOURCES,
    "aime_1983_2024",
    "apps",
    "asdiv",
    "dapo_math",
    "eurus2_code",
    "gretel_text_to_sql",
    "gsm8k",
    "math500",
    "numina_math",
    "openscience",
    "rlvr_math",
    "verifiable_code",
    "rlvr_ifeval",
    "reasoning_gym_generated",
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
SOURCE_DEFINITIONS = (
    {name: family.SOURCES[name] for name, family in FAMILY_SOURCES.items()} | atlas_code.SOURCES | python_tasks.SOURCES
)
SOURCE_FACTORIES = {
    "nemotron_if": instruction_following.recipe,
    "hardmath": hardmath.recipe,
    "hendrycks_math": hendrycks_math.recipe,
    "deepscaler": deepscaler.recipe,
    "if_calendar": if_calendar.recipe,
    "multichallenge": multichallenge.recipe,
    "calendar": calendar_tasks.recipe,
    "reasoning_gym": reasoning_tasks.reasoning_recipe,
    "all_puzzles": reasoning_tasks.puzzle_recipe,
    "knowledge_openqa": qa_tasks.knowledge_recipe,
    "science_openqa": qa_tasks.science_recipe,
}
SOURCE_NAMES = (
    *SOURCE_FACTORIES,
    *FAMILY_SOURCES,
    *executable_tasks.CONFIGS,
    *atlas_code.CONFIGS,
    *python_tasks.SOURCES,
    "nemo_actions",
    "structured_outputs",
    "competitive_coding",
    *ADDITIONAL_SOURCE_NAMES,
)


def source_recipe(name: str, image: str | None) -> DatasetRecipe:
    """Bind a pinned source; executable conversion happens inside audit workers."""
    if name in NEMOTRON_SOURCES:
        return recipe_for_source(NEMOTRON_SOURCES[name])
    if name in ADDITIONAL_SOURCE_NAMES:
        return cast(RecipeModule, import_module(f"taskcompendium.pipeline.datasets.{name}")).recipe()
    if name in FAMILY_SOURCES:
        return FAMILY_SOURCES[name].recipe_for_source(name)
    if name == "nemo_actions":
        return nemo_actions.recipe
    if name in SOURCE_FACTORIES:
        return SOURCE_FACTORIES[name]()
    if name == "structured_outputs":
        recipe = structured_outputs.recipe()
        converter = convert_nemotron_structured_outputs
        converter_revision = "structured-outputs-v1"
    elif image is None or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-fA-F]{64}", image) is None:
        raise ValueError(f"Executable source {name} requires an immutable grader image")
    elif name == "competitive_coding":
        recipe = competitive_coding.recipe(image, timeout=SANDBOX_TIMEOUT, memory_mb=SANDBOX_MEMORY_MB)
        converter = convert_competitive_coding
        converter_revision = "competitive-coding-v1"
    elif name in python_tasks.SOURCES:
        recipe = python_tasks.recipe_for_source(name, image, timeout=SANDBOX_TIMEOUT, memory_mb=SANDBOX_MEMORY_MB)
        converter = convert_python
        converter_revision = "python-unit-tests-v1"
    elif name in atlas_code.CONFIGS:
        recipe = atlas_code.recipe_for_source(name, image, timeout=SANDBOX_TIMEOUT, memory_mb=SANDBOX_MEMORY_MB)
        converter = NEXT_CODE_CONVERTERS[name]
        converter_revision = f"{name}-v1"
    else:
        recipe = executable_tasks.recipe(name, image, timeout=SANDBOX_TIMEOUT, memory_mb=SANDBOX_MEMORY_MB)
        converter = None
        converter_revision = f"{name}-v1"
    normalize = recipe.normalize

    def normalize_raw(row: RawRow) -> TaskSpec | NormalizedTask | ImportRejection:
        prepared = converted_row(row.data, name, converter=converter)
        result = normalize(RawRow(row.id, row.source, prepared))
        if isinstance(result, ImportRejection):
            return result
        changes = tuple(
            NormalizationChange.model_validate(change) for change in prepared["converted"]["normalization_changes"]
        )
        if isinstance(result, NormalizedTask):
            return NormalizedTask(result.task, changes + result.changes)
        return NormalizedTask(result, changes)

    suite = recipe.check_suite
    assert suite is not None
    return replace(
        recipe,
        version=f"{recipe.version}-raw-conversion-v1",
        normalize=normalize_raw,
        check_suite=replace(suite, parameters={**suite.parameters, "converter_revision": converter_revision}),
    )
