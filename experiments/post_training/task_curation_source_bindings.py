# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire sampled Atlas source recipes without reading or converting their rows."""

import hashlib
import re
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import (
    advanced_calculations,
    arc_inductive,
    arc_transductive,
    atlas_code,
    calendar_tasks,
    code_contests,
    codenet,
    codereview,
    competitive_coding,
    curriculum_easy,
    curriculum_medium,
    deepscaler,
    e2egit,
    e2egit_large,
    executable_tasks,
    glaive_code,
    hardmath,
    hendrycks_math,
    if_calendar,
    indirect_injection,
    knowledge_mcqa,
    math_gym,
    math_openreasoning,
    math_oracle,
    math_prism,
    math_stack,
    multichallenge,
    multifile,
    nemo_actions,
    nemotron_structured_outputs,
    pymethods,
    pymethods_large,
    qa_abstention,
    qa_tasks,
    reasoning_tasks,
    safety,
    stack_overflow,
    stack_pytest,
    superuser,
    swe_rebench,
    swesmith,
    tezos,
    unitsyn_large,
    unix,
    web_search_mcqa,
    wizard_orca,
)
from taskcompendium.pipeline.datasets.nemotron_ultra_catalog import NEMOTRON_MODULES
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

from experiments.post_training.task_curation_competitive import convert_competitive_coding
from experiments.post_training.task_curation_executable import converted_row
from experiments.post_training.task_curation_next_code import CONVERTERS as NEXT_CODE_CONVERTERS
from experiments.post_training.tasktrove.converters.nemotron_structured_outputs import (
    convert_nemotron_structured_outputs,
)
from experiments.post_training.tasktrove.converters.python_unit_tests import convert as convert_python


class SnapshotRecipeModule(Protocol):
    def recipe(self, snapshot: Path) -> DatasetRecipe: ...


ADDITIONAL_SOURCE_NAMES = (
    *NEMOTRON_MODULES,
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
    "hh_harmless_base",
    "hh_helpful_base",
    "hh_helpful_online",
    "hh_helpful_rejection_sampled",
    "kto_mix",
    "nemotron_if",
    "rlvr_ifeval",
    "reasoning_gym_generated",
)

RUBRIC_SOURCES = {
    "codereview": codereview,
    "glaive_code": glaive_code,
    "if_calendar": if_calendar,
    "multichallenge": multichallenge,
    "safety": safety,
    "stack_overflow": stack_overflow,
    "superuser": superuser,
    "tezos": tezos,
    "unix": unix,
    "wizard_orca": wizard_orca,
}
MATH_SOURCES = {
    "math_prism": math_prism,
    "math_stack": math_stack,
    "math_gym": math_gym,
    "math_oracle": math_oracle,
}
SOURCE_FACTORIES = {
    **{name: module.recipe for name, module in MATH_SOURCES.items()},
    "swe_rebench": swe_rebench.recipe,
    "swesmith": swesmith.recipe,
    "hardmath": hardmath.recipe,
    "hendrycks_math": hendrycks_math.recipe,
    "deepscaler": deepscaler.recipe,
    **{name: module.recipe for name, module in RUBRIC_SOURCES.items()},
    "calendar": calendar_tasks.recipe,
    "reasoning_gym": reasoning_tasks.reasoning_recipe,
    "all_puzzles": reasoning_tasks.puzzle_recipe,
    "nemo_actions": nemo_actions.snapshot_recipe,
    "knowledge_openqa": qa_tasks.knowledge_recipe,
    "science_openqa": qa_tasks.science_recipe,
    "math_openreasoning": math_openreasoning.recipe,
    "advanced_calculations": advanced_calculations.recipe,
    "knowledge_mcqa": knowledge_mcqa.recipe,
    "web_search_mcqa": web_search_mcqa.recipe,
    "qa_abstention": qa_abstention.recipe,
    "arc_transductive": arc_transductive.recipe,
    "arc_inductive": arc_inductive.recipe,
    "indirect_injection": indirect_injection.recipe,
}
PYTHON_SOURCES = {
    "curriculum_easy": curriculum_easy,
    "curriculum_medium": curriculum_medium,
    "e2egit_large": e2egit_large,
    "e2egit": e2egit,
    "multifile": multifile,
    "pymethods_large": pymethods_large,
    "pymethods": pymethods,
    "stack_pytest": stack_pytest,
    "unitsyn_large": unitsyn_large,
}
SOURCE_NAMES = (
    *SOURCE_FACTORIES,
    *executable_tasks.CONFIGS,
    *atlas_code.CONFIGS,
    *PYTHON_SOURCES,
    "structured_outputs",
    "competitive_coding",
    *ADDITIONAL_SOURCE_NAMES,
)


def converter_digest() -> str:
    """Include borrowed converter code in the audit artifact's identity."""
    directory = Path(__file__).parent
    files = [
        directory / "task_curation_executable.py",
        directory / "task_curation_next_code.py",
        directory / "task_curation_competitive.py",
    ]
    files.extend(sorted((directory / "tasktrove").rglob("*.py")))
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(directory)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def source_recipe(name: str, snapshot: Path, image: str | None) -> DatasetRecipe:
    """Bind raw samples; executable conversion happens inside audit workers."""
    if name in ADDITIONAL_SOURCE_NAMES:
        return cast(SnapshotRecipeModule, import_module(f"taskcompendium.pipeline.datasets.{name}")).recipe(snapshot)
    if name in SOURCE_FACTORIES:
        return SOURCE_FACTORIES[name](snapshot)
    if name == "structured_outputs":
        recipe = nemotron_structured_outputs.recipe(snapshot)
        converter = convert_nemotron_structured_outputs
    elif image is None or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-fA-F]{64}", image) is None:
        raise ValueError(f"Executable source {name} requires an immutable grader image")
    elif name == "competitive_coding":
        recipe = competitive_coding.recipe(snapshot, image, timeout=120.0, memory_mb=512)
        converter = convert_competitive_coding
    elif name in PYTHON_SOURCES:
        recipe = PYTHON_SOURCES[name].recipe(snapshot, image, timeout=120.0, memory_mb=512)
        converter = convert_python
    elif name in atlas_code.CONFIGS:
        factory = {"code_contests": code_contests.recipe, "codenet": codenet.recipe}[name]
        recipe = factory(snapshot, image, timeout=120.0, memory_mb=512)
        converter = NEXT_CODE_CONVERTERS[name]
    else:
        recipe = executable_tasks.recipe(name, snapshot, image, timeout=120.0, memory_mb=512)
        converter = None
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
        check_suite=replace(suite, parameters={**suite.parameters, "converter_sha256": converter_digest()}),
    )
