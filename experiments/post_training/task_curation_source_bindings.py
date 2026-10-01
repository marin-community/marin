# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire sampled Atlas source recipes without reading or converting their rows."""

import hashlib
import re
from dataclasses import replace
from pathlib import Path

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import (
    advanced_calculations,
    arc_inductive,
    arc_transductive,
    atlas_code,
    calendar_tasks,
    code_contests,
    codenet,
    executable_tasks,
    indirect_injection,
    knowledge_mcqa,
    math_openreasoning,
    nemo_actions,
    qa_abstention,
    qa_tasks,
    reasoning_tasks,
    web_search_mcqa,
)
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    NormalizationChange,
    NormalizedTask,
    RawRow,
)

from experiments.post_training.task_curation_executable import converted_row
from experiments.post_training.task_curation_next_code import CONVERTERS as NEXT_CODE_CONVERTERS

SOURCE_FACTORIES = {
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
SOURCE_NAMES = (*SOURCE_FACTORIES, *executable_tasks.CONFIGS, *atlas_code.CONFIGS)


def converter_digest() -> str:
    """Include borrowed converter code in the audit artifact's identity."""
    directory = Path(__file__).parent
    files = [directory / "task_curation_executable.py", directory / "task_curation_next_code.py"]
    files.extend(sorted((directory / "tasktrove").rglob("*.py")))
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(directory)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def source_recipe(name: str, snapshot: Path, image: str | None) -> DatasetRecipe:
    """Bind raw samples; executable conversion happens inside audit workers."""
    if name in SOURCE_FACTORIES:
        return SOURCE_FACTORIES[name](snapshot)
    if image is None or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-fA-F]{64}", image) is None:
        raise ValueError(f"Executable source {name} requires an immutable grader image")
    if name in atlas_code.CONFIGS:
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
        if not isinstance(result, TaskSpec):
            raise TypeError("Executable source normalizers must produce a TaskSpec")
        changes = tuple(
            NormalizationChange.model_validate(change) for change in prepared["converted"]["normalization_changes"]
        )
        return NormalizedTask(result, changes)

    suite = recipe.check_suite
    assert suite is not None
    return replace(
        recipe,
        version=f"{recipe.version}-raw-conversion-v1",
        normalize=normalize_raw,
        check_suite=replace(suite, parameters={**suite.parameters, "converter_sha256": converter_digest()}),
    )
