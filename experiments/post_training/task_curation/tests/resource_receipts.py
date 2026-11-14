# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit capacity and production-receipt fixtures for reviewed pipeline inputs."""

from dataclasses import dataclass, replace
from pathlib import Path

from taskcompendium.models import ExecutionRequirements, TaskSpec
from taskcompendium.pipeline.conversion import convert_source_row
from taskcompendium.pipeline.deduplication import unique_task_indices
from taskcompendium.pipeline.models import Converter, ImportRejection, NormalizedTask
from taskcompendium.pipeline.oracle_filter import oracle_recipe_identity
from taskcompendium.pipeline.sources import conversion_context, source_shards, staged_raw_file_rows

from experiments.post_training.task_curation.pipeline import CurationRecipe, source_recipe
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt

FIXTURE_REQUIREMENTS = ExecutionRequirements(
    cpus=1, memory_mb=2048, storage_mb=2048, agent_timeout_sec=300, verifier_timeout_sec=60
)


@dataclass(frozen=True)
class ResourceConverter:
    convert: Converter

    def __call__(self, row, context):
        result = self.convert(row, context)
        if isinstance(result, TaskSpec):
            return result.model_copy(update={"execution_requirements": FIXTURE_REQUIREMENTS})
        if isinstance(result, NormalizedTask):
            return replace(result, task=result.task.model_copy(update={"execution_requirements": FIXTURE_REQUIREMENTS}))
        return result


def resource_recipe(recipe: CurationRecipe) -> CurationRecipe:
    return replace(recipe, convert=ResourceConverter(recipe.convert))


def fixture_receipt(recipe: CurationRecipe, root: Path, inputs=None, overrides=None, converter=None):
    declared = source_recipe(recipe, inputs or {}, None)
    conversion = replace(declared, convert=converter) if converter is not None else declared
    tasks = []
    input_rows = 0
    for shard in source_shards(str(root), declared.source, overrides):
        for row in staged_raw_file_rows(str(root), shard, declared.source, conversion_context(declared)):
            input_rows += 1
            result = convert_source_row(row, conversion).result
            if not isinstance(result, ImportRejection):
                tasks.append(result.task)
    kept, dropped = unique_task_indices(tasks)
    return oracle_receipt([tasks[index] for index in kept], oracle_recipe_identity(declared)).model_copy(
        update={"input_rows": input_rows - len(dropped), "conversion_rejections": input_rows - len(tasks)}
    )


def write_fixture_receipt(recipe: CurationRecipe, root: Path, destination: Path) -> list[str]:
    destination.write_text(fixture_receipt(recipe, root).model_dump_json())
    return ["--oracle-filter-receipt", str(destination)]
