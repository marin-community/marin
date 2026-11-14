# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact source exclusions retain distinct contracts and stable canonical task IDs."""

import json
from dataclasses import replace

import pyarrow.parquet as pq
import pytest
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from taskcompendium.identity import task_identity
from taskcompendium.models import ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.conversion import ConvertedRow
from taskcompendium.pipeline.deduplication import deduplicate_population, unique_documents, unique_task_indices
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode, run_source_pipeline
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import fixture_recipe, svamp_row_task


def contract(identifier: str, answer: str = "2") -> TaskSpec:
    result = svamp_row_task(
        RawRow(
            identifier,
            Source(dataset="fixture", revision="1", row=identifier, importer_revision="1"),
            {"Body": "Aya has two apples.", "Question": "How many apples?", "Answer": answer},
        )
    )
    assert isinstance(result, TaskSpec)
    return result


@pytest.mark.parametrize("reverse", [False, True])
def test_unique_contracts_keep_lowest_id_and_preserve_different_graders(tmp_path, reverse):
    tasks = [contract("z"), contract("a"), contract("different-grader", "3")]
    if reverse:
        tasks.reverse()
    kept, dropped = unique_task_indices(tasks)
    assert [tasks[index].id for index in kept] == ["a", "different-grader"]
    assert dropped == [{"task_id": "z", "kept_id": "a", "reason": "DUPLICATE_WITHIN_SOURCE"}]
    disk_drops = []
    documents = list(unique_documents(((task.model_dump(mode="json"), {}) for task in tasks), disk_drops.append))
    assert [document["id"] for document, _runtime in documents] == ["a", "different-grader"]
    assert disk_drops == dropped
    converted = [ConvertedRow(RawRow(task.id, task.source, {}), {}, None, NormalizedTask(task, ())) for task in tasks]
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        population = deduplicate_population(
            Dataset.from_list([converted[::2], converted[1::2]]).flat_map(iter), context, str(tmp_path / "dedup")
        )
        retained = context.execute(population.rows()).results
    manifest = [row for file in (tmp_path / "dedup/drops").glob("*.parquet") for row in pq.read_table(file).to_pylist()]
    assert manifest == dropped
    assert sorted(row.raw.id for row in retained) == ["a", "different-grader"]


@pytest.mark.parametrize("role", ["worker", "verifier", "oracle"])
def test_contract_identity_preserves_private_files_and_environment(role):
    task = contract("a")
    changed = task.model_copy(
        update={"resources": ResourceGroups(**{role: (inline_resource("reference", b"different"),)})}
    )
    assert task_identity(task) != task_identity(changed)
    changed_environment = task.model_copy(
        update={
            "environment_requirements": task.environment_requirements.model_copy(update={"docker_image": "ubuntu:24.04"})
        }
    )
    assert task_identity(task) != task_identity(changed_environment)


def test_quick_drops_duplicates_across_files_before_export(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {"Body": "Aya has two apples.", "Question": "How many apples?", "Answer": answer} for answer in ("2", "2", "3")
    ]
    for index, row in enumerate(rows):
        (source / f"{index}.jsonl").write_text(json.dumps(row) + "\n")
    recipe = fixture_recipe(lambda row, _context: svamp_row_task(row), rubric=None, controls=None)
    recipe = replace(recipe, source=replace(recipe.source, patterns=("*.jsonl",)))
    with ZephyrContext(max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            mode=SourceProcessingMode.QUICK,
            canonical_source=recipe.name,
        )
    root = tmp_path / "output"
    tasks = [
        TaskSpec.model_validate_json(row["task_json"])
        for file in (root / "normalize").glob("*.parquet")
        for row in pq.read_table(file).to_pylist()
    ]
    drops = [
        row for file in (root / "within-source-dedup/drops").glob("*.parquet") for row in pq.read_table(file).to_pylist()
    ]
    assert len(tasks) == 2 and len(drops) == 1
    assert drops[0]["kept_id"] < drops[0]["task_id"]
    assert drops[0]["kept_id"] in {task.id for task in tasks}
    manifest = json.loads((root / "manifest.json").read_text())
    assert (
        manifest["within_source_dedup"]["raw_count"],
        manifest["converted_rows"],
        manifest["within_source_dedup"]["within_source_duplicates"],
    ) == (3, 2, 1)
    assert result.input_rows == 3
