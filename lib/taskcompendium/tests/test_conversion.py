# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from taskcompendium.convert.answers import exact_answer_task, source_defect
from taskcompendium.models import ResourceGroups, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.conversion import convert_raw_row, convert_source_row
from taskcompendium.pipeline.execution_telemetry import SourceTelemetry
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    Controls,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    ReviewRubric,
    SourceRecipe,
)
from taskcompendium.pipeline.source_processing import (
    ConversionResult,
    SourceProcessingMode,
    run_source_pipeline,
    write_conversion,
)
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.runtime.resources import inline_resource, resource_bytes


def convert_answer(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    if not row.data["answer"]:
        return source_defect("missing_answer", "The source has no answer")
    task = exact_answer_task(row, prompt=row.data["prompt"], answers=(row.data["answer"],), ignore_case=False)
    return cast(TaskSpec, task).model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("context.txt", b"color context"),))}
    )


def whole_parquet(file: StoragePath, _context: ConversionContext):
    yield from load_parquet(str(file))


def selected_answer(row: dict, _context: ConversionContext) -> bool:
    return row["keep"]


def decode_answer(row: dict, _context: ConversionContext) -> dict:
    return {**row, "prompt": f"Decoded: {row['prompt']}"}


def test_recipe_free_conversion_persists_caller_identity_and_execution_counters(tmp_path: Path):
    row = RawRow(
        "caller-task",
        Source(dataset="colors", revision="pinned", row="archive:17", importer_revision="2"),
        {"prompt": "Name a color", "answer": "red"},
    )
    conversion = ConversionContext(inputs={}, grader_environment=None)
    output = tmp_path / "output"
    telemetry = SourceTelemetry("colors", str(output))
    with ZephyrContext(client=LocalClient(), max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with telemetry.record():
            with telemetry.phase("convert_write") as phase:
                result = write_conversion(
                    Dataset.from_list([row]).map(partial(convert_raw_row, convert=convert_answer, context=conversion)),
                    context,
                    str(output),
                    telemetry=phase,
                )
    (record,) = [row for shard in Path(result.normalized_path).glob("*.parquet") for row in load_parquet(str(shard))]
    task = TaskSpec.model_validate_json(record["task_json"])
    assert task.id == "caller-task"
    assert task.source.row == "archive:17"
    assert cast(TextMessage, task.context.events[0]).content == "Name a color"
    report = json.loads((output / "telemetry.json").read_text())
    execution = report["phases"][0]["executions"][0]
    assert execution["counters"]["source/normalize/attempts"] == 1
    assert execution["counters"]["source/normalize/task_rows"] == 1


def test_conversion_retains_payload_before_decoder_rewrites():
    recipe = SourceRecipe(
        name="colors",
        version="1",
        source=SourceFiles("colors", "pinned", ("rows.jsonl",), SourceFormat.JSONL, decode=decode_answer),
        convert=convert_answer,
        intended_use=IntendedUse.TRAIN,
        rubric=None,
        controls=None,
    )
    original = {"path": "archive/task", "prompt": "Name a color", "answer": "red", "binary": b"original bytes"}
    converted = convert_source_row({"locator": "rows.jsonl:7", "data": original}, recipe)
    assert converted.original_data == original
    task = cast(NormalizedTask, converted.result).task
    assert cast(TextMessage, task.context.events[0]).content == "Decoded: Name a color"


def test_split_parquet_matches_whole_file_tasks_and_rejections_after_selection(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    rows = [
        {"path": f"row-{index}", "prompt": "Name a color", "answer": "red" if index % 7 else "", "keep": index % 5 != 0}
        for index in range(24)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source / "rows.parquet", row_group_size=3)
    recipe = SourceRecipe(
        name="colors",
        version="1",
        source=SourceFiles(
            "colors",
            "pinned",
            ("rows.parquet",),
            SourceFormat.PARQUET,
            select=selected_answer,
            decode=decode_answer,
        ),
        convert=convert_answer,
        intended_use=IntendedUse.TRAIN,
        rubric=None,
        controls=None,
    )
    with ZephyrContext(client=LocalClient(), max_workers=2, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        whole = run_source_pipeline(
            replace(recipe, source=replace(recipe.source, read=whole_parquet)),
            context,
            str(source),
            str(tmp_path / "whole"),
            mode=SourceProcessingMode.QUICK,
            canonical_source=recipe.name,
        )
        split = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "split"),
            parquet_shard_bytes=1,
            mode=SourceProcessingMode.QUICK,
            canonical_source=recipe.name,
        )
    whole, split = cast(ConversionResult, whole), cast(ConversionResult, split)
    whole_files = list(Path(whole.normalized_path).glob("*.parquet"))
    split_files = list(Path(split.normalized_path).glob("*.parquet"))
    assert len(whole_files) == 1
    assert len(split_files) == 8
    whole_rows = {row["source_row"]: row for file in whole_files for row in load_parquet(str(file))}
    split_rows = {row["source_row"]: row for file in split_files for row in load_parquet(str(file))}
    assert whole_rows == split_rows
    assert set(split_rows) == {f"rows.parquet:{index}" for index in range(24) if rows[index]["keep"]}
    assert split.input_rows == whole.input_rows == 19
    assert split.converted_rows == whole.converted_rows == 16
    assert split.rejections == whole.rejections == {"source_defect:missing_answer": 3}
    task = TaskSpec.model_validate_json(split_rows["rows.parquet:23"]["task_json"])
    assert cast(TextMessage, task.context.events[0]).content == "Decoded: Name a color"


def test_conversion_retains_duplicate_tasks_and_rejections_without_review_or_controls(tmp_path: Path):
    source = tmp_path / "input"
    source.mkdir()
    rows = [
        {"path": "first", "prompt": "Name a color", "answer": "red"},
        {"path": "duplicate", "prompt": "Name a color", "answer": "red"},
        {"path": "invalid", "prompt": "Name a color", "answer": ""},
    ]
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    recipe = SourceRecipe(
        name="colors",
        version="1",
        source=SourceFiles("colors", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert_answer,
        intended_use=IntendedUse.TRAIN,
        rubric=ReviewRubric(id="colors", version="1", criteria=("Check correctness",)),
        controls=Controls(golden=reference_reply),
        resource_budget_bytes=1,
    )
    with ZephyrContext(client=LocalClient(), max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        result = run_source_pipeline(
            recipe,
            context,
            str(source),
            str(tmp_path / "output"),
            mode=SourceProcessingMode.QUICK,
            canonical_source=recipe.name,
        )
    result = cast(ConversionResult, result)
    records = [row for shard in Path(result.normalized_path).glob("*.parquet") for row in load_parquet(str(shard))]
    assert result.input_rows == 3
    assert result.converted_rows == 2
    assert result.rejections == {"source_defect:missing_answer": 1}
    assert [row["original_path"] for row in records] == ["first", "duplicate", "invalid"]
    tasks = [TaskSpec.model_validate_json(row["task_json"]) for row in records if row["task_json"]]
    assert tasks[0].context == tasks[1].context
    assert tasks[0].id != tasks[1].id
    assert resource_bytes(tasks[0].resources.worker[0]) == b"color context"
    reviewed = normalize_row({"locator": "rows.jsonl:0", "data": rows[0]}, recipe)
    assert reviewed["audit"]["normalization_rejection"]["reason"] == "resources_over_budget"
    assert records[2]["normalization_reason"] == "missing_answer"
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["reviewed"] is False
    assert manifest["verified"] is False
    assert not (tmp_path / "output" / "final").exists()
