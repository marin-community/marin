# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from taskcompendium.convert.answers import exact_answer_task, source_defect
from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.delivery import rewritten_task
from taskcompendium.convert.environment import grading_environment
from taskcompendium.convert.script_grader import script_package
from taskcompendium.models import ResourceGroups, ScriptGrader, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.conversion import convert_raw_row, convert_source_row
from taskcompendium.pipeline.execution_telemetry import SourceTelemetry
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import (
    Controls,
    Converter,
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

GRADE = b"from colors.scoring import score\nprint(score())\n"
SCORER = b"def score():\n    return 1.0\n"
IMAGE = "grader@sha256:" + "c" * 64


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


def convert_script_answer(row: RawRow, context: ConversionContext) -> TaskSpec | NormalizedTask | ImportRejection:
    if not row.data["answer"]:
        return source_defect("missing_answer", "The source has no answer")
    package = script_package(
        tuple(inline_resource(name, path.read_bytes()) for name, path in context.inputs.items()),
        {"words": [row.data["answer"]]},
        environment=required_grader_environment(context),
        timeout=30,
        answer_path="/app/answer.txt",
    )
    task = conversation_task(
        row,
        events=(TextMessage(role="user", content=f"{row.data['prompt']}. Return one color."),),
        package=package,
    )
    return rewritten_task(task, original=row.data["prompt"], reason="response_format")


@pytest.fixture(params=[convert_answer, convert_script_answer], ids=["answer", "packaged-script"])
def conversion_inputs(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Converter, ConversionContext]:
    convert = cast(Converter, request.param)
    if convert is convert_answer:
        return convert, ConversionContext(inputs={}, grader_environment=None)
    files = {"grade.py": GRADE, "colors/__init__.py": b"", "colors/scoring.py": SCORER}
    inputs = {}
    for name, content in files.items():
        path = tmp_path / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        inputs[name] = StoragePath(str(path))
    return convert, ConversionContext(inputs=inputs, grader_environment=grading_environment(IMAGE))


def test_custom_conversion_writes_tasks_rejections_and_execution_evidence_without_recipe(
    tmp_path: Path, conversion_inputs: tuple[Converter, ConversionContext]
):
    convert, conversion = conversion_inputs
    rows = [
        RawRow(
            f"custom-{index}",
            Source(dataset="custom/colors", revision="pinned", row=name, importer_revision="2"),
            {"path": name, "prompt": "Name a color", "answer": answer},
        )
        for index, (name, answer) in enumerate((("first", "red"), ("duplicate", "red"), ("invalid", "")))
    ]
    output = tmp_path / "output"
    # Ingestion evidence may share the output root without being a conversion output.
    output.mkdir()
    (output / "input.json").write_text('{"revision": "pinned"}')
    telemetry = SourceTelemetry("custom/colors", str(output))
    with ZephyrContext(client=LocalClient(), max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
        with telemetry.record():
            with telemetry.phase("convert_write") as phase:
                result = write_conversion(
                    Dataset.from_list(rows).map(partial(convert_raw_row, convert=convert, context=conversion)),
                    context,
                    str(output),
                    telemetry=phase,
                )
    records = sorted(
        (row for shard in Path(result.normalized_path).glob("*.parquet") for row in load_parquet(str(shard))),
        key=lambda row: row["task_id"],
    )
    assert (result.input_rows, result.converted_rows, result.rejections) == (3, 2, {"source_defect:missing_answer": 1})
    assert [row["original_path"] for row in records] == ["first", "duplicate", "invalid"]
    tasks = [TaskSpec.model_validate_json(row["task_json"]) for row in records[:2]]
    assert [task.id for task in tasks] == ["custom-0", "custom-1"]
    assert [task.source for task in tasks] == [row.source for row in rows[:2]]
    assert tasks[0].context == tasks[1].context
    if convert is convert_script_answer:
        grader = cast(ScriptGrader, tasks[0].grader)
        assert grader.environment == conversion.grader_environment
        resources = {resource.path: resource_bytes(resource) for resource in tasks[0].resources.verifier}
        assert resources["grade.py"] == GRADE
        assert resources["colors/scoring.py"] == SCORER
        assert json.loads(resources["config.json"]) == {"words": ["red"]}
        assert records[0]["normalization_changes"] == [
            {
                "field": "instruction",
                "reason": "response_format",
                "original": "Name a color",
                "replacement": "Name a color. Return one color.",
            }
        ]
    else:
        assert resource_bytes(tasks[0].resources.worker[0]) == b"color context"
    assert records[2]["normalization_reason"] == "missing_answer"
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["reviewed"] is False and manifest["verified"] is False
    assert not (output / "final").exists()
    report = json.loads((output / "telemetry.json").read_text())
    assert report["status"] == "completed"
    assert [phase["phase"] for phase in report["phases"]] == ["convert_write"]
    execution = report["phases"][0]["executions"][0]
    assert execution["execution_id"]
    assert execution["counters"]["source/normalize/attempts"] == 3
    assert execution["counters"]["source/normalize/task_rows"] == 2
    assert execution["counters"]["source/normalize/source_defect"] == 1
    assert "source/decode/attempts" not in execution["counters"]


def test_standard_conversion_retains_payload_before_decoder_rewrites():
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
    assert converted.original_path == "archive/task"
    assert converted.raw.source.row == "rows.jsonl:7"
    assert converted.raw.data["prompt"] == "Decoded: Name a color"
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
