# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from fray.local_backend import LocalClient
from zephyr.context import ZephyrContext
from zephyr.readers import load_parquet

from taskcompendium.convert.answers import exact_answer_task, source_defect
from taskcompendium.models import ResourceGroups, TaskSpec
from taskcompendium.pipeline.controls import reference_reply
from taskcompendium.pipeline.conversion import run_conversion
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import Controls, ImportRejection, IntendedUse, RawRow, ReviewRubric, SourceRecipe
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.runtime.resources import inline_resource, resource_bytes


def convert_answer(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    if not row.data["answer"]:
        return source_defect("missing_answer", "The source has no answer")
    task = exact_answer_task(row, prompt=row.data["prompt"], answers=(row.data["answer"],), ignore_case=False)
    assert isinstance(task, TaskSpec)
    return task.model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("context.txt", b"color context"),))}
    )


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
        result = run_conversion(recipe, context, str(source), str(tmp_path / "output"))
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
