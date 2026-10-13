# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.pipeline.assessment import _original_files
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.source_evidence import OriginalRow, SourceEvidenceFile, original_rows
from taskcompendium.pipeline.sources import prepared_component_path, prepared_shared_path, raw_input_sha256


@pytest.mark.parametrize("representation", ["parquet", "prepared", "jsonl", "prepared_override"])
def test_original_evidence_recovers_source_ordinals_and_bounds_previews(tmp_path, representation):
    data = [{"text": f"row {index}\u2028" + "x" * 100} for index in range(6)]
    logical = "source.parquet" if representation == "parquet" else "source.jsonl"
    path = tmp_path / logical
    component = "component" if representation in ("prepared", "prepared_override") else None
    source_format = SourceFormat.PARQUET if representation == "parquet" else SourceFormat.JSONL
    spec = SourceFiles("fixture", "pin", (logical,), source_format, prepared_component=component)
    prepared_path = None
    if representation == "parquet":
        table = pa.Table.from_pylist(data)
        with pq.ParquetWriter(path, table.schema) as writer:
            writer.write_table(table.slice(0, 3))
            writer.write_table(table.slice(0, 0))
            writer.write_table(table.slice(3))
    elif representation == "prepared":
        prepared = prepared_component_path(StoragePath(str(path)), component)
        prepared_path = str(prepared)
        prepared.parent.mkdirs()
        table = pa.Table.from_pylist(
            [
                {"source_index": 8, "raw_json": "unselected invalid JSON"},
                {"source_index": 5, "raw_json": json.dumps(data[5])},
                {"source_index": 2, "raw_json": json.dumps(data[2])},
            ]
        )
        with pq.ParquetWriter(str(prepared), table.schema) as writer:
            writer.write_table(table.slice(0, 1))
            writer.write_table(table.slice(0, 0))
            writer.write_table(table.slice(1))
    else:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in data))
    rows = {index: OriginalRow(f"task-{index}", f"{logical}:{index}", raw_input_sha256(data[index])) for index in (2, 5)}
    recovered = list(
        original_rows(
            SourceEvidenceFile(logical, str(path), rows, prepared_path=prepared_path),
            spec=spec,
            context=ConversionContext({}, None),
            preview_characters=40,
        )
    )
    assert len(recovered) == 2
    assert {row["task_id"] for row in recovered} == {"task-2", "task-5"}
    for row in recovered:
        index = int(row["task_id"].split("-")[1])
        evidence = row["evidence"]["source"]
        assert evidence["raw_input_sha256"] == rows[index].raw_input_sha256
        assert f"row {index}" in evidence["preview"] and len(evidence["preview"]) == 40
        assert evidence["truncated"] and row["omissions"] == ["source: original row preview truncated"]


@pytest.mark.parametrize("failure", ["changed", "missing"])
def test_original_evidence_rejects_changed_or_missing_rows(tmp_path, failure):
    path = tmp_path / "source.parquet"
    data = {"bytes": b"original binary"}
    pq.write_table(pa.Table.from_pylist([data]), path)
    spec = SourceFiles("fixture", "pin", (path.name,), SourceFormat.PARQUET)
    index = 1 if failure == "missing" else 0
    expected = raw_input_sha256({"bytes": b"different binary"}) if failure == "changed" else raw_input_sha256(data)
    file = SourceEvidenceFile(path.name, str(path), {index: OriginalRow("task", f"{path.name}:{index}", expected)})
    with pytest.raises(ValueError, match="Original source row"):
        list(original_rows(file, spec=spec, context=ConversionContext({}, None), preview_characters=40))


@pytest.mark.parametrize("layout, override", [("2", False), ("3", False), ("3", True)])
def test_recovery_uses_recorded_acquisition_layout_and_local_overrides(tmp_path, layout, override):
    raw = tmp_path / "source.jsonl"
    data = {"text": "original"}
    spec = SourceFiles("fixture", "pin", (raw.name,), SourceFormat.JSONL, prepared_component="selected")
    prepared = (
        prepared_shared_path(StoragePath(str(raw)))
        if layout == "2"
        else prepared_component_path(StoragePath(str(raw)), "selected")
    )
    if override:
        raw.write_text(json.dumps(data) + "\n")
    else:
        prepared.parent.mkdirs()
        pq.write_table(pa.Table.from_pylist([{"source_index": 0, "raw_json": json.dumps(data)}]), str(prepared))
    records = {"task": [{"source_locator": f"{raw.name}:0", "raw_input_sha256": raw_input_sha256(data)}]}
    download = {
        "source_input": str(tmp_path),
        "files": {"prepared_component": "selected", "prepared_source_revision": layout},
        "staged_files": [raw.name],
        "source_file_overrides": {raw.name: {"path": str(raw)}} if override else {},
    }
    files = _original_files(records, download)
    recovered = list(original_rows(files[0], spec=spec, context=ConversionContext({}, None), preview_characters=100))
    assert recovered[0]["evidence"]["source"]["preview"] == json.dumps(data, sort_keys=True, separators=(",", ":"))
    assert files[0].prepared_path == (None if override else str(prepared))


def test_recorded_generated_content_cannot_be_downgraded_to_regeneration_failure(tmp_path):
    path = tmp_path / "recorded.parquet"
    pq.write_table(pa.Table.from_pylist([{"source_index": 0, "raw_json": '{"text":"changed"}'}]), path)
    rows = {0: OriginalRow("task", "generator.whl:0", raw_input_sha256({"text": "original"}))}
    spec = SourceFiles("fixture", "pin", ("generator.whl",), SourceFormat.GENERATED)
    file = SourceEvidenceFile("generator.whl", "unused.whl", rows, prepared_path=str(path))
    with pytest.raises(ValueError, match="Original source row changed"):
        list(original_rows(file, spec=spec, context=ConversionContext({}, None), preview_characters=100))


@dataclass(frozen=True)
class GeneratedEvidenceRows:
    count: int = 1

    def size(self, _path, _context):
        return 2

    def __call__(self, _path, _context, _part, indices):
        for index in sorted(indices):
            yield index, {"text": "retained" if index == 0 else "changed generation"}


def test_irreproducible_generator_keeps_identity_without_fabricating_original_content(tmp_path):
    path = tmp_path / "generator.whl"
    expected = raw_input_sha256({"text": "retained"})
    rows = {index: OriginalRow(f"task-{index}", f"{path.name}:{index}", expected) for index in (0, 1)}
    spec = SourceFiles("fixture", "pin", (path.name,), SourceFormat.GENERATED, parts=GeneratedEvidenceRows())
    recovered = list(
        original_rows(
            SourceEvidenceFile(path.name, str(path), rows),
            spec=spec,
            context=ConversionContext({}, None),
            preview_characters=100,
        )
    )
    assert len(recovered) == 2 and recovered[0]["evidence"]["source"]["available"]
    missing = recovered[1]
    assert missing["task_id"] == "task-1" and not missing["evidence"]["source"]["available"]
    assert missing["evidence"]["source"]["raw_input_sha256"] == expected
    assert "preview" not in missing["evidence"]["source"]
    assert missing["omissions"] == ["source: original generated row could not be reproduced"]
