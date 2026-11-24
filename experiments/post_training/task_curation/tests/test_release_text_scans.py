# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained text-scan release stage (catalog text preflight).

The fixtures build a real composed handoff — release and coverage documents sealed
over small real parquet assignment files whose text needs the catalog codec (control
characters, reserved-prefix literals, binary payloads) — and run the real regional
scan and merge paths, including the dry-run slice and every refusal.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.release_stages import text_scans
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

REGION_A = "alpha-region"
REGION_B = "beta-region"
RELEASE_ID = "testrelease"
SOURCE_ALPHA = "Fixture:alpha"
SOURCE_BETA = "Fixture:beta"
# Values that trigger the codec: a control character, a reserved-prefix literal, bytes.
CONTROL_TEXT = "prompt with a \x1f control character"
PREFIX_TEXT = text_scans.catalog_text.PREFIX + "utf8:0:fixture"
DEEP_TEXT = "meta \x0c value"
ASSIGNMENT_SCHEMA = pa.schema(
    [
        ("source_id", pa.string()),
        ("task_id", pa.string()),
        ("prompt", pa.string()),
        ("notes", pa.string()),
        ("payload", pa.binary()),
        ("meta", pa.struct([("a~b", pa.string())])),
    ]
)
SOURCE_REVISION = "a" * 40


def _rows(source_id: str, task_prefix: str, *, count: int = 2) -> list[dict]:
    """One affected row carrying every codec-triggering shape, then clean rows."""
    rows = [
        {
            "source_id": source_id,
            "task_id": f"{task_prefix}-1",
            "prompt": CONTROL_TEXT,
            "notes": PREFIX_TEXT,
            "payload": b"\x00\x01binary",
            "meta": {"a~b": DEEP_TEXT},
        }
    ]
    rows += [
        {
            "source_id": source_id,
            "task_id": f"{task_prefix}-{index}",
            "prompt": "clean prompt",
            "notes": "clean notes",
            "payload": None,
            "meta": {"a~b": "fine"},
        }
        for index in range(2, count + 1)
    ]
    return rows


def _write_assignment(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=ASSIGNMENT_SCHEMA), path, compression="zstd")
    return str(path)


def _seal_json(tmp_path: Path, name: str, document: dict) -> ArtifactRef:
    path = tmp_path / name
    path.write_bytes(json.dumps(document).encode())
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _handoff(tmp_path: Path, *, extra_full_path: str | None = None) -> ArtifactRef:
    """A composed handoff over two regional full files and one regional sample file."""
    full_a = _write_assignment(tmp_path / "alpha" / "full-a.parquet", _rows(SOURCE_ALPHA, "alpha"))
    full_b = _write_assignment(tmp_path / "beta" / "full-b.parquet", _rows(SOURCE_BETA, "beta"))
    sample_b = _write_assignment(tmp_path / "beta" / "sample-b.parquet", _rows(SOURCE_BETA, "beta-sample", count=1))
    release = {
        "paths": [full_a, full_b] + ([extra_full_path] if extra_full_path else []),
        "task_count": 4,
        "sources": [
            {"source_id": SOURCE_ALPHA, "task_count": 2},
            {"source_id": SOURCE_BETA, "task_count": 2},
        ],
    }
    coverage = {"sample_paths": [sample_b], "sample_task_count": 1}
    return _seal_json(
        tmp_path,
        "composed-handoff.json",
        {
            "release_id": RELEASE_ID,
            "release": _seal_json(tmp_path, "release.json", release).model_dump(),
            "coverage": _seal_json(tmp_path, "coverage.json", coverage).model_dump(),
            "confidence": _seal_json(tmp_path, "confidence.json", {"levels": []}).model_dump(),
        },
    )


def _inputs(tmp_path: Path, handoff: ArtifactRef, *, max_files: int = 0) -> text_scans.TextScanInputs:
    return text_scans.TextScanInputs(
        release_id=RELEASE_ID,
        source_revision=SOURCE_REVISION,
        handoff=handoff,
        output_root=str(tmp_path / "scans"),
        slice_output_root=str(tmp_path / "slice"),
        regions=(REGION_A, REGION_B),
        region_prefixes={REGION_A: str(tmp_path / "alpha") + "/", REGION_B: str(tmp_path / "beta") + "/"},
        max_files=max_files,
    )


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def _alert_rows(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]


def _receipt_document(tmp_path: Path, region: str) -> dict:
    path = tmp_path / "scans" / f"text-scan-{region}" / "regional-scan-receipt.json"
    return json.loads(path.read_text())


def test_regional_scan_seals_region_receipt_with_counts(tmp_path):
    inputs = _inputs(tmp_path, _handoff(tmp_path))
    proof = text_scans.regional_scan(inputs, REGION_A, dry_run=False, feeds=_feeds(tmp_path))
    receipt_path = tmp_path / "scans" / f"text-scan-{REGION_A}" / "regional-scan-receipt.json"
    assert proof["receipt"]["uri"] == str(receipt_path)
    document = json.loads(receipt_path.read_text())
    assert document["kind"] == f"atlas-{RELEASE_ID}-regional-text-scan"
    assert document["release_id"] == RELEASE_ID
    assert document["region"] == REGION_A
    assert document["rows"] == {SOURCE_ALPHA: 2}
    assert document["bad_rows"] == {SOURCE_ALPHA: 1}
    assert document["unsupported_identity_fields"] == 0
    assert document["files_scanned"] == [str(tmp_path / "alpha" / "full-a.parquet")]
    assert document["codec_sha256"] == text_scans.codec_sha256()
    assert document["all_rows_encoded_and_json_serialized"] is True
    assert document["affected_rows_lossless_roundtrip"] is True
    assert document["sql_writes"] == 0
    result = document["file_results"][0]
    assert result["phase"] == "full" and result["unsupported_identity_fields"] == 0
    assert sorted(entry["field"] for entry in document["fields"]) == ["/meta/a~0b", "/notes", "/payload", "/prompt"]
    diagnostics = pq.read_table(result["diagnostics"][0]["uri"])
    assert diagnostics.num_rows == 4
    assert diagnostics.column("identity_field").to_pylist() == [False] * 4
    assert diagnostics.column("source_id").to_pylist() == [SOURCE_ALPHA] * 4
    rows = _alert_rows(tmp_path)
    assert rows[0]["kind"] == text_scans.alerts.KIND_START
    assert rows[0]["key"] == f"atlas-release:{RELEASE_ID}:text_scans"
    assert rows[-1]["kind"] == text_scans.alerts.KIND_FINISH


def test_regional_scan_dry_run_writes_only_under_slice_root(tmp_path):
    inputs = _inputs(tmp_path, _handoff(tmp_path), max_files=1)
    proof = text_scans.regional_scan(inputs, REGION_B, dry_run=True, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True
    # Region beta holds the second full file and the sample file; the slice takes the first.
    assert proof["files_scanned"] == [str(tmp_path / "beta" / "full-b.parquet")]
    assert not (tmp_path / "scans").exists()
    slice_root = tmp_path / "slice" / f"text-scan-{REGION_B}"
    assert (slice_root / "regional-scan-receipt.json").exists()
    # Diagnostics names carry the document index of the file within the release paths.
    assert (slice_root / "diagnostics" / "full-00001.parquet").exists()


def test_regional_scan_refuses_file_outside_region_prefix(tmp_path):
    stray = str(tmp_path / "stray" / "full-c.parquet")
    inputs = _inputs(tmp_path, _handoff(tmp_path, extra_full_path=stray))
    with pytest.raises(ValueError, match="outside every configured region prefix"):
        text_scans.regional_scan(inputs, REGION_A, dry_run=False, feeds=_feeds(tmp_path))
    rows = _alert_rows(tmp_path)
    assert rows[-1]["kind"] == text_scans.alerts.KIND_FAIL
    assert rows[-1]["key"] == f"atlas-release:{RELEASE_ID}:text_scans"


def test_regional_scan_refuses_failed_roundtrip(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, _handoff(tmp_path))
    monkeypatch.setattr(
        text_scans.catalog_text,
        "decode_catalog_value",
        lambda value: {} if isinstance(value, dict) else value,
    )
    with pytest.raises(AssertionError, match=SOURCE_ALPHA):
        text_scans.regional_scan(inputs, REGION_A, dry_run=False, feeds=_feeds(tmp_path))
    rows = _alert_rows(tmp_path)
    assert rows[-1]["kind"] == text_scans.alerts.KIND_FAIL


def test_regional_scan_refuses_write_slice_and_unknown_region(tmp_path):
    inputs = _inputs(tmp_path, _handoff(tmp_path), max_files=2)
    with pytest.raises(ValueError, match="write-mode regional scan covers every file"):
        text_scans.regional_scan(inputs, REGION_A, dry_run=False, feeds=_feeds(tmp_path))
    with pytest.raises(ValueError, match="not one of the typed regions"):
        text_scans.regional_scan(inputs, "missing-region", dry_run=True, feeds=_feeds(tmp_path))


def test_merge_seals_pass_receipt_from_regional_scans(tmp_path):
    inputs = _inputs(tmp_path, _handoff(tmp_path))
    feeds = _feeds(tmp_path)
    receipts = tuple(
        ArtifactRef(**text_scans.regional_scan(inputs, region, dry_run=False, feeds=feeds)["receipt"])
        for region in (REGION_A, REGION_B)
    )
    proof = text_scans.merge_region_receipts(inputs, receipts, dry_run=False, feeds=feeds)
    assert proof["kind"] == f"atlas-{RELEASE_ID}-curriculum-catalog-text-preflight"
    assert proof["status"] == "PASS" and proof["dry_run"] is False
    assert proof["task_count"] == 4 and proof["sample_task_count"] == 1 and proof["source_count"] == 2
    assert proof["unsupported_identity_fields"] == 0 and proof["manifest_fields"] == []
    assert proof["affected_rows"] == {"full": 2, "sample": 1}
    assert proof["affected_sources"] == {"full": {SOURCE_ALPHA: 1, SOURCE_BETA: 1}, "sample": {SOURCE_BETA: 1}}
    assert [entry["region"] for entry in proof["regions"]] == [REGION_A, REGION_B]
    assert proof["regions"][0]["receipt"] == receipts[0].model_dump()
    counts = {(row["source_id"], row["field"]): row["count"] for row in proof["field_counts"]}
    assert len(counts) == 8
    assert counts[(SOURCE_ALPHA, "/meta/a~0b")] == 1
    assert counts[(SOURCE_BETA, "/prompt")] == 2
    sealed_path = tmp_path / "scans" / "scan-receipt.json"
    sealed = json.loads(sealed_path.read_text())
    assert sealed["status"] == "PASS" and sealed["codec_sha256"] == text_scans.codec_sha256()
    assert sealed["handoff"] == {"uri": inputs.handoff.uri, "sha256": inputs.handoff.sha256}
    # A dry-run merge performs the same verification with zero writes.
    sealed_path.unlink()
    dry = text_scans.merge_region_receipts(inputs, receipts, dry_run=True, feeds=feeds)
    assert dry["status"] == "PASS" and dry["dry_run"] is True
    assert not sealed_path.exists()
    rows = _alert_rows(tmp_path)
    assert rows[-1]["kind"] == text_scans.alerts.KIND_FINISH
    assert rows[-1]["key"] == f"atlas-release:{RELEASE_ID}:text_scans_merge"
    assert not any(row["kind"] == text_scans.alerts.KIND_FAIL for row in rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("codec", "codec digests"),
        ("handoff", "handoff references"),
        ("rows", "disagree with the release sources"),
        ("unsupported", "unsupported identity fields"),
        ("region", "claim the same region"),
        ("files", "differ from the sealed document"),
    ],
)
def test_merge_refusals(tmp_path, mutation, message):
    inputs = _inputs(tmp_path, _handoff(tmp_path))
    feeds = _feeds(tmp_path)
    scanned = {
        region: text_scans.regional_scan(inputs, region, dry_run=False, feeds=feeds)["receipt"]
        for region in (REGION_A, REGION_B)
    }
    document = _receipt_document(tmp_path, REGION_A)
    if mutation == "codec":
        doctored = _seal_json(tmp_path, "doctored.json", {**document, "codec_sha256": "e" * 64})
    elif mutation == "handoff":
        doctored = _seal_json(
            tmp_path, "doctored.json", {**document, "handoff": {"uri": "/elsewhere/handoff.json", "sha256": "d" * 64}}
        )
    elif mutation == "rows":
        drifted_result = {**document["file_results"][0], "rows": {SOURCE_ALPHA: 3}}
        doctored = _seal_json(tmp_path, "doctored.json", {**document, "file_results": [drifted_result]})
    elif mutation == "unsupported":
        drifted_result = {**document["file_results"][0], "unsupported_identity_fields": 1}
        doctored = _seal_json(tmp_path, "doctored.json", {**document, "file_results": [drifted_result]})
    elif mutation == "region":
        doctored = _seal_json(tmp_path, "doctored.json", {**document, "region": REGION_B})
    else:
        drifted = {**document["file_results"][0], "input": "/stray/full.parquet"}
        doctored = _seal_json(
            tmp_path,
            "doctored.json",
            {**document, "files_scanned": ["/stray/full.parquet"], "file_results": [drifted]},
        )
    receipts = (doctored, ArtifactRef(**scanned[REGION_B]))
    with pytest.raises(ValueError, match=message):
        text_scans.merge_region_receipts(inputs, receipts, dry_run=True, feeds=feeds)
    rows = _alert_rows(tmp_path)
    assert rows[-1]["kind"] == text_scans.alerts.KIND_FAIL
    assert rows[-1]["key"] == f"atlas-release:{RELEASE_ID}:text_scans_merge"


def test_merge_requires_receipts_and_inputs_validate(tmp_path):
    inputs = _inputs(tmp_path, _handoff(tmp_path))
    with pytest.raises(ValueError, match="requires the regional receipt references"):
        text_scans.merge_region_receipts(inputs, (), dry_run=True, feeds=_feeds(tmp_path))
    base = {
        "release_id": RELEASE_ID,
        "source_revision": SOURCE_REVISION,
        "handoff": ArtifactRef(uri=str(tmp_path / "handoff.json"), sha256="c" * 64),
        "output_root": str(tmp_path / "scans"),
        "slice_output_root": str(tmp_path / "slice"),
        "regions": (REGION_A,),
        "region_prefixes": {REGION_A: str(tmp_path / "alpha") + "/"},
    }
    text_scans.TextScanInputs(**base).validate()
    with pytest.raises(ValueError, match="exactly the typed regions"):
        text_scans.TextScanInputs(**{**base, "regions": (REGION_A, REGION_B)}).validate()
    with pytest.raises(ValueError, match="end with '/'"):
        text_scans.TextScanInputs(**{**base, "region_prefixes": {REGION_A: str(tmp_path / "alpha")}}).validate()
    with pytest.raises(ValueError, match="max_files"):
        text_scans.TextScanInputs(**{**base, "max_files": -1}).validate()
