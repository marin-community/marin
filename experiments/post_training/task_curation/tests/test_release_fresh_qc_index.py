# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the fresh-QC-index stage (the graph's staged-panel index producer)."""

import dataclasses
import gzip
import hashlib
import json
from pathlib import Path

import pytest
from marina.client import MarinaRequestError
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import fresh_qc_index as stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

KEY = "qc:" + "b" * 64


def _declarations(tmp_path: Path) -> ArtifactRef:
    document = {
        "Marin:fixture": {
            "id": "Marin:fixture",
            "data_restriction": "open",
            "independent_qc": {"evidence_url": f"file://{tmp_path}/review"},
        }
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "declarations.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _completion(tmp_path: Path, *, status: str = "PASS") -> ArtifactRef:
    document = {
        "kind": "atlas-testrelease-prior-qc-browse",
        "status": status,
        "results": [
            {
                "source_id": "Marin:fixture",
                "pipeline_key": KEY,
                "original_receipt": {"artifact": {"uri": "s3://bucket/review", "sha256": "b" * 64}, "current_pin": True},
            }
        ],
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "browse" / "completion.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _packed(content: dict) -> dict:
    raw = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return {
        "encoding": "gzip+base64",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "payload": __import__("base64").b64encode(gzip.compress(raw, mtime=0)).decode("ascii"),
    }


class _FakeQuery:
    """The prior load's review tables for one panel."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.tables = {
            "atlas_assessments": [
                {
                    "pipeline_key": KEY,
                    "source_id": "Marin:fixture",
                    "scope": "sample",
                    "model": {"model": "m"},
                    "rubric": "{}",
                    "curation": {},
                    "curation_uri": "s3://bucket/review",
                    "traces": [],
                }
            ],
            "atlas_populations": [
                {"pipeline_key": KEY, "cohort": "admitted", "population": 50, "sample_count": 2},
            ],
            "atlas_items": [
                {
                    "pipeline_key": KEY,
                    "task_id": f"t{i}",
                    "cohort": "admitted",
                    "status": "complete",
                    "verdict": "good",
                    "explanation": None,
                    "evidence_ids": [],
                    "packet_uri": None,
                    "has_task": True,
                    "omissions": None,
                    "exclusion_reason": None,
                    "confidence": None,
                    "reference_status": None,
                    "defects": None,
                    "provenance": None,
                    "replaced_task_sha256": None,
                    "repair_advisories": None,
                }
                for i in range(2)
            ],
            "atlas_raw_reviews": [
                {
                    "pipeline_key": KEY,
                    "task_id": f"t{i}",
                    "record": _packed({"typed_observation": {"task_id": f"t{i}"}}),
                    "reviewer": {"model": "m"},
                }
                for i in range(2)
            ],
            "atlas_review_batches": [
                {"pipeline_key": KEY, "batch_key": "k1", "record": {"attempts": []}, "chunk_count": 0},
            ],
            "atlas_review_batch_chunks": [],
            "atlas_review_batch_tasks": [
                {"pipeline_key": KEY, "task_id": "t0", "batch_key": "k1"},
                {"pipeline_key": KEY, "task_id": "t1", "batch_key": "k1"},
            ],
        }

    def query(self, sql: str, params: dict) -> dict:
        for table, rows in self.tables.items():
            if f"FROM {table}" in sql:
                order = [c.strip() for c in sql.split("ORDER BY ")[1].split(",")]
                order = [c.split()[0] for c in order]
                filtered = [r for r in rows if r.get("pipeline_key") == params.get("key")]
                for index, column in enumerate(order):
                    after = params.get(f"after_{index}")
                    if after is not None:
                        filtered = [r for r in filtered if str(r.get(column)) > str(after)]
                limit = 2
                if "LIMIT " in sql:
                    limit = int(sql.rsplit("LIMIT ", 1)[1].strip())
                return {"rows": filtered[:limit]}
        raise AssertionError(f"unexpected query: {sql[:80]}")


def _inputs(tmp_path: Path, **overrides) -> stage.FreshQcIndexInputs:
    inputs = stage.FreshQcIndexInputs(
        release_id="testrelease",
        prior_load_id="c" * 32,
        target_load_id="a" * 32,
        snapshot_id="assembly-test-v1",
        source_revision="e" * 40,
        output_root=str(tmp_path / "release"),
        service_url="https://marina.example.test",
        applet_id="applet",
        declarations=_declarations(tmp_path),
        qc_completion=_completion(tmp_path),
    )
    return dataclasses.replace(inputs, **overrides) if overrides else inputs


@pytest.fixture
def fake_query(tmp_path: Path, monkeypatch):
    query = _FakeQuery(tmp_path)
    opened = {}

    def fake_open(url, applet_id, control):
        opened["url"] = url
        return query.query

    monkeypatch.setattr("experiments.post_training.task_curation.release_stages.paced_query.open_paced_query", fake_open)
    monkeypatch.setattr("experiments.post_training.task_curation.release_stages.paced_query.RetryControl", lambda: None)
    query.tables["atlas_review_batch_chunks"] = [
        {"pipeline_key": KEY, "batch_key": "k1", "chunk_index": 0, "payload": "abc"},
        {"pipeline_key": KEY, "batch_key": "k1", "chunk_index": 1, "payload": "def"},
    ]
    return query


def test_write_exports_each_panel_from_the_prior_load(tmp_path, fake_query):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=False)
    assert proof["load_id"] == "a" * 32 and proof["snapshot_id"] == "assembly-test-v1"
    index = json.loads(StoragePath(proof["index"]["uri"]).read_text())
    row = index["receipts"][0]
    assert row["source_id"] == "Marin:fixture" and row["status"] == "complete" and row["records"] == 2
    document = json.loads(gzip.decompress(StoragePath(row["reference"]["uri"]).read_bytes()))
    assert document["receipt"]["artifact"]["sha256"] == "b" * 64
    assert document["assessment"]["source_id"] == "Marin:fixture"
    assert document["raw"][0]["record"]["typed_observation"]["task_id"] == "t0"
    assert document["data_restriction"] == "open"
    assert index["provenance"]["declarations"]["sha256"] == inputs.declarations.sha256


def test_dry_run_verifies_without_writing(tmp_path, fake_query):
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=True)
    assert proof["panels"] == 1
    assert not StoragePath(f"{inputs.output_root}/{stage.STAGED_DIR}").exists()
    assert not StoragePath(f"{inputs.output_root}/fresh-qc-index.json").exists()


def test_reassembles_a_chunked_batch_record(tmp_path, fake_query):
    fake_query.tables["atlas_review_batches"][0] = {
        "pipeline_key": KEY,
        "batch_key": "k1",
        "record": {"payload_chunks": 2},
        "chunk_count": 2,
    }
    inputs = _inputs(tmp_path)
    proof = stage.run_fresh_qc_index(inputs, dry_run=False)
    index = json.loads(StoragePath(proof["index"]["uri"]).read_text())
    document = json.loads(gzip.decompress(StoragePath(index["receipts"][0]["reference"]["uri"]).read_bytes()))
    assert document["batches"][0]["record"]["payload"] == "abcdef"
    assert "payload_chunks" not in document["batches"][0]["record"]


def test_refuses_a_panel_without_a_frozen_declaration(tmp_path, fake_query):
    inputs = _inputs(tmp_path)
    declarations = StoragePath(inputs.declarations.uri)
    document = json.loads(declarations.read_text())
    document["Marin:other"] = {"id": "Marin:other", "data_restriction": "open"}
    document.pop("Marin:fixture")
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    declarations.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(
        inputs, declarations=ArtifactRef(uri=str(declarations), sha256=hashlib.sha256(raw).hexdigest())
    )
    with pytest.raises(ValueError, match="no frozen declaration"):
        stage.run_fresh_qc_index(broken, dry_run=True)


def test_refuses_items_and_raw_that_disagree(tmp_path, fake_query):
    fake_query.tables["atlas_items"] = fake_query.tables["atlas_items"][:1]
    with pytest.raises(ValueError, match="1 items and 2 raw rows"):
        stage.run_fresh_qc_index(_inputs(tmp_path), dry_run=True)


def test_refuses_a_failed_browse_completion(tmp_path, fake_query):
    inputs = _inputs(tmp_path)
    path = StoragePath(inputs.qc_completion.uri)
    document = json.loads(path.read_text())
    document["status"] = "FAIL"
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw, auto_mkdir=True)
    broken = dataclasses.replace(
        inputs, qc_completion=ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())
    )
    with pytest.raises(ValueError, match="needs a PASS"):
        stage.run_fresh_qc_index(broken, dry_run=True)


def test_paged_halves_through_the_response_cap():
    """A page exceeding the applet's response cap halves until it fits, and the
    composite keyset still collects every row."""

    class _CapQuery:
        def __init__(self, rows):
            self.rows = rows
            self.pages = []

        def __call__(self, sql, params):
            limit = int(sql.rsplit("LIMIT ", 1)[1])
            self.pages.append(limit)
            if limit > 3:
                raise MarinaRequestError(413, "query response exceeds 5242880 bytes")
            order_sql = sql.split("ORDER BY ")[1].split(" LIMIT ")[0]
            order = [c.strip() for c in order_sql.split(",")]
            filtered = self.rows
            for index, column in enumerate(order):
                after = params.get(f"after_{index}")
                if after is not None:
                    filtered = [r for r in filtered if str(r.get(column)) > str(after)]
            return {"rows": filtered[:limit]}

    rows = [{"task_id": f"t{i}", "reviewer": "r", "record": {"n": i}} for i in range(7)]
    query = _CapQuery(rows)
    out = stage._paged(
        query,
        "SELECT task_id,record,reviewer FROM atlas_raw_reviews"
        " WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY task_id",
        {"load": "l", "key": "k"},
        ("task_id", "reviewer"),
    )
    assert [row["record"]["n"] for row in out] == list(range(7))
    assert query.pages[0] == stage.PAGE_ROWS and 3 in query.pages


def test_paged_refuses_a_row_larger_than_the_cap():
    class _Always413:
        def __call__(self, sql, params):
            raise MarinaRequestError(413, "query response exceeds 5242880 bytes")

    with pytest.raises(MarinaRequestError):
        stage._paged(
            _Always413(),
            "SELECT task_id FROM atlas_items WHERE load_id=:load AND pipeline_key=:key{predicate} ORDER BY task_id",
            {"load": "l", "key": "k"},
            ("task_id",),
        )
