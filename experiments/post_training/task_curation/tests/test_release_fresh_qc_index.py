# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the fresh-QC-index stage (the graph's staged-panel index producer)."""

import dataclasses
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest
from marin.inference.openai_chat import OpenAIChatClient
from marina.client import MarinaRequestError
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.full_qc import ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.release_stages import fresh_qc_index as stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.sample_qc import review_panel

KEY = "qc:" + "b" * 64


def _declarations(tmp_path: Path) -> ArtifactRef:
    document = {
        "Marin:fixture": {
            "id": "Marin:fixture",
            "dataset_id": "fixture",
            "dataset_revision": "old",
            "pipeline": None,
            "data_restriction": "open",
            "independent_qc": {"evidence_url": "s3://bucket/review"},
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
                "original_receipt": {
                    "artifact": {"uri": "s3://bucket/review/.artifact.json", "sha256": "b" * 64},
                    "panel": {"dataset_id": "fixture", "dataset_revision": "old", "sample_size": 2},
                    "current_pin": True,
                },
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
        fresh_panels={},
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


@pytest.fixture
def fresh_panel(tmp_path, sample_panel, review_server):
    reviewer = ChatReviewer(
        OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2),
        local_reviewer(ReviewerVersion.V3),
        query_cache_root=str(tmp_path / "cache"),
    )
    review = review_panel(sample_panel, reviewer, str(tmp_path / "reviews"))
    uri = str(StoragePath(review.path) / ".artifact.json")
    reference = ArtifactRef(uri=uri, sha256=hashlib.sha256(StoragePath(uri).read_bytes()).hexdigest())
    declaration = {
        "id": sample_panel.source_id,
        "data_restriction": "open",
        "dataset_id": sample_panel.dataset_id,
        "dataset_revision": sample_panel.dataset_revision,
        "pipeline": {"version": sample_panel.converter_version},
        "actor_census_input": {
            "task_specs_sha256": sample_panel.task_specs_sha256,
            "measured_tasks": sample_panel.retained_tasks,
        },
        "independent_qc": {"evidence_url": review.path},
    }
    return reference, declaration


@pytest.mark.parametrize("carry_prior", [True, False])
def test_cli_combines_carried_panel_with_selected_current_artifact(
    tmp_path, fake_query, fresh_panel, monkeypatch, capsys, carry_prior
):
    reference, declaration = fresh_panel
    inputs = _inputs(tmp_path, fresh_panels={declaration["id"]: reference})
    path = StoragePath(inputs.declarations.uri)
    old = json.loads(path.read_text())["Marin:fixture"]
    declarations = {"Marin:carried": {**old, "id": "Marin:carried"}, declaration["id"]: declaration}
    path.write_text(json.dumps(declarations))
    completion_path = StoragePath(inputs.qc_completion.uri)
    completion = json.loads(completion_path.read_text())
    completion["results"][0]["source_id"] = "Marin:carried"
    if not carry_prior:
        completion["results"] = []
    completion_path.write_text(json.dumps(completion))
    fake_query.tables["atlas_assessments"][0]["source_id"] = "Marin:carried"
    document = dataclasses.asdict(inputs)
    document["declarations"] = {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    document["qc_completion"] = {
        "uri": str(completion_path),
        "sha256": hashlib.sha256(completion_path.read_bytes()).hexdigest(),
    }
    document["fresh_panels"] = {declaration["id"]: reference.model_dump(mode="json")}
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["fresh_qc_index", "--inputs", str(config), "--mode", "write"])

    stage.main()

    proof = json.loads(capsys.readouterr().out)
    index = json.loads(StoragePath(proof["index"]["uri"]).read_text())
    panels = {row["source_id"]: row for row in index["receipts"]}
    assert set(panels) == ({"Marin:carried", declaration["id"]} if carry_prior else {declaration["id"]})
    current = json.loads(gzip.decompress(StoragePath(panels[declaration["id"]]["reference"]["uri"]).read_bytes()))
    assert current["receipt"]["artifact"] == reference.model_dump(mode="json")
    assert current["receipt"]["current_pin"] is True
    assert {row["task_id"] for row in current["items"]} == {"kept", "dropped", "unshaped"}
    assert len(current["raw"]) == 3
    if carry_prior:
        carried = json.loads(gzip.decompress(StoragePath(panels["Marin:carried"]["reference"]["uri"]).read_bytes()))
        assert carried["receipt"] == completion["results"][0]["original_receipt"]


@pytest.mark.parametrize("fault", ["digest", "source", "overlap"])
def test_fresh_panel_refuses_wrong_evidence(tmp_path, fake_query, fresh_panel, fault):
    reference, declaration = fresh_panel
    inputs = _inputs(tmp_path, fresh_panels={declaration["id"]: reference})
    completion_path = StoragePath(inputs.qc_completion.uri)
    completion = json.loads(completion_path.read_text())
    if fault != "overlap":
        completion["results"] = []
    completion_path.write_text(json.dumps(completion))
    inputs = dataclasses.replace(
        inputs,
        qc_completion=ArtifactRef(
            uri=str(completion_path), sha256=hashlib.sha256(completion_path.read_bytes()).hexdigest()
        ),
    )
    if fault == "digest":
        inputs = dataclasses.replace(
            inputs, fresh_panels={declaration["id"]: reference.model_copy(update={"sha256": "0" * 64})}
        )
    if fault == "source":
        declaration = {**declaration, "id": "Marin:wrong"}
        inputs = dataclasses.replace(inputs, fresh_panels={declaration["id"]: reference})
    path = StoragePath(inputs.declarations.uri)
    path.write_text(json.dumps({declaration["id"]: declaration}))
    inputs = dataclasses.replace(
        inputs, declarations=ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    )
    with pytest.raises(ValueError):
        stage.run_fresh_qc_index(inputs, dry_run=False)
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


@pytest.mark.parametrize("fault", [None, "changed_retained_pin", "corrupt_retained_panel"])
def test_cli_refreshes_selected_panel_and_reuses_verified_staged_bytes(
    tmp_path, fake_query, fresh_panel, monkeypatch, capsys, fault
):
    reference, declaration = fresh_panel
    inputs = _inputs(tmp_path, fresh_panels={declaration["id"]: reference})
    path = StoragePath(inputs.declarations.uri)
    declarations = json.loads(path.read_text())
    declarations[declaration["id"]] = declaration
    path.write_text(json.dumps(declarations))
    inputs = dataclasses.replace(
        inputs, declarations=ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    )
    original = stage.run_fresh_qc_index(inputs, dry_run=False)
    prior = json.loads(StoragePath(original["index"]["uri"]).read_bytes())
    retained = next(row for row in prior["receipts"] if row["source_id"] == "Marin:fixture")
    replacement = StoragePath(str(tmp_path / "replacement" / ".artifact.json"))
    replacement.write_bytes(StoragePath(reference.uri).read_bytes(), auto_mkdir=True)
    fresh = ArtifactRef(uri=str(replacement), sha256=hashlib.sha256(replacement.read_bytes()).hexdigest())
    declarations[declaration["id"]] = {**declaration, "independent_qc": {"evidence_url": str(replacement)}}
    if fault == "changed_retained_pin":
        declarations["Marin:fixture"]["dataset_revision"] = "unreviewed"
    if fault == "corrupt_retained_panel":
        StoragePath(retained["reference"]["uri"]).write_bytes(b"corrupt")
    current_path = StoragePath(str(tmp_path / "current-declarations.json"))
    current_path.write_text(json.dumps(declarations))
    refresh = dataclasses.replace(
        inputs,
        declarations=ArtifactRef(uri=str(current_path), sha256=hashlib.sha256(current_path.read_bytes()).hexdigest()),
        prior_index=ArtifactRef(**original["index"]),
        fresh_panels={declaration["id"]: fresh},
        snapshot_id="assembly-current-v2",
        source_revision="f" * 40,
        output_root=str(tmp_path / "refreshed"),
    )

    def no_sql(*args):
        raise AssertionError("index refresh must use sealed artifacts without SQL")

    monkeypatch.setattr(stage.paced_query, "open_paced_query", no_sql)
    if fault:
        with pytest.raises(ValueError):
            stage.run_fresh_qc_index(refresh, dry_run=False)
        assert not StoragePath(f"{refresh.output_root}/fresh-qc-index.json").exists()
        return
    stage.run_fresh_qc_index(refresh, dry_run=True)
    assert not StoragePath(refresh.output_root).exists()
    config = tmp_path / "refresh-inputs.json"
    document = dataclasses.asdict(refresh)
    for field in ("declarations", "qc_completion", "prior_index"):
        document[field] = getattr(refresh, field).model_dump(mode="json")
    document["fresh_panels"] = {declaration["id"]: fresh.model_dump(mode="json")}
    config.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["fresh_qc_index", "--inputs", str(config), "--mode", "write"])
    stage.main()
    result = json.loads(capsys.readouterr().out)
    current = json.loads(StoragePath(result["index"]["uri"]).read_bytes())
    panels = {row["source_id"]: row for row in current["receipts"]}
    assert panels["Marin:fixture"] == retained
    replaced = json.loads(gzip.decompress(StoragePath(panels[declaration["id"]]["reference"]["uri"]).read_bytes()))
    assert replaced["receipt"]["artifact"] == fresh.model_dump(mode="json")
    assert replaced["receipt"]["current_pin"] is True
    assert {row["task_id"] for row in replaced["items"]} == {"kept", "dropped", "unshaped"}
    assert current["provenance"]["prior_index"] == original["index"]
    assert current["snapshot_id"] == "assembly-current-v2"


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
