# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained catalog-load stage (R40 gate slice 2)."""

import dataclasses
import gzip
import hashlib
import json
import sys
import threading

import pytest
from finestore.rl.difficulty import Solver
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import BackendFit, BackendSourceMap, BackendSurvey, FitSupport

from experiments.post_training.task_curation.difficulty.ledger import append_rows
from experiments.post_training.task_curation.difficulty.seal import seal_ledger
from experiments.post_training.task_curation.difficulty.state import CatalogSource, LedgerRow, Outcome
from experiments.post_training.task_curation.release_stages import catalog_load as cl
from experiments.post_training.task_curation.release_stages import sizing
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog import load_reviews as load_reviews_module

PRIOR = "b" * 32
TARGET = "c" * 32
MEASURED_SHA = "a" * 64
MEASURED_SCHEMA = "e" * 64


def _write_json(tmp_path, name, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ArtifactRef(uri=str(path), sha256=digest)


def _inputs(tmp_path, **overrides) -> cl.CatalogLoadInputs:
    contracts = overrides.pop("contract_rows", [])
    contracts_ref = _write_json(tmp_path, "contracts.json", {"contracts": contracts})
    base = dict(
        release_id="rtest",
        prior_load_id=PRIOR,
        target_load_id=TARGET,
        snapshot_id="assembly-test-v1",
        source_revision="rev-1",
        catalog_root="/tmp/catalog",
        receipt_root=str(tmp_path / "releases/rtest"),
        publication=_write_json(tmp_path, "publication.json", {"backend": {}, "identity": {}}),
        staging={
            "prior_qc_completion": _write_json(tmp_path, "browse.json", {}),
            "fresh_qc_index": _write_json(tmp_path, "qc.json", {}),
            "contracts": contracts_ref,
            "history_receipt": _write_json(tmp_path, "history.json", {}),
        },
        expected_source_count=2,
        expected_contract_count=len(contracts) or 1,
        expected_contract_tasks=sum(row["input"]["measured_tasks"] for row in contracts) or 10,
        expected_selected_panels=1,
        expected_browse_source_count=1,
        measured_module="does-not-matter",
        measured_revision="rev",
        measured_sha256=MEASURED_SHA,
        measured_schema_sha256=MEASURED_SCHEMA,
    )
    return cl.CatalogLoadInputs(**{**base, **overrides})


def test_inputs_validate_load_ids_and_staging(tmp_path):
    inputs = _inputs(tmp_path)
    inputs.validate()
    with pytest.raises(ValueError, match="32-hex"):
        dataclasses.replace(inputs, target_load_id="not-a-load-id").validate()
    staging = dict(inputs.staging)
    del staging["history_receipt"]
    with pytest.raises(ValueError, match="history_receipt"):
        dataclasses.replace(inputs, staging=staging).validate()


def test_read_ref_refuses_digest_mismatch(tmp_path):
    ref = _write_json(tmp_path, "doc.json", {"a": 1})
    tampered = ArtifactRef(uri=ref.uri, sha256="0" * 64)
    with pytest.raises(ValueError, match="digest mismatch"):
        cl.read_ref(tampered)


def test_dry_run_refuses_contract_count_mismatch(tmp_path):
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}], expected_contract_count=2)
    with pytest.raises(ValueError, match="derived contracts"):
        cl.run_dry_run(inputs)


def test_dry_run_refuses_measured_dependency_drift(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}])
    monkeypatch.setattr(cl, "backend_schema_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="measured"):
        cl.run_dry_run(inputs)


def _aligned_state(tmp_path, monkeypatch):
    """The full aligned-state fixture: publication, catalog, measured module, review stub."""
    inputs = _inputs(tmp_path, contract_rows=[{"input": {"measured_tasks": 10}}])
    monkeypatch.setattr(cl, "backend_schema_sha256", lambda: MEASURED_SCHEMA)

    source_file = tmp_path / "task_models.py"
    source_file.write_text("# measured module\n")
    inputs = dataclasses.replace(
        inputs,
        measured_module=str(source_file),
        measured_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )

    class FakeSnapshot:
        snapshot_id = "assembly-test-v1"

    class FakeCatalog:
        def __init__(self, root):
            self.root = root

        def current(self):
            return FakeSnapshot()

        def sources(self, snapshot_id, *, pending_stage_of=None):
            assert pending_stage_of is not None  # pre-stage Pending rows need the computed stage
            return [{"id": "s1"}, {"id": "s2"}]

    monkeypatch.setattr(cl, "CurationCatalog", FakeCatalog)
    monkeypatch.setattr(
        cl,
        "canonical_snapshot_source",
        lambda row: type(
            "S",
            (),
            {
                "id": row["id"],
                "declaration": type("D", (), {"model_dump": lambda self, mode=None: {}})(),
            },
        )(),
    )
    reviewed = {}
    monkeypatch.setattr(
        cl,
        "review_admissions",
        lambda declarations, survey, available: reviewed.update(
            {source_id: sorted(available) for source_id in declarations}
        )
        or None,
    )
    # The survey model itself is exercised in test_release_admission_check; this
    # test pins the load-level wiring.
    monkeypatch.setattr(cl, "canonical_backend_survey", lambda document: BackendSurvey.model_construct(sources=[]))

    backend_path = tmp_path / "backend.json"
    backend_path.write_text(json.dumps({"sources": [{"source_id": "s1"}, {"source_id": "s2"}]}))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps({"fingerprints": {"s1": {}, "s2": {}}}))
    publication = {
        "backend": {"uri": str(backend_path), "sha256": hashlib.sha256(backend_path.read_bytes()).hexdigest()},
        "identity": {"uri": str(identity_path), "sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest()},
    }
    publication_path = tmp_path / "publication.json"
    publication_path.write_text(json.dumps(publication))
    inputs = dataclasses.replace(
        inputs,
        publication=ArtifactRef(
            uri=str(publication_path), sha256=hashlib.sha256(publication_path.read_bytes()).hexdigest()
        ),
    )
    return inputs, reviewed


def test_dry_run_passes_with_aligned_state(tmp_path, monkeypatch):
    inputs, reviewed = _aligned_state(tmp_path, monkeypatch)
    proof = cl.run_dry_run(inputs)
    assert proof["sql_writes"] == 0
    assert proof["target_load_id"] == TARGET
    assert proof["contracts"] == 1
    # The dry run sweeps every source through the admission review (coordinator
    # design input): the real sweep is behavior-tested in test_release_admission_check.
    assert sorted(reviewed) == ["s1", "s2"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_cli_verifies_json_difficulty_snapshot_before_loading(tmp_path, monkeypatch, capsys, corrupt):
    inputs, _ = _aligned_state(tmp_path, monkeypatch)
    ledger = str(tmp_path / "ledger")
    append_rows(
        ledger,
        [
            LedgerRow(
                "population",
                "s1",
                "a" * 64,
                Solver.SMALL,
                "attempt",
                Outcome.SCORED,
                None,
                "run",
                0.0,
                "2026-11-27T11:59:00+00:00",
                task_id="task",
                reward=1.0,
            )
        ],
    )
    source = CatalogSource("s1", "population", (("task", "a" * 64),), "queue", "digest")
    reference = seal_ledger(ledger, [source], str(tmp_path / "sealed"), {}, "2026-11-27T12:00:00+00:00")
    if corrupt:
        (StoragePath(reference.uri).parent / "cells.jsonl").write_bytes(b"changed")
    document = dataclasses.asdict(inputs)
    document["publication"] = inputs.publication.model_dump(mode="json")
    document["staging"] = {name: ref.model_dump(mode="json") for name, ref in inputs.staging.items()}
    document["difficulty_snapshot"] = reference.model_dump(mode="json")
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["catalog_load", "--inputs", str(path), "--mode", "dry-run"])
    if corrupt:
        with pytest.raises(ValueError, match="Sealed snapshot file differs"):
            cl.main()
        return
    cl.main()
    proof = json.loads(capsys.readouterr().out.split("CATALOG-LOAD-DRY-RUN ", 1)[1])
    assert proof["sql_writes"] == 0 and proof["source_count"] == 2
    assert proof["target_load_id"] == TARGET


def test_dry_run_refuses_a_blocked_admission(tmp_path, monkeypatch):
    """A blocked source fails the dry run — never the cluster job (2026-11-23)."""
    inputs, _ = _aligned_state(tmp_path, monkeypatch)

    def blocked_review(declarations, survey, available):
        raise ValueError("Admission blocked for 1 source(s): s2: no surveyed production backend can run the source")

    monkeypatch.setattr(cl, "review_admissions", blocked_review)
    with pytest.raises(ValueError, match="Admission blocked"):
        cl.run_dry_run(inputs)


def test_dry_run_refuses_a_backend_blocker_without_a_mapped_reason(tmp_path, monkeypatch):
    """R41 (R40 item 47): the load fails closed on blockers the taxonomy cannot label.

    The Backend evidence page refuses to render an unmapped blocker, so a survey
    that drifted from the taxonomy must refuse the load — not crash the page the
    way R40's unsupported-environment-kind blockers did at the final browser sweep.
    """
    inputs, _ = _aligned_state(tmp_path, monkeypatch)
    unmapped = BackendSourceMap.model_construct(
        source_id="s1",
        basis="converted_spec_census",
        task_count=2,
        inputs={},
        demands={},
        resource_ranges={},
        backends={
            "local": BackendFit.model_construct(
                support=FitSupport.UNSUPPORTED,
                task_counts={FitSupport.UNSUPPORTED: 2},
                roles={"agent": {FitSupport.UNSUPPORTED: 2}, "grader": {FitSupport.ROLE_ABSENT: 2}},
                blockers={"agent:some_mystery_demand": 2},
                not_applicable={},
                blocker_reasons={},
                validation_errors={},
            )
        },
        role_backend_assignments={},
        grading_environment=None,
    )
    monkeypatch.setattr(
        cl, "canonical_backend_survey", lambda document: BackendSurvey.model_construct(sources=[unmapped])
    )
    with pytest.raises(ValueError, match="no mapped reason"):
        cl.run_dry_run(inputs)


def test_verify_history_receipt_refuses_wrong_load(tmp_path):
    history = {
        "load_id": "d" * 32,
        "prior_load_id": PRIOR,
        "selected_pointer_changed": False,
        "prior_assessment_headers": 5,
    }
    inputs = _inputs(tmp_path, staging={"history_receipt": _write_json(tmp_path, "h.json", history)})
    with pytest.raises(ValueError, match="history receipt targets"):
        cl.verify_history_receipt(inputs)


def test_history_receipt_refuses_changed_pointer(tmp_path):
    history = {
        "load_id": TARGET,
        "prior_load_id": PRIOR,
        "selected_pointer_changed": True,
        "prior_assessment_headers": 5,
    }
    inputs = _inputs(tmp_path, staging={"history_receipt": _write_json(tmp_path, "h.json", history)})
    with pytest.raises(ValueError, match="changed selection pointer"):
        cl.verify_history_receipt(inputs)


def test_write_receipt_refuses_overwrite(tmp_path):
    inputs = _inputs(tmp_path)
    receipt_dir = tmp_path / "releases/rtest/catalog-load"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "receipt.json").write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        cl._write_receipt(inputs, {"kind": "catalog-load"})


def test_failed_run_emits_alert(tmp_path):
    feed = tmp_path / "alerts.jsonl"
    inputs = _inputs(tmp_path, staging={})
    with pytest.raises(ValueError, match="staging refs must include"):
        cl.run_catalog_load(inputs, dry_run=True, feeds=(feed,))
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FAIL"]


def test_fresh_qc_panels_replay_by_scope_and_report_pin_currency(tmp_path):
    """SAMPLE panels hold their drawn sample, FULL panels every retained task; a panel at an
    older pin replays and reports its currency."""

    def staged_panel(name: str, scope: str, records: int, *, sample_size=None, retained=None, current_pin=True):
        document = {
            "items": [{}] * records,
            "raw": [{}] * records,
            "assessment": {"scope": scope},
            "receipt": {"current_pin": current_pin, "panel": {"sample_size": sample_size, "retained_tasks": retained}},
        }
        raw = gzip.compress(json.dumps(document).encode())
        path = tmp_path / f"{name}.json.gz"
        path.write_bytes(raw)
        return {
            "source_id": f"Test:{name}",
            "status": "complete",
            "records": records,
            "reference": {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        }

    inputs = _inputs(tmp_path)
    loader = lambda staged, load_id, source_id, query: {"items": len(staged["items"])}  # noqa: E731
    n = cl.QC_PANEL_RECORDS
    sample = cl._stage_qc_panel(
        inputs, staged_panel("sample", "sample", n, sample_size=n, current_pin=False), loader, None
    )
    assert sample["current_pin"] is False and sample["counts"] == {"items": n}
    small = staged_panel("small", "sample", 91, sample_size=91, retained=91)
    assert cl._stage_qc_panel(inputs, small, loader, None)["counts"] == {"items": 91}
    full = staged_panel("full", "full", 2 * n, retained=2 * n)
    assert cl._stage_qc_panel(inputs, full, loader, None)["counts"] == {"items": 2 * n}
    with pytest.raises(ValueError, match="is not a complete panel"):
        cl._stage_qc_panel(inputs, staged_panel("short", "sample", n - 1, sample_size=n), loader, None)


def test_orphaned_review_panels_of_removed_sources_are_dropped():
    """History staging carries the prior load's panels; a source outside the load loses its panel."""
    calls = []

    def query(sql, parameters):
        calls.append(sql.split()[0])
        return (
            {"rows": [{"source_id": "Test:removed-b"}, {"source_id": "Test:removed-a"}]}
            if sql.startswith("SELECT")
            else {}
        )

    assert cl._drop_orphaned_review_panels(query, "load") == ["Test:removed-a", "Test:removed-b"]
    assert calls == ["SELECT", "DELETE"]
    calls.clear()
    assert (
        cl._drop_orphaned_review_panels(lambda sql, parameters: calls.append(sql.split()[0]) or {"rows": []}, "load")
        == []
    )
    assert calls == ["SELECT"]


# --- QC replay: progress total and the segmented parallel fan-out -----------------
class _ReplayQuery:
    """Serves the replay's SQL shapes from in-memory tables and records every statement."""

    def __init__(self, declarations: dict):
        self.declarations = declarations
        self.tables: dict[str, list[dict]] = {}
        self.statements: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, sql: str, parameters: dict):
        with self._lock:
            return self._serve(" ".join(sql.split()), parameters)

    def _serve(self, text: str, parameters: dict):
        upper = text.upper()
        self.statements.append((text, dict(parameters)))
        if "FROM ATLAS_SOURCES" in upper:
            return {"rows": [{"declaration": self.declarations[parameters["source"]]}]}
        if upper.startswith(("CREATE TABLE", "ALTER TABLE", "UPDATE")):
            return {"rows": [], "row_count": 0}
        if "AS MATCHED" in upper:
            table = text.split(" JOIN ")[1].split(" ")[0]
            columns = [column.strip().split()[0] for column in text.split(" AS expected(")[1].split(")")[0].split(",")]
            expected = json.loads(parameters["rows"])
            matched = 0
            for row in expected:
                projected = {column: row.get(column) for column in columns}
                if any(
                    all(stored.get(column) == value for column, value in projected.items())
                    and all(stored.get(column) == row.get(column) for column in ("pipeline_key",))
                    for stored in self.tables[table]
                ):
                    matched += 1
            return {"rows": [{"matched": matched}]}
        if upper.startswith("INSERT INTO ATLAS_REVIEW_CURRENT"):
            self.tables.setdefault("atlas_review_current", []).append(
                {
                    "load_id": parameters["load"],
                    "source_id": parameters["source"],
                    "pipeline_key": parameters["key"],
                    "receipt": json.loads(parameters["receipt"]),
                }
            )
            return {"rows": []}
        if "SELECT PIPELINE_KEY,RECEIPT FROM ATLAS_REVIEW_CURRENT" in upper:
            rows = [
                {"pipeline_key": row["pipeline_key"], "receipt": row["receipt"]}
                for row in self.tables["atlas_review_current"]
                if row["load_id"] == parameters["load"] and row["source_id"] == parameters["source"]
            ]
            return {"rows": rows}
        if "SELECT SOURCE_ID,PIPELINE_KEY,RECEIPT FROM ATLAS_REVIEW_CURRENT" in upper:
            source_ids = json.loads(parameters["sources"])
            return {
                "rows": [
                    {name: row[name] for name in ("source_id", "pipeline_key", "receipt")}
                    for row in self.tables["atlas_review_current"]
                    if row["load_id"] == parameters["load"] and row["source_id"] in source_ids
                ]
            }
        if upper.startswith("INSERT INTO ATLAS_") and "JSONB_TO_RECORDSET" in upper:
            table = text.split("INTO ")[1].split(" (")[0].strip()
            columns = [column.strip().split()[0] for column in text.split(" AS r(")[1].split(")")[0].split(",")]
            rows = json.loads(parameters["rows"])
            extra = ("task_id", "batch_key", "chunk_index", "cohort")
            identities = ("pipeline_key", *(column for column in extra if column in columns))
            stored = self.tables.setdefault(table, [])
            inserted = 0
            for row in rows:
                projected = {"load_id": parameters["load"], **{column: row.get(column) for column in columns}}
                key = tuple(projected.get(column) for column in ("load_id", *identities))
                if any(tuple(existing.get(column) for column in ("load_id", *identities)) == key for existing in stored):
                    continue
                stored.append(projected)
                inserted += 1
            return {"rows": [], "row_count": inserted}
        raise AssertionError(f"unhandled replay SQL in fake: {upper[:160]}")


def _staged_panel_document(source_id: str, *, records: int, batches: int = 1, chunk: bool = False) -> dict:
    """A staged panel the loader accepts: consistent ids, shared batches, exact scope."""
    sha = hashlib.sha256(source_id.encode()).hexdigest()
    key = "qc:" + sha
    artifact = {"uri": f"s3://bucket/{source_id}/.artifact.json", "sha256": sha}
    payload = "x" * 1_200_000 if chunk else "x" * 10
    items = [
        {"pipeline_key": key, "task_id": f"{source_id}#t{index}", "task": {"id": f"{source_id}#t{index}"}}
        for index in range(records)
    ]
    raw = [
        {"pipeline_key": key, "task_id": item["task_id"], "record": {"verdict": "pass"}, "reviewer": "r"}
        for item in items
    ]
    batch_rows = [
        {"pipeline_key": key, "batch_key": f"{source_id}#b{index}", "record": {"payload": payload, "note": "batch"}}
        for index in range(batches)
    ]
    links = [
        {"pipeline_key": key, "task_id": item["task_id"], "batch_key": batch_rows[0]["batch_key"]} for item in items
    ]
    return {
        "data_restriction": "open",
        "assessment": {
            "pipeline_key": key,
            "source_id": source_id,
            "scope": "sample",
            "load_id": "prior",
            "records": records,
        },
        "populations": [{"pipeline_key": key, "cohort": "qc", "population": [], "sample_count": records}],
        "items": items,
        "raw": raw,
        "batches": batch_rows,
        "batch_links": links,
        "receipt": {
            "artifact": artifact,
            "panel": {"sample_size": records, "retained_tasks": records},
            "current_pin": True,
            "mapping": None,
            "mapping_counts": None,
            "raw_messages_missing": 0,
            "linked_tasks": len(links),
        },
    }


def _declaration_for(source_id: str, sha: str) -> dict:
    return {
        "id": source_id,
        "data_restriction": "open",
        "pipeline": {"version": "converter-1"},
        "independent_qc": {"evidence_url": f"s3://bucket/{source_id}/.artifact.json"},
        "actor_census_input": {"task_specs_sha256": sha},
    }


def _replay_fixture(records: int, batches: int = 1, chunk: bool = False):
    """A staged document plus the declaration its loader checks against."""
    document = _staged_panel_document("Replay:fixture", records=records, batches=batches, chunk=chunk)
    sha = document["receipt"]["artifact"]["sha256"]
    return document, _declaration_for("Replay:fixture", sha)


def test_load_content_reviews_preserves_its_statement_sequence(monkeypatch):
    """The refactor keeps the loader's exact SQL order: checks, DDL, insert+verify pairs, pointer last."""
    document, declaration = _replay_fixture(records=3, batches=1, chunk=True)
    fake = _ReplayQuery({"Replay:fixture": declaration})
    seen_summaries: list[str] = []
    monkeypatch.setattr(
        load_reviews_module,
        "materialize_cohort_summaries",
        lambda extract, load_id, *, pipeline_key: seen_summaries.append(pipeline_key) or 0,
    )
    counts = load_reviews_module.load_content_reviews(document, "target", "Replay:fixture", fake)
    kinds = [text.split("(")[0].split()[0] for text, _ in fake.statements]
    assert kinds[0] == "SELECT"  # the declaration check runs first
    assert kinds.count("CREATE") == 5
    assert kinds[6] == "ALTER" and kinds[7] == "UPDATE"
    inserts = [
        index
        for index, (text, _) in enumerate(fake.statements)
        if text.upper().startswith("INSERT INTO ATLAS_") and "REVIEW_CURRENT" not in text.upper()
    ]
    verifies = [index for index, (text, _) in enumerate(fake.statements) if "AS MATCHED" in text.upper()]
    assert len(inserts) == len(verifies)
    assert [insert + 1 for insert in inserts] == verifies  # every table body verifies before the next statement
    pointer = next(
        index
        for index, (text, _) in enumerate(fake.statements)
        if text.upper().startswith("INSERT INTO ATLAS_REVIEW_CURRENT")
    )
    assert seen_summaries == [document["assessment"]["pipeline_key"]]
    assert pointer > verifies[-1]
    assert counts["atlas_raw_reviews"] == 3 and counts["atlas_review_batch_chunks"] == 3


def test_replay_total_matches_the_loader_row_groups():
    document, _ = _replay_fixture(records=5, batches=2)
    total = cl._replay_record_total([document])
    expected = sum(len(rows) for _table, rows, _columns in load_reviews_module.panel_row_groups(document))
    assert total == expected == 5 + 5 + 5 + 2 + 0 + 1 + 1  # raw, items, links, batches, chunks, assessments, populations


def test_replay_progress_reports_done_total_rate_and_eta():
    lines: list[str] = []
    progress = cl._ReplayProgress(100, interval_s=0.0, log=lines.append)
    progress("atlas_items", 40)
    progress("atlas_items", 60)
    progress.finish()
    assert len(lines) == 3
    assert "done=40/100 (40.0%" in lines[0] and "eta" in lines[0]
    assert lines[1].startswith("PROGRESS replay done=100/100 (100.0%")
    assert lines[2].startswith("PROGRESS replay done=100/100 (100.0%")
    with pytest.raises(ValueError, match="positive"):
        cl._ReplayProgress(0)


def test_replay_segments_bound_the_largest_panel():
    rows = [{"task_id": f"t{index}"} for index in range(45_000)]
    pieces = cl._segment_rows(rows)
    assert len(pieces) == 3
    assert all(len(piece) <= cl._REPLAY_SEGMENT_ROWS for piece in pieces)
    assert [len(piece) for piece in pieces] == [cl._REPLAY_SEGMENT_ROWS, cl._REPLAY_SEGMENT_ROWS, 5_000]
    flattened = [row for piece in pieces for row in piece]
    assert flattened == rows


def test_parallel_replay_matches_serial_byte_identical(tmp_path, monkeypatch):
    """The segmented fan-out stores exactly what the serial replay stores."""
    document, declaration = _replay_fixture(records=25, batches=1, chunk=True)
    inputs = _inputs(tmp_path)
    monkeypatch.setattr(
        load_reviews_module,
        "materialize_cohort_summaries",
        lambda extract, load_id, *, pipeline_key: 0,
    )

    def run(parallel: bool):
        fake = _ReplayQuery({"Replay:fixture": declaration})
        total = cl._replay_record_total([document])
        progress = cl._ReplayProgress(total, interval_s=10**9, log=lambda line: None)
        if parallel:
            results = cl._replay_qc_panels_parallel(
                inputs,
                [{"source_id": "Replay:fixture", "reference": document["receipt"]["artifact"]}],
                [document],
                total,
                progress,
                fake,
                lambda: fake,
                cl.RetryControl(),
            )
        else:
            counts = load_reviews_module.load_content_reviews(document, TARGET, "Replay:fixture", fake, progress)
            results = [{"source_id": "Replay:fixture", "counts": counts}]
        progress.finish()
        return results, fake

    serial_results, serial_fake = run(parallel=False)
    parallel_results, parallel_fake = run(parallel=True)
    assert serial_results[0]["counts"] == parallel_results[0]["counts"]
    for table, rows in serial_fake.tables.items():
        assert sorted(json.dumps(row, sort_keys=True) for row in rows) == sorted(
            json.dumps(row, sort_keys=True) for row in parallel_fake.tables.get(table, [])
        ), table
    assert set(parallel_fake.tables) == set(serial_fake.tables)


def test_parallel_replay_preserves_the_original_sql_failure(tmp_path, monkeypatch):
    document, declaration = _replay_fixture(records=3)

    class FailedAssessmentQuery(_ReplayQuery):
        def _serve(self, text, parameters):
            if text.startswith("INSERT INTO atlas_assessments"):
                raise ValueError("assessment insert refused")
            return super()._serve(text, parameters)

    query = FailedAssessmentQuery({"Replay:fixture": declaration})
    monkeypatch.setattr(load_reviews_module, "materialize_cohort_summaries", lambda *args, **kwargs: 0)
    total = cl._replay_record_total([document])
    with pytest.raises(ExceptionGroup) as failure:
        cl._replay_qc_panels_parallel(
            _inputs(tmp_path),
            [{"source_id": "Replay:fixture", "reference": document["receipt"]["artifact"]}],
            [document],
            total,
            cl._ReplayProgress(total),
            query,
            lambda: query,
            cl.RetryControl(),
        )
    primary = failure.value.subgroup(ValueError)
    assert primary is not None
    assert any(str(error) == "assessment insert refused" for error in primary.exceptions)
    assert "atlas_review_current" not in query.tables


def test_catalog_replay_commits_parent_batches_before_child_links(tmp_path, monkeypatch):
    documents = [_staged_panel_document(name, records=3) for name in ("slow", "fast")]
    declarations = {
        name: _declaration_for(name, document["receipt"]["artifact"]["sha256"])
        for name, document in zip(("slow", "fast"), documents, strict=True)
    }
    release_parent = threading.Event()
    parent_started = threading.Event()
    slow_key = documents[0]["assessment"]["pipeline_key"]
    fast_key = documents[1]["assessment"]["pipeline_key"]

    class ForeignKeyQuery(_ReplayQuery):
        def __call__(self, sql, parameters):
            if sql.startswith("INSERT INTO atlas_review_batches"):
                key = json.loads(parameters["rows"])[0]["pipeline_key"]
                if key == slow_key:
                    parent_started.set()
                    assert release_parent.wait(10), "independent panel never reached the SQL boundary"
                if key == fast_key:
                    assert parent_started.wait(10)
                    result = super().__call__(sql, parameters)
                    release_parent.set()
                    return result
            if sql.startswith("INSERT INTO atlas_review_batch_tasks"):
                rows = json.loads(parameters["rows"])
                stored = self.tables.get("atlas_review_batches", [])
                if any(
                    not any(
                        parent["pipeline_key"] == row["pipeline_key"] and parent["batch_key"] == row["batch_key"]
                        for parent in stored
                    )
                    for row in rows
                ):
                    release_parent.set()
                    raise ValueError("review batch foreign key violation")
            return super().__call__(sql, parameters)

    query = ForeignKeyQuery(declarations)
    monkeypatch.setattr(load_reviews_module, "materialize_cohort_summaries", lambda *args, **kwargs: 0)
    panels = []
    for source_id, document in zip(("slow", "fast"), documents, strict=True):
        raw = gzip.compress(json.dumps(document).encode())
        path = tmp_path / f"{source_id}.json.gz"
        path.write_bytes(raw)
        panels.append(
            {
                "source_id": source_id,
                "status": "complete",
                "records": 3,
                "reference": {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
            }
        )
    results = cl.replay_catalog_reviews(
        _inputs(tmp_path),
        panels,
        {"results": []},
        declarations,
        query,
        lambda: query,
        cl.RetryControl(),
    )
    assert {result["source_id"] for result in results} == {"slow", "fast"}
    assert len(query.tables["atlas_review_batches"]) == 2
    assert len(query.tables["atlas_review_batch_tasks"]) == 6
    assert {row["source_id"] for row in query.tables["atlas_review_current"]} == {"slow", "fast"}


@pytest.mark.parametrize("stored_change", ["unchanged", "scope", "unknown_source"])
def test_assessment_resume_accepts_only_the_sealed_partial_qc_headers(tmp_path, stored_change):
    panels = []
    headers = {}
    for source_id in ("first", "second"):
        staged = _staged_panel_document(source_id, records=3)
        raw = gzip.compress(json.dumps(staged).encode())
        path = tmp_path / f"{source_id}.json.gz"
        path.write_bytes(raw)
        panels.append(
            {
                "source_id": source_id,
                "status": "complete",
                "records": 3,
                "reference": {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
            }
        )
        names = [column.strip().split()[0] for column in cl.load_module.TABLE_COLUMNS["assessments"].split(",")]
        headers[staged["assessment"]["pipeline_key"]] = {name: staged["assessment"].get(name) for name in names}
    changed = next(iter(headers.values()))
    if stored_change == "scope":
        changed["scope"] = "full"
    if stored_change == "unknown_source":
        changed["source_id"] = "unplanned"

    def query(sql, parameters):
        if "COUNT(*) AS n" in sql:
            return {"rows": [{"n": 5}]}
        if "NOT EXISTS" in sql:
            return {"rows": [{"pipeline_key": key, "source_id": value["source_id"]} for key, value in headers.items()]}
        return {"rows": [headers[parameters["key"]]]}

    if stored_change == "unchanged":
        assert set(cl.verify_assessment_resume(_inputs(tmp_path), 3, panels, query)) == set(headers)
    else:
        with pytest.raises(ValueError, match=r"sealed QC panel|unexpected assessment"):
            cl.verify_assessment_resume(_inputs(tmp_path), 3, panels, query)


@pytest.mark.parametrize("changed_pointer", [False, True])
def test_catalog_reviews_reuse_copied_payload_and_replay_new_panel(tmp_path, monkeypatch, changed_pointer):
    copied = _staged_panel_document("retained", records=2)
    fresh = _staged_panel_document("Replay:fixture", records=3)
    declaration = _declaration_for("retained", copied["receipt"]["artifact"]["sha256"])
    fresh_declaration = _declaration_for("Replay:fixture", fresh["receipt"]["artifact"]["sha256"])
    copied["receipt"]["current_pin"] = False
    fresh["receipt"]["current_pin"] = False
    copy_key = copied["assessment"]["pipeline_key"]
    fake = _ReplayQuery({"retained": declaration, "Replay:fixture": fresh_declaration})
    monkeypatch.setattr(load_reviews_module, "materialize_cohort_summaries", lambda *args, **kwargs: 0)
    load_reviews_module.load_content_reviews(copied, TARGET, "retained", fake)
    prior_rows = {table: list(rows) for table, rows in fake.tables.items()}
    if changed_pointer:
        fake.tables["atlas_review_current"][0]["receipt"] = {"changed": True}
    copy_receipt = {
        "source_id": "retained",
        "catalog_load_id": TARGET,
        "pipeline_key": copy_key,
        "original_receipt": copied["receipt"],
        "selected_receipt": copied["receipt"],
        "presentation_pin_currency_preserved": True,
        "qa_bindings_changed": False,
        "task_or_reviewer_identities_changed": False,
        "tables": {"atlas_raw_reviews": {"rows": 2, "mismatches": 0}},
    }
    raw = gzip.compress(json.dumps(fresh).encode())
    fresh_path = tmp_path / "fresh.json.gz"
    fresh_path.write_bytes(raw)
    retained_panel = {
        "source_id": "retained",
        "status": "complete",
        "records": 2,
        "identity": hashlib.sha256(json.dumps(copied["receipt"], sort_keys=True).encode()).hexdigest(),
        "reference": {"uri": str(tmp_path / "unavailable-copied-payload.json.gz"), "sha256": "a" * 64},
    }
    fresh_panel = {
        "source_id": "Replay:fixture",
        "status": "complete",
        "records": 3,
        "reference": {"uri": str(fresh_path), "sha256": hashlib.sha256(raw).hexdigest()},
    }
    args = (
        _inputs(tmp_path),
        [retained_panel, fresh_panel],
        {"results": [copy_receipt]},
        {"retained": declaration},
        fake,
        lambda: fake,
        cl.RetryControl(),
    )
    if changed_pointer:
        with pytest.raises(ValueError, match="selection changed"):
            cl.replay_catalog_reviews(*args)
        assert fake.tables["atlas_raw_reviews"] == prior_rows["atlas_raw_reviews"]
        return
    results = cl.replay_catalog_reviews(*args)
    assert [row["source_id"] for row in results] == ["retained", "Replay:fixture"]
    assert results[0]["retained_sql_copy"] is True
    assert results[0]["review"] == copied["receipt"]
    assert fake.tables["atlas_raw_reviews"][:2] == prior_rows["atlas_raw_reviews"]
    assert len(fake.tables["atlas_raw_reviews"]) == 5
    assert {row["source_id"] for row in fake.tables["atlas_review_current"]} == {"retained", "Replay:fixture"}


def test_replay_sizing_scales_with_records():
    anchor = sizing.catalog_replay_sizing(sizing._CATALOG_REPLAY_ANCHOR_RECORDS)
    assert anchor.workers == sizing._CATALOG_REPLAY_ANCHOR_WORKERS
    assert sizing.catalog_replay_sizing(1000).workers == 2  # the floor
    assert sizing.catalog_replay_sizing(10**9).workers == 16  # the cap
    with pytest.raises(ValueError, match="positive"):
        sizing.catalog_replay_sizing(0)
