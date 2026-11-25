# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained QC-browse release stage.

The fake query serves the SQL shapes the stage issues against in-memory tables and
records every statement, so tests can assert the zero-write dry run, the
copy-then-verify sequence with pointer publication last, and resume without recopying.
"""

import dataclasses
import hashlib
import json
import re
import sys
import threading

import pytest

from experiments.post_training.task_curation.release_stages import sizing
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.qc_browse import (
    COPY_TABLES,
    PLAN_PAGE_ROWS,
    QcBrowseInputs,
    declaration_binding,
    main,
    pin_currency,
    plan_qc_browse,
    run_qc_browse_copy,
)

PRIOR = "b" * 32
TARGET = "c" * 32
FEED_NAME = "alerts.jsonl"


def make_declaration(source_id, *, revision="rev-1", withheld=False):
    declaration = {
        "id": source_id,
        "dataset_id": "dataset-" + source_id,
        "dataset_revision": revision,
        "pipeline": {"version": "converter-1"},
        "actor_census_input": {"task_specs_sha256": "specs-" + source_id, "measured_tasks": 5},
        "independent_qc": {"evidence_url": f"s3://bucket/artifacts/{source_id}/.artifact.json"},
    }
    if withheld:
        declaration["data_restriction"] = "private_always"
    return declaration


def make_panel(declaration, *, current_pin=True):
    uri = declaration["independent_qc"]["evidence_url"]
    sha = hashlib.sha256(uri.encode()).hexdigest()
    return {
        "source_id": declaration["id"],
        "pipeline_key": "qc:" + sha,
        "receipt": {
            "artifact": {"uri": uri, "sha256": sha},
            "panel": {
                "dataset_id": declaration["dataset_id"],
                "dataset_revision": declaration["dataset_revision"],
                "converter_version": declaration["pipeline"]["version"],
                "task_specs_sha256": declaration["actor_census_input"]["task_specs_sha256"],
                "retained_tasks": declaration["actor_census_input"]["measured_tasks"],
            },
            "current_pin": current_pin,
        },
    }


def base_fixtures():
    """Two selected panels: one current, one stale but consistently flagged as stale."""
    declarations = {f"source-{index}": make_declaration(f"source-{index}") for index in (1, 2)}
    panels = [make_panel(declarations["source-1"])]
    stale = make_panel(declarations["source-2"])
    declarations["source-2"]["dataset_revision"] = "rev-2"
    stale["receipt"]["current_pin"] = pin_currency(declarations["source-2"], stale["receipt"])
    panels.append(stale)
    return panels, declarations


def seal_declarations(tmp_path, declarations) -> ArtifactRef:
    raw = (json.dumps(declarations, sort_keys=True, indent=2) + "\n").encode()
    path = tmp_path / "declarations.json"
    path.write_bytes(raw)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


_PAGE_BYTES_OF = {
    "atlas_review_batches": lambda row: len(json.dumps(row["record"])),
    "atlas_review_batch_chunks": lambda row: len(row["payload"]),
    "atlas_review_batch_tasks": lambda row: 128,
    "atlas_raw_reviews": lambda row: len(json.dumps(row["record"])) + len(row["reviewer"] or ""),
}


class FakeQuery:
    """Serves the QC-browse SQL shapes from in-memory state and records every statement.

    One statement at a time under a lock: the real paced client serves one statement
    per call, and the parallel copy runs several clients against shared in-memory
    tables.
    """

    def __init__(self, panels, declarations, *, prior=PRIOR, target=TARGET):
        self.prior = prior
        self.target = target
        self.current_pointer = prior
        self.declarations = dict(declarations)
        self.source_bindings = {sid: declaration_binding(decl) for sid, decl in declarations.items()}
        self.review_current = {
            prior: {panel["source_id"]: (panel["pipeline_key"], panel["receipt"]) for panel in panels},
            target: {},
        }
        self.tables = {table.name: [] for table in COPY_TABLES}
        self.statements: list[tuple[str, dict]] = []
        self._lock = threading.Lock()

    def __call__(self, sql, parameters):
        with self._lock:
            return self._serve(sql, parameters)

    def _serve(self, sql, parameters):
        self.statements.append((sql, parameters))
        upper = " ".join(sql.split()).upper()
        if "FROM ATLAS_CURRENT" in upper:
            return {"rows": [{"load_id": self.current_pointer}]}
        if upper.startswith("INSERT INTO ATLAS_REVIEW_CURRENT"):
            self._publish_pointer(parameters)
            return {"rows": []}
        if "FROM ATLAS_REVIEW_CURRENT" in upper:
            return {"rows": self._review_current_rows(upper, parameters)}
        if "FROM ATLAS_SOURCES" in upper:
            binding = self.source_bindings.get(parameters["source"])
            return {"rows": [] if binding is None else [{"binding": binding}]}
        if upper.startswith("INSERT INTO "):
            self._insert_copy(upper, parameters)
            return {"rows": []}
        if "AS BYTES" in upper:
            return {"rows": self._page_rows(upper, parameters)}
        if "BOUNDED_CHECK" in upper:
            return {"rows": [self._bounded_check(upper, parameters)]}
        if "AS OLD_ROWS" in upper:
            return {"rows": [self._counts(upper, parameters)]}
        raise AssertionError(f"unhandled SQL in fake: {sql[:160]}")

    def _table(self, upper):
        for table in COPY_TABLES:
            if table.name.upper() in upper:
                return table
        raise AssertionError(f"no copy table in SQL: {upper[:160]}")

    def _publish_pointer(self, parameters):
        bucket = self.review_current.setdefault(parameters["new"], {})
        if parameters["source"] not in bucket:  # ON CONFLICT DO NOTHING
            bucket[parameters["source"]] = (parameters["key"], json.loads(parameters["receipt"]))

    def _review_current_rows(self, upper, parameters):
        bucket = self.review_current.get(parameters["load"], {})
        if ":AFTER" in upper:  # plan enumeration page
            after = parameters.get("after")
            ordered = sorted(bucket)
            return [
                {"source_id": sid, "pipeline_key": bucket[sid][0], "receipt": bucket[sid][1]}
                for sid in ordered
                if after is None or sid > after
            ][:PLAN_PAGE_ROWS]
        entry = bucket.get(parameters["source"])
        return [] if entry is None else [{"pipeline_key": entry[0], "receipt": entry[1]}]

    def _in_tuples(self, table, parameters):
        tuples = []
        index = 0
        while f"id_{index}_{table.keys[0]}" in parameters:
            tuples.append(tuple(parameters[f"id_{index}_{column}"] for column in table.keys))
            index += 1
        return tuples

    def _page_rows(self, upper, parameters):
        table = self._table(upper)
        rows = [
            row
            for row in self.tables[table.name]
            if row["load_id"] == parameters["old"] and row["pipeline_key"] == parameters["key"]
        ]
        if "after_0" in parameters:
            after = tuple(parameters[f"after_{index}"] for index in range(len(table.keys)))
            rows = [row for row in rows if tuple(row[column] for column in table.keys) > after]
        rows.sort(key=lambda row: tuple(row[column] for column in table.keys))
        limit = int(re.search(r"LIMIT (\d+)", upper).group(1))
        return [
            {**{column: row[column] for column in table.keys}, "bytes": _PAGE_BYTES_OF[table.name](row)}
            for row in rows[:limit]
        ]

    def _insert_copy(self, upper, parameters):
        table = self._table(upper)
        in_tuples = self._in_tuples(table, parameters)
        existing = {
            tuple(row[column] for column in table.keys)
            for row in self.tables[table.name]
            if row["load_id"] == parameters["new"]
        }
        for row in list(self.tables[table.name]):
            if row["load_id"] != parameters["old"] or row["pipeline_key"] != parameters["key"]:
                continue
            if tuple(row[column] for column in table.keys) not in in_tuples:
                continue
            copy = {**row, "load_id": parameters["new"]}
            if tuple(copy[column] for column in table.keys) not in existing:  # ON CONFLICT DO NOTHING
                self.tables[table.name].append(copy)

    def _bounded_check(self, upper, parameters):
        table = self._table(upper)
        in_tuples = self._in_tuples(table, parameters)
        previous = [
            row
            for row in self.tables[table.name]
            if row["load_id"] == parameters["old"]
            and row["pipeline_key"] == parameters["key"]
            and tuple(row[column] for column in table.keys) in in_tuples
        ]
        current = {
            tuple(row[column] for column in table.keys): {column: row[column] for column in table.columns}
            for row in self.tables[table.name]
            if row["load_id"] == parameters["new"] and row["pipeline_key"] == parameters["key"]
        }
        mismatches = sum(
            1
            for row in previous
            if current.get(tuple(row[column] for column in table.keys))
            != {column: row[column] for column in table.columns}
        )
        return {"rows": len(previous), "mismatches": mismatches}

    def _counts(self, upper, parameters):
        table = self._table(upper)
        rows = self.tables[table.name]
        return {
            "old_rows": sum(
                1 for row in rows if row["load_id"] == parameters["old"] and row["pipeline_key"] == parameters["key"]
            ),
            "new_rows": sum(
                1 for row in rows if row["load_id"] == parameters["new"] and row["pipeline_key"] == parameters["key"]
            ),
        }


def _copy_inserts(statements):
    """The copy-table INSERT statements (every INSERT except the pointer publication)."""
    return [
        params
        for sql, params in statements
        if sql.upper().startswith("INSERT INTO ") and "ATLAS_REVIEW_CURRENT" not in sql.upper()
    ]


def seed_prior_tables(fake, panel, *, batches=2, chunks=2, tasks=2, reviews=2):
    """Populate the fake's prior-load review tables for one panel."""
    key = panel["pipeline_key"]
    fake.tables["atlas_review_batches"].extend(
        {
            "load_id": fake.prior,
            "pipeline_key": key,
            "batch_key": f"{key}#b{index}",
            "record": {"batch": index},
            "chunk_count": 2,
        }
        for index in range(batches)
    )
    fake.tables["atlas_review_batch_chunks"].extend(
        {
            "load_id": fake.prior,
            "pipeline_key": key,
            "batch_key": f"{key}#b0",
            "chunk_index": index,
            "payload": json.dumps({"chunk": index}),
        }
        for index in range(chunks)
    )
    fake.tables["atlas_review_batch_tasks"].extend(
        {"load_id": fake.prior, "pipeline_key": key, "task_id": f"{key}#t{index}", "batch_key": f"{key}#b0"}
        for index in range(tasks)
    )
    fake.tables["atlas_raw_reviews"].extend(
        {
            "load_id": fake.prior,
            "pipeline_key": key,
            "task_id": f"{key}#t{index}",
            "record": {"verdict": "pass"},
            "reviewer": "reviewer-a",
        }
        for index in range(reviews)
    )


def make_inputs(tmp_path, panels, declarations, *, expected_panels=2, exclusions=(), workers=1):
    """Seal the frozen declarations doc and build typed inputs plus a seeded fake query."""
    inputs = QcBrowseInputs(
        release_id="rtest",
        prior_load_id=PRIOR,
        target_load_id=TARGET,
        source_revision="rev-test-1",
        declarations=seal_declarations(tmp_path, declarations),
        output_root=str(tmp_path / "release"),
        expected_panels=expected_panels,
        workers=workers,
        exclusions=exclusions,
    )
    fake = FakeQuery(panels, declarations)
    for panel in panels:
        seed_prior_tables(fake, panel)
    return inputs, fake


def test_plan_dry_run_enumerates_validates_and_writes_nothing(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    proof = plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))
    plan = proof["plan"]
    assert plan["kind"] == "atlas-rtest-qc-browse-copy-plan"
    assert [panel["source_id"] for panel in plan["panels"]] == ["source-1", "source-2"]
    assert plan["source_count"] == 2 and plan["excluded_stale_panels"] == []
    assert plan["declarations_ref"] == {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256}
    assert proof["dry_run"] is True and proof["sql_writes"] == 0
    assert proof["selected_pointer_changed"] is False
    assert not (tmp_path / "release/prior-qc-browse/plan.json").exists()
    assert not any(sql.upper().startswith("INSERT") for sql, _ in fake.statements)


def test_plan_dry_run_applies_typed_exclusions(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations, expected_panels=1, exclusions=("source-2",))
    proof = plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))
    assert [panel["source_id"] for panel in proof["plan"]["panels"]] == ["source-1"]
    assert proof["plan"]["excluded_stale_panels"] == ["source-2"]


def test_plan_write_seals_once_and_rerun_is_byte_identical(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    first = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(feed,))
    assert first["release_id"] == inputs.release_id
    path = tmp_path / "release/prior-qc-browse/plan.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == first["plan_ref"]["sha256"]
    second = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(feed,))
    assert second["plan_ref"] == first["plan_ref"]
    assert path.read_bytes() == raw


def test_plan_refuses_moved_current_pointer(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    fake.current_pointer = "d" * 32
    with pytest.raises(ValueError, match="wrong load"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_plan_refuses_withheld_panel(tmp_path):
    panels, declarations = base_fixtures()
    declarations["source-1"] = make_declaration("source-1", withheld=True)
    panels[0] = make_panel(declarations["source-1"])
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    with pytest.raises(ValueError, match="withheld"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_plan_refuses_declaration_binding_drift(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    fake.source_bindings["source-1"] = {"actor_census_input": None, "evidence_url": None}
    with pytest.raises(ValueError, match="binding"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_plan_refuses_pin_currency_mismatch(tmp_path):
    panels, declarations = base_fixtures()
    panels[0]["receipt"]["current_pin"] = not pin_currency(declarations["source-1"], panels[0]["receipt"])
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    with pytest.raises(ValueError, match="currency"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_plan_refuses_panel_count_mismatch(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations, expected_panels=3)
    with pytest.raises(ValueError, match="expected 3"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_plan_refuses_missing_declaration(tmp_path):
    panels, declarations = base_fixtures()
    del declarations["source-2"]
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    with pytest.raises(ValueError, match="no frozen declaration"):
        plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(tmp_path / FEED_NAME,))


def test_copy_verifies_then_publishes_and_seals_completion(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan_proof = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(feed,))
    proof = run_qc_browse_copy(inputs, plan_proof["plan_ref"], query=fake, feeds=(feed,))
    assert proof["kind"] == "atlas-rtest-qc-browse-retention"
    assert proof["status"] == "PASS"
    assert [row["source_id"] for row in proof["results"]] == ["source-1", "source-2"]
    assert proof["qa_bindings_changed"] is False
    assert proof["task_or_reviewer_identities_changed"] is False
    assert proof["selected_pointer_changed"] is False
    assert proof["retry_policy"] == {"wait_seconds": 300, "per_statement_retries": 3, "shared_run_retries": 10}
    completion = json.loads((tmp_path / "release/prior-qc-browse/completion.json").read_text())
    assert completion["status"] == "PASS" and completion["target_load_id"] == TARGET
    for panel in plan_proof["plan"]["panels"]:
        digest = hashlib.sha256(panel["source_id"].encode()).hexdigest()
        receipt = json.loads((tmp_path / f"release/prior-qc-browse/panels/{digest}.json").read_text())
        assert receipt["catalog_load_id"] == TARGET
        assert receipt["tables"]["atlas_raw_reviews"]["rows"] == 2
    for table in COPY_TABLES:
        assert any(row["load_id"] == TARGET for row in fake.tables[table.name])
    pointer_at = {}
    last_verify = {}
    for index, (sql, params) in enumerate(fake.statements):
        upper = " ".join(sql.split()).upper()
        if upper.startswith("INSERT INTO ATLAS_REVIEW_CURRENT"):
            pointer_at[params["source"]] = index
        if "BOUNDED_CHECK" in upper:
            last_verify[params["key"]] = index
    for panel in plan_proof["plan"]["panels"]:
        assert pointer_at[panel["source_id"]] > last_verify[panel["pipeline_key"]]
    table_order = []
    for sql, _ in fake.statements:
        upper = " ".join(sql.split()).upper()
        if "AS BYTES" in upper:
            for table in COPY_TABLES:
                if table.name.upper() in upper and table.name not in table_order:
                    table_order.append(table.name)
    assert table_order == [table.name for table in COPY_TABLES]


def test_copy_resumes_from_panel_receipts_without_recopying(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan_proof = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(feed,))
    run_qc_browse_copy(inputs, plan_proof["plan_ref"], query=fake, feeds=(feed,))
    assert len(_copy_inserts(fake.statements)) > 0
    # Simulate the resumed run: completion not yet present, every panel already staged.
    (tmp_path / "release/prior-qc-browse/completion.json").unlink()
    before = len(fake.statements)
    second = run_qc_browse_copy(inputs, plan_proof["plan_ref"], query=fake, feeds=(feed,))
    assert second["status"] == "PASS" and second["source_count"] == 2
    resumed = fake.statements[before:]
    assert not any(sql.upper().startswith("INSERT") for sql, _ in resumed)
    pointer_checks = sum(
        1 for sql, params in resumed if "FROM ATLAS_REVIEW_CURRENT" in sql.upper() and params.get("load") == TARGET
    )
    assert pointer_checks == len(plan_proof["plan"]["panels"])


def test_copy_resumes_one_panel_and_copies_the_rest_from_plan_document(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan = plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(feed,))["plan"]
    staged, fresh = plan["panels"]
    # The first panel is already staged by an interrupted run.
    fake.review_current[TARGET][staged["source_id"]] = (staged["pipeline_key"], staged["receipt"])
    receipt_path = (
        tmp_path
        / "release/prior-qc-browse/panels"
        / (hashlib.sha256(staged["source_id"].encode()).hexdigest() + ".json")
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "source_id": staged["source_id"],
                "pipeline_key": staged["pipeline_key"],
                "original_receipt": staged["receipt"],
                "selected_receipt": staged["receipt"],
                "presentation_pin_currency_preserved": True,
                "tables": {},
                "qa_bindings_changed": False,
                "task_or_reviewer_identities_changed": False,
                "catalog_load_id": TARGET,
                "source_revision": "rev-test-1",
                "completed_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )
    proof = run_qc_browse_copy(inputs, plan, query=fake, feeds=(feed,))
    assert proof["status"] == "PASS"
    copies: dict[str, int] = {}
    for params in _copy_inserts(fake.statements):
        copies[params["source"]] = copies.get(params["source"], 0) + 1
    assert staged["source_id"] not in copies
    assert copies[fresh["source_id"]] > 0


def test_copy_refuses_tampered_plan_digest(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan_ref = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(feed,))["plan_ref"]
    with pytest.raises(ValueError, match="digest mismatch"):
        run_qc_browse_copy(inputs, {"uri": plan_ref["uri"], "sha256": "0" * 64}, query=fake, feeds=(feed,))


class PointerMovesMidRun:
    """Wraps the fake and flips the served selection after the first pointer check."""

    def __init__(self, inner):
        self.inner = inner
        self.pointer_checks = 0

    def __call__(self, sql, parameters):
        upper = " ".join(sql.split()).upper()
        if "FROM ATLAS_CURRENT" in upper:
            self.pointer_checks += 1
            if self.pointer_checks > 1:
                self.inner.current_pointer = self.inner.target
        return self.inner(sql, parameters)


def test_copy_refuses_when_selection_moves_mid_run(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan = plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(feed,))["plan"]
    with pytest.raises(ValueError, match="after every panel"):
        run_qc_browse_copy(inputs, plan, query=PointerMovesMidRun(fake), feeds=(feed,))
    assert not (tmp_path / "release/prior-qc-browse/completion.json").exists()


def test_copy_refuses_dry_run_mode(tmp_path, monkeypatch):
    panels, declarations = base_fixtures()
    inputs, _ = make_inputs(tmp_path, panels, declarations)
    document = dataclasses.asdict(inputs)
    document["declarations"] = {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256}
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps(document))
    monkeypatch.setattr(sys, "argv", ["qc-browse", "--inputs", str(inputs_path), "--step", "copy", "--mode", "dry-run"])
    with pytest.raises(ValueError, match="no dry-run mode"):
        main()


def test_alerts_record_start_finish_and_failure(tmp_path):
    panels, declarations = base_fixtures()
    inputs, fake = make_inputs(tmp_path, panels, declarations)
    feed = tmp_path / FEED_NAME
    plan_proof = plan_qc_browse(inputs, query=fake, dry_run=True, feeds=(feed,))
    run_qc_browse_copy(inputs, plan_proof["plan"], query=fake, feeds=(feed,))
    bad_plan = {**plan_proof["plan"], "kind": "atlas-other-qc-browse-copy-plan"}
    with pytest.raises(ValueError, match="does not bind release"):
        run_qc_browse_copy(inputs, bad_plan, query=fake, feeds=(feed,))
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [(row["key"], row["kind"]) for row in rows] == [
        ("atlas-release:rtest:qc_browse_plan", "STAGE_START"),
        ("atlas-release:rtest:qc_browse_plan", "STAGE_FINISH"),
        ("atlas-release:rtest:qc_browse", "STAGE_START"),
        ("atlas-release:rtest:qc_browse", "STAGE_FINISH"),
        ("atlas-release:rtest:qc_browse", "STAGE_START"),
        ("atlas-release:rtest:qc_browse", "STAGE_FAIL"),
    ]


def _six_panels():
    """Six independent panels so a three-worker copy really runs concurrently."""
    declarations = {f"source-{index}": make_declaration(f"source-{index}") for index in range(1, 7)}
    panels = [make_panel(declarations[f"source-{index}"]) for index in range(1, 7)]
    return panels, declarations


def _copy_without_timestamps(proof: dict) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "completed_at"} for row in proof["results"]]


def test_copy_fans_out_across_workers_with_identical_results(tmp_path):
    """The worker fan-out copies the same panels as a serial run, in declared order."""
    panels, declarations = _six_panels()

    def inputs(root, workers):
        typed, fake = make_inputs(tmp_path, panels, declarations, expected_panels=6, workers=workers)
        return dataclasses.replace(typed, output_root=str(root / "release")), fake

    parallel_root, serial_root = tmp_path / "parallel", tmp_path / "serial"
    parallel_inputs, parallel_fake = inputs(parallel_root, workers=3)
    serial_inputs, serial_fake = inputs(serial_root, workers=1)
    parallel_plan = plan_qc_browse(parallel_inputs, query=parallel_fake, dry_run=False, feeds=(tmp_path / FEED_NAME,))
    serial_plan = plan_qc_browse(serial_inputs, query=serial_fake, dry_run=False, feeds=(tmp_path / FEED_NAME,))

    parallel = run_qc_browse_copy(
        parallel_inputs, parallel_plan["plan_ref"], query_factory=lambda: parallel_fake, feeds=(tmp_path / FEED_NAME,)
    )
    serial = run_qc_browse_copy(
        serial_inputs, serial_plan["plan_ref"], query_factory=lambda: serial_fake, feeds=(tmp_path / FEED_NAME,)
    )
    assert [row["source_id"] for row in parallel["results"]] == [row["source_id"] for row in serial["results"]]
    assert _copy_without_timestamps(parallel) == _copy_without_timestamps(serial)
    for field in ("status", "source_count", "qa_bindings_changed", "selected_pointer_changed"):
        assert parallel[field] == serial[field], field
    # Every panel sealed its own receipt at the source-keyed path, exactly once.
    for panel in parallel_plan["plan"]["panels"]:
        digest = hashlib.sha256(panel["source_id"].encode()).hexdigest()
        receipt = json.loads((parallel_root / f"release/prior-qc-browse/panels/{digest}.json").read_text())
        assert receipt["tables"]["atlas_raw_reviews"]["rows"] == 2
    for table in COPY_TABLES:
        assert len([row for row in parallel_fake.tables[table.name] if row["load_id"] == TARGET]) == len(
            [row for row in serial_fake.tables[table.name] if row["load_id"] == TARGET]
        )


def test_copy_refuses_parallel_workers_without_a_factory(tmp_path):
    panels, declarations = _six_panels()
    inputs, fake = make_inputs(tmp_path, panels, declarations, expected_panels=6, workers=3)
    plan_proof = plan_qc_browse(inputs, query=fake, dry_run=False, feeds=(tmp_path / FEED_NAME,))
    with pytest.raises(ValueError, match="query_factory"):
        run_qc_browse_copy(inputs, plan_proof["plan_ref"], query=fake, feeds=(tmp_path / FEED_NAME,))


def test_qc_browse_sizing_scales_with_panels():
    anchor = sizing.qc_browse_sizing(sizing._QC_BROWSE_ANCHOR_PANELS)
    assert anchor.workers == sizing._QC_BROWSE_ANCHOR_WORKERS
    assert sizing.qc_browse_sizing(40).workers == 2  # the floor: a stalled copy is observable
    assert sizing.qc_browse_sizing(1000).workers == 16  # the cap
    with pytest.raises(ValueError, match="positive"):
        sizing.qc_browse_sizing(0)
