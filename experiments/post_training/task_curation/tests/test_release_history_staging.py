# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained history-staging stage (R40 gate slice).

The fakes answer the SQL shapes the stage issues; every test asserts on behavior the
r39 failure made real: wrong-target refusal, dry-run write-freedom, and coverage
verification before the receipt.
"""

import dataclasses
import json

import pytest

from experiments.post_training.task_curation.release_stages.history_staging import (
    HistoryStagingInputs,
    assert_plan_target,
    plan_writes,
    read_prior_state,
    run_history_staging,
)
from experiments.post_training.task_curation.release_stages.sizing import HistoryStagingSizing

PRIOR = "b" * 32
TARGET = "c" * 32

INPUTS = HistoryStagingInputs(
    release_id="rtest",
    prior_load_id=PRIOR,
    target_load_id=TARGET,
    snapshot_id="assembly-test-v1",
    source_revision="rev-test-1",
    catalog_root="s3://test/catalog",
    receipt_root="s3://test/releases/rtest",
    expected_prior_sources=3,
)
SIZING = HistoryStagingSizing(workers=2, item_batch_rows=64)


class FakeQuery:
    """Answers the SQL statements the stage issues, and records them."""

    def __init__(self):
        self.current_pointer = PRIOR
        self.prior_sources = [
            {"id": "s1", "assessment_key": "k1"},
            {"id": "s2", "assessment_key": "k2"},
            {"id": "s3", "assessment_key": None},
        ]
        self.headers = [{"pipeline_key": "k1"}, {"pipeline_key": "k2"}]
        self.summaries = [
            {"pipeline_key": "k1", "cohort": "c", "verdict_counts": {"pass": 3}},
            {"pipeline_key": "k2", "cohort": "c", "verdict_counts": {"pass": 2}},
        ]
        self.populations = [
            {"pipeline_key": "k1", "cohort": "c", "population": 1, "sample_count": 3},
            {"pipeline_key": "k2", "cohort": "c", "population": 1, "sample_count": 2},
        ]
        self.statements: list[tuple[str, dict]] = []
        self.load_rows: list[dict] = []

    def __call__(self, sql, parameters):
        self.statements.append((sql, parameters))
        upper = " ".join(sql.split()).upper()
        if "FROM ATLAS_CURRENT" in upper:
            return {"rows": [{"load_id": self.current_pointer}]}
        if "FROM ATLAS_SOURCES" in upper:
            return {"rows": self.prior_sources}
        if "FROM ATLAS_ASSESSMENTS" in upper:
            after = parameters.get("after")
            return {"rows": [row for row in self.headers if after is None or row["pipeline_key"] > after][:2]}
        if "FROM ATLAS_COHORT_SUMMARIES" in upper:
            return {"rows": self.summaries}
        if "FROM ATLAS_POPULATIONS" in upper:
            return {"rows": self.populations}
        if upper.startswith("INSERT INTO ATLAS_LOADS"):
            self.load_rows.append(parameters)
            return {"rows": []}
        if "FROM ATLAS_LOADS" in upper:
            inserted = [row for row in self.load_rows if row["load"] == parameters.get("load")]
            return {"rows": [{"snapshot_id": row["snapshot"]} for row in inserted]}
        raise AssertionError(f"unhandled SQL in fake: {sql[:120]}")


def _wrong_pointer_query() -> FakeQuery:
    query = FakeQuery()
    query.current_pointer = "d" * 32
    return query


def test_read_prior_state_verifies_pointer_inventory_and_totals():
    state = read_prior_state(FakeQuery(), INPUTS)
    assert state.latest_load_id == PRIOR
    assert state.source_count == 3
    assert state.assessment_key_count == 2
    assert state.item_total == 5


def test_read_prior_state_refuses_wrong_current_pointer():
    with pytest.raises(ValueError, match="wrong load"):
        read_prior_state(_wrong_pointer_query(), INPUTS)


def test_read_prior_state_refuses_source_count_mismatch():
    query = FakeQuery()
    query.prior_sources = query.prior_sources[:1]
    with pytest.raises(ValueError, match="expected 3"):
        read_prior_state(query, INPUTS)


def test_read_prior_state_refuses_population_summary_mismatch():
    query = FakeQuery()
    query.populations = [{**query.populations[0], "sample_count": 99}, query.populations[1]]
    with pytest.raises(ValueError, match="population"):
        read_prior_state(query, INPUTS)


def test_plan_and_target_check_agree():
    state = read_prior_state(FakeQuery(), INPUTS)
    plan = plan_writes(state, INPUTS, SIZING)
    assert plan.target_load_id == TARGET
    assert plan.load_row["snapshot_id"] == "assembly-test-v1"
    assert_plan_target(plan, INPUTS, state)


def test_target_check_refuses_drifted_plan():
    state = read_prior_state(FakeQuery(), INPUTS)
    plan = dataclasses.replace(plan_writes(state, INPUTS, SIZING), target_load_id="d" * 32)
    with pytest.raises(ValueError, match="manifest input"):
        assert_plan_target(plan, INPUTS, state)


def test_dry_run_executes_no_writes(tmp_path):
    query = FakeQuery()
    proof = run_history_staging(INPUTS, dry_run=True, sizing=SIZING, query=query, feeds=(tmp_path / "alerts.jsonl",))
    assert proof["dry_run"] is True
    assert proof["load_id"] == TARGET
    assert query.load_rows == []
    assert not any(sql.upper().startswith(("INSERT", "CREATE")) for sql, _ in query.statements)


def test_write_mode_stages_copies_verifies_and_writes_receipt(tmp_path):
    query = FakeQuery()
    seen: dict[tuple, int] = {}

    def retain(query_arg, factory, columns, prior, target, *, workers, item_batch_rows, stop):
        seen[(prior, target, workers, item_batch_rows)] = workers
        return {"assessments": 2}

    inputs = dataclasses.replace(INPUTS, receipt_root=str(tmp_path / "releases/rtest"))
    proof = run_history_staging(
        inputs,
        dry_run=False,
        sizing=SIZING,
        query=query,
        query_factory=lambda: query,
        retain=retain,
        feeds=(tmp_path / "alerts.jsonl",),
    )
    assert (PRIOR, TARGET, 2, 64) in seen
    assert proof["copied"] == {"assessments": 2}
    assert query.load_rows and query.load_rows[0]["load"] == TARGET
    receipt = json.loads((tmp_path / "releases/rtest/history-staging/receipt.json").read_text())
    assert receipt["load_id"] == TARGET
    assert proof["receipt"]["uri"].endswith("receipt.json")


def test_write_mode_refuses_existing_receipt(tmp_path):
    query = FakeQuery()
    inputs = dataclasses.replace(INPUTS, receipt_root=str(tmp_path / "releases/rtest"))
    receipt_dir = tmp_path / "releases/rtest/history-staging"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "receipt.json").write_text("{}\n")
    with pytest.raises(ValueError, match="already exists"):
        run_history_staging(
            inputs,
            dry_run=False,
            sizing=SIZING,
            query=query,
            query_factory=lambda: query,
            retain=lambda *args, **kwargs: {},
            feeds=(tmp_path / "alerts.jsonl",),
        )


def test_alerts_record_start_finish_and_failure(tmp_path):
    feed = tmp_path / "alerts.jsonl"
    run_history_staging(INPUTS, dry_run=True, sizing=SIZING, query=FakeQuery(), feeds=(feed,))
    with pytest.raises(ValueError, match="wrong load"):
        run_history_staging(INPUTS, dry_run=True, sizing=SIZING, query=_wrong_pointer_query(), feeds=(feed,))
    rows = [json.loads(line) for line in feed.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["STAGE_START", "STAGE_FINISH", "STAGE_START", "STAGE_FAIL"]
    assert all(row["key"] == "atlas-release:rtest:history_staging" for row in rows)
