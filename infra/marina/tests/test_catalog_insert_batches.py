# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound indexed catalog writes while preserving large evidence and resume identity."""

import json
import time

import pytest
from sqlalchemy import create_engine, text

from infra.marina.apps.rl_data_catalog.load import TABLE_COLUMNS, insert_rows
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL

COMMITTED_PREFIX_ROWS = 13


def execute_batch(engine, sql, parameters):
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL statement_timeout = '2s'"))
        result = connection.execute(text(sql), parameters)
    return {"rows": [], "row_count": result.rowcount}


@pytest.fixture
def batch_database(database_url):
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            for statement in SCHEMA_SQL.split(";"):
                if statement.strip():
                    connection.execute(text(statement))
        yield engine
    finally:
        engine.dispose()


def test_indexed_item_batches_resume_without_overwriting_evidence(batch_database, record_property):
    rows = [
        {
            "pipeline_key": "full-content:example",
            "task_id": f"task-{index:06d}",
            "cohort": "admitted",
            "status": "reviewed",
            "verdict": "some_issues",
            "explanation": "Recorded finding " * (1100 if index % 17 < 3 else 30),
            "evidence_ids": ["task"],
            "packet_uri": "s3://example/packet.json",
            "has_task": True,
            "omissions": [],
            "exclusion_reason": None,
            "defects": ["wrong_reference"],
            "confidence": "high",
            "reference_status": "conflict",
            "provenance": None,
            "replaced_task_sha256": None,
            "repair_advisories": [],
        }
        for index in range(503)
    ]
    timings = []
    batch_sizes = []
    bodies = []

    def query(sql, parameters):
        body = parameters["rows"]
        size = len(json.loads(body))
        assert size <= 5
        assert len(body.encode()) <= 35_000
        started = time.monotonic()
        result = execute_batch(batch_database, sql, parameters)
        timings.append(time.monotonic() - started)
        batch_sizes.append(size)
        bodies.append(len(body.encode()))
        return result

    # Model a resumed load with an already committed prefix. Re-staging that
    # boundary must retain the original judgment, not overwrite it with a retry.
    assert (
        insert_rows(query, "atlas_items", TABLE_COLUMNS["items"], "resumed", rows[:COMMITTED_PREFIX_ROWS])
        == COMMITTED_PREFIX_ROWS
    )
    retry = [
        {**row, "explanation": "A later retry must not replace this"} for row in rows[:COMMITTED_PREFIX_ROWS]
    ] + rows[COMMITTED_PREFIX_ROWS:]
    assert insert_rows(query, "atlas_items", TABLE_COLUMNS["items"], "resumed", retry, conflict_nothing=True) == (
        len(rows) - COMMITTED_PREFIX_ROWS
    )
    with batch_database.begin() as connection:
        stored = dict(
            connection.execute(text("SELECT task_id,explanation FROM atlas_items WHERE load_id='resumed'")).all()
        )
    assert stored == {row["task_id"]: row["explanation"] for row in rows}
    record_property("statement_count", len(timings))
    record_property("max_statement_seconds", max(timings))
    record_property("max_rows", max(batch_sizes))
    record_property("max_json_bytes", max(bodies))


def test_large_packets_stay_intact_in_single_row_statements(batch_database):
    packets = [
        {"pipeline_key": "panel", "task_id": str(index), "packet": json.dumps({"evidence": "x" * 120_000})}
        for index in range(3)
    ]

    def query(sql, parameters):
        assert len(json.loads(parameters["rows"])) == 1
        return execute_batch(batch_database, sql, parameters)

    assert insert_rows(query, "atlas_packets", TABLE_COLUMNS["packets"], "large", packets) == 3
    with batch_database.begin() as connection:
        stored = dict(connection.execute(text("SELECT task_id,packet FROM atlas_packets WHERE load_id='large'")).all())
    assert stored == {row["task_id"]: row["packet"] for row in packets}
