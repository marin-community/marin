# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound indexed catalog writes while preserving large evidence and resume identity."""

import json
import threading
import time

import pytest
from marina.applets import validate_query
from sqlalchemy import create_engine, text

from infra.marina.apps.rl_data_catalog.historical_assessments import retain_prior_assessments
from infra.marina.apps.rl_data_catalog.load import TABLE_COLUMNS, insert_rows
from infra.marina.apps.rl_data_catalog.parallel_history import CHECKPOINT_TABLE, retain_prior_assessments_parallel
from infra.marina.apps.rl_data_catalog.server.cohort_summaries import materialize_cohort_summaries
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL

COMMITTED_PREFIX_ROWS = 13


def execute_statement(engine, sql, parameters):
    validate_query(sql)
    with engine.begin() as connection:
        connection.execute(text("SET LOCAL statement_timeout = '2s'"))
        result = connection.execute(text(sql), parameters)
        rows = [dict(row) for row in result.mappings()] if result.returns_rows else []
    return {"rows": rows, "row_count": result.rowcount}


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
        result = execute_statement(batch_database, sql, parameters)
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
        return execute_statement(batch_database, sql, parameters)

    assert insert_rows(query, "atlas_packets", TABLE_COLUMNS["packets"], "large", packets) == 3
    with batch_database.begin() as connection:
        stored = dict(connection.execute(text("SELECT task_id,packet FROM atlas_packets WHERE load_id='large'")).all())
    assert stored == {row["task_id"]: row["packet"] for row in packets}


def test_retained_history_copies_only_bounded_missing_rows(batch_database, record_property):
    items = [
        {
            "pipeline_key": "retired",
            "task_id": f"task-{index:06d}",
            "cohort": "admitted",
            "status": "reviewed",
            "verdict": "good",
            "explanation": "Original archived judgment",
            "evidence_ids": ["task"],
            "packet_uri": "s3://example/original",
            "has_task": True,
            "omissions": [],
            "exclusion_reason": None,
            "defects": [],
        }
        for index in range(503)
    ]
    packets = [
        {"pipeline_key": "retired", "task_id": item["task_id"], "packet": "archived evidence " * 7000}
        for item in items[:25]
    ]

    def seed(sql, parameters):
        return execute_statement(batch_database, sql, parameters)

    assessment = {
        "pipeline_key": "retired",
        "source_id": "source",
        "scope": "full",
        "model": {},
        "rubric": "Archived",
        "curation": {},
        "curation_uri": "s3://example/curation",
        "traces": [],
    }
    insert_rows(seed, "atlas_assessments", TABLE_COLUMNS["assessments"], "old", [assessment])
    insert_rows(
        seed,
        "atlas_populations",
        TABLE_COLUMNS["populations"],
        "old",
        [{"pipeline_key": "retired", "cohort": "admitted", "population": len(items), "sample_count": len(items)}],
    )
    with batch_database.begin() as connection:
        connection.execute(text("INSERT INTO atlas_cohort_summaries VALUES ('old','retired','admitted',1,'{}'::jsonb)"))
    insert_rows(seed, "atlas_items", TABLE_COLUMNS["items"], "old", items)
    insert_rows(seed, "atlas_packets", TABLE_COLUMNS["packets"], "old", packets)
    # Previous hash-bucket attempts can leave holes rather than a sorted prefix.
    committed = [{**item} for item in items[::3]]
    committed[0]["explanation"] = "Already staged judgment stays unchanged"
    insert_rows(seed, "atlas_items", TABLE_COLUMNS["items"], "new", committed)
    insert_rows(seed, "atlas_packets", TABLE_COLUMNS["packets"], "new", packets[::3])
    maximum = {"items": 0, "packets": 0}
    elapsed = []

    def retained_counts():
        with batch_database.begin() as connection:
            return dict(
                connection.execute(
                    text(
                        "SELECT 'items', count(*) FROM atlas_items WHERE load_id='new' "
                        "UNION ALL SELECT 'packets', count(*) FROM atlas_packets WHERE load_id='new'"
                    )
                ).all()
            )

    def query(sql, parameters):
        before = retained_counts()
        started = time.monotonic()
        result = execute_statement(batch_database, sql, parameters)
        elapsed.append(time.monotonic() - started)
        after = retained_counts()
        for table, limit in (("items", 32), ("packets", 8)):
            inserted = after[table] - before[table]
            assert inserted <= limit
            maximum[table] = max(maximum[table], inserted)
        return result

    copied = retain_prior_assessments(query, TABLE_COLUMNS, "old", "new")
    assert copied == {
        "populations": 1,
        "items": len(items) - len(committed),
        "packets": len(packets) - len(packets[::3]),
        "assessments": 1,
    }
    assert retain_prior_assessments(query, TABLE_COLUMNS, "old", "new") == dict.fromkeys(copied, 0)
    with batch_database.begin() as connection:
        restored = dict(
            connection.execute(text("SELECT task_id,explanation FROM atlas_items WHERE load_id='new'")).all()
        )
        restored_packets = dict(
            connection.execute(text("SELECT task_id,packet FROM atlas_packets WHERE load_id='new'")).all()
        )
    expected = {item["task_id"]: item["explanation"] for item in items}
    expected.update({item["task_id"]: item["explanation"] for item in committed})
    assert restored == expected
    assert restored_packets == {item["task_id"]: item["packet"] for item in packets}
    record_property("retained_max_item_rows", maximum["items"])
    record_property("retained_max_packet_rows", maximum["packets"])
    record_property("retained_max_statement_seconds", max(elapsed))


def test_parallel_ranges_resume_after_copy_commits_before_checkpoint(batch_database, record_property):
    population = 12307
    batch_rows = 1024
    with batch_database.begin() as c:
        c.execute(
            text(
                "INSERT INTO "
                "atlas_assessments(load_id,pipeline_key,source_id,scope,model,rubric,curation,curation_uri,traces) "
                "VALUES ('parallel-old','retired','source','full','{}','Archived','{}','s3://example/archived','[]')"
            )
        )
        c.execute(
            text(
                "INSERT INTO atlas_populations(load_id,pipeline_key,cohort,population,sample_count) VALUES "
                "('parallel-old','retired','admitted',:n,:n)"
            ),
            {"n": population},
        )
        c.execute(text("INSERT INTO atlas_cohort_summaries VALUES ('parallel-old','retired','admitted',1,'{}')"))
        c.execute(
            text(
                "INSERT INTO "
                "atlas_items(load_id,pipeline_key,task_id,cohort,status,verdict,explanation,evidence_i"
                "ds,packet_uri,has_task,omissions,defects) "
                "SELECT "
                "'parallel-old','retired',lpad(i::text,8,'0'),'admitted','reviewed','good','original-'"
                "||i,'[]','s3://fixture/original',true,'[]','[]' "
                "FROM generate_series(0,:last) i"
            ),
            {"last": population - 1},
        )
        c.execute(
            text(
                "INSERT INTO atlas_items SELECT "
                "'parallel-new',pipeline_key,task_id,cohort,status,verdict,'already-staged',evidence_i"
                "ds,packet_uri,has_task,omissions,exclusion_reason,confidence,reference_status,defects"
                ",provenance,replaced_task_sha256,repair_advisories "
                "FROM atlas_items WHERE load_id='parallel-old' AND mod(task_id::int,11)=0"
            )
        )
        c.execute(
            text(
                "INSERT INTO atlas_packets(load_id,pipeline_key,task_id,packet) SELECT "
                "'parallel-old','retired',lpad(i::text,8,'0'),repeat('evidence-',15000) FROM generate_series(0,100) i"
            )
        )
    stopped = threading.Event()
    lock = threading.Lock()
    interrupted = False
    worker_threads = set()
    durations = []

    def query(sql, parameters):
        nonlocal interrupted
        start = time.monotonic()
        result = execute_statement(batch_database, sql, parameters)
        with lock:
            durations.append(time.monotonic() - start)
            if sql.startswith("INSERT INTO atlas_items"):
                assert sum(key.startswith("task_id_") for key in parameters) <= batch_rows
                worker_threads.add(threading.get_ident())
                if not interrupted:
                    interrupted = True
                    raise RuntimeError("network lost after committed copy")
        return result

    with pytest.raises(RuntimeError):
        retain_prior_assessments_parallel(
            query,
            lambda: query,
            TABLE_COLUMNS,
            "parallel-old",
            "parallel-new",
            workers=4,
            item_batch_rows=batch_rows,
            stop=stopped,
        )
    with batch_database.connect() as c:
        assert c.execute(text("SELECT count(*) FROM atlas_assessments WHERE load_id='parallel-new'")).scalar_one() == 0
        committed = c.execute(text("SELECT count(*) FROM atlas_items WHERE load_id='parallel-new'")).scalar_one()
        assert committed > population // 11
    result = retain_prior_assessments_parallel(
        query,
        lambda: query,
        TABLE_COLUMNS,
        "parallel-old",
        "parallel-new",
        workers=4,
        item_batch_rows=batch_rows,
        stop=threading.Event(),
    )
    with batch_database.connect() as c:
        actual = dict(c.execute(text("SELECT task_id,explanation FROM atlas_items WHERE load_id='parallel-new'")).all())
        expected = {f"{i:08d}": "already-staged" if i % 11 == 0 else f"original-{i}" for i in range(population)}
        assert actual == expected
        assert c.execute(text("SELECT count(*) FROM atlas_packets WHERE load_id='parallel-new'")).scalar_one() == 101
        assert c.execute(text("SELECT count(*) FROM atlas_assessments WHERE load_id='parallel-new'")).scalar_one() == 1
        segments = c.execute(
            text(f"SELECT count(*) FROM {CHECKPOINT_TABLE} WHERE load_id='parallel-new' AND completed")
        ).scalar_one()
        assert segments >= 8
    assert result["items"] == population - committed
    assert len(worker_threads) >= 2
    assert retain_prior_assessments_parallel(
        query,
        lambda: query,
        TABLE_COLUMNS,
        "parallel-old",
        "parallel-new",
        workers=4,
        item_batch_rows=batch_rows,
        stop=threading.Event(),
    ) == dict.fromkeys(result, 0)
    record_property("parallel_copy_threads", len(worker_threads))
    record_property("max_parallel_query_wall_seconds", max(durations))


@pytest.mark.parametrize("count,scope", [(0, "sample"), (1, "sample"), (8193, "full"), (110600, "full")])
def test_bounded_summary_matches_canonical_view_and_resumes(batch_database, count, scope, record_property):
    load_id = f"summary-{scope}-{count}"
    parameters = {"load": load_id, "count": count, "scope": scope}
    with batch_database.begin() as connection:
        connection.execute(text("INSERT INTO atlas_loads VALUES (:load,'fixture',:load,now(),now())"), parameters)
        connection.execute(
            text(
                "INSERT INTO atlas_assessments VALUES (:load,'summary','source',:scope,"
                "'{}','fixture','{}','s3://fixture','[]')"
            ),
            parameters,
        )
        connection.execute(
            text(
                "INSERT INTO atlas_populations VALUES "
                "(:load,'summary','admitted',:count,(:count+1)/2),"
                "(:load,'summary','excluded',:count,:count/2),"
                "(:load,'summary','empty',0,0)"
            ),
            parameters,
        )
        connection.execute(
            text(
                """
            INSERT INTO atlas_items (load_id,pipeline_key,task_id,cohort,status,verdict,explanation,
                evidence_ids,packet_uri,has_task,omissions,confidence,reference_status,defects)
            SELECT :load,'summary',lpad(n::text,8,'0'),
                CASE WHEN mod(n,2)=1 THEN 'admitted' ELSE 'excluded' END,
                CASE WHEN mod(n,11)=0 THEN 'unavailable' ELSE 'reviewed' END,
                (ARRAY['good','some_issues','bad','unknown'])[mod(n,4)+1],repeat('Frozen evidence ',100),
                '[]'::jsonb,'s3://fixture',mod(n,3)=0,
                CASE WHEN mod(n,13)=0 THEN '["missing"]'::jsonb ELSE '[]'::jsonb END,
                (ARRAY['high','medium','low',NULL])[mod(n,4)+1],
                CASE WHEN mod(n,17)=0 THEN 'conflict' ELSE NULL END,
                CASE WHEN mod(n,7)=0 THEN '["wrong_reference","ambiguity"]'::jsonb ELSE '[]'::jsonb END
            FROM generate_series(1,:count) n
        """
            ),
            parameters,
        )
        expected = [
            dict(row)
            for row in connection.execute(
                text(
                    "SELECT cohort,to_jsonb(s)-'load_id'-'pipeline_key'-'cohort' AS summary "
                    "FROM atlas_cohort_summary s WHERE load_id=:load ORDER BY cohort"
                ),
                parameters,
            ).mappings()
        ]
    statements = []

    def query(sql, params):
        started = time.perf_counter()
        result = execute_statement(batch_database, sql, params)["rows"]
        statements.append({"seconds": time.perf_counter() - started, "sql": sql})
        return result

    assert materialize_cohort_summaries(query, load_id) == 1
    with batch_database.connect() as connection:
        actual = [
            dict(row)
            for row in connection.execute(
                text("SELECT cohort,summary FROM atlas_cohort_summaries WHERE load_id=:load ORDER BY cohort"), parameters
            ).mappings()
        ]
    assert actual == expected
    assert materialize_cohort_summaries(query, load_id) == 0
    assert all("FROM atlas_cohort_summary " not in row["sql"] for row in statements)
    record_property("maximum_statement_seconds", max(row["seconds"] for row in statements))
    record_property("statement_count", len(statements))
