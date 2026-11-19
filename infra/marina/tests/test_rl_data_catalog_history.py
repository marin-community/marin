# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retiring a source preserves its complete assessment and private packet evidence."""

import sqlite3

from infra.marina.apps.rl_data_catalog.historical_assessments import retain_prior_assessments


def test_refresh_retains_prior_evidence_without_overwriting_current_assessments():
    columns = {
        table: "pipeline_key TEXT, task_id TEXT, payload TEXT"
        for table in ("assessments", "populations", "items", "packets")
    }
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE TABLE atlas_cohort_summaries (load_id TEXT, pipeline_key TEXT)")
        for table in columns:
            database.execute(f"CREATE TABLE atlas_{table} (load_id TEXT, {columns[table]})")
            database.executemany(
                f"INSERT INTO atlas_{table} VALUES (?, ?, ?, ?)",
                [
                    ("prior", "retired-source", "retired-task", f"archived-{table}"),
                    ("prior", "active-source", "active-task", "previous"),
                    ("new", "active-source", "active-task", "current"),
                ],
            )

        database.execute("INSERT INTO atlas_cohort_summaries SELECT load_id,pipeline_key FROM atlas_assessments")
        statements = []

        def query(sql, parameters):
            cursor = database.execute(sql, parameters)
            statements.append(sql)
            rows = (
                [dict(zip([column[0] for column in cursor.description], row, strict=True)) for row in cursor.fetchall()]
                if cursor.description
                else []
            )
            return {"row_count": cursor.rowcount, "rows": rows}

        assert retain_prior_assessments(query, columns, "prior", "new") == dict.fromkeys(columns, 1)
        for table in columns:
            assert dict(database.execute(f"SELECT pipeline_key, payload FROM atlas_{table} WHERE load_id = 'new'")) == {
                "retired-source": f"archived-{table}",
                "active-source": "current",
            }
        copies = [statement for statement in statements if statement.startswith("INSERT")]
        assert len(copies) == len(columns) * 1  # one missing pipeline per table
        assert retain_prior_assessments(query, columns, "prior", "new") == dict.fromkeys(columns, 0)


def test_refresh_copies_each_missing_pipeline_in_its_own_statement():
    """The gateway bound: N missing pipelines cost N bounded copy statements."""
    columns = {
        table: "pipeline_key TEXT, task_id TEXT, payload TEXT"
        for table in ("assessments", "populations", "items", "packets")
    }
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE TABLE atlas_cohort_summaries (load_id TEXT, pipeline_key TEXT)")
        for table in columns:
            database.execute(f"CREATE TABLE atlas_{table} (load_id TEXT, {columns[table]})")
            database.executemany(
                f"INSERT INTO atlas_{table} VALUES (?, ?, ?, ?)",
                [("prior", f"retired-{index}", f"task-{index}", f"archived-{table}-{index}") for index in range(5)],
            )
        database.execute("INSERT INTO atlas_cohort_summaries SELECT load_id,pipeline_key FROM atlas_assessments")
        statements = []

        def query(sql, parameters):
            cursor = database.execute(sql, parameters)
            statements.append(sql)
            rows = (
                [dict(zip([column[0] for column in cursor.description], row, strict=True)) for row in cursor.fetchall()]
                if cursor.description
                else []
            )
            return {"row_count": cursor.rowcount, "rows": rows}

        assert retain_prior_assessments(query, columns, "prior", "new") == {table: 5 for table in columns}
        copies = [statement for statement in statements if statement.startswith("INSERT")]
        # 4 tables x 5 pipelines = 20 bounded copies, never one unbounded statement
        assert len(copies) == 20
        assert all("previous.pipeline_key = :pipeline_key" in copy for copy in copies)
