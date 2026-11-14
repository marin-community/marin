# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retiring a source preserves its complete assessment and private packet evidence."""

import sqlite3

from infra.marina.apps.rl_data_catalog.historical_assessments import retain_prior_assessments


def test_refresh_retains_prior_evidence_without_overwriting_current_assessments():
    columns = {table: "pipeline_key TEXT, payload TEXT" for table in ("assessments", "populations", "items", "packets")}
    with sqlite3.connect(":memory:") as database:
        for table in columns:
            database.execute(f"CREATE TABLE atlas_{table} (load_id TEXT, pipeline_key TEXT, payload TEXT)")
            database.executemany(
                f"INSERT INTO atlas_{table} VALUES (?, ?, ?)",
                [
                    ("prior", "retired-source", f"archived-{table}"),
                    ("prior", "active-source", "previous"),
                    ("new", "active-source", "current"),
                ],
            )

        def query(sql, parameters):
            cursor = database.execute(sql, parameters)
            return {"row_count": cursor.rowcount, "rows": []}

        assert retain_prior_assessments(query, columns, "prior", "new") == dict.fromkeys(columns, 1)
        for table in columns:
            assert dict(database.execute(f"SELECT pipeline_key, payload FROM atlas_{table} WHERE load_id = 'new'")) == {
                "retired-source": f"archived-{table}",
                "active-source": "current",
            }
        assert retain_prior_assessments(query, columns, "prior", "new") == dict.fromkeys(columns, 0)
