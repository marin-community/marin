# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from ducky.query_log import QueryLog, QueryLogRow, now_utc
from finelog.client.log_client import schema_from_dataclass


def test_row_is_a_valid_finelog_schema():
    """QueryLogRow must infer a finelog schema, else registration/writes silently fail."""
    schema = schema_from_dataclass(QueryLogRow)
    assert schema.key_column == "ts"  # segments keyed on time for range-scan pruning
    by_name = {c.name: c for c in schema.columns}
    assert set(by_name) == {
        "ts",
        "query_id",
        "sql",
        "status",
        "cached",
        "elapsed_ms",
        "total_rows",
        "result_bytes",
        "result_path",
        "error",
    }
    assert all(c.nullable for c in schema.columns)  # finelog registers every column nullable


def test_now_utc_is_naive():
    assert now_utc().tzinfo is None
    assert isinstance(now_utc(), datetime)


def test_record_swallows_table_write_failures():
    """A finelog write blowing up must not propagate to the query path."""

    class _BoomTable:
        def write(self, rows):
            raise RuntimeError("finelog down")

    log = QueryLog(log_client=None, table=_BoomTable())  # type: ignore[arg-type]
    row = QueryLogRow(ts=now_utc(), query_id="a" * 32, sql="SELECT 1", status="done", cached=False)
    log.record(row)  # does not raise
