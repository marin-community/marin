# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for shaping a finelog result into Grafana rows."""

from datetime import UTC, datetime

import pytest
from conftest import finelog_result
from results import rows_to_json, substitute_time_macros

START = datetime(2026, 7, 17, 3, 0, 0, tzinfo=UTC)
END = datetime(2026, 7, 17, 4, 0, 0, tzinfo=UTC)


def test_substitute_time_macros_inserts_tz_naive_utc_literals():
    sql = substitute_time_macros("WHERE ts >= {{from}} AND ts < {{to}}", START, END)
    assert sql == "WHERE ts >= TIMESTAMP '2026-07-17 03:00:00' AND ts < TIMESTAMP '2026-07-17 04:00:00'"


def test_substitute_time_macros_leaves_sql_without_macros_untouched():
    assert substitute_time_macros("SELECT 1", None, None) == "SELECT 1"


def test_substitute_time_macros_rejects_a_macro_without_its_bound():
    with pytest.raises(ValueError, match="no matching time bound"):
        substitute_time_macros("WHERE ts >= {{from}}", None, END)


def test_rows_to_json_reads_naive_timestamps_as_utc_millis():
    # finelog stores timestamps tz-naive in UTC; reading them as the server's local
    # time would shift every point, and Grafana plots millis, not datetimes.
    naive = datetime(2025, 7, 8, 12, 0, 0)
    rows = rows_to_json(finelog_result(t=[naive], value=[3.0]))
    assert rows == [{"t": round(naive.replace(tzinfo=UTC).timestamp() * 1000), "value": 3.0}]
    assert isinstance(rows[0]["t"], int)


def test_rows_to_json_flattens_json_labels_into_columns():
    table = finelog_result(value=[3.0], labels=['{"region": "us-east5", "scope": "pool"}'])
    assert rows_to_json(table) == [{"value": 3.0, "label_region": "us-east5", "label_scope": "pool"}]


def test_rows_to_json_keeps_a_row_whose_labels_do_not_parse():
    # One malformed cell is schema drift; it must not blank the panel.
    assert rows_to_json(finelog_result(value=[1.0], labels=["{not json"])) == [{"value": 1.0, "labels": "{not json"}]


def test_rows_to_json_passes_through_rows_without_a_labels_column():
    assert rows_to_json(finelog_result(task_id=["a"], value=[5.0])) == [{"task_id": "a", "value": 5.0}]
