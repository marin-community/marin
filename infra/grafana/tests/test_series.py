# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the finelog-rows-to-Grafana-series conversion."""

from datetime import UTC, datetime

import pyarrow as pa
import pytest
from conftest import result_table as _table
from series import UNLABELLED, build_sql, to_json_rows, to_series

NAMESPACE = "infra.canary.metrics"
START = datetime(2026, 7, 17, 3, 0, 0, tzinfo=UTC)
END = datetime(2026, 7, 17, 4, 0, 0, tzinfo=UTC)


def test_build_sql_narrows_by_metric_and_time_window():
    sql = build_sql(NAMESPACE, "worker_healthy", START, END, limit=100)
    assert f'FROM "{NAMESPACE}"' in sql
    assert "WHERE metric = 'worker_healthy'" in sql
    # tz-naive UTC literals: finelog stores timestamps naive, so a tz-aware
    # literal would raise a comparison error in the engine.
    assert "collected_at >= TIMESTAMP '2026-07-17 03:00:00'" in sql
    assert "collected_at < TIMESTAMP '2026-07-17 04:00:00'" in sql
    assert "LIMIT 100" in sql


def test_build_sql_casts_time_axis_to_int64_micros():
    # The bridge converts micros->millis itself; the cast has to be present and
    # aliased for to_series to find the column.
    assert "arrow_cast(collected_at, 'Int64') AS collected_us" in build_sql(NAMESPACE, "m", START, END, limit=1)


@pytest.mark.parametrize(
    "metric",
    [
        "worker_healthy'; DROP TABLE x --",
        "a b",
        "",
        "1_leading_digit",
        'quote"d',
    ],
)
def test_build_sql_rejects_non_identifier_metric(metric):
    # Identifiers are validated, not escaped, so nothing a caller supplies can
    # reach the engine as SQL.
    with pytest.raises(ValueError, match="bare identifier"):
        build_sql(NAMESPACE, metric, START, END, limit=1)


def test_build_sql_rejects_inverted_window():
    with pytest.raises(ValueError, match="must be after"):
        build_sql(NAMESPACE, "m", END, START, limit=1)


def test_converts_micros_to_millis():
    # A raw micros value fed to Grafana lands ~58000 years in the future; finelog
    # stores TIMESTAMP_MS columns at microsecond precision.
    table = _table([("worker_healthy", 3.0, {"region": "us-east5"}, 1_752_000_000_000_000)])
    assert to_series(table)[0].time_ms == 1_752_000_000_000


def test_groups_by_label_the_engine_cannot_slice():
    table = _table(
        [
            ("worker_healthy", 3.0, {"region": "us-east5"}, 1_000_000),
            ("worker_healthy", 5.0, {"region": "us-central2"}, 1_000_000),
            ("worker_healthy", 4.0, {"region": "us-east5"}, 2_000_000),
        ]
    )
    points = to_series(table, group_by="region")
    assert {(p.series, p.value) for p in points} == {
        ("us-east5", 3.0),
        ("us-central2", 5.0),
        ("us-east5", 4.0),
    }


def test_ungrouped_rows_collapse_to_one_series_named_for_the_metric():
    table = _table([("worker_healthy", 3.0, {"scope": "fleet"}, 1_000_000)])
    assert to_series(table)[0].series == "worker_healthy"


def test_match_filters_rows_by_label():
    # probes emits both a fleet rollup and per-region rows under one metric;
    # picking one out is only possible after decoding labels.
    table = _table(
        [
            ("worker_healthy", 8.0, {"scope": "fleet"}, 1_000_000),
            ("worker_healthy", 3.0, {"region": "us-east5"}, 1_000_000),
        ]
    )
    assert [p.value for p in to_series(table, match={"scope": "fleet"})] == [8.0]


def test_rows_missing_the_group_label_get_a_sentinel_series():
    table = _table([("worker_healthy", 8.0, {"scope": "fleet"}, 1_000_000)])
    assert to_series(table, group_by="region")[0].series == UNLABELLED


def test_drops_rows_with_unparseable_labels():
    # One malformed cell is schema drift; it must not blank the panel.
    table = pa.table(
        {
            "metric": ["m", "m"],
            "value": [1.0, 2.0],
            "labels": ["{not json", '{"region": "us-east5"}'],
            "collected_us": [1_000_000, 2_000_000],
        }
    )
    assert [p.value for p in to_series(table, group_by="region")] == [2.0]


def test_rejects_non_identifier_group_by():
    with pytest.raises(ValueError, match="bare identifier"):
        to_series(_table([]), group_by="region; DROP TABLE x")


def test_rejects_result_missing_expected_columns():
    with pytest.raises(ValueError, match="missing columns"):
        to_series(pa.table({"metric": ["m"]}))


def test_to_json_rows_emits_long_format_for_infinity():
    table = _table([("worker_healthy", 3.0, {"region": "us-east5"}, 1_752_000_000_000_000)])
    assert to_json_rows(to_series(table, group_by="region")) == [
        {"time": 1_752_000_000_000, "series": "us-east5", "value": 3.0}
    ]
