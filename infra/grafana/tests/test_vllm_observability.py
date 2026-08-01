# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import duckdb
import pytest
from vllm_observability import (
    VLLM_MAX_POINTS,
    VLLM_MAX_RESULT_ROWS,
    VLLM_MAX_WINDOW_MS,
    VllmIdentityField,
    vllm_overview_query,
)

START_MS = 120_000
END_MS = 180_000
BUCKET_MS = 60_000


def _attributes(temporality: str | None = None, **labels: str) -> str:
    values = dict(labels)
    if temporality is not None:
        values["source_temporality"] = temporality
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _resource(job: str, replica: str) -> str:
    return json.dumps({"job_id": job, "worker": replica}, sort_keys=True, separators=(",", ":"))


def _record(
    replica: str,
    name: str,
    value: float,
    timestamp_ms: int,
    attributes: str,
    *,
    job: str = "/serve",
    kind: str = "gauge",
) -> tuple:
    return ("cw-a", "vllm", name, kind, value, _resource(job, replica), attributes, timestamp_ms)


def _database(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    database.execute(
        """
        CREATE TABLE telemetry_v1(
            cluster VARCHAR,
            service VARCHAR,
            name VARCHAR,
            kind VARCHAR,
            value DOUBLE,
            resource_attributes_json VARCHAR,
            attributes_json VARCHAR,
            timestamp_ms BIGINT,
            seq BIGINT
        )
        """
    )
    database.execute(
        "CREATE MACRO json_get(document, field_name) "
        "AS json_extract_string(document, concat('$.', field_name))"
    )
    if rows:
        database.executemany(
            "INSERT INTO telemetry_v1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(*row, seq) for seq, row in enumerate(rows)],
        )
    return database


def _query_rows(database: duckdb.DuckDBPyConnection, job: str = "/serve") -> list[dict]:
    query = vllm_overview_query(VllmIdentityField.JOB_ID, job, START_MS, END_MS, BUCKET_MS)
    result = database.execute(query.sql)
    columns = [description[0] for description in result.description]
    return [dict(zip(columns, values, strict=True)) for values in result.fetchall()]


def _one(rows: list[dict], section: str, metric: str, stat: str, series: str | None = None) -> dict:
    matches = [
        row
        for row in rows
        if row["section"] == section
        and row["metric"] == metric
        and row["stat"] == stat
        and (series is None or row["series"] == series)
    ]
    assert len(matches) == 1
    return matches[0]


def test_query_preserves_replicas_discards_resets_and_keeps_native_counter_deltas():
    cumulative = _attributes("cumulative_snapshot")
    current = _attributes("current_snapshot")
    rows: list[tuple] = []

    for timestamp_ms, value in zip((105_000, 120_000, 135_000, 150_000, 165_000), (100, 110, 4, 9, 14)):
        rows.append(_record("a", "generation_tokens_total", value, timestamp_ms, cumulative))
    for timestamp_ms, value in zip((105_000, 120_000, 135_000, 150_000, 165_000), (50, 55, 60, 65, 70)):
        rows.append(_record("b", "generation_tokens_total", value, timestamp_ms, cumulative))
    rows.extend(
        [
            _record("a", "prompt_tokens_total", 0, 105_000, cumulative),
            _record("a", "prompt_tokens_total", 12, 135_000, cumulative),
            _record("b", "prompt_tokens_total", 0, 105_000, cumulative),
            _record("b", "prompt_tokens_total", 8, 135_000, cumulative),
        ]
    )

    rows.extend(
        [
            _record("a", "num_preemptions_total", 2, 135_000, _attributes(), kind="counter"),
            _record("a", "num_preemptions_total", 1, 165_000, _attributes(), kind="counter"),
        ]
    )

    for replica, running, waiting, kv in (
        ("a", (2, 4), (1, 1), (0.5, 0.7)),
        ("b", (3, 5), (0, 2), (0.3, 0.5)),
    ):
        for timestamp_ms, running_value, waiting_value, kv_value in zip(
            (120_000, 150_000), running, waiting, kv, strict=True
        ):
            rows.extend(
                [
                    _record(replica, "num_requests_running", running_value, timestamp_ms, current),
                    _record(replica, "num_requests_waiting", waiting_value, timestamp_ms, current),
                    _record(replica, "kv_cache_usage_perc", kv_value, timestamp_ms, current),
                ]
            )

    rows.extend(
        _histogram_records(
            "a",
            {
                105_000: (10, 10, 5, 8, 10),
                135_000: (12, 12, 6, 10, 12),
                150_000: (1, 1, 0, 1, 1),
                165_000: (3, 3, 1, 2, 3),
            },
        )
    )
    for family, total_sum, count in (
        ("inter_token_latency_seconds", 0.4, 4),
        ("request_queue_time_seconds", 0.6, 3),
        ("e2e_request_latency_seconds", 4.5, 3),
    ):
        rows.extend(
            [
                _record("a", f"{family}_sum", 0, 105_000, cumulative),
                _record("a", f"{family}_count", 0, 105_000, cumulative),
                _record("a", f"{family}_sum", total_sum, 135_000, cumulative),
                _record("a", f"{family}_count", count, 135_000, cumulative),
            ]
        )
    rows.extend(
        _histogram_records(
            "b",
            {
                105_000: (4, 4, 3, 4, 4),
                135_000: (5, 6, 4, 5, 6),
                165_000: (7, 8, 5, 7, 8),
            },
        )
    )

    rows.extend(
        [
            _record("a", "request_success_total", 0, 105_000, _attributes("cumulative_snapshot", finished_reason="stop")),
            _record("a", "request_success_total", 2, 120_000, _attributes("cumulative_snapshot", finished_reason="stop")),
            _record("a", "request_success_total", 0, 135_000, _attributes("cumulative_snapshot", finished_reason="stop")),
            _record("a", "request_success_total", 1, 150_000, _attributes("cumulative_snapshot", finished_reason="stop")),
            _record("b", "request_success_total", 0, 105_000, _attributes("cumulative_snapshot", finished_reason="length")),
            _record("b", "request_success_total", 1, 135_000, _attributes("cumulative_snapshot", finished_reason="length")),
            _record("b", "request_success_total", 2, 165_000, _attributes("cumulative_snapshot", finished_reason="length")),
        ]
    )

    result = _query_rows(_database(rows))

    assert _one(result, "counter_total", "prompt_tokens", "total")["value"] == pytest.approx(20)
    assert _one(result, "counter_total", "generated_tokens", "total")["value"] == pytest.approx(40)
    assert _one(result, "token_rate", "prompt_tokens", "rate")["value"] == pytest.approx(20 / 60)
    assert _one(result, "token_rate", "generated_tokens", "rate")["value"] == pytest.approx(40 / 60)
    assert _one(result, "counter_total", "preemptions", "total")["value"] == pytest.approx(3)
    assert _one(result, "saturation_summary", "num_requests_running", "average")["value"] == pytest.approx(7)
    assert _one(result, "saturation_summary", "num_requests_waiting", "peak")["value"] == pytest.approx(2)
    assert _one(result, "saturation_summary", "kv_cache_usage", "average")["value"] == pytest.approx(0.5)
    assert _one(result, "latency", "ttft", "mean")["value"] == pytest.approx(0.875)
    assert _one(result, "latency", "ttft", "p50")["value"] == pytest.approx(0.5)
    assert _one(result, "latency", "ttft", "p90")["value"] is None
    assert _one(result, "latency", "tpot", "mean")["value"] == pytest.approx(0.1)
    assert _one(result, "latency", "queue", "mean")["value"] == pytest.approx(0.2)
    assert _one(result, "latency", "e2e", "mean")["value"] == pytest.approx(1.5)
    assert _one(result, "request_outcome", "requests", "total", "stop")["value"] == pytest.approx(3)
    assert _one(result, "request_outcome", "requests", "total", "length")["value"] == pytest.approx(2)
    freshness = _one(result, "freshness", "telemetry", "latest_sample_age")
    assert (freshness["status"], freshness["value"], freshness["gap_seconds"]) == ("fresh", 15.0, 15.0)


def _histogram_records(replica: str, snapshots: dict[int, tuple[float, float, float, float, float]]) -> list[tuple]:
    rows = []
    for timestamp_ms, (total_sum, count, at_half, at_one, at_infinity) in snapshots.items():
        rows.extend(
            [
                _record(replica, "time_to_first_token_seconds_sum", total_sum, timestamp_ms, _attributes("cumulative_snapshot")),
                _record(replica, "time_to_first_token_seconds_count", count, timestamp_ms, _attributes("cumulative_snapshot")),
                _record(
                    replica,
                    "time_to_first_token_seconds_bucket",
                    at_half,
                    timestamp_ms,
                    _attributes("cumulative_snapshot", le="0.5"),
                ),
                _record(
                    replica,
                    "time_to_first_token_seconds_bucket",
                    at_one,
                    timestamp_ms,
                    _attributes("cumulative_snapshot", le="1.0"),
                ),
                _record(
                    replica,
                    "time_to_first_token_seconds_bucket",
                    at_infinity,
                    timestamp_ms,
                    _attributes("cumulative_snapshot", le="+Inf"),
                ),
            ]
        )
    return rows


def test_query_returns_explicit_no_data_row():
    rows = _query_rows(_database([]))
    assert rows == [
        {
            "t": None,
            "section": "freshness",
            "metric": "telemetry",
            "stat": "latest_sample_age",
            "series": "telemetry",
            "value": None,
            "unit": "s",
            "status": "no_data",
            "samples": 0,
            "gap_seconds": None,
        }
    ]


def test_query_quotes_identity_as_data():
    cumulative = _attributes("cumulative_snapshot")
    database = _database(
        [
            _record("a", "generation_tokens_total", 0, 105_000, cumulative, job="job'quoted"),
            _record("a", "generation_tokens_total", 5, 135_000, cumulative, job="job'quoted"),
            _record("b", "generation_tokens_total", 0, 105_000, cumulative, job="other"),
            _record("b", "generation_tokens_total", 500, 135_000, cumulative, job="other"),
        ]
    )

    rows = _query_rows(database, job="job'quoted")

    assert _one(rows, "counter_total", "generated_tokens", "total")["value"] == pytest.approx(5)


def test_query_enforces_window_bucket_and_result_safety_contract():
    with pytest.raises(ValueError, match="7 days"):
        vllm_overview_query(VllmIdentityField.JOB_ID, "/serve", 0, VLLM_MAX_WINDOW_MS + 1, 60_000)
    with pytest.raises(ValueError, match="positive"):
        vllm_overview_query(VllmIdentityField.JOB_ID, "/serve", 0, 60_000, 0)

    query = vllm_overview_query(VllmIdentityField.JOB_ID, "/serve", 0, VLLM_MAX_WINDOW_MS, 1)
    assert query.bucket_ms >= VLLM_MAX_WINDOW_MS / VLLM_MAX_POINTS
    assert f"LIMIT {VLLM_MAX_RESULT_ROWS}" in query.sql
    assert "timestamp_ms >= 0" in query.sql
    assert f"timestamp_ms < {VLLM_MAX_WINDOW_MS}" in query.sql
    assert "json_get(resource_attributes_json, 'job_id') = '/serve'" in query.sql
