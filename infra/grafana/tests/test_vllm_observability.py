# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace as Record
from uuid import uuid4

import duckdb
import httpx
import pyarrow as pa
import pytest
from config import ClusterTarget
from conftest import bridge_config
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from dashboard_stitch import stitch_all
from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from finelog.errors import StatsError
from rigging.timing import Duration, ExponentialBackoff
from server import create_app
from starlette.testclient import TestClient
from vllm_observability import (
    VLLM_DETAIL_MAX_WINDOW_MS,
    VLLM_OVERVIEW_SECTIONS,
)


@pytest.mark.parametrize("filename", ["inference.json", "inference_overview.json"])
def test_inference_identity_selector_reads_request_state_rows(filename: str) -> None:
    database = duckdb.connect()
    columns = """service VARCHAR, job_id VARCHAR, run_id VARCHAR, execution_uid VARCHAR,
        timestamp_ms BIGINT, name VARCHAR, attributes_json VARCHAR"""
    database.execute(f'CREATE TABLE "telemetry_v1.vllm"({columns})')
    database.execute(f'CREATE TABLE "telemetry_v1.marinskyrl"({columns})')
    database.execute("CREATE MACRO json_get(d, f) AS json_extract_string(d, concat('$.', f))")
    database.execute(
        """INSERT INTO "telemetry_v1.vllm" VALUES
           ('vllm', '/serve', 'serve-run', 'serve-execution', 1000, 'num_requests_running', '{}'),
           ('vllm', '/metric-only', 'metric-run', 'metric-execution', 2000, 'generation_tokens_total', '{}'),
           ('vllm', '/later', 'later-run', 'later-execution', 28800000, 'num_requests_running', '{}')"""
    )
    database.execute(
        """INSERT INTO "telemetry_v1.marinskyrl" VALUES
           ('marinskyrl', '/train', 'r1', 'e1', 3000, 'num_requests_running', '{"metric_source":"vllm"}'),
           ('marinskyrl', '/foreign', 'r2', 'e2', 4000, 'num_requests_running', '{"metric_source":"ray"}')"""
    )

    dashboard = json.loads((Path(__file__).parents[1] / "dashboards" / filename).read_text())
    variable = next(item for item in dashboard["templating"]["list"] if item["name"] == "identity")
    sql = next(
        item["value"] for item in variable["query"]["infinityQuery"]["url_options"]["params"] if item["key"] == "sql"
    )
    sql = sql.replace("${identity_kind}", "job_id")
    short_sql = sql.replace("{{from}}", "TIMESTAMP '1970-01-01 00:00:00'").replace(
        "{{to}}", "TIMESTAMP '1970-01-01 00:04:00'"
    )
    long_sql = sql.replace("{{from}}", "TIMESTAMP '1970-01-01 00:00:00'").replace(
        "{{to}}", "TIMESTAMP '1970-01-01 09:00:00'"
    )

    assert database.execute(short_sql).fetchall() == [("/serve",), ("/train",)]
    assert database.execute(long_sql).fetchall() == [("/serve",), ("/train",)]


def _vllm_projection_database():
    database = duckdb.connect()
    columns = """cluster VARCHAR, service VARCHAR, job_id VARCHAR, name VARCHAR, kind VARCHAR,
        value DOUBLE, resource_attributes_json VARCHAR, attributes_json VARCHAR, timestamp_ms BIGINT, seq BIGINT,
        body_json VARCHAR"""
    for table in ("telemetry_v1.marinskyrl", "telemetry_v1.vllm"):
        database.execute(f'CREATE TABLE "{table}"({columns})')
    database.execute("CREATE MACRO json_get(d, f) AS json_extract_string(d, concat('$.', f))")
    # Finelog and DuckDB name the same struct constructor differently.
    database.execute(
        """CREATE MACRO named_struct(k1, v1, k2, v2, k3, v3, k4, v4, k5, v5,
                                     k6 := NULL, v6 := NULL, k7 := NULL, v7 := NULL,
                                     k8 := NULL, v8 := NULL, k9 := NULL, v9 := NULL,
                                     k10 := NULL, v10 := NULL, k11 := NULL, v11 := NULL)
                     AS struct_pack(timestamp_ms := v1, seq := v2, value := v3,
                                    body_json := v4, publication_id := v5,
                                    histogram_count := v6, histogram_sum := v7,
                                    histogram_bounds := v8, producer_epoch := v9,
                                    source_sequence := v10, histogram_bins := v11)"""
    )
    return database


_HISTOGRAM_ATTRIBUTES = {
    "engine": "engine-a",
    "engine_index": "0",
    "metric_source": "vllm",
    "source_kind": "histogram",
    "source_temporality": "cumulative_snapshot",
}


def _histogram_rows(family, bounds, point, attributes=_HISTOGRAM_ATTRIBUTES):
    timestamp, sequence, bins, total = point
    common = {key: value for key, value in attributes.items() if key != "histogram_publication_id"}
    labels = json.dumps(common, sort_keys=True, separators=(",", ":"))

    def scalar_labels(extra):
        return json.dumps({**attributes, **extra}, sort_keys=True, separators=(",", ":"))

    count = sum(bins)
    body = {
        "encoding": "explicit_bucket_v1",
        "aggregation_temporality": "cumulative",
        "explicit_bounds": bounds,
        "bucket_counts": bins,
        "count": count,
        "sum": total,
        "producer_epoch": attributes["engine"],
        "sequence": sequence,
    }
    scalar = []
    cumulative = 0
    for bound, amount in zip((*bounds, float("inf")), bins, strict=True):
        cumulative += amount
        label = "+Inf" if bound == float("inf") else str(int(bound)) if float(bound).is_integer() else str(bound)
        scalar.append((family + "_bucket", "gauge", float(cumulative), None, scalar_labels({"le": label}), timestamp))
    scalar.extend(
        (family + "_" + component, "gauge", float(value), None, scalar_labels({}), timestamp)
        for component, value in (("count", count), ("sum", total))
    )
    return scalar, (family, "histogram", None, json.dumps(body), labels, timestamp)


def _insert_histograms(database, rows):
    database.executemany(
        """INSERT INTO "telemetry_v1.marinskyrl"
           (cluster, service, job_id, name, kind, value, body_json,
            resource_attributes_json, attributes_json, timestamp_ms, seq)
           VALUES ('cw-a', 'marinskyrl', '/train', ?, ?, ?, ?, '{}', ?, ?, ?)""",
        [(*row, seq) for seq, row in enumerate(rows)],
    )


def _histogram_results(database, start_ms=0):
    source = Record(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=lambda sql, *, max_rows: database.execute(sql).fetch_arrow_table(),
    )
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    params = {"identity_kind": "job_id", "identity": "/train", "from": start_ms, "to": 90_000, "bucket_ms": 15_000}
    with TestClient(app) as client:
        detail = client.get("/finelog/marin/v1/vllm/overview", params=params)
        summary = client.get(
            "/finelog/marin/v1/vllm/overview",
            params={**params, "to": VLLM_DETAIL_MAX_WINDOW_MS + 90_000, "view": "run_summary"},
        )
    assert detail.status_code == summary.status_code == 200
    return detail.json(), summary.json()


@pytest.mark.parametrize(
    "scenario",
    [
        "scalar_conflicting",
        "scalar_staggered",
        "scalar_bucket_reset",
        "scalar_middle_bin_reset",
        "structured_middle_bin_reset",
        "structured_unidentified_fast",
        "dual_conflicting",
        "dual_missing",
        "dual_missing_middle_bucket",
        "dual_partial",
    ],
)
def test_vllm_histogram_publication_faults(scenario):
    database = _vllm_projection_database()
    points = [(0, 0, (0, 0, 0), 0.0), (15_000, 1, (1, 2, 0), 0.3), (30_000, 2, (3, 2, 0), 0.8)]
    if scenario == "scalar_conflicting":
        points.insert(2, (15_000, 1, (2, 3, 0), 0.5))
    elif scenario == "scalar_bucket_reset":
        points[-1] = (30_000, 2, (0, 5, 0), 0.8)
    elif scenario.endswith("middle_bin_reset"):
        points[-1] = (30_000, 2, (3, 1, 1), 0.8)
    rows = []
    for timestamp, sequence, bins, total in points:
        attributes = dict(_HISTOGRAM_ATTRIBUTES)
        if scenario.startswith("dual") or scenario in ("scalar_conflicting", "scalar_staggered"):
            attributes = {"histogram_publication_id": f"engine-a:{sequence}", **attributes}
        scalar, structured = _histogram_rows(
            "time_to_first_token_seconds", (0.1, 1.0), (timestamp, sequence, bins, total), attributes
        )
        if scenario.startswith(("scalar", "dual")):
            for name, kind, value, body, labels, sample_time in scalar:
                if scenario == "scalar_staggered":
                    fields = json.loads(labels)
                    labels = json.dumps(
                        {"histogram_publication_id": fields.pop("histogram_publication_id"), **fields},
                        separators=(",", ":"),
                    )
                if sequence == 1:
                    if scenario == "dual_partial" and name != "time_to_first_token_seconds_bucket":
                        continue
                    if scenario == "dual_partial" and json.loads(labels)["le"] != "0.1":
                        continue
                    if scenario == "dual_missing_middle_bucket" and json.loads(labels).get("le") == "1":
                        continue
                    if scenario == "scalar_staggered":
                        sample_time = 16_000 if name.endswith("_sum") else 17_000
                rows.append((name, kind, value, body, labels, sample_time))
        if not scenario.startswith("scalar") and not (
            scenario in ("dual_missing", "dual_missing_middle_bucket", "dual_partial") and sequence == 1
        ):
            name, kind, value, body_json, labels, sample_time = structured
            body = json.loads(body_json)
            if scenario == "structured_unidentified_fast":
                del body["producer_epoch"], body["sequence"]
                sample_time //= 3
            rows.append((name, kind, value, json.dumps(body), labels, sample_time))
            if scenario == "dual_conflicting" and sequence == 1:
                rows.append(
                    (
                        name,
                        kind,
                        value,
                        json.dumps({**body, "bucket_counts": [0, sum(bins), 0]}),
                        rows[-1][4],
                        sample_time,
                    )
                )
    _insert_histograms(database, rows)
    start = 30_000 if scenario == "scalar_conflicting" else 16_500 if scenario == "scalar_staggered" else 0
    result, summary = _histogram_results(database, start)
    stats = {
        row["stat"]: (row["value"], row["samples"])
        for row in result
        if row["section"] == "latency" and row["metric"] == "ttft" and row["t"] is None
    }
    reset = scenario == "scalar_bucket_reset" or scenario.endswith("middle_bin_reset")
    count = 3 if reset else 5
    assert stats["mean"] == pytest.approx((0.1 if reset else 0.16, count))
    assert stats["p50"] == pytest.approx((1.0 if reset else 0.1, count))
    assert stats["p90"] == stats["p99"] == pytest.approx((1.0, count))
    samples = (
        1
        if reset or scenario in ("dual_conflicting", "dual_missing_middle_bucket", "dual_partial", "scalar_conflicting")
        else 2
    )
    assert [(row["value"], row["samples"]) for row in summary if row["metric"] == "ttft_observations"] == [
        (count, samples)
    ]


def test_vllm_target_histograms_weight_producers_without_dual_or_reset_counts():
    database = _vllm_projection_database()
    rows = []
    for family, bounds, small_total, big_total in (
        ("request_generation_tokens", (10.0, 100.0), 20.0, 800.0),
        ("request_time_per_output_token_seconds", (0.02, 0.1), 0.04, 0.8),
    ):
        for timestamp, count, total in ((0, 0, 0.0), (15_000, 2, small_total), (30_000, 1, small_total / 2)):
            labels = {"histogram_publication_id": f"engine-a:{timestamp // 15_000}", **_HISTOGRAM_ATTRIBUTES}
            scalar, structured = _histogram_rows(
                family, bounds, (timestamp, timestamp // 15_000, (count, 0, 0), total), labels
            )
            rows.extend(scalar if timestamp < 30_000 else [])
            rows.append(structured)
        for timestamp, count, total in ((0, 0, 0.0), (15_000, 8, big_total), (30_000, 10, big_total * 1.25)):
            scalar, _ = _histogram_rows(
                family,
                bounds,
                (timestamp, timestamp // 15_000, (0, count, 0), total),
                {**_HISTOGRAM_ATTRIBUTES, "engine": "engine-b", "engine_index": "1"},
            )
            rows.extend(scalar)
    _insert_histograms(database, rows)
    result, _ = _histogram_results(database)
    means = {
        row["metric"]: (row["value"], row["samples"])
        for row in result
        if row["stat"] == "mean" and row["metric"] in ("output_tokens", "tpot")
    }
    assert means == {"output_tokens": pytest.approx((85.0, 12)), "tpot": pytest.approx((1.04 / 12, 12))}
    assert [
        (row["value"], row["samples"])
        for row in result
        if row["metric"] == "tpot" and row["stat"] == "mean_over_time" and row["t"] == 30_000
    ] == [pytest.approx((0.1, 2))]
    assert {row["series"]: row["value"] for row in result if row["section"] == "output_length_distribution"} == {
        "10": 2,
        "100": 10,
        "+Inf": 0,
    }


@pytest.mark.parametrize("conflicting", [False, True])
def test_structured_histogram_exact_large_counts_duplicate_loss_and_reset(conflicting):
    database = _vllm_projection_database()
    first = (1 << 53) + 1
    rows = []
    for timestamp, sequence, count, total, arrival in (
        (0, 0, first, 0.0, 5),
        (15_000, 1, first + 2, 0.2, 2),
        (15_000, 1, first + (4 if conflicting else 2), 0.4 if conflicting else 0.2, 8),
        (45_000, 3, first + 5, 0.5, 1),
        (60_000, 4, 1, 0.1, 7),
        (75_000, 5, 3, 0.3, 3),
    ):
        _, row = _histogram_rows("time_to_first_token_seconds", (0.1,), (timestamp, sequence, (count, 0), total))
        rows.append((*row, arrival))
    database.executemany(
        """INSERT INTO "telemetry_v1.marinskyrl"
           (cluster, service, job_id, name, kind, value, body_json,
            resource_attributes_json, attributes_json, timestamp_ms, seq)
           VALUES ('cw-a', 'marinskyrl', '/train', ?, ?, ?, ?, '{}', ?, ?, ?)""",
        rows,
    )
    scalar, _ = _histogram_rows(
        "time_to_first_token_seconds",
        (0.1,),
        (30_000, 2, (first + 3, 0), 0.3),
        {"histogram_publication_id": "engine-a:2", **_HISTOGRAM_ATTRIBUTES},
    )
    _insert_histograms(database, scalar)
    for start in (0, 30_000):
        result, summary = _histogram_results(database, start)
        count = 7 if conflicting or start == 0 else 5
        assert [
            (row["value"], row["samples"])
            for row in result
            if row["section"] == "latency" and row["metric"] == "ttft" and row["stat"] == "mean"
        ] == [(pytest.approx(0.1), count)]
        times = {
            row["t"]: row["samples"]
            for row in result
            if row["section"] == "latency" and row["metric"] == "ttft" and row["stat"] == "mean_over_time"
        }
        expected = {45_000: 5, 75_000: 2} if conflicting else {15_000: 2, 45_000: 3, 75_000: 2}
        assert times == {t: value for t, value in expected.items() if t >= start}
        assert [row["value"] for row in summary if row["metric"] == "ttft_observations"] == [count]


@pytest.mark.parametrize(
    ("legacy_prefix", "bounds", "duplicate_last", "count", "mean", "p50"),
    [
        pytest.param(False, (0.1, 0.1, 0.2, 0.2), False, 4, 0.15, None, id="structured_schema_change"),
        pytest.param(True, (0.1, 0.1, 0.2, 0.2), False, 4, 0.15, None, id="scalar_prefix_schema_change"),
        pytest.param(False, (1, 1.0, 1.0, 1.0), False, 5, 0.14, 1.0, id="equivalent_numeric_bounds"),
        pytest.param(False, (1, 1.0, 1.0, 1.0), True, 3, 0.1, 1.0, id="conflicting_numeric_copy"),
    ],
)
def test_structured_histogram_bound_change_keeps_means_and_withholds_mixed_schema_tails(
    legacy_prefix, bounds, duplicate_last, count, mean, p50
):
    database = _vllm_projection_database()
    rows = []
    for sequence, (bound, amount, total) in enumerate(zip(bounds, (0, 2, 3, 5), (0.0, 0.2, 0.3, 0.7), strict=True)):
        scalar, structured = _histogram_rows(
            "time_to_first_token_seconds", (bound,), (sequence * 15_000, sequence, (amount, 0), total)
        )
        rows.extend(scalar if legacy_prefix and sequence < 2 else [structured])
        if duplicate_last and sequence == 3:
            body = json.loads(structured[3])
            body["explicit_bounds"] = [int(bound)]
            rows.append((*structured[:3], json.dumps(body), *structured[4:]))
    _insert_histograms(database, rows)
    result, summary = _histogram_results(database)
    stats = {
        row["stat"]: (row["value"], row["samples"])
        for row in result
        if row["section"] == "latency" and row["metric"] == "ttft" and row["t"] is None
    }
    assert stats["mean"] == pytest.approx((mean, count))
    assert stats["p50"] == ((None, count) if p50 is None else pytest.approx((p50, count)))
    assert [row["value"] for row in summary if row["metric"] == "ttft_observations"] == [count]


@pytest.mark.integration
def test_vllm_histograms_roundtrip_native_finelog(tmp_path):
    server = require_embedded_server()(log_dir=str(tmp_path / "finelog"))
    client = LogClient.connect(server.address)
    try:
        rows = []
        for point in ((0, 0, (0, 0, 0), 0.0), (15_000, 1, (1, 2, 0), 0.3), (30_000, 2, (3, 2, 0), 0.8)):
            scalar, structured = _histogram_rows(
                "time_to_first_token_seconds",
                (0.1, 1.0),
                point,
                {"histogram_publication_id": f"engine-a:{point[1]}", **_HISTOGRAM_ATTRIBUTES},
            )
            rows.extend([*scalar, structured])
        records = [
            {
                "name": name,
                "kind": kind,
                "value": value,
                "body": json.loads(body) if body is not None else None,
                "timestamp_ms": timestamp,
                "attributes": json.loads(labels),
            }
            for name, kind, value, body, labels, timestamp in rows
        ]
        with httpx.Client(base_url=server.address, timeout=5) as ingest:
            for service in ("vllm", "marinskyrl"):
                batch_id = str(uuid4())
                response = ingest.post(
                    "/v1/telemetry",
                    headers={"Idempotency-Key": batch_id},
                    json={
                        "version": 1,
                        "batch_id": batch_id,
                        "resource": {"service": service, "job_id": "/train" if service == "marinskyrl" else "/other"},
                        "records": records,
                    },
                )
                assert response.status_code == 200, response.text
        ExponentialBackoff().wait_until_or_raise(
            lambda: client.query("SELECT COUNT(*) FROM \"telemetry_v1.marinskyrl\" WHERE job_id = '/train'")
            .column(0)[0]
            .as_py()
            == len(records),
            timeout=Duration.from_seconds(10),
            error_message="Histogram records were accepted but did not become query-visible",
        )
        source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=client.query)
        app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
        params = {"identity_kind": "job_id", "identity": "/train", "from": 0, "to": 45_000, "bucket_ms": 15_000}
        with TestClient(app) as bridge:
            detail = bridge.get("/finelog/marin/v1/vllm/overview", params=params)
            summary = bridge.get(
                "/finelog/marin/v1/vllm/overview",
                params={**params, "to": VLLM_DETAIL_MAX_WINDOW_MS + 45_000, "view": "run_summary"},
            )
        assert detail.status_code == summary.status_code == 200
        assert [
            (row["value"], row["samples"]) for row in detail.json() if row["metric"] == "ttft" and row["stat"] == "mean"
        ] == [(pytest.approx(0.16), 5)]
        assert [row["value"] for row in summary.json() if row["metric"] == "ttft_observations"] == [5]
    finally:
        client.close()
        server.stop()


def _embedded_overview_app(invalid_histogram, histogram_format="scalar"):
    database = _vllm_projection_database()

    def sample(name, value, timestamp, **labels):
        attributes = {"metric_source": "vllm", "engine": "physical-a", "engine_index": "0", **labels}
        return name, value, json.dumps(attributes), timestamp

    cumulative = {"source_temporality": "cumulative_snapshot"}
    snapshot = {"source_temporality": "current_snapshot"}
    samples = [
        sample("generation_tokens_total", value, timestamp, **cumulative)
        for timestamp, value in ((0, 0), (15_000, 150), (30_000, 300))
    ]
    samples += [
        sample("prompt_tokens_total", value, timestamp, **cumulative)
        for timestamp, value in ((0, 0), (15_000, 30), (30_000, 60))
    ]
    samples += [
        sample("request_success_total", value, timestamp, finished_reason="stop", **cumulative)
        for timestamp, value in ((0, 0), (15_000, 1))
    ]
    samples += [
        sample(name, value, timestamp, finished_reason=reason, **cumulative)
        for name, reason, final in (("request_success_total", "length", 1), ("request_failure_total", "error", 100))
        for timestamp, value in ((0, 0), (15_000, final))
    ]
    histograms = []
    for family, scale in (
        ("request_time_per_output_token_seconds", 0.1),
        ("time_to_first_token_seconds", 2),
        ("inter_token_latency_seconds", 0.02),
        ("request_prefill_time_seconds", 1),
        ("request_decode_time_seconds", 10),
        ("request_generation_tokens", 100),
    ):
        if invalid_histogram == "missing_component" and family == "inter_token_latency_seconds":
            continue  # Older MSRL runs did not export native ITL.
        bounds = (0.1 * scale, float(scale))
        for sequence, bins, total in ((0, (0, 0, 0), 0), (1, (1, 2, 0), 0.3), (2, (3, 2, 0), 0.8)):
            attributes = {**_HISTOGRAM_ATTRIBUTES, "engine": "physical-a"}
            if histogram_format == "dual":
                attributes["histogram_publication_id"] = f"physical-a:{sequence}"
            scalar, structured = _histogram_rows(
                family, bounds, (sequence * 15_000, sequence, bins, total * scale), attributes
            )
            if histogram_format in ("scalar", "dual") or (histogram_format == "mixed" and sequence < 2):
                histograms.extend(scalar)
            if histogram_format in ("structured", "dual") or (histogram_format == "mixed" and sequence > 0):
                histograms.append(structured)
    _insert_histograms(database, histograms)
    # A second engine must not skew the fleet mean/tails with a reset or partial scrape.
    for component in ("sum", "count", "bucket"):
        labels = {**cumulative, "engine": "physical-b", "engine_index": "1"}
        if component == "bucket":
            labels["le"] = "100"
        samples.append(sample(f"request_prefill_time_seconds_{component}", 10, 0, **labels))
        if invalid_histogram == "missing_component" and component == "count":
            continue
        value = 5 if invalid_histogram == "reset" and component == "sum" else 110
        samples.append(sample(f"request_prefill_time_seconds_{component}", value, 15_000, **labels))
    samples += [
        sample("num_requests_running", 2, 15_000, **snapshot),
        sample("num_requests_waiting", 1, 15_000, **snapshot),
        sample("kv_cache_usage_perc", 0.25, 15_000, **snapshot),
        sample("kv_cache_usage_perc", 0.75, 15_000, engine="physical-b", engine_index="1", **snapshot),
    ]
    samples += [
        sample("num_requests_running", 1, timestamp, engine="physical-b", engine_index="1", **snapshot)
        for timestamp in range(0, 240_000, 15_000)
    ]
    samples += [
        sample("num_requests_waiting", 0, timestamp, engine="physical-b", engine_index="1", **snapshot)
        for timestamp in range(0, 240_000, 15_000)
    ]
    database.executemany(
        """INSERT INTO "telemetry_v1.marinskyrl"
           (cluster, service, job_id, name, kind, value, resource_attributes_json,
            attributes_json, timestamp_ms, seq)
           VALUES ('cw-a', 'marinskyrl', '/train', ?, 'gauge', ?, '{"worker":"driver"}', ?, ?, ?)""",
        [(*row, seq) for seq, row in enumerate(samples)],
    )

    queries = []

    def query(sql, *, max_rows):
        queries.append(sql)
        return database.execute(sql).fetch_arrow_table()

    source = Record(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=query,
    )
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    dashboard_dir = Path(__file__).parents[1] / "dashboards"
    dashboards = stitch_all(dashboard_dir, dashboard_dir / "panels")
    params = {"identity_kind": "job_id", "identity": "/train", "from": 0, "to": 240_000, "bucket_ms": 15_000}
    return app, dashboards, params, queries


def _assert_dashboard_panels(client, dashboards, params):
    for filename in ("inference.json", "inference_overview.json"):
        for panel in dashboards[filename]["panels"]:
            for target in panel.get("targets", []):
                target_params = {param["key"]: param["value"] for param in target["url_options"]["params"]}
                view = target_params["view"]
                result = client.get(
                    f"/finelog/marin{target['url']}",
                    params={**params, "view": view, "bucket_ms": target_params["bucket_ms"]},
                )
                assert result.status_code == 200, panel["title"]
                for row in result.json():
                    assert row["section"] == view
                    assert all(column["selector"] in row for column in target["columns"])


def _assert_overview_rows(rows, invalid_histogram):
    assert {row["section"] for row in rows} == VLLM_OVERVIEW_SECTIONS - {"run_timeline"}
    assert [row["status"] for row in rows if row["section"] == "diagnostic_status"] == ["detail", "check_count_gap"]
    values = {(row["section"], row["metric"], row["stat"], row["series"], row["t"]): row["value"] for row in rows}
    for metric, mean, p50, tail in (
        ("ttft", 0.32, 0.2, 2),
        ("inter_token_latency", 0.0032, 0.002, 0.02),
        ("tpot", 0.016, 0.01, 0.1),
        ("output_tokens", 16, 10, 100),
    ):
        if metric == "inter_token_latency" and invalid_histogram == "missing_component":
            continue
        for stat, expected in (("mean", mean), ("p50", p50), ("p90", tail), ("p99", tail)):
            assert [
                (row["value"], row["samples"])
                for row in rows
                if row["metric"] == metric and row["stat"] == stat and row["t"] is None
            ] == [pytest.approx((expected, 5))]
    assert values[("request_outcome", "requests", "total", "stop", None)] == 1
    assert (
        values[("length_finish_fraction", "length_finish_fraction", "fraction", "length / all engine finishes", None)]
        == 0.5
    )
    assert [(row["series"], row["value"]) for row in rows if row["section"] == "output_length_distribution"] == [
        ("10", 3),
        ("100", 2),
        ("+Inf", 0),
    ]
    assert values[("request_rate", "requests", "rate", "stop", 15_000)] == pytest.approx(1 / 15)
    assert values[("latency", "tpot", "mean_over_time", "time per output token", 15_000)] == pytest.approx(0.01)
    assert values[("latency", "tpot", "mean_over_time", "time per output token", 30_000)] == pytest.approx(0.025)
    assert values[("latency", "ttft", "mean_over_time", "ttft", 15_000)] == pytest.approx(0.2)
    assert values[("latency", "ttft", "mean_over_time", "ttft", 30_000)] == pytest.approx(0.5)
    assert values[("latency", "prefill", "mean", "prefill", None)] == pytest.approx(0.16)
    assert values[("latency", "prefill", "p90", "prefill", None)] == 1
    assert values[("latency", "decode", "mean", "decode", None)] == pytest.approx(1.6)
    assert values[("workload", "output_tokens", "mean", "output_tokens", None)] == 16
    histogram_rows = [row for row in rows if row["section"] in ("latency", "workload")]
    assert all(row["samples"] == {None: 5, 15_000: 3, 30_000: 2}[row["t"]] for row in histogram_rows)
    assert all(row["unit"] == "tokens" for row in histogram_rows if row["metric"] == "output_tokens")
    assert values[("saturation", "num_requests_running", "value", "num_requests_running", 15_000)] == 3
    assert values[("saturation", "num_requests_in_flight", "value", "num_requests_in_flight", 15_000)] == 4
    assert values[("saturation", "kv_cache_usage", "value", "kv_cache_usage", 15_000)] == 0.5
    assert values[("saturation", "kv_cache_usage_peak", "value", "kv_cache_usage_peak", 15_000)] == 0.75
    engines = {
        (row["series"].split(" @ ", 1)[0], row["metric"]): row["value"]
        for row in rows
        if row["section"] == "engine_summary"
    }
    assert engines[("physical-a", "running_mean")] == 2
    assert engines[("physical-b", "running_mean")] == 1
    assert engines[("physical-a", "waiting_mean")] == 1
    assert engines[("physical-b", "kv_cache_peak")] == 0.75
    assert engines[("physical-a", "generated_tokens_per_second")] == 10
    if invalid_histogram == "missing_component":
        assert ("physical-a", "inter_token_latency_mean") not in engines
    else:
        assert engines[("physical-a", "inter_token_latency_mean")] == pytest.approx(0.0032)
    assert ("physical-b", "generated_tokens_per_second") not in engines  # Missing is not idle.
    freshness = {row["series"].rsplit(":", 1)[-1]: row["status"] for row in rows if row["section"] == "freshness_detail"}
    assert freshness == {"physical-a": "stale_or_stopped", "physical-b": "fresh"}


@pytest.mark.parametrize(
    ("histogram_format", "invalid_histogram"),
    [
        ("scalar", "reset"),
        ("scalar", "missing_component"),
        ("structured", "reset"),
        ("mixed", "reset"),
        ("dual", "reset"),
    ],
)
def test_dashboard_vllm_overview_end_to_end(histogram_format, invalid_histogram):
    app, dashboards, params, queries = _embedded_overview_app(invalid_histogram, histogram_format)
    path = "/v1/vllm/overview"

    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{path}", params={**params, "view": "token_rate"})
        saturation = client.get(f"/finelog/marin{path}", params={**params, "view": "saturation"})
        all_rows = client.get(f"/finelog/marin{path}", params=params)
        _assert_dashboard_panels(client, dashboards, params)
        long_params = {**params, "to": VLLM_DETAIL_MAX_WINDOW_MS + 15_000}
        long_summary = client.get(f"/finelog/marin{path}", params={**long_params, "view": "run_summary"})
        _assert_dashboard_panels(client, dashboards, long_params)

    assert len(queries) == 2  # One external query per range serves both pages and all panels.
    assert response.status_code == 200
    assert saturation.status_code == 200
    assert all_rows.status_code == 200
    token_rows = response.json()
    saturation_rows = saturation.json()
    assert [row["t"] for row in token_rows] == sorted(row["t"] for row in token_rows)
    assert [row["t"] for row in saturation_rows] == sorted(row["t"] for row in saturation_rows)
    assert {(row["metric"], row["t"]): row["value"] for row in token_rows} == {
        ("generated_tokens", 15_000): 10,
        ("generated_tokens", 30_000): 10,
        ("prompt_tokens", 15_000): 2,
        ("prompt_tokens", 30_000): 2,
    }
    _assert_overview_rows(all_rows.json(), invalid_histogram)
    summary_rows = long_summary.json()
    assert [row["value"] for row in summary_rows if row["metric"] == "generated_tokens"] == [300]
    if invalid_histogram == "missing_component":
        assert not any(row["metric"] == "inter_token_latency" for row in summary_rows)
    else:
        assert [row["value"] for row in summary_rows if row["metric"] == "inter_token_latency"] == [
            pytest.approx(0.0032)
        ]
    assert [row["value"] for row in summary_rows if row["metric"] == "ttft_observations"] == [5]


def _standalone_overview_app():
    database = _vllm_projection_database()
    for cluster, values in (("cw-a", (100, 250, 10)), ("cw-b", (100, 160, 220))):
        for timestamp, value in zip((0, 60_000, 120_000), values, strict=True):
            database.execute(
                """INSERT INTO "telemetry_v1.vllm"
                (cluster, service, job_id, name, kind, value, resource_attributes_json,
                 attributes_json, timestamp_ms, seq) VALUES
                (?, 'vllm', '/serve', 'generation_tokens_total', 'gauge', ?, '{}',
                 '{"source_temporality":"cumulative_snapshot"}', ?, 0),
                (?, 'vllm', '/serve', 'num_requests_waiting', 'gauge', 0, '{}',
                 '{"source_temporality":"current_snapshot"}', ?, 0),
                (?, 'vllm', '/serve', 'request_success_total', 'counter', 2, '{}',
                 '{"finished_reason":"stop"}', ?, 0)""",
                [cluster, value, timestamp, cluster, timestamp, cluster, timestamp],
            )
    source = Record(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=lambda sql, *, max_rows: database.execute(sql).fetch_arrow_table(),
    )
    return create_app(bridge_config(), {"marin": source}, {}, None, None, None)


def _assert_standalone_rows(rows, rate):
    assert [row["value"] for row in rows if row["section"] == "token_rate"] == [rate]
    assert [row["value"] for row in rows if row["section"] == "request_outcome"] == [4]
    assert [row["value"] for row in rows if row["section"] == "saturation"] == [0]
    assert not any(row["metric"] in ("inter_token_latency", "inter_token_latency_mean") for row in rows)
    assert [row["status"] for row in rows if row["section"] == "telemetry_health"] == ["unknown"]


@pytest.fixture
def standalone_overview_client():
    with TestClient(_standalone_overview_app()) as client:
        yield client


def test_standalone_reset_window_change(standalone_overview_client):
    params = {"identity_kind": "job_id", "identity": "/serve", "from": 60_000, "to": 120_000, "bucket_ms": 15_000}
    first = standalone_overview_client.get("/finelog/marin/v1/vllm/overview", params=params)
    manual = standalone_overview_client.get(
        "/finelog/marin/v1/vllm/overview",
        params={**params, "identity": "/absent", "identity_override": "/serve", "view": "token_rate"},
    )
    reset = standalone_overview_client.get(
        "/finelog/marin/v1/vllm/overview", params={**params, "from": 120_000, "to": 180_000}
    )
    assert first.status_code == manual.status_code == reset.status_code == 200
    assert manual.json() == [row for row in first.json() if row["section"] == "token_rate"]
    _assert_standalone_rows(first.json(), 3.5)
    _assert_standalone_rows(reset.json(), 1.0)


def test_standalone_absent_identity(standalone_overview_client):
    params = {"identity_kind": "job_id", "identity": "/absent", "from": 60_000, "to": 120_000, "bucket_ms": 15_000}
    missing = standalone_overview_client.get("/finelog/marin/v1/vllm/overview", params=params)
    assert missing.status_code == 200
    rows = missing.json()
    assert [row["status"] for row in rows if row["section"] == "freshness"] == ["no_data"]
    assert [row["status"] for row in rows if row["section"] == "diagnostic_status"] == ["empty", "no_conclusion"]
    assert not any(row["section"] == "token_rate" for row in rows)


def test_standalone_long_range_has_summary_and_no_detail(standalone_overview_client):
    params = {
        "identity_kind": "job_id",
        "identity": "/serve",
        "from": 60_000,
        "to": VLLM_DETAIL_MAX_WINDOW_MS + 120_000,
        "bucket_ms": 15_000,
    }
    summary = standalone_overview_client.get("/finelog/marin/v1/vllm/overview", params=params)
    detail = standalone_overview_client.get("/finelog/marin/v1/vllm/overview", params={**params, "view": "token_rate"})
    empty = standalone_overview_client.get("/finelog/marin/v1/vllm/overview", params={**params, "identity": "/absent"})
    assert summary.status_code == detail.status_code == empty.status_code == 200
    rows = summary.json()
    assert [row["status"] for row in rows if row["section"] == "diagnostic_status"] == ["summary_only", "no_conclusion"]
    assert [row["value"] for row in rows if row["section"] == "run_summary" and row["metric"] == "generated_tokens"] == [
        270
    ]
    assert [row["value"] for row in rows if row["section"] == "run_timeline"] == [270]
    assert [row["value"] for row in rows if row["metric"] == "request_success_total"] == [8]
    assert detail.json() == []
    assert [row["status"] for row in empty.json()] == ["empty", "no_conclusion"]


def test_run_triage_count_gap_can_be_a_partial_window_without_a_failed_request():
    database = _vllm_projection_database()
    rows = [
        (name, value, timestamp, labels)
        for name, values, labels in (
            ("time_to_first_token_seconds_count", (0, 2, 2, 2), {}),
            ("time_to_first_token_seconds_sum", (0, 0.2, 0.2, 0.2), {}),
            ("time_to_first_token_seconds_bucket", (0, 2, 2, 2), {"le": "+Inf"}),
            ("request_success_total", (0, 0, 1, 2), {"finished_reason": "stop"}),
        )
        for timestamp, value in zip((0, 60_000, 5 * 3_600_000, 9 * 3_600_000), values, strict=True)
    ]
    database.executemany(
        """INSERT INTO "telemetry_v1.vllm"
           (cluster, service, job_id, name, kind, value, resource_attributes_json,
            attributes_json, timestamp_ms, seq) VALUES
           ('cw-a', 'vllm', '/other-serve', ?, 'gauge', ?, '{}', ?, ?, 0)""",
        [
            (
                name,
                value,
                json.dumps(
                    {
                        "source_temporality": "cumulative_snapshot",
                        **labels,
                    }
                ),
                timestamp,
            )
            for name, value, timestamp, labels in rows
        ],
    )
    source = Record(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=lambda sql, *, max_rows: database.execute(sql).fetch_arrow_table(),
    )
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    params = {
        "identity_kind": "job_id",
        "identity": "/other-serve",
        "from": 0,
        "bucket_ms": 15_000,
        "view": "diagnostic_status",
    }
    with TestClient(app) as client:
        short = client.get("/finelog/marin/v1/vllm/overview", params={**params, "to": 2 * 3_600_000})
        partial = client.get("/finelog/marin/v1/vllm/overview", params={**params, "to": 8 * 3_600_000})
        complete = client.get("/finelog/marin/v1/vllm/overview", params={**params, "to": 10 * 3_600_000})
    assert short.status_code == partial.status_code == complete.status_code == 200
    assert [row["status"] for row in short.json()] == ["detail", "check_count_gap"]
    assert (
        short.json()[1]["ttft_observations"],
        short.json()[1]["request_success_finishes"],
        short.json()[1]["value"],
    ) == (2, 0, 2)
    assert [row["status"] for row in partial.json()] == ["summary_only", "check_count_gap"]
    assert (
        partial.json()[1]["ttft_observations"],
        partial.json()[1]["request_success_finishes"],
        partial.json()[1]["value"],
    ) == (2, 1, 1)
    assert [row["status"] for row in complete.json()] == ["summary_only", "no_conclusion"]
    assert (
        complete.json()[1]["ttft_observations"],
        complete.json()[1]["request_success_finishes"],
        complete.json()[1]["value"],
    ) == (2, 2, 0)


def test_sample_budget_returns_cached_error_instead_of_partial_panels():
    points = pa.ListArray.from_arrays(
        pa.array([0, 1_000_001]), pa.repeat(pa.scalar({"timestamp_ms": 0, "seq": 0, "value": 0.0}), 1_000_001)
    )
    series = pa.table({"points": points})
    queries = []

    def query(sql, *, max_rows):
        queries.append(sql)
        return series

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    params = {"identity_kind": "job_id", "identity": "/dense", "from": 0, "to": 300_000, "bucket_ms": 15_000}
    with TestClient(app) as client:
        for view in ("token_rate", "engine_summary"):
            response = client.get("/finelog/marin/v1/vllm/overview", params={**params, "view": view})
            assert response.status_code == 200
            assert response.json() == []
        status = client.get("/finelog/marin/v1/vllm/overview", params={**params, "view": "diagnostic_status"})
        assert [row["status"] for row in status.json()] == ["sample_limit", "unavailable"]
    assert len(queries) == 1


def test_finelog_timeout_is_visible_and_reuses_cached_failure():
    queries = []

    def query(sql, *, max_rows):
        queries.append(sql)
        error = StatsError("query failed")
        error.__cause__ = ConnectError(Code.DEADLINE_EXCEEDED, "deadline exceeded")
        raise error

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    params = {"identity_kind": "job_id", "identity": "/slow", "from": 0, "to": 300_000, "bucket_ms": 15_000}
    with TestClient(app) as client:
        tokens = client.get("/finelog/marin/v1/vllm/overview", params={**params, "view": "token_rate"})
        status = client.get("/finelog/marin/v1/vllm/overview", params={**params, "view": "diagnostic_status"})
    assert status.status_code == tokens.status_code == 200
    assert [row["status"] for row in status.json()] == ["query_timeout", "unavailable"]
    assert tokens.json() == []
    assert len(queries) == 1
