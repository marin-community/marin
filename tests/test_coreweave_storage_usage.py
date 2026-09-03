# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from finelog.client import schema_from_dataclass

from scripts.ops.storage.coreweave_usage import StorageMetric, StorageUsage, StorageUsageError, collect_storage_usage

TIB = 1024**4


def test_storage_usage_is_a_finelog_schema() -> None:
    schema = schema_from_dataclass(StorageUsage)

    assert schema.key_column == "zone"
    assert {column.name for column in schema.columns} == {
        "provider",
        "metric",
        "zone",
        "bucket",
        "storage_class",
        "value_bytes",
        "observed_at",
        "collected_at",
    }


@contextmanager
def _prometheus_server(usage_payload: dict, quota_payload: dict) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlsplit(self.path).query)["query"][0]
            payload = quota_payload if "cwobject_quota_info" in query else usage_payload
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _payload(metric: dict[str, str], timestamp: dt.datetime, value: float) -> dict:
    return {
        "status": "success",
        "data": {"result": [{"metric": metric, "value": [timestamp.timestamp(), str(value)]}]},
    }


def test_collect_storage_usage_records_bucket_usage_and_live_zone_quota() -> None:
    collected_at = dt.datetime(2026, 8, 15, 2, tzinfo=dt.UTC)
    usage_observed_at = collected_at - dt.timedelta(minutes=25)
    quota_observed_at = collected_at - dt.timedelta(minutes=10)
    usage_payload = _payload(
        {"bucket_name": "marin-us-east-02a", "zone": "US-EAST-02A", "storage_class": "STANDARD"},
        usage_observed_at,
        700 * TIB,
    )
    quota_payload = _payload(
        {"quota_zone": "US-EAST-02A", "storage_class": "STANDARD"},
        quota_observed_at,
        900 * TIB,
    )

    with _prometheus_server(usage_payload, quota_payload) as server_url:
        samples = collect_storage_usage(requests.Session(), server_url, collected_at)

    samples_by_metric = {sample.metric: sample for sample in samples}
    usage = samples_by_metric[StorageMetric.USED_BYTES]
    assert (usage.zone, usage.bucket, usage.storage_class) == (
        "US-EAST-02A",
        "marin-us-east-02a",
        "STANDARD",
    )
    assert (usage.value_bytes, usage.observed_at, usage.collected_at) == (
        700 * TIB,
        usage_observed_at,
        collected_at,
    )

    quota = samples_by_metric[StorageMetric.QUOTA_BYTES]
    assert (quota.zone, quota.bucket, quota.storage_class) == ("US-EAST-02A", None, "STANDARD")
    assert (quota.value_bytes, quota.observed_at, quota.collected_at) == (
        900 * TIB,
        quota_observed_at,
        collected_at,
    )


def test_collect_storage_usage_fails_when_a_usage_zone_has_no_quota() -> None:
    collected_at = dt.datetime(2026, 8, 15, 2, tzinfo=dt.UTC)
    usage_payload = _payload(
        {"bucket_name": "marin-us-east-02a", "zone": "US-EAST-02A", "storage_class": "STANDARD"},
        collected_at,
        700 * TIB,
    )
    quota_payload = {"status": "success", "data": {"result": []}}

    with _prometheus_server(usage_payload, quota_payload) as server_url:
        with pytest.raises(StorageUsageError, match=r"quota.*US-EAST-02A"):
            collect_storage_usage(requests.Session(), server_url, collected_at)
