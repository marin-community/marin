# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.cost_manager import run as cost_manager_run
from scripts.cost_manager.backends import coreweave
from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow

GIB = 1024**3
TIB = 1024**4
HOT_STORAGE_GIB_HOUR_RATE = 0.06 / 730


def test_provider_enabled_env_controls_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_name = "test-provider"
    token_env = "TEST_PROVIDER_TOKEN"
    calls: list[str] = []

    def fetch(config: Mapping[str, Any], _window: DateWindow) -> list[CostEvent]:
        calls.append(config["name"])
        return []

    monkeypatch.setitem(cost_manager_run.BACKENDS, provider_name, fetch)
    provider = {"name": provider_name, "enabled": True, "enabled_env": token_env}
    window = DateWindow.trailing(1, today=dt.date(2026, 8, 13))

    monkeypatch.delenv(token_env, raising=False)
    assert cost_manager_run._run_backends([provider], window, set()) == ([], [])
    assert calls == []

    monkeypatch.setenv(token_env, "token")
    assert cost_manager_run._run_backends([provider], window, set()) == ([], [])
    assert calls == [provider_name]


@contextmanager
def _prometheus_server(payload_by_query: Mapping[str, dict]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            query = parse_qs(urlsplit(self.path).query)["query"][0]
            payload = payload_by_query[query]
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


def _timestamp(day: dt.date, hour: int) -> float:
    return dt.datetime.combine(day, dt.time(hour), tzinfo=dt.UTC).timestamp()


def test_coreweave_storage_fetch_keeps_daily_bucket_bytes_quota_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    day = dt.date(2026, 8, 12)
    storage_payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"bucket_name": "marin-us-east-02a", "zone": "US-EAST-02A"},
                    "values": [
                        [_timestamp(day, 1), str(2 * GIB)],
                        [_timestamp(day, 0), str(GIB)],
                    ],
                }
            ]
        },
    }
    quota_payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"quota_zone": "US-EAST-02A", "storage_class": "STANDARD"},
                    "values": [[_timestamp(day, 1), str(900 * TIB)]],
                }
            ]
        },
    }
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test-token")
    with _prometheus_server({"storage query": storage_payload, "quota query": quota_payload}) as server_url:
        events = coreweave.fetch(
            {
                "prometheus_url": server_url,
                "step_seconds": 3600,
                "rate_card": [
                    {
                        "category": "storage",
                        "query": "storage query",
                        "detail_label": "bucket_name",
                        "region_label": "zone",
                        "usage_unit": "bytes",
                        "unit_divisor": GIB,
                        "unit_rate": HOT_STORAGE_GIB_HOUR_RATE,
                    },
                    {
                        "category": "storage_quota",
                        "query": "quota query",
                        "detail_label": "storage_class",
                        "region_label": "quota_zone",
                        "usage_unit": "bytes",
                        "unit_rate": 0,
                    },
                ],
            },
            DateWindow(day, day),
        )

    assert len(events) == 2
    events_by_category = {event.category: event for event in events}
    storage = events_by_category["storage"]
    assert (storage.detail, storage.region) == ("marin-us-east-02a", "US-EAST-02A")
    assert (storage.usage_amount, storage.usage_unit) == (2 * GIB, "bytes")
    assert storage.cost == pytest.approx(3 * HOT_STORAGE_GIB_HOUR_RATE)

    quota = events_by_category["storage_quota"]
    assert (quota.detail, quota.region) == ("STANDARD", "US-EAST-02A")
    assert (quota.usage_amount, quota.usage_unit) == (900 * TIB, "bytes")
    assert quota.cost == 0


def test_coreweave_storage_fetch_fails_when_a_required_metric_has_no_series(monkeypatch: pytest.MonkeyPatch) -> None:
    day = dt.date(2026, 8, 12)
    storage_payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"bucket_name": "marin-us-east-02a", "zone": "US-EAST-02A"},
                    "values": [[_timestamp(day, 1), str(2 * GIB)]],
                }
            ]
        },
    }
    empty_payload = {"status": "success", "data": {"result": []}}
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test-token")
    with _prometheus_server({"storage query": storage_payload, "quota query": empty_payload}) as server_url:
        with pytest.raises(CostFetchError, match=r"storage_quota.*no usage series"):
            coreweave.fetch(
                {
                    "prometheus_url": server_url,
                    "rate_card": [
                        {
                            "category": "storage",
                            "query": "storage query",
                            "detail_label": "bucket_name",
                            "region_label": "zone",
                            "unit_rate": 1,
                        },
                        {
                            "category": "storage_quota",
                            "query": "quota query",
                            "detail_label": "storage_class",
                            "region_label": "quota_zone",
                            "unit_rate": 0,
                        },
                    ],
                },
                DateWindow(day, day),
            )
