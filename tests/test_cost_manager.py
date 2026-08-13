# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from scripts.cost_manager.backends import coreweave
from scripts.cost_manager.cost_event import CostFetchError, DateWindow, cost_event
from scripts.cost_manager.slack_alert import AlertMetric, evaluate_alerts, parse_alert_rules

GIB = 1024**3
TIB = 1024**4
HOT_STORAGE_GIB_HOUR_RATE = 0.06 / 730


@contextmanager
def _prometheus_server(payload: dict) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
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


def test_coreweave_storage_fetch_keeps_daily_bucket_bytes_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    day = dt.date(2026, 8, 12)
    payload = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"bucket_name": "marin-us-east-02a", "region": "US-EAST-02"},
                    "values": [
                        [_timestamp(day, 1), str(2 * GIB)],
                        [_timestamp(day, 0), str(GIB)],
                    ],
                }
            ]
        },
    }
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test-token")
    with _prometheus_server(payload) as server_url:
        events = coreweave.fetch(
            {
                "prometheus_url": server_url,
                "step_seconds": 3600,
                "rate_card": [
                    {
                        "category": "storage",
                        "query": "storage query",
                        "detail_label": "bucket_name",
                        "region_label": "region",
                        "usage_unit": "bytes",
                        "unit_divisor": GIB,
                        "unit_rate": HOT_STORAGE_GIB_HOUR_RATE,
                    }
                ],
            },
            DateWindow(day, day),
        )

    assert len(events) == 1
    event = events[0]
    assert (event.detail, event.region) == ("marin-us-east-02a", "US-EAST-02")
    assert (event.usage_amount, event.usage_unit) == (2 * GIB, "bytes")
    assert event.cost == pytest.approx(3 * HOT_STORAGE_GIB_HOUR_RATE)


def test_coreweave_storage_fetch_fails_when_the_metric_has_no_series(monkeypatch: pytest.MonkeyPatch) -> None:
    day = dt.date(2026, 8, 12)
    monkeypatch.setenv("COREWEAVE_API_TOKEN", "test-token")
    with _prometheus_server({"status": "success", "data": {"result": []}}) as server_url:
        with pytest.raises(CostFetchError, match="no usage series"):
            coreweave.fetch(
                {
                    "prometheus_url": server_url,
                    "rate_card": [{"category": "storage", "query": "storage query", "unit_rate": 1}],
                },
                DateWindow(day, day),
            )


def test_storage_alert_uses_current_day_and_bucket_filter() -> None:
    today = dt.date(2026, 8, 13)
    rules = parse_alert_rules(
        [
            {
                "name": "coreweave-east-storage",
                "metric": "usage_amount",
                "threshold": 80 * TIB,
                "provider": "coreweave",
                "category": "storage",
                "detail": "marin-us-east-02a",
                "window": "current_day",
            }
        ]
    )
    events = [
        cost_event(
            provider="coreweave",
            day=today,
            category="storage",
            detail="marin-us-east-02a",
            cost=1.0,
            usage_amount=81 * TIB,
            usage_unit="bytes",
        ),
        cost_event(
            provider="coreweave",
            day=today - dt.timedelta(days=1),
            category="storage",
            detail="marin-us-east-02a",
            cost=1.0,
            usage_amount=10 * TIB,
            usage_unit="bytes",
        ),
        cost_event(
            provider="coreweave",
            day=today,
            category="storage",
            detail="marin-us-west-04a",
            cost=1.0,
            usage_amount=90 * TIB,
            usage_unit="bytes",
        ),
    ]

    (breach,) = evaluate_alerts(events, rules, window=DateWindow.trailing(3, today=today), today=today)

    assert breach.metric is AlertMetric.USAGE_AMOUNT
    assert breach.observed_value == 81 * TIB
    assert breach.threshold_value == 80 * TIB
    assert breach.unit == "bytes"
    assert breach.scope == "coreweave / storage / marin-us-east-02a"


def test_cost_alert_sums_the_configured_window() -> None:
    today = dt.date(2026, 8, 13)
    rules = parse_alert_rules(
        [{"name": "openai-window", "metric": "cost", "threshold": 5, "provider": "openai", "window": "window_total"}]
    )
    events = [
        cost_event(provider="openai", day=today - dt.timedelta(days=1), category="api", detail="tokens", cost=3),
        cost_event(provider="openai", day=today, category="api", detail="tokens", cost=4),
        cost_event(provider="anthropic", day=today, category="api", detail="tokens", cost=20),
    ]

    (breach,) = evaluate_alerts(events, rules, window=DateWindow.trailing(2, today=today), today=today)

    assert breach.metric is AlertMetric.COST
    assert breach.observed_value == 7
    assert breach.threshold_value == 5
    assert breach.unit == "USD"
