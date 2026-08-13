# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave costs — usage-based estimate.

CoreWeave exposes **no dollar-denominated billing API**; billed amounts live
only in the Billing Insights console. What is available programmatically is a
Prometheus-compatible usage API at ``observe.coreweave.com`` exposing
``billing:*`` metrics (instance counts, object-storage bytes, ...). This
backend reads those usage series over the window and multiplies them by an
operator-supplied **rate card** to produce *estimated* costs. Rows are tagged
``amount_kind="estimated"`` and will not reconcile exactly with the invoice
(contracts, discounts, taxes, billing-cycle effects).

Each rate-card entry is::

    {category, query, unit_rate, unit_divisor, detail_label, region_label, usage_unit}

``query`` is PromQL returning an instantaneous usage quantity (e.g. instance
count). Sampling at ``step_seconds`` and summing ``value * step_hours`` over a
UTC day approximates resource-hours, which times ``unit_rate`` ($/unit/hour)
gives the estimated daily cost. Series are grouped by ``detail_label``.
"""

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

from scripts.cost_manager.cost_event import AmountKind, CostEvent, CostFetchError, DateWindow, cost_event, require_env

logger = logging.getLogger(__name__)

PROVIDER = "coreweave"
DEFAULT_PROMETHEUS_URL = "https://observe.coreweave.com"
DEFAULT_STEP_SECONDS = 3600
REQUEST_TIMEOUT = 60.0
# observe.coreweave.com sits behind Cloudflare, which rejects non-browser
# clients; present a browser User-Agent to get past the bot challenge.
_BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


@dataclass(frozen=True)
class _DailyUsage:
    """Integrated usage and the last gauge sample for one UTC day."""

    unit_hours: float
    last_value: float


def fetch(config: Mapping[str, Any], window: DateWindow) -> list[CostEvent]:
    token = require_env(
        config.get("api_token_env", "COREWEAVE_API_TOKEN"),
        provider=PROVIDER,
        purpose="CoreWeave token with the Observability Viewer role",
    )
    base_url = str(config.get("prometheus_url", DEFAULT_PROMETHEUS_URL)).rstrip("/")
    step_seconds = int(config.get("step_seconds", DEFAULT_STEP_SECONDS))
    rate_card = list(config.get("rate_card", []))
    if not rate_card:
        raise CostFetchError(
            "coreweave: config.rate_card is empty — CoreWeave has no dollar API, so a "
            "{category, query, unit_rate, detail_label} rate card is required to estimate cost"
        )

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": _BROWSER_USER_AGENT})
    step_hours = step_seconds / 3600.0

    window_days = set(window.days())
    events: list[CostEvent] = []
    for entry in rate_card:
        series = _query_range(session, base_url, entry["query"], window, step_seconds)
        daily_usage = _daily_usage(
            series,
            window_days=window_days,
            detail_label=entry.get("detail_label"),
            region_label=entry.get("region_label"),
            step_hours=step_hours,
        )
        unit_rate = float(entry["unit_rate"])
        unit_divisor = float(entry.get("unit_divisor", 1.0))
        if unit_divisor <= 0:
            raise CostFetchError(f"coreweave: unit_divisor must be positive, got {unit_divisor}")
        usage_unit = entry.get("usage_unit")
        ordered_usage = sorted(daily_usage.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or ""))
        for (day, detail, region), usage in ordered_usage:
            events.append(
                cost_event(
                    provider=PROVIDER,
                    day=day,
                    category=str(entry["category"]),
                    detail=detail,
                    cost=usage.unit_hours / unit_divisor * unit_rate,
                    amount_kind=AmountKind.ESTIMATED,
                    region=region,
                    usage_amount=usage.last_value if usage_unit else None,
                    usage_unit=str(usage_unit) if usage_unit else None,
                )
            )
    if not events:
        raise CostFetchError(f"coreweave: no usage series for {window.start}..{window.end}")
    logger.info("coreweave: estimated %d cost rows for %s..%s", len(events), window.start, window.end)
    return events


def _query_range(
    session: requests.Session, base_url: str, query: str, window: DateWindow, step_seconds: int
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "start": int(window.start_dt.timestamp()),
        "end": int(window.end_exclusive_dt.timestamp()),
        "step": step_seconds,
    }
    response = session.get(f"{base_url}/api/v1/query_range", params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code in (401, 403):
        raise CostFetchError(
            f"coreweave: {response.status_code} from {base_url} — the token may lack the "
            f"Observability Viewer role, or Cloudflare blocked the request: {response.text[:200]}"
        )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise CostFetchError(f"coreweave: query_range returned status {payload.get('status')!r} for {query!r}")
    return payload.get("data", {}).get("result", []) or []


def _series_label(labels: Mapping[str, Any], label: str | None, *, default: str | None) -> str | None:
    if label is None:
        return default
    if label not in labels:
        raise CostFetchError(f"coreweave: response series has no {label!r} label")
    return str(labels[label])


def _daily_usage(
    series: list[dict[str, Any]],
    *,
    window_days: set[dt.date],
    detail_label: str | None,
    region_label: str | None,
    step_hours: float,
) -> dict[tuple[dt.date, str, str | None], _DailyUsage]:
    """Integrate samples and keep the last gauge value for each UTC day.

    Prometheus ``query_range`` treats both range endpoints as inclusive, so a
    window ending at midnight returns the next day's first sample; samples whose
    day falls outside ``window_days`` are dropped so no out-of-window row is
    emitted.
    """
    totals: dict[tuple[dt.date, str, str | None], float] = defaultdict(float)
    latest: dict[tuple[dt.date, str, str | None], tuple[float, float]] = {}
    for item in series:
        labels = item.get("metric", {})
        detail = _series_label(labels, detail_label, default="total")
        assert detail is not None
        region = _series_label(labels, region_label, default=None)
        for sample_ts, raw_value in item.get("values", []) or []:
            timestamp = float(sample_ts)
            day = dt.datetime.fromtimestamp(timestamp, tz=dt.UTC).date()
            if day not in window_days:
                continue
            value = float(raw_value)
            key = (day, detail, region)
            totals[key] += value * step_hours
            if key not in latest or timestamp > latest[key][0]:
                latest[key] = (timestamp, value)
    return {key: _DailyUsage(unit_hours=total, last_value=latest[key][1]) for key, total in totals.items()}
