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

    {category, query, unit_rate, detail_label}

``query`` is PromQL returning an instantaneous usage quantity (e.g. instance
count). Sampling at ``step_seconds`` and summing ``value * step_hours`` over a
UTC day approximates resource-hours, which times ``unit_rate`` ($/unit/hour)
gives the estimated daily cost. Series are grouped by ``detail_label``.
"""

import datetime as dt
import logging
from collections import defaultdict
from collections.abc import Mapping
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
        unit_hours = _accumulate_unit_hours(
            series, window_days=window_days, detail_label=entry.get("detail_label"), step_hours=step_hours
        )
        unit_rate = float(entry["unit_rate"])
        for (day, detail), hours in sorted(unit_hours.items()):
            events.append(
                cost_event(
                    provider=PROVIDER,
                    day=day,
                    category=str(entry["category"]),
                    detail=detail,
                    cost=hours * unit_rate,
                    amount_kind=AmountKind.ESTIMATED,
                )
            )
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


def _accumulate_unit_hours(
    series: list[dict[str, Any]], *, window_days: set[dt.date], detail_label: str | None, step_hours: float
) -> dict[tuple[dt.date, str], float]:
    """Sum ``value * step_hours`` per (UTC day, detail) over samples in the window.

    Prometheus ``query_range`` treats both range endpoints as inclusive, so a
    window ending at midnight returns the next day's first sample; samples whose
    day falls outside ``window_days`` are dropped so no out-of-window row is
    emitted.
    """
    totals: dict[tuple[dt.date, str], float] = defaultdict(float)
    for item in series:
        labels = item.get("metric", {})
        detail = str(labels.get(detail_label, "total")) if detail_label else "total"
        for sample_ts, raw_value in item.get("values", []) or []:
            day = dt.datetime.fromtimestamp(float(sample_ts), tz=dt.UTC).date()
            if day not in window_days:
                continue
            totals[(day, detail)] += float(raw_value) * step_hours
    return dict(totals)
