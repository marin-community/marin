# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic organization costs.

Reads the Anthropic Admin Cost Report API
(``GET /v1/organizations/cost_report``). Requires an **Admin API key**
(``sk-ant-admin01-...``); a standard key is rejected.

Two gotchas baked in here:

- ``amount`` is a decimal **string in the lowest currency unit** (USD cents),
  so it is divided by 100 to get dollars.
- bucket boundaries live on the bucket (``starting_at``), not inside each
  result.

Priority-tier spend is excluded from this endpoint (it is reported via the
usage endpoint), so totals here track on-demand cost.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import requests

from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow, cost_event, require_env

logger = logging.getLogger(__name__)

COST_REPORT_URL = "https://api.anthropic.com/v1/organizations/cost_report"
ANTHROPIC_VERSION = "2023-06-01"
PROVIDER = "anthropic"
# The cost endpoint caps a single page at 31 daily buckets.
MAX_BUCKETS = 31
# Amounts are reported in cents; convert to dollars.
CENTS_PER_DOLLAR = Decimal(100)
REQUEST_TIMEOUT = 30.0


def fetch(config: Mapping[str, Any], window: DateWindow) -> list[CostEvent]:
    api_key = require_env(
        config.get("api_key_env", "ANTHROPIC_ADMIN_KEY"),
        provider=PROVIDER,
        purpose="organization Admin API key (sk-ant-admin01-...)",
    )
    base_url = config.get("base_url", COST_REPORT_URL)
    group_by = list(config.get("group_by", ["description"]))

    params: dict[str, Any] = {
        "starting_at": _rfc3339(window.start_dt),
        "ending_at": _rfc3339(window.end_exclusive_dt),
        "bucket_width": "1d",
        "limit": min(len(window.days()), MAX_BUCKETS),
    }
    if group_by:
        params["group_by[]"] = group_by

    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    session = requests.Session()
    events: list[CostEvent] = []
    page: str | None = None
    while True:
        if page:
            params["page"] = page
        payload = _get(session, base_url, headers, params)
        events.extend(_buckets_to_events(payload.get("data", []) or []))
        if not payload.get("has_more"):
            break
        page = payload.get("next_page")
        if not page:
            break
    logger.info("anthropic: fetched %d cost rows for %s..%s", len(events), window.start, window.end)
    return events


def _rfc3339(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(session: requests.Session, url: str, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code in (401, 403):
        raise CostFetchError(
            f"anthropic: {response.status_code} from Cost Report API — the key is likely "
            f"not an org Admin key: {response.text[:200]}"
        )
    response.raise_for_status()
    return response.json()


def _buckets_to_events(buckets: list[dict[str, Any]]) -> list[CostEvent]:
    events: list[CostEvent] = []
    for bucket in buckets:
        day = dt.datetime.fromisoformat(bucket["starting_at"]).astimezone(dt.UTC).date()
        for result in bucket.get("results", []) or []:
            raw_amount = result.get("amount")
            if raw_amount is None:
                continue
            dollars = float(Decimal(str(raw_amount)) / CENTS_PER_DOLLAR)
            detail = result.get("description") or result.get("model") or result.get("workspace_id") or "total"
            events.append(
                cost_event(
                    provider=PROVIDER,
                    day=day,
                    category="api",
                    detail=str(detail),
                    cost=dollars,
                    currency=str(result.get("currency", "USD")).upper(),
                )
            )
    return events
