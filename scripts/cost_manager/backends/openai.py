# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI organization costs.

Reads the OpenAI Costs API (``GET /v1/organization/costs``), which returns
daily cost buckets for the whole organization. Requires an **organization
Admin API key** (a project ``sk-proj-...`` key is rejected); the key's owner
needs the dashboard "Usage" permission.

Each daily bucket carries one or more results; grouping by ``line_item`` splits
a day into per-feature rows (e.g. a model's input/output tokens), which become
the :class:`CostEvent` ``detail``. Amounts are already in dollars.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping
from typing import Any

import requests

from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow, cost_event, require_env

logger = logging.getLogger(__name__)

COSTS_URL = "https://api.openai.com/v1/organization/costs"
PROVIDER = "openai"
# The API caps a single page at 180 daily buckets.
MAX_BUCKETS = 180
REQUEST_TIMEOUT = 30.0


def fetch(config: Mapping[str, Any], window: DateWindow) -> list[CostEvent]:
    api_key = require_env(
        config.get("api_key_env", "OPENAI_ADMIN_KEY"),
        provider=PROVIDER,
        purpose="organization Admin API key with the 'Usage' permission",
    )
    base_url = config.get("base_url", COSTS_URL)
    group_by = list(config.get("group_by", ["line_item"]))

    params: dict[str, Any] = {
        "start_time": int(window.start_dt.timestamp()),
        "end_time": int(window.end_exclusive_dt.timestamp()),
        "bucket_width": "1d",
        "limit": min(len(window.days()), MAX_BUCKETS),
    }
    if group_by:
        params["group_by"] = group_by

    headers = {"Authorization": f"Bearer {api_key}"}
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
    logger.info("openai: fetched %d cost rows for %s..%s", len(events), window.start, window.end)
    return events


def _get(session: requests.Session, url: str, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code in (401, 403):
        raise CostFetchError(
            f"openai: {response.status_code} from Costs API — the key likely lacks org "
            f"Admin/Usage access: {response.text[:200]}"
        )
    response.raise_for_status()
    return response.json()


def _buckets_to_events(buckets: list[dict[str, Any]]) -> list[CostEvent]:
    events: list[CostEvent] = []
    for bucket in buckets:
        day = dt.datetime.fromtimestamp(int(bucket["start_time"]), tz=dt.UTC).date()
        for result in bucket.get("results", []) or []:
            amount = result.get("amount", {})
            value = amount.get("value")
            if value is None:
                continue
            detail = result.get("line_item") or result.get("project_id") or "total"
            events.append(
                cost_event(
                    provider=PROVIDER,
                    day=day,
                    category="api",
                    detail=str(detail),
                    cost=float(value),
                    currency=str(amount.get("currency", "usd")).upper(),
                )
            )
    return events
