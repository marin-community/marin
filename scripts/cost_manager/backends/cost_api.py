# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared client for the cursor-paginated provider cost APIs.

OpenAI's Costs API and Anthropic's Cost Report share the same wire contract: a
GET returns a page of daily buckets plus ``has_more`` / ``next_page``, and the
prior ``next_page`` is passed back as ``page``. This module owns the GET (with a
clear error on auth/permission failures) and the page loop, so each backend
only maps its buckets to :class:`CostEvent`.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import requests

from scripts.cost_manager.cost_event import CostFetchError

DEFAULT_TIMEOUT = 30.0


def get_json(
    session: requests.Session,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    *,
    provider: str,
    api_label: str,
    permission_hint: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """GET ``url`` and return JSON; raise :class:`CostFetchError` on 401/403."""
    response = session.get(url, headers=dict(headers), params=dict(params), timeout=timeout)
    if response.status_code in (401, 403):
        raise CostFetchError(
            f"{provider}: {response.status_code} from {api_label} — {permission_hint}: {response.text[:200]}"
        )
    response.raise_for_status()
    return response.json()


def paginate(
    session: requests.Session,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, Any],
    *,
    provider: str,
    api_label: str,
    permission_hint: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[dict[str, Any]]:
    """Yield each response page, following the ``page`` cursor to exhaustion."""
    page_params = dict(params)
    page: str | None = None
    while True:
        if page:
            page_params["page"] = page
        payload = get_json(
            session,
            url,
            headers,
            page_params,
            provider=provider,
            api_label=api_label,
            permission_hint=permission_hint,
            timeout=timeout,
        )
        yield payload
        if not payload.get("has_more"):
            return
        page = payload.get("next_page")
        if not page:
            return
