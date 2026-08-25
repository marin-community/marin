# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared GraphQL transport and upstream-error mapping for bridge sources."""

import httpx
from errors import UpstreamError


def graphql_data(
    client: httpx.Client,
    *,
    source: str,
    url: str,
    query: str,
    variables: dict,
) -> dict:
    """Post a GraphQL query and return its data payload."""
    try:
        response = client.post(url, json={"query": query, "variables": variables})
    except httpx.TransportError as err:
        raise UpstreamError(source, f"graphql unreachable ({err})", status_code=504) from err
    if response.status_code != 200:
        raise UpstreamError(source, f"graphql returned {response.status_code}", status_code=502)
    payload = response.json()
    if payload.get("errors"):
        raise UpstreamError(source, f"graphql errors: {payload['errors']}", status_code=502)
    if not payload.get("data"):
        raise UpstreamError(source, "graphql returned no data", status_code=502)
    return payload["data"]
