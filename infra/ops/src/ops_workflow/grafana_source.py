# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only projection of active alerts from Grafana's Alertmanager API."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx
from rigging.auth import TokenProvider

from ops_workflow.grafana import GrafanaAlert, grafana_group_metadata

GRAFANA_ALERTS_PATH = "/api/alertmanager/grafana/api/v2/alerts"
OPS_RECEIVER = "ops-agent"
SOURCE_VERSION = "grafana-api-v1"


@dataclass(frozen=True)
class PolledGrafanaAlert:
    """One firing Grafana instance with deterministic workflow grouping."""

    alert: GrafanaAlert
    receiver: str
    group_key: str
    group_labels: Mapping[str, str]
    title: str


@dataclass(frozen=True)
class GrafanaSnapshot:
    """A complete, successfully read snapshot of active Grafana instances."""

    observed_at: datetime
    alerts: tuple[PolledGrafanaAlert, ...]


class GrafanaAlertSource(Protocol):
    """Source capable of returning one complete Grafana alert snapshot."""

    async def snapshot(self) -> GrafanaSnapshot:
        """Return all currently active alert instances."""


class GrafanaApiAlertSource:
    """Read active alerts from Grafana through an IAP-authenticated API request."""

    def __init__(
        self,
        base_url: str,
        token_provider: TokenProvider,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}{GRAFANA_ALERTS_PATH}"
        self._token_provider = token_provider
        self._transport = transport

    async def snapshot(self) -> GrafanaSnapshot:
        token = await asyncio.to_thread(self._token_provider.get_token)
        if not token:
            raise RuntimeError("IAP token provider returned no token")
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=10,
            follow_redirects=False,
        ) as client:
            response = await client.get(self._url, headers={"Proxy-Authorization": f"Bearer {token}"})
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("Grafana alerts response must be an array")
        return snapshot_from_api_alerts(payload, observed_at=datetime.now(UTC))


def snapshot_from_api_alerts(alerts: Sequence[object], *, observed_at: datetime) -> GrafanaSnapshot:
    """Normalize Grafana Alertmanager API alerts into workflow instances."""

    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    normalized = tuple(_alert_from_api(item) for item in alerts)
    fingerprints = [item.alert.fingerprint for item in normalized]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Grafana snapshot contains duplicate alert fingerprints")
    return GrafanaSnapshot(observed_at=observed_at.astimezone(UTC), alerts=normalized)


def _alert_from_api(value: object) -> PolledGrafanaAlert:
    if not isinstance(value, Mapping):
        raise ValueError("Grafana alert must be an object")
    labels = _string_map(value.get("labels"), "labels")
    annotations = _string_map(value.get("annotations"), "annotations")
    alert_name = labels.get("alertname")
    if not alert_name:
        raise ValueError("Grafana alert labels must include alertname")
    fingerprint = _string(value.get("fingerprint"), "fingerprint")
    generator_url = value.get("generatorURL", "")
    if not isinstance(generator_url, str):
        raise ValueError("generatorURL must be a string")
    starts_at = _timestamp(value.get("startsAt"), "startsAt")
    cluster = labels.get("cluster", "")
    group = grafana_group_metadata(alert_name, cluster)
    group_key = json.dumps(group.labels, sort_keys=True, separators=(",", ":"))
    return PolledGrafanaAlert(
        alert=GrafanaAlert(
            fingerprint=fingerprint,
            status="firing",
            labels=labels,
            annotations=annotations,
            values={},
            starts_at=starts_at,
            ends_at=None,
            generator_url=generator_url,
            silence_url="",
            dashboard_url="",
            panel_url="",
        ),
        receiver=OPS_RECEIVER,
        group_key=group_key,
        group_labels=group.labels,
        title=group.title,
    )


def _string_map(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ValueError(f"{field} must contain string keys and values")
    return dict(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _timestamp(value: object, field: str) -> datetime:
    text = _string(value, field)
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.astimezone(UTC)
