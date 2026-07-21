# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalized Grafana alert delivery types used by the polling adapter."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

GRAFANA_BASE_URL = "https://grafana.oa.dev"


@dataclass(frozen=True)
class GrafanaGroupMetadata:
    """Labels and title shared by firing and resolved group projections."""

    labels: Mapping[str, str]
    title: str


def grafana_group_metadata(alert_name: str, cluster: str) -> GrafanaGroupMetadata:
    """Build the stable workflow presentation for one Grafana alert group."""

    return GrafanaGroupMetadata(
        labels={"alertname": alert_name, "cluster": cluster},
        title=f"{alert_name} · {cluster}" if cluster else alert_name,
    )


@dataclass(frozen=True)
class GrafanaAlert:
    """One alert instance from a Grafana-owned group."""

    fingerprint: str
    status: str
    labels: Mapping[str, str]
    annotations: Mapping[str, str]
    values: Mapping[str, object]
    starts_at: datetime
    ends_at: datetime | None
    generator_url: str
    silence_url: str
    dashboard_url: str
    panel_url: str

    @property
    def alert_name(self) -> str:
        return self.labels.get("alertname", "UnnamedAlert")

    @property
    def severity(self) -> str:
        return self.labels.get("severity", "warning").lower()

    @property
    def summary(self) -> str:
        return self.annotations.get("summary") or self.annotations.get("description") or self.alert_name


@dataclass(frozen=True)
class GrafanaNotification:
    """One grouped firing or resolved projection from Grafana."""

    receiver: str
    status: str
    org_id: int
    version: str
    group_key: str
    group_labels: Mapping[str, str]
    common_labels: Mapping[str, str]
    common_annotations: Mapping[str, str]
    external_url: str
    title: str
    message: str
    alerts: tuple[GrafanaAlert, ...]


@dataclass(frozen=True)
class GrafanaDelivery:
    """A normalized, deterministic delivery derived from a SQL poll."""

    notification: GrafanaNotification
    source_timestamp: datetime
    delivery_key: str
    body_sha256: str
    normalized_payload: Mapping[str, object]
