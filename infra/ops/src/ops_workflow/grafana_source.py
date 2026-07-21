# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only projection of firing alert instances from Grafana PostgreSQL."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

import psycopg
from psycopg.rows import dict_row

from ops_workflow.grafana import GrafanaAlert

ACTIVE_GRAFANA_STATE = "Alerting"
OPS_RECEIVER = "ops-agent"
SOURCE_VERSION = "grafana-postgres-v1"

ALERT_INSTANCE_QUERY = """
    SELECT
        instance.rule_org_id,
        instance.rule_uid,
        instance.labels,
        instance.labels_hash,
        instance.current_state_since,
        instance.last_eval_time,
        instance.fired_at,
        instance.annotations AS instance_annotations,
        instance.last_result,
        rule.title AS rule_title,
        rule.labels AS rule_labels,
        rule.annotations AS rule_annotations
    FROM public.alert_instance AS instance
    JOIN public.alert_rule AS rule
      ON rule.org_id = instance.rule_org_id AND rule.uid = instance.rule_uid
    WHERE instance.current_state = %s
    ORDER BY instance.rule_org_id, instance.rule_uid, instance.labels_hash
"""


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
    """A complete, successfully read snapshot of firing Grafana instances."""

    observed_at: datetime
    alerts: tuple[PolledGrafanaAlert, ...]


class GrafanaAlertSource(Protocol):
    """Source capable of returning one complete Grafana alert snapshot."""

    async def snapshot(self) -> GrafanaSnapshot:
        """Return all currently firing alert instances."""


class PostgresGrafanaAlertSource:
    """Read active Grafana instances through a dedicated PostgreSQL role."""

    def __init__(self, database_url: str, *, password: str | None = None) -> None:
        self._database_url = database_url
        self._password = password

    async def snapshot(self) -> GrafanaSnapshot:
        connection = await psycopg.AsyncConnection.connect(
            self._database_url,
            password=self._password,
            row_factory=dict_row,
            connect_timeout=10,
            options="-c statement_timeout=10000",
        )
        typed = cast(psycopg.AsyncConnection[dict[str, Any]], connection)
        async with typed:
            async with typed.transaction():
                await typed.execute("SET TRANSACTION READ ONLY")
                cursor = await typed.execute(ALERT_INSTANCE_QUERY, (ACTIVE_GRAFANA_STATE,))
                rows = await cursor.fetchall()
        observed_at = datetime.now(UTC)
        return snapshot_from_rows(rows, observed_at=observed_at)


def snapshot_from_rows(rows: Sequence[Mapping[str, object]], *, observed_at: datetime) -> GrafanaSnapshot:
    """Normalize Grafana's SQL serialization into workflow alert instances."""

    if observed_at.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware")
    alerts = tuple(_alert_from_row(row) for row in rows)
    fingerprints = [item.alert.fingerprint for item in alerts]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Grafana snapshot contains duplicate alert fingerprints")
    return GrafanaSnapshot(observed_at=observed_at.astimezone(UTC), alerts=alerts)


def _alert_from_row(row: Mapping[str, object]) -> PolledGrafanaAlert:
    org_id = _integer(row, "rule_org_id")
    rule_uid = _string(row, "rule_uid")
    alert_name = _string(row, "rule_title")
    labels = {
        **_string_object(row.get("rule_labels"), "rule_labels"),
        **_tuple_labels(row.get("labels")),
        "alertname": alert_name,
        "grafana_rule_uid": rule_uid,
    }
    annotations = {
        **_string_object(row.get("rule_annotations"), "rule_annotations"),
        **_string_object(row.get("instance_annotations"), "instance_annotations"),
    }
    result = _object(row.get("last_result"), "last_result", empty={})
    values = result.get("values", {})
    if not isinstance(values, dict):
        raise ValueError("last_result.values must be an object")

    labels_hash = _string(row, "labels_hash")
    fingerprint = f"{org_id}:{rule_uid}:{labels_hash}"
    starts_at = _optional_unix_time(row.get("fired_at")) or _unix_time(row, "current_state_since")
    cluster = labels.get("cluster", "")
    group_labels = {"alertname": alert_name, "cluster": cluster}
    group_key = json.dumps(group_labels, sort_keys=True, separators=(",", ":"))
    title = f"{alert_name} · {cluster}" if cluster else alert_name
    generator_url = f"https://grafana.oa.dev/alerting/grafana/{rule_uid}/view"
    return PolledGrafanaAlert(
        alert=GrafanaAlert(
            fingerprint=fingerprint,
            status="firing",
            labels=labels,
            annotations=annotations,
            values=values,
            starts_at=starts_at,
            ends_at=None,
            generator_url=generator_url,
            silence_url="",
            dashboard_url="",
            panel_url="",
        ),
        receiver=OPS_RECEIVER,
        group_key=group_key,
        group_labels=group_labels,
        title=title,
    )


def _decoded(value: object, field: str, *, empty: object) -> object:
    if value is None or value == "":
        return empty
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} is not valid JSON") from error
    return value


def _object(value: object, field: str, *, empty: dict[str, object]) -> dict[str, object]:
    decoded = _decoded(value, field, empty=empty)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must be an object")
    return decoded


def _string_object(value: object, field: str) -> dict[str, str]:
    decoded = _object(value, field, empty={})
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in decoded.items()):
        raise ValueError(f"{field} must contain string keys and values")
    return cast(dict[str, str], decoded)


def _tuple_labels(value: object) -> dict[str, str]:
    decoded = _decoded(value, "labels", empty=[])
    if not isinstance(decoded, list):
        raise ValueError("labels must be an array of [name, value] pairs")
    labels: dict[str, str] = {}
    for pair in decoded:
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise ValueError("labels must be an array of [name, value] pairs")
        name, label_value = pair
        if name in labels:
            raise ValueError(f"labels contains duplicate key {name!r}")
        labels[name] = label_value
    return labels


def _string(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _unix_time(row: Mapping[str, object], field: str) -> datetime:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be Unix seconds")
    return datetime.fromtimestamp(value, tz=UTC)


def _optional_unix_time(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("fired_at must be Unix seconds or null")
    return datetime.fromtimestamp(value, tz=UTC)
