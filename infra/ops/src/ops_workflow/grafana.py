# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authentication and normalization for Grafana Alerting webhooks."""

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MAX_BODY_BYTES = 1024 * 1024
MAX_ALERTS = 5_000
MAX_FIELD_BYTES = 16 * 1024
# Cloud Tasks may retry an authenticated internal delivery for one day. Keep a
# narrow future-clock allowance while accepting delayed, HMAC-covered tasks.
MAX_DELIVERY_DELAY_SECONDS = 25 * 60 * 60
MAX_FUTURE_SKEW_SECONDS = 5 * 60
SIGNATURE_HEADER = "x-grafana-alerting-signature"
TIMESTAMP_HEADER = "x-grafana-alerting-signature-timestamp"


class GrafanaWebhookError(ValueError):
    """A stable webhook authentication or payload rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GrafanaAlert:
    """One alert instance from a Grafana notification group."""

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
    """One grouped firing or resolved delivery from Grafana Alertmanager."""

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
class VerifiedGrafanaWebhook:
    """A verified delivery with deterministic retry identity."""

    notification: GrafanaNotification
    source_timestamp: datetime
    delivery_key: str
    body_sha256: str
    normalized_payload: Mapping[str, object]


def verify_grafana_webhook(
    raw_body: bytes,
    *,
    signature: str,
    timestamp: str,
    secret: bytes,
    now: datetime,
) -> VerifiedGrafanaWebhook:
    """Verify Grafana's timestamped HMAC and normalize its default JSON payload."""

    if len(raw_body) > MAX_BODY_BYTES:
        raise GrafanaWebhookError("payload_too_large", "webhook exceeds the 1 MiB limit")
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise GrafanaWebhookError("invalid_signature", "signature must be lowercase hexadecimal SHA-256")
    source_timestamp = _unix_timestamp(timestamp)
    current = _to_utc(now, "now")
    delivery_age = (current - source_timestamp).total_seconds()
    if delivery_age < -MAX_FUTURE_SKEW_SECONDS or delivery_age > MAX_DELIVERY_DELAY_SECONDS:
        raise GrafanaWebhookError("replay", "webhook timestamp is outside the replay window")
    signed = timestamp.encode() + b":" + raw_body
    expected = hmac.new(secret, signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise GrafanaWebhookError("invalid_signature", "signature does not match")

    normalized = _payload(raw_body)
    body_sha256 = hashlib.sha256(raw_body).hexdigest()
    return VerifiedGrafanaWebhook(
        notification=_notification(normalized),
        source_timestamp=source_timestamp,
        delivery_key=body_sha256,
        body_sha256=body_sha256,
        normalized_payload=normalized,
    )


def _payload(raw_body: bytes) -> Mapping[str, object]:
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GrafanaWebhookError("invalid_payload", "webhook must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise GrafanaWebhookError("invalid_payload", "webhook root must be an object")
    truncated_alerts = payload.get("truncatedAlerts", 0)
    if isinstance(truncated_alerts, bool) or not isinstance(truncated_alerts, int):
        raise GrafanaWebhookError("invalid_payload", "webhook.truncatedAlerts must be an integer")
    if truncated_alerts != 0:
        raise GrafanaWebhookError("truncated_alerts", "Grafana truncated the alert group; configure maxAlerts=0")
    alerts = payload.get("alerts")
    if not isinstance(alerts, list) or not alerts:
        raise GrafanaWebhookError("invalid_payload", "webhook alerts must be a non-empty array")
    if len(alerts) > MAX_ALERTS:
        raise GrafanaWebhookError("too_many_alerts", f"webhook exceeds the {MAX_ALERTS} alert limit")
    return payload


def _notification(payload: Mapping[str, Any]) -> GrafanaNotification:
    status = _status(payload, "status", "webhook")
    org_id = payload.get("orgId")
    if isinstance(org_id, bool) or not isinstance(org_id, int):
        raise GrafanaWebhookError("invalid_payload", "webhook.orgId must be an integer")
    return GrafanaNotification(
        receiver=_string(payload, "receiver", "webhook"),
        status=status,
        org_id=org_id,
        version=_string(payload, "version", "webhook"),
        group_key=_string(payload, "groupKey", "webhook"),
        group_labels=_string_map(payload.get("groupLabels"), "webhook.groupLabels"),
        common_labels=_string_map(payload.get("commonLabels"), "webhook.commonLabels"),
        common_annotations=_string_map(payload.get("commonAnnotations"), "webhook.commonAnnotations"),
        external_url=_optional_string(payload.get("externalURL"), "webhook.externalURL"),
        title=_optional_string(payload.get("title"), "webhook.title"),
        message=_optional_string(payload.get("message"), "webhook.message"),
        alerts=tuple(_alert(item, index) for index, item in enumerate(payload["alerts"])),
    )


def _alert(value: object, index: int) -> GrafanaAlert:
    if not isinstance(value, dict):
        raise GrafanaWebhookError("invalid_payload", f"webhook.alerts[{index}] must be an object")
    prefix = f"webhook.alerts[{index}]"
    ends_at = _timestamp(value, "endsAt", prefix)
    if ends_at.year == 1:
        ends_at = None
    return GrafanaAlert(
        fingerprint=_string(value, "fingerprint", prefix),
        status=_status(value, "status", prefix),
        labels=_string_map(value.get("labels"), f"{prefix}.labels"),
        annotations=_string_map(value.get("annotations"), f"{prefix}.annotations"),
        values=_object_map(value.get("values"), f"{prefix}.values"),
        starts_at=_timestamp(value, "startsAt", prefix),
        ends_at=ends_at,
        generator_url=_optional_string(value.get("generatorURL"), f"{prefix}.generatorURL"),
        silence_url=_optional_string(value.get("silenceURL"), f"{prefix}.silenceURL"),
        dashboard_url=_optional_string(value.get("dashboardURL"), f"{prefix}.dashboardURL"),
        panel_url=_optional_string(value.get("panelURL"), f"{prefix}.panelURL"),
    )


def _status(payload: Mapping[str, Any], key: str, prefix: str) -> str:
    value = _string(payload, key, prefix)
    if value not in ("firing", "resolved"):
        raise GrafanaWebhookError("invalid_payload", f"{prefix}.{key} must be firing or resolved")
    return value


def _string(payload: Mapping[str, Any], key: str, prefix: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise GrafanaWebhookError("invalid_payload", f"{prefix}.{key} must be a non-empty string")
    return _bounded(value, f"{prefix}.{key}")


def _optional_string(value: object, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise GrafanaWebhookError("invalid_payload", f"{field} must be a string")
    return _bounded(value, field)


def _bounded(value: str, field: str) -> str:
    if len(value.encode()) > MAX_FIELD_BYTES:
        raise GrafanaWebhookError("invalid_payload", f"{field} exceeds {MAX_FIELD_BYTES} bytes")
    return value


def _string_map(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, dict):
        raise GrafanaWebhookError("invalid_payload", f"{field} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise GrafanaWebhookError("invalid_payload", f"{field} must contain string keys and values")
        result[_bounded(key, field)] = _bounded(item, field)
    return result


def _object_map(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise GrafanaWebhookError("invalid_payload", f"{field} must be an object")
    return value


def _timestamp(payload: Mapping[str, Any], key: str, prefix: str) -> datetime:
    value = _string(payload, key, prefix)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise GrafanaWebhookError("invalid_payload", f"{prefix}.{key} must be RFC 3339") from error
    return _to_utc(parsed, f"{prefix}.{key}")


def _unix_timestamp(value: str) -> datetime:
    try:
        return datetime.fromtimestamp(int(value), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError) as error:
        raise GrafanaWebhookError("invalid_timestamp", "timestamp must be Unix seconds") from error


def _to_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise GrafanaWebhookError("invalid_payload", f"{field} must include a timezone")
    return value.astimezone(UTC)
