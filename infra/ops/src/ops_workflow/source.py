# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authentication and validation for Kubernetes Warning snapshots."""

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MAX_BODY_BYTES = 1024 * 1024
MAX_EVENTS = 5_000
MAX_MESSAGE_BYTES = 8 * 1024
MAX_FIELD_BYTES = 512
REPLAY_WINDOW_SECONDS = 5 * 60
SCHEMA_VERSION = 1

_INVALID_PAYLOAD = "invalid_payload"
_INVALID_SIGNATURE = "invalid_signature"
_INVALID_TIMESTAMP = "invalid_timestamp"

_CLUSTER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SourceRequestError(ValueError):
    """A stable source-ingestion rejection."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WarningEvent:
    """One normalized Kubernetes Warning event."""

    event_uid: str
    resource_version: str
    object_uid: str
    namespace: str
    object_kind: str
    object_name: str
    reason: str
    message: str
    first_seen_at: datetime
    last_seen_at: datetime
    count: int
    reporting_controller: str


@dataclass(frozen=True)
class WarningSnapshot:
    """One complete, source-ordered cluster snapshot."""

    cluster: str
    observed_at: datetime
    events: tuple[WarningEvent, ...]


@dataclass(frozen=True)
class VerifiedSnapshot:
    """An authenticated snapshot and its retry identity."""

    snapshot: WarningSnapshot
    key_id: str
    source_timestamp: datetime
    delivery_key: str
    body_sha256: str


def verify_snapshot(
    raw_body: bytes,
    *,
    key_id: str,
    timestamp: str,
    signature: str,
    keys: Mapping[str, bytes],
    now: datetime,
    allowed_clusters: frozenset[str] | None = None,
) -> VerifiedSnapshot:
    """Authenticate and normalize a complete Warning snapshot.

    Args:
        raw_body: Exact request bytes used by the sender to compute its HMAC.
        key_id: Named current or overlap key.
        timestamp: Unix seconds encoded as decimal text.
        signature: Lowercase hexadecimal HMAC-SHA256.
        keys: Accepted key identifiers and secret bytes.
        now: Current UTC time, injected for deterministic replay checks.
        allowed_clusters: Optional configured cluster allowlist.

    Returns:
        The verified snapshot and deterministic retry identity.

    Raises:
        SourceRequestError: If authentication, replay, or payload validation fails.
    """

    if len(raw_body) > MAX_BODY_BYTES:
        raise SourceRequestError("payload_too_large", "snapshot exceeds the 1 MiB limit")

    secret = keys.get(key_id)
    if secret is None:
        raise SourceRequestError("unknown_key", "source key is not accepted")

    source_timestamp = _source_timestamp(timestamp)
    current_time = _to_utc(now, "now")
    if abs((current_time - source_timestamp).total_seconds()) > REPLAY_WINDOW_SECONDS:
        raise SourceRequestError("replay", "source timestamp is outside the replay window")

    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise SourceRequestError(_INVALID_SIGNATURE, "source signature must be lowercase hex")
    signed = timestamp.encode() + b"\n" + raw_body
    expected = hmac.new(secret, signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SourceRequestError(_INVALID_SIGNATURE, "source signature does not match")

    snapshot = _snapshot(raw_body, allowed_clusters=allowed_clusters)
    if abs((snapshot.observed_at - source_timestamp).total_seconds()) > REPLAY_WINDOW_SECONDS:
        raise SourceRequestError(_INVALID_PAYLOAD, "snapshot observed_at is outside the source timestamp window")
    delivery_material = key_id.encode() + b"\n" + timestamp.encode() + b"\n" + raw_body
    return VerifiedSnapshot(
        snapshot=snapshot,
        key_id=key_id,
        source_timestamp=source_timestamp,
        delivery_key=hashlib.sha256(delivery_material).hexdigest(),
        body_sha256=hashlib.sha256(raw_body).hexdigest(),
    )


def _snapshot(raw_body: bytes, *, allowed_clusters: frozenset[str] | None) -> WarningSnapshot:
    try:
        payload = json.loads(raw_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceRequestError(_INVALID_PAYLOAD, "snapshot must be UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise SourceRequestError(_INVALID_PAYLOAD, "snapshot root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SourceRequestError("unsupported_schema", "snapshot schema_version must be 1")

    cluster = _string(payload, "cluster", MAX_FIELD_BYTES)
    if not _CLUSTER_PATTERN.fullmatch(cluster):
        raise SourceRequestError(_INVALID_PAYLOAD, "cluster has an invalid format")
    if allowed_clusters is not None and cluster not in allowed_clusters:
        raise SourceRequestError("unknown_cluster", "cluster is not configured")

    observed_at = _timestamp(payload, "observed_at")
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise SourceRequestError(_INVALID_PAYLOAD, "events must be an array")
    if len(raw_events) > MAX_EVENTS:
        raise SourceRequestError("too_many_events", f"snapshot exceeds the {MAX_EVENTS} event limit")

    events: list[WarningEvent] = []
    event_uids: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        event = _event(raw_event, index)
        if event.event_uid in event_uids:
            raise SourceRequestError("duplicate_event", f"events[{index}].event_uid is duplicated")
        event_uids.add(event.event_uid)
        events.append(event)
    return WarningSnapshot(cluster=cluster, observed_at=observed_at, events=tuple(events))


def _event(raw_event: Any, index: int) -> WarningEvent:
    if not isinstance(raw_event, dict):
        raise SourceRequestError(_INVALID_PAYLOAD, f"events[{index}] must be an object")
    first_seen_at = _timestamp(raw_event, "first_seen_at", prefix=f"events[{index}]")
    last_seen_at = _timestamp(raw_event, "last_seen_at", prefix=f"events[{index}]")
    if last_seen_at < first_seen_at:
        raise SourceRequestError(_INVALID_PAYLOAD, f"events[{index}].last_seen_at precedes first_seen_at")
    count = raw_event.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SourceRequestError(_INVALID_PAYLOAD, f"events[{index}].count must be a positive integer")

    return WarningEvent(
        event_uid=_string(raw_event, "event_uid", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        resource_version=_string(raw_event, "resource_version", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        object_uid=_string(raw_event, "object_uid", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        namespace=_string(raw_event, "namespace", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        object_kind=_string(raw_event, "object_kind", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        object_name=_string(raw_event, "object_name", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        reason=_string(raw_event, "reason", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
        message=_string(raw_event, "message", MAX_MESSAGE_BYTES, prefix=f"events[{index}]"),
        first_seen_at=first_seen_at,
        last_seen_at=last_seen_at,
        count=count,
        reporting_controller=_string(raw_event, "reporting_controller", MAX_FIELD_BYTES, prefix=f"events[{index}]"),
    )


def _string(payload: Mapping[str, Any], key: str, max_bytes: int, *, prefix: str = "snapshot") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SourceRequestError(_INVALID_PAYLOAD, f"{prefix}.{key} must be a non-empty string")
    if len(value.encode()) > max_bytes:
        raise SourceRequestError(_INVALID_PAYLOAD, f"{prefix}.{key} exceeds {max_bytes} bytes")
    return value


def _timestamp(payload: Mapping[str, Any], key: str, *, prefix: str = "snapshot") -> datetime:
    value = _string(payload, key, MAX_FIELD_BYTES, prefix=prefix)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SourceRequestError(_INVALID_PAYLOAD, f"{prefix}.{key} must be an RFC 3339 timestamp") from error
    try:
        return _to_utc(parsed, f"{prefix}.{key}")
    except ValueError as error:
        raise SourceRequestError(_INVALID_PAYLOAD, str(error)) from error


def _source_timestamp(value: str) -> datetime:
    try:
        seconds = int(value)
    except ValueError as error:
        raise SourceRequestError(_INVALID_TIMESTAMP, "source timestamp must be Unix seconds") from error
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise SourceRequestError(_INVALID_TIMESTAMP, "source timestamp is out of range") from error


def _to_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value.astimezone(UTC)
