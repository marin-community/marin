# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
from ops_workflow.source import SourceRequestError, verify_snapshot

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)
SECRET = b"test-source-secret"


@dataclass(frozen=True)
class SignedRequest:
    body: bytes
    timestamp: str
    signature: str


def warning_event_payload(uid: str = "event-1", count: int = 3) -> dict:
    return {
        "event_uid": uid,
        "resource_version": "42",
        "object_uid": "pod-1",
        "namespace": "kube-system",
        "object_kind": "Pod",
        "object_name": "node-local-dns-dcb4s",
        "reason": "DNSConfigForming",
        "message": "Nameserver limits were exceeded",
        "first_seen_at": "2026-07-21T15:00:00Z",
        "last_seen_at": "2026-07-21T15:59:00Z",
        "count": count,
        "reporting_controller": "kubelet",
    }


def signed_snapshot_request(
    *,
    events: list[dict] | None = None,
    timestamp: datetime = NOW,
    observed_at: str = "2026-07-21T16:00:00Z",
) -> SignedRequest:
    payload = {
        "schema_version": 1,
        "cluster": "cw-us-east-08a",
        "observed_at": observed_at,
        "events": events if events is not None else [warning_event_payload()],
    }
    raw_body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    timestamp_text = str(int(timestamp.timestamp()))
    signature = hmac.new(SECRET, timestamp_text.encode() + b"\n" + raw_body, hashlib.sha256).hexdigest()
    return SignedRequest(body=raw_body, timestamp=timestamp_text, signature=signature)


def verify(raw_body: bytes, timestamp: str, signature: str):
    return verify_snapshot(
        raw_body,
        key_id="current",
        timestamp=timestamp,
        signature=signature,
        keys={"current": SECRET},
        now=NOW,
        allowed_clusters=frozenset(("cw-us-east-08a",)),
    )


def test_verify_snapshot_exact_retry_has_same_delivery_identity() -> None:
    request = signed_snapshot_request()

    first = verify(request.body, request.timestamp, request.signature)
    retry = verify(request.body, request.timestamp, request.signature)

    assert first.delivery_key == retry.delivery_key
    assert first.snapshot.events[0].event_uid == "event-1"
    assert first.snapshot.events[0].count == 3


def test_verify_snapshot_rejects_request_outside_replay_window() -> None:
    request = signed_snapshot_request(timestamp=NOW - timedelta(minutes=6))

    with pytest.raises(SourceRequestError) as error:
        verify(request.body, request.timestamp, request.signature)

    assert error.value.code == "replay"


def test_verify_snapshot_rejects_duplicate_event_identity() -> None:
    event = warning_event_payload()
    request = signed_snapshot_request(events=[event, event])

    with pytest.raises(SourceRequestError) as error:
        verify(request.body, request.timestamp, request.signature)

    assert error.value.code == "duplicate_event"


def test_verify_snapshot_rejects_body_changed_after_signing() -> None:
    request = signed_snapshot_request()
    changed_body = request.body.replace(b"Nameserver", b"ResolverXX")

    with pytest.raises(SourceRequestError) as error:
        verify(changed_body, request.timestamp, request.signature)

    assert error.value.code == "invalid_signature"


def test_verify_snapshot_rejects_future_observation_that_would_poison_high_water() -> None:
    request = signed_snapshot_request(observed_at="2026-07-22T16:00:00Z")

    with pytest.raises(SourceRequestError) as error:
        verify(request.body, request.timestamp, request.signature)

    assert error.value.code == "invalid_payload"
