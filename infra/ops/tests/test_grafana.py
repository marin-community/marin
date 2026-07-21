# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from ops_workflow.grafana import GrafanaWebhookError, verify_grafana_webhook

FIXTURE = Path(__file__).parent.parent / "fixtures" / "dns-warning-firing.json"
SECRET = b"test-grafana-secret"
NOW = datetime(2026, 7, 21, 17, 0, tzinfo=UTC)


def _signature(body: bytes, timestamp: str) -> str:
    return hmac.new(SECRET, timestamp.encode() + b":" + body, hashlib.sha256).hexdigest()


def _verify(body: bytes, *, now: datetime = NOW, timestamp: str | None = None):
    timestamp = timestamp or str(int(NOW.timestamp()))
    return verify_grafana_webhook(
        body,
        signature=_signature(body, timestamp),
        timestamp=timestamp,
        secret=SECRET,
        now=now,
    )


def test_verified_webhook_preserves_grafana_group_and_fingerprint_identity():
    body = FIXTURE.read_bytes()

    verified = _verify(body)

    notification = verified.notification
    assert notification.receiver == "ops-agent"
    assert notification.group_key == '{}:{alertname="DNSConfigForming", cluster="cw-us-east-08a"}'
    assert [alert.fingerprint for alert in notification.alerts] == ["2b05ef3b1641c79a", "ef356383208c86c5"]
    assert all(alert.ends_at is None for alert in notification.alerts)
    assert verified.delivery_key == hashlib.sha256(body).hexdigest()


def test_signature_covers_timestamp_and_exact_raw_body():
    body = FIXTURE.read_bytes()
    timestamp = str(int(NOW.timestamp()))
    signature = _signature(body, timestamp)

    with pytest.raises(GrafanaWebhookError, match="signature does not match") as error:
        verify_grafana_webhook(
            body + b" ",
            signature=signature,
            timestamp=timestamp,
            secret=SECRET,
            now=NOW,
        )

    assert error.value.code == "invalid_signature"


def test_cloud_tasks_retry_within_delivery_window_is_accepted():
    body = FIXTURE.read_bytes()
    timestamp = str(int((NOW - timedelta(hours=24)).timestamp()))

    assert _verify(body, timestamp=timestamp).notification.status == "firing"


def test_timestamp_older_than_task_retry_window_is_rejected_before_payload_is_applied():
    body = FIXTURE.read_bytes()
    timestamp = str(int((NOW - timedelta(hours=26)).timestamp()))

    with pytest.raises(GrafanaWebhookError, match="outside the replay window") as error:
        _verify(body, timestamp=timestamp)

    assert error.value.code == "replay"


def test_timestamp_beyond_future_clock_skew_is_rejected():
    body = FIXTURE.read_bytes()
    timestamp = str(int((NOW + timedelta(minutes=6)).timestamp()))

    with pytest.raises(GrafanaWebhookError, match="outside the replay window") as error:
        _verify(body, timestamp=timestamp)

    assert error.value.code == "replay"


@pytest.mark.parametrize("truncated", [1, "1", True])
def test_truncated_or_malformed_group_is_rejected(truncated: object):
    payload = json.loads(FIXTURE.read_bytes())
    payload["truncatedAlerts"] = truncated
    body = json.dumps(payload).encode()

    with pytest.raises(GrafanaWebhookError) as error:
        _verify(body)

    expected = "truncated_alerts" if truncated == 1 and not isinstance(truncated, bool) else "invalid_payload"
    assert error.value.code == expected
