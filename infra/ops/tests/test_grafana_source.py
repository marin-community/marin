# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

import httpx
import pytest
from ops_workflow.grafana_source import GrafanaApiAlertSource, snapshot_from_api_alerts

NOW = datetime(2026, 7, 22, 1, 14, 27, tzinfo=UTC)


class StaticTokenSource:
    def get_token(self) -> str:
        return "test-iap-jwt"


def _alert(*, fingerprint: str = "4798bb3") -> dict[str, object]:
    return {
        "annotations": {
            "summary": "cw-rno2a: a watched control-plane component is in backoff",
            "runbook_url": "https://example.test/runbook",
        },
        "endsAt": "2026-07-22T01:19:12Z",
        "fingerprint": fingerprint,
        "generatorURL": "https://grafana.oa.dev/alerting/grafana/k8s-control-plane-crashloop/view?orgId=1",
        "labels": {
            "alertname": "ControlPlaneCrashLooping",
            "cluster": "cw-rno2a",
            "grafana_folder": "Alerts",
            "severity": "critical",
        },
        "receivers": [{"name": "ops-critical"}],
        "startsAt": "2026-07-22T01:08:12Z",
        "status": {"inhibitedBy": [], "silencedBy": [], "state": "active"},
        "updatedAt": "2026-07-22T01:14:12Z",
    }


@pytest.mark.anyio
async def test_snapshot_reads_active_alerts_through_iap() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/alertmanager/grafana/api/v2/alerts"
        assert request.headers["proxy-authorization"] == "Bearer test-iap-jwt"
        return httpx.Response(200, json=[_alert()])

    source = GrafanaApiAlertSource(
        "https://grafana.example.test",
        StaticTokenSource(),
        transport=httpx.MockTransport(handler),
    )

    snapshot = await source.snapshot()

    assert len(snapshot.alerts) == 1
    item = snapshot.alerts[0]
    assert item.alert.fingerprint == "4798bb3"
    assert item.alert.labels == {
        "alertname": "ControlPlaneCrashLooping",
        "cluster": "cw-rno2a",
        "grafana_folder": "Alerts",
        "severity": "critical",
    }
    assert item.alert.annotations["summary"] == "cw-rno2a: a watched control-plane component is in backoff"
    assert item.alert.values == {}
    assert item.alert.starts_at == datetime(2026, 7, 22, 1, 8, 12, tzinfo=UTC)
    assert item.group_labels == {"alertname": "ControlPlaneCrashLooping", "cluster": "cw-rno2a"}
    assert item.group_key == '{"alertname":"ControlPlaneCrashLooping","cluster":"cw-rno2a"}'


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_snapshot_rejects_duplicate_grafana_instance_identity() -> None:
    with pytest.raises(ValueError, match="duplicate alert fingerprints"):
        snapshot_from_api_alerts([_alert(), _alert()], observed_at=NOW)
