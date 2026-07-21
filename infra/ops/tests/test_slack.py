# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from ops_workflow.result import OpsResult, parse_ops_result
from ops_workflow.slack import SlackWebhook, SlackWebhookError, escalation_draft

CASE_ID = "8c592336-b43c-4a5a-88bc-1f13dd861680"
TURN_ID = "40fea6a1-10a1-4d42-983e-a8dbfc3f971b"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _result() -> OpsResult:
    return parse_ops_result(
        json.dumps(
            {
                "schema_version": 2,
                "case_id": CASE_ID,
                "ops_turn_id": TURN_ID,
                "outcome": "action_recommended",
                "summary": "Image cleanup freed no space.",
                "evidence": [],
                "action_taken": "none",
                "recommended_next_step": "Inspect disk consumers.",
                "escalation": {"severity": "error", "reason": "Node disk pressure is increasing."},
            }
        ),
        case_id=CASE_ID,
        turn_id=TURN_ID,
    )


def test_warning_escalation_uses_signal_generation_for_deduplication():
    case = {"id": CASE_ID, "title": "Image filesystem pressure"}
    signals = [
        {"fingerprint": "grafana-b", "signal_generation": 2, "severity": "warning"},
        {"fingerprint": "grafana-a", "signal_generation": 1, "severity": "warning"},
    ]

    draft = escalation_draft(result=_result(), case=case, signals=signals, public_url="https://ops.oa.dev")
    reverse_order = escalation_draft(
        result=_result(),
        case=case,
        signals=list(reversed(signals)),
        public_url="https://ops.oa.dev",
    )

    assert draft is not None
    assert reverse_order is not None
    assert draft.incident_key == reverse_order.incident_key
    assert "grafana-a:1, grafana-b:2" in draft.message
    assert f"https://ops.oa.dev/cases/{CASE_ID}" in draft.message


def test_agent_does_not_duplicate_grafana_error_notification():
    draft = escalation_draft(
        result=_result(),
        case={"id": CASE_ID, "title": "Image filesystem pressure"},
        signals=[{"fingerprint": "grafana-a", "signal_generation": 1, "severity": "error"}],
        public_url="https://ops.oa.dev",
    )

    assert draft is None


@pytest.mark.anyio
async def test_webhook_failure_does_not_expose_secret_url(monkeypatch: pytest.MonkeyPatch):
    secret_url = "https://hooks.slack.com/services/secret-token"
    response = httpx.Response(500, request=httpx.Request("POST", secret_url))

    async def fail_request(*_args: object, **_kwargs: object) -> httpx.Response:
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", fail_request)
    webhook = SlackWebhook(secret_url)
    try:
        with pytest.raises(SlackWebhookError, match="HTTP 500") as raised:
            await webhook.send("test")
        assert "secret-token" not in str(raised.value)
    finally:
        await webhook.close()
