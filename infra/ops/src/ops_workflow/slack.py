# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Slack escalation formatting and webhook delivery."""

import hashlib
import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

from ops_workflow.result import EscalationSeverity, OpsResult

SLACK_POST_TIMEOUT = 10.0
GRAFANA_SLACK_SEVERITIES = frozenset({"error", "critical"})


class SlackWebhookError(RuntimeError):
    """A delivery failure that never includes the secret webhook URL."""


@dataclass(frozen=True)
class SlackEscalationDraft:
    incident_key: str
    severity: EscalationSeverity
    reason: str
    message: str


@dataclass(frozen=True)
class SlackDelivery:
    """Leased outbox record ready for one webhook attempt."""

    id: str
    message: str
    attempts: int


class SlackDeliveryStore(Protocol):
    async def claim_slack_escalation(self) -> SlackDelivery | None: ...
    async def slack_escalation_sent(self, escalation_id: str) -> None: ...
    async def slack_escalation_retry(self, escalation_id: str, error: str) -> None: ...


class SlackWebhook:
    """Post fixed messages to one Slack incoming webhook."""

    def __init__(self, webhook_url: str) -> None:
        self._client = httpx.AsyncClient(timeout=SLACK_POST_TIMEOUT)
        self._webhook_url = webhook_url

    async def send(self, message: str) -> None:
        try:
            response = await self._client.post(self._webhook_url, json={"text": message})
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise SlackWebhookError(f"Slack webhook returned HTTP {error.response.status_code}") from None
        except httpx.HTTPError:
            raise SlackWebhookError("Slack webhook request failed") from None

    async def close(self) -> None:
        await self._client.aclose()


class SlackDispatcher:
    """Deliver one durable escalation per reconciliation iteration."""

    def __init__(self, store: SlackDeliveryStore, webhook: SlackWebhook) -> None:
        self._store = store
        self._webhook = webhook

    async def reconcile(self) -> None:
        delivery = await self._store.claim_slack_escalation()
        if delivery is None:
            return
        try:
            await self._webhook.send(delivery.message)
        except SlackWebhookError as error:
            await self._store.slack_escalation_retry(delivery.id, str(error))
            return
        await self._store.slack_escalation_sent(delivery.id)

    async def close(self) -> None:
        await self._webhook.close()


def escalation_draft(
    *,
    result: OpsResult,
    case: Mapping[str, object],
    signals: Sequence[Mapping[str, object]],
    public_url: str,
) -> SlackEscalationDraft | None:
    """Build a fixed Slack message and agent-independent incident key."""

    request = result.escalation
    if request is None:
        return None
    if any(str(signal.get("severity", "")).lower() in GRAFANA_SLACK_SEVERITIES for signal in signals):
        return None
    signal_keys = sorted(f"{signal['fingerprint']}:{signal['signal_generation']}" for signal in signals)
    source_key = "|".join(signal_keys) if signal_keys else f"manual:{case['id']}"
    incident_key = hashlib.sha256(source_key.encode()).hexdigest()
    case_url = f"{public_url.rstrip('/')}/cases/{case['id']}"
    signal_summary = ", ".join(signal_keys) if signal_keys else "manual investigation"
    message = "\n".join(
        (
            f":rotating_light: *Ops agent escalation · {_escape(str(case['title']))}*",
            f"Severity: `{request.severity.value}`",
            f"Reason: {_escape(request.reason)}",
            f"Summary: {_escape(result.summary)}",
            f"Signals: `{_escape(signal_summary)}`",
            f"Case: <{case_url}|open in ops.oa.dev>",
        )
    )
    return SlackEscalationDraft(
        incident_key=incident_key,
        severity=request.severity,
        reason=request.reason,
        message=message,
    )


def _escape(value: str) -> str:
    return html.escape(value, quote=False)
