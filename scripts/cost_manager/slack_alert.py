# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold alerts: ping Slack when a cost slice exceeds a configured limit.

A run evaluates a list of :class:`AlertRule`s against the fetched
:class:`~scripts.cost_manager.cost_event.CostEvent`s. Each rule sums the cost of
a slice (an optional provider/category filter) over either the most recent
complete UTC day or the whole fetch window, and fires an :class:`AlertBreach`
when that sum exceeds ``max_usd``. Breaches are formatted into one message and
POSTed to a Slack incoming webhook — the same ``{"text": ...}`` contract as the
repo's ``notify-slack`` GitHub action.

Computation (:func:`evaluate_alerts`, :func:`format_slack_message`) is separate
from I/O (:func:`post_slack_message`) so the threshold logic is testable without
the network and the runner can print-instead-of-post on a dry run.
"""

import datetime as dt
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import requests

from scripts.cost_manager.cost_event import CostEvent, DateWindow

logger = logging.getLogger(__name__)

POST_TIMEOUT = 10.0


class AlertWindow(StrEnum):
    """Which span a rule's threshold is measured over."""

    # The most recent fully-elapsed UTC day (today is always partial).
    LATEST_DAY = "latest_day"
    # The entire trailing fetch window.
    WINDOW_TOTAL = "window_total"


@dataclass(frozen=True)
class AlertRule:
    """A spend ceiling over a cost slice.

    ``provider``/``category`` are optional filters; ``None`` matches every value
    for that field, so a rule with neither set covers total spend.
    """

    name: str
    max_usd: float
    provider: str | None = None
    category: str | None = None
    window: AlertWindow = AlertWindow.LATEST_DAY


@dataclass(frozen=True)
class AlertBreach:
    """A rule whose measured spend exceeded its threshold."""

    rule_name: str
    scope: str
    window_label: str
    observed_usd: float
    threshold_usd: float


def parse_alert_rules(raw_rules: Iterable[dict[str, Any]]) -> list[AlertRule]:
    """Build :class:`AlertRule`s from the ``alerts.rules`` config list."""
    rules: list[AlertRule] = []
    for raw in raw_rules:
        name = raw.get("name")
        if not name:
            raise ValueError(f"alert rule is missing 'name': {raw!r}")
        if "max_usd" not in raw:
            raise ValueError(f"alert rule {name!r} is missing 'max_usd'")
        window = AlertWindow(raw.get("window", AlertWindow.LATEST_DAY))
        rules.append(
            AlertRule(
                name=str(name),
                max_usd=float(raw["max_usd"]),
                provider=raw.get("provider"),
                category=raw.get("category"),
                window=window,
            )
        )
    return rules


def _scope_label(rule: AlertRule) -> str:
    parts = [rule.provider or "all providers"]
    if rule.category is not None:
        parts.append(rule.category)
    return " / ".join(parts)


def _matches(event: CostEvent, rule: AlertRule) -> bool:
    if rule.provider is not None and event.provider != rule.provider:
        return False
    return rule.category is None or event.category == rule.category


def evaluate_alerts(
    events: list[CostEvent], rules: list[AlertRule], *, window: DateWindow, today: dt.date
) -> list[AlertBreach]:
    """Return one :class:`AlertBreach` per rule whose slice exceeds its threshold."""
    latest_complete_day = today - dt.timedelta(days=1)
    window_label_total = f"{window.start.isoformat()}..{window.end.isoformat()}"

    breaches: list[AlertBreach] = []
    for rule in rules:
        if rule.window is AlertWindow.LATEST_DAY:
            target = latest_complete_day.isoformat()
            observed = sum(e.cost for e in events if _matches(e, rule) and e.usage_date == target)
            window_label = target
        else:
            observed = sum(e.cost for e in events if _matches(e, rule))
            window_label = window_label_total

        if observed > rule.max_usd:
            breaches.append(
                AlertBreach(
                    rule_name=rule.name,
                    scope=_scope_label(rule),
                    window_label=window_label,
                    observed_usd=observed,
                    threshold_usd=rule.max_usd,
                )
            )
    return breaches


def format_slack_message(breaches: list[AlertBreach]) -> str:
    """Render breaches as a Slack mrkdwn message body."""
    lines = [":rotating_light: *Cost alert* — spend exceeded a configured threshold"]
    for b in breaches:
        lines.append(
            f"• {b.scope} ({b.window_label}): ${b.observed_usd:,.2f} > ${b.threshold_usd:,.2f} [`{b.rule_name}`]"
        )
    lines.append("Source: finelog `cost.events` (scripts/cost_manager).")
    return "\n".join(lines)


def post_slack_message(webhook_url: str, text: str, *, timeout: float = POST_TIMEOUT) -> None:
    """POST ``{"text": text}`` to a Slack incoming webhook; raise on a non-2xx reply."""
    response = requests.post(webhook_url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
