# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Threshold alerts: ping Slack when a CostEvent slice exceeds a configured limit.

A run evaluates a list of :class:`AlertRule`s against the fetched
:class:`~scripts.cost_manager.cost_event.CostEvent`s. A rule selects the cost or
the provider usage gauge. Cost rules can use the most recent completed UTC day
or the complete fetch window. Usage rules can use the current or most recent
completed day. The rule can filter the provider, category, and detail.

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
    # The current partial UTC day. Usage gauges use this for early alerts.
    CURRENT_DAY = "current_day"
    # The entire trailing fetch window.
    WINDOW_TOTAL = "window_total"


class AlertMetric(StrEnum):
    """The numeric CostEvent field that an alert rule measures."""

    COST = "cost"
    USAGE_AMOUNT = "usage_amount"


@dataclass(frozen=True)
class AlertRule:
    """A ceiling over a CostEvent slice.

    ``provider``, ``category``, and ``detail`` are optional filters. ``None``
    matches each value in that field.
    """

    name: str
    metric: AlertMetric
    threshold: float
    provider: str | None = None
    category: str | None = None
    detail: str | None = None
    window: AlertWindow = AlertWindow.LATEST_DAY


@dataclass(frozen=True)
class AlertBreach:
    """A rule whose measured value exceeded its threshold."""

    rule_name: str
    scope: str
    window_label: str
    metric: AlertMetric
    observed_value: float
    threshold_value: float
    unit: str


def parse_alert_rules(raw_rules: Iterable[dict[str, Any]]) -> list[AlertRule]:
    """Build :class:`AlertRule`s from the ``alerts.rules`` config list."""
    rules: list[AlertRule] = []
    for raw in raw_rules:
        name = raw.get("name")
        if not name:
            raise ValueError(f"alert rule is missing 'name': {raw!r}")
        if "metric" not in raw:
            raise ValueError(f"alert rule {name!r} is missing 'metric'")
        if "threshold" not in raw:
            raise ValueError(f"alert rule {name!r} is missing 'threshold'")
        metric = AlertMetric(raw["metric"])
        window = AlertWindow(raw.get("window", AlertWindow.LATEST_DAY))
        if metric is AlertMetric.USAGE_AMOUNT and window is AlertWindow.WINDOW_TOTAL:
            raise ValueError(f"alert rule {name!r}: usage_amount cannot use window_total")
        rules.append(
            AlertRule(
                name=str(name),
                metric=metric,
                threshold=float(raw["threshold"]),
                provider=raw.get("provider"),
                category=raw.get("category"),
                detail=raw.get("detail"),
                window=window,
            )
        )
    return rules


def _scope_label(rule: AlertRule) -> str:
    parts = [rule.provider or "all providers"]
    if rule.category is not None:
        parts.append(rule.category)
    if rule.detail is not None:
        parts.append(rule.detail)
    return " / ".join(parts)


def _matches(event: CostEvent, rule: AlertRule) -> bool:
    if rule.provider is not None and event.provider != rule.provider:
        return False
    if rule.category is not None and event.category != rule.category:
        return False
    return rule.detail is None or event.detail == rule.detail


def _value_and_unit(events: list[CostEvent], metric: AlertMetric) -> tuple[float, str]:
    if metric is AlertMetric.COST:
        return sum(event.cost for event in events), "USD"
    with_usage = [event for event in events if event.usage_amount is not None]
    units = {event.usage_unit for event in with_usage}
    if None in units:
        raise ValueError("usage alert matched an event without usage_unit")
    if len(units) > 1:
        raise ValueError(f"usage alert matched different units: {sorted(units)}")
    unit = next(iter(units), "value")
    assert unit is not None
    return sum(event.usage_amount for event in with_usage if event.usage_amount is not None), unit


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
            matched = [e for e in events if _matches(e, rule) and e.usage_date == target]
            window_label = target
        elif rule.window is AlertWindow.CURRENT_DAY:
            target = today.isoformat()
            matched = [e for e in events if _matches(e, rule) and e.usage_date == target]
            window_label = target
        else:
            matched = [e for e in events if _matches(e, rule)]
            window_label = window_label_total

        observed, unit = _value_and_unit(matched, rule.metric)
        if observed > rule.threshold:
            breaches.append(
                AlertBreach(
                    rule_name=rule.name,
                    scope=_scope_label(rule),
                    window_label=window_label,
                    metric=rule.metric,
                    observed_value=observed,
                    threshold_value=rule.threshold,
                    unit=unit,
                )
            )
    return breaches


def format_slack_message(breaches: list[AlertBreach]) -> str:
    """Render breaches as a Slack mrkdwn message body."""
    lines = [":rotating_light: *Threshold alert* — a measured value exceeded its configured ceiling"]
    for breach in breaches:
        if breach.unit == "USD":
            observed = f"${breach.observed_value:,.2f}"
            threshold = f"${breach.threshold_value:,.2f}"
        elif breach.unit == "bytes":
            observed = f"{breach.observed_value / 1024**4:,.2f} TiB"
            threshold = f"{breach.threshold_value / 1024**4:,.2f} TiB"
        else:
            observed = f"{breach.observed_value:,.2f} {breach.unit}"
            threshold = f"{breach.threshold_value:,.2f} {breach.unit}"
        lines.append(f"• {breach.scope} ({breach.window_label}): {observed} > {threshold} [`{breach.rule_name}`]")
    lines.append("Source: finelog `cost.events` (scripts/cost_manager).")
    return "\n".join(lines)


def post_slack_message(webhook_url: str, text: str, *, timeout: float = POST_TIMEOUT) -> None:
    """POST ``{"text": text}`` to a Slack incoming webhook; raise on a non-2xx reply."""
    response = requests.post(webhook_url, json={"text": text}, timeout=timeout)
    response.raise_for_status()
