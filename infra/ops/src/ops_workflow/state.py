# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared workflow states and alert priority policy."""

from enum import StrEnum


class SignalState(StrEnum):
    FIRING = "firing"
    RESOLVED = "resolved"


class CaseState(StrEnum):
    PENDING = "pending"
    INVESTIGATING = "investigating"
    WAITING_HUMAN = "waiting_human"
    INVESTIGATED = "investigated"
    FAILED = "failed"
    ARCHIVED = "archived"


class SignalDisposition(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    STALE = "stale"


def severity_priority(severity: str) -> int:
    """Return the case queue priority for a Grafana severity label."""

    return {
        "critical": 100,
        "error": 90,
        "warning": 50,
        "info": 10,
    }.get(severity.lower(), 50)
