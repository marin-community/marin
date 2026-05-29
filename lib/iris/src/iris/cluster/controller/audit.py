# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structured audit-line emitter for controller state transitions.

This module is intentionally free of DB and state-machine dependencies; it just
formats and emits ``logger.info`` lines that the Iris log server captures for
later querying.
"""

import logging

# Preserve the historical log channel name so external audit-line consumers
# (and tests that pin the channel via ``caplog.at_level(..., logger=...)``)
# continue to observe events on ``iris.cluster.controller.transitions``.
logger = logging.getLogger("iris.cluster.controller.transitions")


def log_event(
    action: str,
    entity_id: str,
    *,
    trigger: str | None = None,
    **details: object,
) -> None:
    """Emit a semi-structured audit line for a controller state transition.

    Each call produces one ``logger.info`` line of the shape::

        event=<action> entity=<entity_id> trigger=<trigger> k=v ...

    ``trigger`` names the upstream event when this is derived (e.g.
    ``trigger=heartbeat_applied`` on cascaded job terminations); callers omit
    it for externally-caused events and the line renders ``trigger=-``.

    These lines are captured by the Iris log server and queried via the normal
    ``iris process logs`` / log-store DuckDB interface — there is no SQLite
    audit table.
    """
    extras = " ".join(f"{k}={v}" for k, v in details.items() if v is not None)
    logger.info(
        "event=%s entity=%s trigger=%s %s",
        action,
        entity_id,
        trigger or "-",
        extras,
    )
