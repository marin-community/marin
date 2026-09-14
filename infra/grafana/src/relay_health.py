# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alert projection for Finelog's direct relay heartbeat."""

from collections.abc import Iterable, Sequence

from finelog.client import RelaySenderStatus

REQUIRED_RELAY_NAMESPACES = ("telemetry_v1.node_agent",)
RELAY_HEARTBEAT_MAX_AGE_MS = 2 * 60 * 1000
RELAY_PROGRESS_MAX_AGE_MS = 10 * 60 * 1000


def relay_alert_rows(
    statuses: Sequence[RelaySenderStatus],
    expected_clusters: Iterable[str],
    now_ms: int,
) -> list[dict[str, object]]:
    """Return one explicit alert value per required cluster and namespace."""
    by_cluster = {status.cluster: status for status in statuses}
    rows: list[dict[str, object]] = []
    for cluster in sorted(set(expected_clusters) | set(by_cluster)):
        sender = by_cluster.get(cluster)
        for namespace in REQUIRED_RELAY_NAMESPACES:
            state = _relay_state(sender, namespace, now_ms)
            rows.append(
                {
                    "cluster": cluster,
                    "namespace": namespace,
                    "state": state,
                    "value": 0 if state == "healthy" else 1,
                }
            )
    return rows


def _relay_state(sender: RelaySenderStatus | None, namespace: str, now_ms: int) -> str:
    if sender is None:
        return "heartbeat_missing"
    if now_ms - sender.received_at_ms >= RELAY_HEARTBEAT_MAX_AGE_MS:
        return "heartbeat_stale"

    status = next((status for status in sender.namespaces if status.namespace == namespace), None)
    if status is None:
        return "namespace_missing"
    if (
        status.visible_high_water > status.published_high_water
        and now_ms - status.publication_progress_at_ms >= RELAY_PROGRESS_MAX_AGE_MS
    ):
        return "publication_stalled"
    if (
        status.published_high_water > (status.settled_cursor or 0)
        and now_ms - status.cursor_progress_at_ms >= RELAY_PROGRESS_MAX_AGE_MS
    ):
        return "forwarding_stalled"
    return "healthy"
