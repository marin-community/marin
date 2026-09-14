# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alert projection for Finelog's direct relay heartbeat."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

REQUIRED_RELAY_NAMESPACES = ("telemetry_v1.node_agent",)
RELAY_HEARTBEAT_MAX_AGE_MS = 2 * 60 * 1000
RELAY_PROGRESS_MAX_AGE_MS = 10 * 60 * 1000
HEALTHY_RELAY_STATE = "healthy"


@dataclass(frozen=True)
class RelayNamespaceStatus:
    namespace: str
    visible_high_water: int
    published_high_water: int
    settled_cursor: int | None
    publication_progress_at_ms: int
    cursor_progress_at_ms: int


@dataclass(frozen=True)
class RelaySenderStatus:
    cluster: str
    boot_id: str
    report_sequence: int
    target: str
    received_at_ms: int
    namespaces: tuple[RelayNamespaceStatus, ...]


def relay_sender_statuses(payload: dict[str, object]) -> tuple[RelaySenderStatus, ...]:
    """Parse a protobuf-JSON ListRelayStatus response."""
    senders = cast(list[dict[str, object]], payload.get("senders", []))
    return tuple(
        RelaySenderStatus(
            cluster=str(sender["cluster"]),
            boot_id=str(sender["bootId"]),
            report_sequence=int(str(sender["reportSequence"])),
            target=str(sender["target"]),
            received_at_ms=int(str(sender["receivedAtMs"])),
            namespaces=tuple(_namespace_status(item) for item in cast(list[dict[str, object]], sender["namespaces"])),
        )
        for sender in senders
    )


def _namespace_status(item: dict[str, object]) -> RelayNamespaceStatus:
    settled_cursor = item.get("settledCursor")
    return RelayNamespaceStatus(
        namespace=str(item["namespace"]),
        visible_high_water=int(str(item["visibleHighWater"])),
        published_high_water=int(str(item["publishedHighWater"])),
        settled_cursor=None if settled_cursor is None else int(str(settled_cursor)),
        publication_progress_at_ms=int(str(item["publicationProgressAtMs"])),
        cursor_progress_at_ms=int(str(item["cursorProgressAtMs"])),
    )


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
                    "value": 0 if state == HEALTHY_RELAY_STATE else 1,
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
    return HEALTHY_RELAY_STATE
