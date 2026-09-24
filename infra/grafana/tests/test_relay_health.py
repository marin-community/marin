# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from relay_health import RelayNamespaceStatus, RelaySenderStatus, relay_alert_rows, relay_sender_statuses

NOW_MS = 1_000_000
NAMESPACE = "telemetry_v1.node_agent"


def sender(
    *,
    received_at_ms: int = NOW_MS,
    visible: int = 10,
    published: int = 10,
    cursor: int | None = 10,
    publication_progress_at_ms: int = NOW_MS,
    cursor_progress_at_ms: int = NOW_MS,
    namespaces: bool = True,
) -> RelaySenderStatus:
    status = RelayNamespaceStatus(
        namespace=NAMESPACE,
        visible_high_water=visible,
        published_high_water=published,
        settled_cursor=cursor,
        publication_progress_at_ms=publication_progress_at_ms,
        cursor_progress_at_ms=cursor_progress_at_ms,
    )
    return RelaySenderStatus(
        cluster="cw-a",
        boot_id="boot",
        report_sequence=1,
        target="https://hub",
        received_at_ms=received_at_ms,
        namespaces=(status,) if namespaces else (),
    )


def state(status: RelaySenderStatus | None) -> str:
    statuses = () if status is None else (status,)
    return relay_alert_rows(statuses, ("cw-a",), NOW_MS)[0]["state"]


def test_relay_alert_distinguishes_missing_heartbeat_namespace_and_each_stalled_stage():
    assert state(None) == "heartbeat_missing"
    assert state(sender(received_at_ms=NOW_MS - 120_000)) == "heartbeat_stale"
    assert state(sender(namespaces=False)) == "namespace_missing"
    assert state(sender(visible=12, published=10, publication_progress_at_ms=NOW_MS - 600_000)) == "publication_stalled"
    assert state(sender(published=10, cursor=8, cursor_progress_at_ms=NOW_MS - 600_000)) == "forwarding_stalled"


def test_relay_alert_allows_recent_progress_and_returns_an_explicit_zero():
    assert relay_alert_rows((sender(visible=12, published=10, cursor=8),), ("cw-a",), NOW_MS) == [
        {
            "cluster": "cw-a",
            "namespace": NAMESPACE,
            "state": "healthy",
            "value": 0,
        }
    ]


def test_relay_status_parser_preserves_an_absent_cursor_and_protobuf_integers():
    (status,) = relay_sender_statuses(
        {
            "senders": [
                {
                    "cluster": "cw-a",
                    "bootId": "boot",
                    "reportSequence": "2",
                    "target": "https://hub",
                    "receivedAtMs": "1000",
                    "namespaces": [
                        {
                            "namespace": NAMESPACE,
                            "visibleHighWater": "12",
                            "publishedHighWater": "10",
                            "publicationProgressAtMs": "900",
                            "cursorProgressAtMs": "800",
                        }
                    ],
                }
            ]
        }
    )
    assert status.report_sequence == 2
    assert status.namespaces[0].settled_cursor is None
