# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add normalized tags to Echo wiki entries."""

import sqlalchemy

DDL = """
ALTER TABLE wiki_entries
ADD COLUMN tags TEXT[] DEFAULT '{}'::text[] NOT NULL,
ADD CONSTRAINT wiki_entries_tags_limit CHECK (cardinality(tags) <= 20);
CREATE INDEX idx_wiki_entries_tags ON wiki_entries USING gin (tags);
"""

IMPORTED_INCIDENT_TAGS = {
    9: (
        "Incident 2026-03-22: Zephyr coordinator thread leaked during shutdown",
        ("ops", "debugging", "incident", "zephyr"),
    ),
    10: ("Incident 2026-03-30: Iris worker-row schema mismatch", ("ops", "debugging", "incident", "iris")),
    11: (
        'Incident 2026-04-21: Iris scheduler thread freeze — "Pending scheduler feedback"',
        ("ops", "debugging", "incident", "iris", "outage", "fixed"),
    ),
    12: (
        "Incident 2026-04-21: Iris TPU worker wedge + reservation-holder poll storm",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    13: (
        "Incident 2026-04-28: Iris Split Slice And Orphan Attempts",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    14: (
        "Incident 2026-06-08: TPU canary ferry: 5 days of timeouts (reservation-job taint + v5p stockout)",
        ("ops", "debugging", "incident", "ci", "canary-ferry", "iris", "degraded", "mitigated"),
    ),
    15: ("Incident 2026-06-11: Marin unit import failure on PR 6216", ("ops", "debugging", "incident", "marin")),
    16: (
        "Incident 2026-06-16: Recycled-IP zombie worker black-holes and kills every task it is assigned",
        ("ops", "debugging", "incident", "iris"),
    ),
    17: (
        "Incident 2026-06-16: Iris control loop stalls ~60s/pass: autoscaler "
        "refresh() serializes blocking GCP describes",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    18: (
        'Incident 2026-07-15: Iris scheduler: non-atomic preempt/place strands "free" workers and thrashes gangs',
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    19: ("Incident 2026-07-20: CoreWeave CI signing-key access failure", ("ops", "debugging", "incident", "coreweave")),
    20: (
        "Incident 2026-07-22: Coreweave Kueue Leader Loss",
        ("ops", "debugging", "incident", "coreweave", "outage", "fixed"),
    ),
    21: (
        "Incident 2026-07-22: Iris Proxy Health Stalls",
        ("ops", "debugging", "incident", "iris", "degraded", "mitigated"),
    ),
    22: (
        "Incident 2026-07-23: Finelog Rno2A Readiness Loop",
        ("ops", "debugging", "incident", "finelog", "degraded", "mitigated"),
    ),
    23: (
        "Incident 2026-07-24: Coreweave Kueue Manager Oom",
        ("ops", "debugging", "incident", "coreweave", "outage", "fixed"),
    ),
    24: (
        "Incident 2026-07-24: Finelog Architecture Mismatch",
        ("ops", "debugging", "incident", "finelog", "degraded", "fixed"),
    ),
    25: (
        "Incident 2026-07-24: Finelog Schema Rollback",
        ("ops", "debugging", "incident", "finelog", "degraded", "mitigated"),
    ),
    26: (
        "Incident 2026-07-24: Gb200 Controller Slow Status",
        ("ops", "debugging", "incident", "coreweave", "degraded", "investigating"),
    ),
    27: (
        "Incident 2026-07-24: Grafana Coreweave Token Newline",
        ("ops", "debugging", "incident", "gcp", "degraded", "mitigated"),
    ),
    28: (
        "Incident 2026-07-24: Harbor dataset revision: configured tag is absent",
        ("ops", "debugging", "incident", "harbor"),
    ),
    29: (
        "Incident 2026-07-24: Iris Coreweave Checkpoint Restore",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    30: (
        "Incident 2026-07-24: Iris RPC metrics: duplicate collector registration",
        ("ops", "debugging", "incident", "iris"),
    ),
    31: (
        "Incident 2026-07-24: Iris Rust Proxy Auth Handoff",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    32: (
        "Incident 2026-07-24: Iris Singleton Kueue Diagnostic Gap",
        ("ops", "debugging", "incident", "iris", "degraded", "mitigated"),
    ),
    33: (
        "Incident 2026-07-24: Loom deployment: remote image context does not rebuild",
        ("ops", "debugging", "incident", "loom"),
    ),
    34: (
        "Incident 2026-07-25: Iris ARM log-shipper CrashLoopBackOff",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    35: (
        "Incident 2026-07-25: Iris Federated Proxy Validation",
        ("ops", "debugging", "incident", "iris", "diagnostic-only", "fixed"),
    ),
    36: (
        "Incident 2026-07-25: Qwen3 Eval Response Cutoff",
        ("ops", "debugging", "incident", "marin-eval", "degraded", "mitigated"),
    ),
    37: (
        "Incident 2026-07-26: Iris Kueue Preemption Misclassified",
        ("ops", "debugging", "incident", "iris", "degraded", "fixed"),
    ),
    38: (
        "Incident 2026-07-27: Iris Finelog Query Latency",
        ("ops", "debugging", "incident", "finelog", "degraded", "investigating"),
    ),
    39: (
        "Incident 2026-07-27: Native package releases: post-publication lock drift",
        ("ops", "debugging", "incident", "native"),
    ),
    40: (
        "Incident 2026-07-27: Vllm Streamer Retry Cache Lock",
        ("ops", "debugging", "incident", "vllm", "degraded", "fixed"),
    ),
    41: (
        "Incident 2026-07-28: CoreWeave US-EAST-08A public VIP blackhole",
        ("ops", "debugging", "incident", "coreweave", "degraded", "investigating"),
    ),
    42: (
        "Incident 2026-07-28: Gb200 First Step Collective Hang",
        ("ops", "debugging", "incident", "iris", "degraded", "mitigated"),
    ),
}


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
    update = sqlalchemy.text(
        "UPDATE wiki_entries SET tags = CAST(:tags AS text[]) WHERE id = :entry_id AND title = :title"
    )
    for entry_id, (title, tags) in IMPORTED_INCIDENT_TAGS.items():
        conn.execute(update, {"entry_id": entry_id, "title": title, "tags": list(tags)})
