# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add append-only node lifecycle events captured by the controller."""

SOURCE_CHECK = (
    "source IN ("
    "'gcp_tpu_api', "
    "'gcp_queued_resource', "
    "'gcp_audit_log', "
    "'iris_autoscaler', "
    "'iris_bootstrap', "
    "'iris_worker_health'"
    ")"
)
REASON_CHECK = (
    "reason IN ("
    "'gcp_preemption', "
    "'cloud_deleted_or_missing', "
    "'cloud_deleting', "
    "'queued_resource_failed', "
    "'queued_resource_suspended', "
    "'iris_scale_down', "
    "'bootstrap_failed', "
    "'heartbeat_lost', "
    "'health_probe_failed'"
    ")"
)
CONFIDENCE_CHECK = "confidence IN ('reported', 'controller_action', 'inferred', 'unknown')"


def migrate(raw_conn) -> None:
    raw_conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS node_lifecycle_events (
            event_id VARCHAR NOT NULL,
            observed_at_ms INTEGER NOT NULL,
            event_time_ms INTEGER,
            provider VARCHAR DEFAULT '' NOT NULL,
            source VARCHAR NOT NULL,
            reason VARCHAR NOT NULL,
            confidence VARCHAR NOT NULL,
            scale_group VARCHAR DEFAULT '' NOT NULL,
            slice_id VARCHAR DEFAULT '' NOT NULL,
            worker_id VARCHAR DEFAULT '' NOT NULL,
            node_name VARCHAR DEFAULT '' NOT NULL,
            zone VARCHAR DEFAULT '' NOT NULL,
            device_type VARCHAR DEFAULT '' NOT NULL,
            device_variant VARCHAR DEFAULT '' NOT NULL,
            capacity_type VARCHAR DEFAULT '' NOT NULL,
            task_id VARCHAR DEFAULT '' NOT NULL,
            attempt_id INTEGER,
            cloud_state VARCHAR DEFAULT '' NOT NULL,
            previous_state VARCHAR DEFAULT '' NOT NULL,
            message VARCHAR DEFAULT '' NOT NULL,
            raw_json VARCHAR DEFAULT '{{}}' NOT NULL,
            PRIMARY KEY (event_id),
            CONSTRAINT node_lifecycle_events_source_check CHECK ({SOURCE_CHECK}),
            CONSTRAINT node_lifecycle_events_reason_check CHECK ({REASON_CHECK}),
            CONSTRAINT node_lifecycle_events_confidence_check CHECK ({CONFIDENCE_CHECK})
        )
        """
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_lifecycle_observed_at " "ON node_lifecycle_events (observed_at_ms)"
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_lifecycle_scale_group_observed "
        "ON node_lifecycle_events (scale_group, observed_at_ms)"
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_lifecycle_slice_observed "
        "ON node_lifecycle_events (slice_id, observed_at_ms)"
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_lifecycle_worker_observed "
        "ON node_lifecycle_events (worker_id, observed_at_ms)"
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_lifecycle_reason_observed "
        "ON node_lifecycle_events (reason, observed_at_ms)"
    )
