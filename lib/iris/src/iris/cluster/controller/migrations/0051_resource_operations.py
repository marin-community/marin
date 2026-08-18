# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add durable generic resource operations."""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_operations (
            operation_id VARCHAR NOT NULL PRIMARY KEY,
            principal_id VARCHAR NOT NULL,
            request_id VARCHAR NOT NULL,
            request_hash VARCHAR NOT NULL,
            verb VARCHAR NOT NULL,
            requested_authority VARCHAR NOT NULL,
            requested_type VARCHAR NOT NULL,
            requested_id VARCHAR NOT NULL,
            requested_uid VARCHAR,
            resolved_authority VARCHAR NOT NULL,
            resolved_type VARCHAR NOT NULL,
            resolved_id VARCHAR NOT NULL,
            resolved_uid VARCHAR NOT NULL,
            update_type_url VARCHAR NOT NULL,
            update_payload BLOB NOT NULL,
            reason VARCHAR NOT NULL DEFAULT '',
            phase VARCHAR NOT NULL CHECK (phase IN ('accepted', 'verifying', 'verified', 'failed')),
            peer_id VARCHAR NOT NULL DEFAULT '',
            remote_operation_id VARCHAR NOT NULL DEFAULT '',
            error_code VARCHAR,
            error_message VARCHAR NOT NULL DEFAULT '',
            accepted_at_ms INTEGER NOT NULL,
            applied_at_ms INTEGER,
            completed_at_ms INTEGER,
            CONSTRAINT resource_operations_principal_request_key UNIQUE (principal_id, request_id)
        )
        """
    )
    raw_conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_resource_operations_phase
        ON resource_operations (phase, accepted_at_ms)
        """
    )
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS resource_operation_targets (
            operation_id VARCHAR NOT NULL
                REFERENCES resource_operations(operation_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            authority_cluster_id VARCHAR NOT NULL,
            resource_type VARCHAR NOT NULL,
            resource_id VARCHAR NOT NULL,
            resource_uid VARCHAR NOT NULL,
            backend_id VARCHAR NOT NULL,
            worker_id VARCHAR,
            PRIMARY KEY (operation_id, ordinal)
        )
        """
    )
