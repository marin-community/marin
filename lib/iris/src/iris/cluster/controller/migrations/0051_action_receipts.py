# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add durable receipts for resource actions."""


def migrate(raw_conn) -> None:
    raw_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS action_receipts (
            action_id VARCHAR NOT NULL PRIMARY KEY,
            authority_cluster_id VARCHAR NOT NULL CHECK (authority_cluster_id <> ''),
            authority_action_id VARCHAR NOT NULL,
            action_kind VARCHAR NOT NULL
                CHECK (action_kind IN ('cancel_job', 'retry_task', 'terminate_attempt')),
            target_kind VARCHAR NOT NULL CHECK (target_kind IN ('job', 'task', 'attempt')),
            target_id VARCHAR NOT NULL,
            expected_target_uid VARCHAR NOT NULL CHECK (expected_target_uid <> ''),
            expected_attempt_uid VARCHAR NOT NULL DEFAULT '',
            backend_id VARCHAR NOT NULL DEFAULT '',
            execution_cluster_id VARCHAR NOT NULL,
            principal_id VARCHAR NOT NULL,
            client_idempotency_key VARCHAR NOT NULL,
            payload_hash VARCHAR NOT NULL,
            state VARCHAR NOT NULL CHECK (state IN ('accepted', 'verifying', 'succeeded', 'failed')),
            result_code VARCHAR NOT NULL
                CHECK (result_code IN (
                    'none', 'satisfied', 'target_absent', 'provider_rejected', 'internal_error'
                )),
            result_message VARCHAR NOT NULL DEFAULT '',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            completed_at_ms INTEGER,
            UNIQUE (authority_cluster_id, authority_action_id),
            UNIQUE (principal_id, action_kind, client_idempotency_key),
            CHECK (
                (action_kind = 'cancel_job' AND target_kind = 'job' AND expected_attempt_uid = '') OR
                (action_kind = 'retry_task' AND target_kind = 'task' AND expected_attempt_uid <> '') OR
                (action_kind = 'terminate_attempt' AND target_kind = 'attempt' AND expected_attempt_uid <> '')
            ),
            CHECK (action_kind = 'cancel_job' OR backend_id <> ''),
            CHECK (
                (state IN ('accepted', 'verifying') AND result_code = 'none' AND completed_at_ms IS NULL) OR
                (state = 'succeeded' AND result_code IN ('satisfied', 'target_absent')
                    AND completed_at_ms IS NOT NULL) OR
                (state = 'failed' AND result_code IN ('provider_rejected', 'internal_error')
                    AND completed_at_ms IS NOT NULL)
            )
        )
        """
    )
    raw_conn.execute("CREATE INDEX IF NOT EXISTS action_receipts_state ON action_receipts (state, updated_at_ms)")
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS action_receipts_target " "ON action_receipts (target_kind, target_id, updated_at_ms)"
    )
    raw_conn.execute(
        "CREATE INDEX IF NOT EXISTS action_receipts_principal ON action_receipts (principal_id, updated_at_ms)"
    )
