use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute_batch(
        r#"
        CREATE TABLE namespaces (
            namespace        TEXT PRIMARY KEY,
            schema_json      TEXT NOT NULL,
            registered_at_ms INTEGER NOT NULL,
            last_modified_ms INTEGER NOT NULL
        );
        CREATE TABLE storage_policies (
            namespace        TEXT PRIMARY KEY,
            max_segments     INTEGER,
            max_bytes        INTEGER,
            max_age_seconds  INTEGER
        );
        CREATE TABLE segments (
            namespace     TEXT    NOT NULL,
            path          TEXT    NOT NULL,
            level         INTEGER NOT NULL,
            min_seq       INTEGER NOT NULL,
            max_seq       INTEGER NOT NULL,
            row_count     INTEGER NOT NULL,
            byte_size     INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL,
            min_key_value TEXT,
            max_key_value TEXT,
            location      TEXT    NOT NULL,
            PRIMARY KEY (namespace, path)
        );
        CREATE INDEX segments_ns_level_minseq ON segments (namespace, level, min_seq);
        CREATE TABLE forward_state (
            target    TEXT    NOT NULL,
            namespace TEXT    NOT NULL,
            cursor    INTEGER NOT NULL,
            PRIMARY KEY (target, namespace)
        );
        "#,
    )
    .map_err(|error| StatsError::Internal(format!("apply catalog migration 0000_init: {error}")))
}
