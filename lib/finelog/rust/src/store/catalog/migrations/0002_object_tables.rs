use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute_batch(
        r#"
        CREATE TABLE table_specs (
            namespace TEXT    NOT NULL,
            version   INTEGER NOT NULL,
            spec_json TEXT    NOT NULL,
            spec_hash BLOB    NOT NULL,
            state     TEXT    NOT NULL,
            PRIMARY KEY (namespace, version)
        );
        CREATE TABLE table_heads (
            namespace                  TEXT    PRIMARY KEY,
            catalog_generation         INTEGER NOT NULL,
            active_table_spec_version  INTEGER NOT NULL,
            desired_table_spec_version INTEGER
        );
        CREATE TABLE object_segments (
            namespace            TEXT    NOT NULL,
            path                 TEXT    NOT NULL,
            table_spec_version   INTEGER NOT NULL,
            source_json          TEXT    NOT NULL,
            migration_backfill   INTEGER NOT NULL DEFAULT 0,
            migration_source_id  TEXT,
            migration_source_rows INTEGER,
            PRIMARY KEY (namespace, path)
        );
        CREATE TABLE table_migrations (
            namespace                TEXT    PRIMARY KEY,
            migration_id             TEXT    NOT NULL,
            from_version             INTEGER NOT NULL,
            to_version               INTEGER NOT NULL,
            phase                    TEXT    NOT NULL,
            fence_seq                INTEGER NOT NULL,
            source_generation        INTEGER NOT NULL,
            rows_total               INTEGER NOT NULL,
            rows_completed           INTEGER NOT NULL,
            phase_updated_at_ms       INTEGER NOT NULL
        );
        "#,
    )
    .map_err(|error| {
        StatsError::Internal(format!(
            "apply catalog migration 0002_object_tables: {error}"
        ))
    })
}
