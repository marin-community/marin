//! Ordered SQLite migrations for the local Finelog catalog.

use rusqlite::{Connection, OptionalExtension};

use crate::errors::StatsError;

#[path = "0000_init.rs"]
mod init;
#[path = "0003_migration_deadline.rs"]
mod migration_deadline;
#[path = "0002_object_tables.rs"]
mod object_tables;
#[path = "0004_segment_artifacts.rs"]
mod segment_artifacts;
#[path = "0001_segment_partitions.rs"]
mod segment_partitions;

type ApplyMigration = fn(&Connection) -> Result<(), StatsError>;

const MIGRATIONS: &[(i64, &str, ApplyMigration)] = &[
    (0, "init", init::apply),
    (1, "segment_partitions", segment_partitions::apply),
    (2, "object_tables", object_tables::apply),
    (3, "migration_deadline", migration_deadline::apply),
    (4, "segment_artifacts", segment_artifacts::apply),
];

fn sqlite_error(error: rusqlite::Error) -> StatsError {
    StatsError::Internal(format!("catalog sqlite migration error: {error}"))
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool, StatsError> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1",
        [table],
        |_| Ok(()),
    )
    .optional()
    .map(|value| value.is_some())
    .map_err(sqlite_error)
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool, StatsError> {
    let mut statement = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(sqlite_error)?;
    let columns = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(sqlite_error)?;
    for existing in columns {
        if existing.map_err(sqlite_error)? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn record_migration(conn: &Connection, version: i64, name: &str) -> Result<(), StatsError> {
    conn.execute(
        "INSERT INTO schema_migrations (version, name, applied_at_ms)
         VALUES (?1, ?2, CAST(strftime('%s', 'now') AS INTEGER) * 1000)",
        rusqlite::params![version, name],
    )
    .map_err(sqlite_error)?;
    Ok(())
}

fn bootstrap_preledger(conn: &mut Connection) -> Result<(), StatsError> {
    if table_exists(conn, "schema_migrations")? {
        return Ok(());
    }
    let transaction = conn.transaction().map_err(sqlite_error)?;
    transaction
        .execute_batch(
            "CREATE TABLE schema_migrations (
                version       INTEGER PRIMARY KEY,
                name          TEXT    NOT NULL,
                applied_at_ms INTEGER NOT NULL
            );",
        )
        .map_err(sqlite_error)?;
    if table_exists(&transaction, "segments")? {
        record_migration(&transaction, 0, "init")?;
        if column_exists(&transaction, "segments", "partition_json")? {
            record_migration(&transaction, 1, "segment_partitions")?;
        }
        if table_exists(&transaction, "table_specs")?
            && table_exists(&transaction, "table_heads")?
            && table_exists(&transaction, "object_segments")?
            && table_exists(&transaction, "table_migrations")?
        {
            record_migration(&transaction, 2, "object_tables")?;
            if column_exists(&transaction, "table_migrations", "observation_deadline_ms")? {
                record_migration(&transaction, 3, "migration_deadline")?;
                if column_exists(&transaction, "object_segments", "artifacts_json")? {
                    record_migration(&transaction, 4, "segment_artifacts")?;
                }
            }
        }
    }
    transaction.commit().map_err(sqlite_error)
}

pub(super) fn migrate(conn: &mut Connection) -> Result<(), StatsError> {
    bootstrap_preledger(conn)?;
    let latest_known = MIGRATIONS.last().expect("catalog has migrations").0;
    let latest_applied: Option<i64> = conn
        .query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
            row.get(0)
        })
        .map_err(sqlite_error)?;
    if latest_applied.is_some_and(|version| version > latest_known) {
        return Err(StatsError::Internal(format!(
            "catalog schema version {} is newer than this binary supports ({latest_known})",
            latest_applied.unwrap_or_default()
        )));
    }

    for &(version, name, apply) in MIGRATIONS {
        let applied = conn
            .query_row(
                "SELECT 1 FROM schema_migrations WHERE version = ?1",
                [version],
                |_| Ok(()),
            )
            .optional()
            .map_err(sqlite_error)?
            .is_some();
        if applied {
            continue;
        }
        let transaction = conn.transaction().map_err(sqlite_error)?;
        apply(&transaction)?;
        record_migration(&transaction, version, name)?;
        transaction.commit().map_err(sqlite_error)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn applied(conn: &Connection) -> Vec<i64> {
        let mut statement = conn
            .prepare("SELECT version FROM schema_migrations ORDER BY version")
            .unwrap();
        statement
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    }

    #[test]
    fn fresh_database_runs_every_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        assert_eq!(applied(&conn), vec![0, 1, 2, 3, 4]);
        assert!(column_exists(&conn, "segments", "partition_json").unwrap());
        assert!(table_exists(&conn, "object_segments").unwrap());
    }

    #[test]
    fn preledger_database_is_stamped_before_upgrade() {
        let mut conn = Connection::open_in_memory().unwrap();
        init::apply(&conn).unwrap();
        migrate(&mut conn).unwrap();
        assert_eq!(applied(&conn), vec![0, 1, 2, 3, 4]);
        assert!(column_exists(&conn, "segments", "partition_json").unwrap());
    }

    #[test]
    fn preledger_partition_schema_is_stamped_without_reapplying_alter() {
        let mut conn = Connection::open_in_memory().unwrap();
        init::apply(&conn).unwrap();
        segment_partitions::apply(&conn).unwrap();
        migrate(&mut conn).unwrap();
        assert_eq!(applied(&conn), vec![0, 1, 2, 3, 4]);
        assert!(table_exists(&conn, "object_segments").unwrap());
    }

    #[test]
    fn preledger_object_schema_is_stamped_before_deadline_upgrade() {
        let mut conn = Connection::open_in_memory().unwrap();
        init::apply(&conn).unwrap();
        segment_partitions::apply(&conn).unwrap();
        object_tables::apply(&conn).unwrap();
        migrate(&mut conn).unwrap();
        assert_eq!(applied(&conn), vec![0, 1, 2, 3, 4]);
        assert!(column_exists(&conn, "table_migrations", "observation_deadline_ms").unwrap());
        assert!(column_exists(&conn, "object_segments", "artifacts_json").unwrap());
    }

    #[test]
    fn future_schema_version_is_rejected() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at_ms) VALUES (99, 'future', 0)",
            [],
        )
        .unwrap();
        let error = migrate(&mut conn).unwrap_err();
        assert!(matches!(error, StatsError::Internal(_)));
    }
}
