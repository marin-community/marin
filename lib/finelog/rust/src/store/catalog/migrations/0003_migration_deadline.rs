use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute(
        "ALTER TABLE table_migrations ADD COLUMN observation_deadline_ms INTEGER NOT NULL DEFAULT 0",
        [],
    )
    .map_err(|error| {
        StatsError::Internal(format!(
            "apply catalog migration 0003_migration_deadline: {error}"
        ))
    })?;
    Ok(())
}
