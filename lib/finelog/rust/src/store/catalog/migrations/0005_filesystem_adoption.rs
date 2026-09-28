use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute(
        "ALTER TABLE table_heads
         ADD COLUMN filesystem_adoption_disabled INTEGER NOT NULL DEFAULT 0",
        [],
    )
    .map_err(|error| {
        StatsError::Internal(format!(
            "apply catalog migration 0005_filesystem_adoption: {error}"
        ))
    })?;
    Ok(())
}
