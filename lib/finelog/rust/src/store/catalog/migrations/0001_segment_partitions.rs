use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute("ALTER TABLE segments ADD COLUMN partition_json TEXT", [])
        .map(|_| ())
        .map_err(|error| {
            StatsError::Internal(format!(
                "apply catalog migration 0001_segment_partitions: {error}"
            ))
        })
}
