use rusqlite::Connection;

use crate::errors::StatsError;

/// Add `object_segments.column_bounds`, a JSON list of a segment's recorded
/// column ranges. NULL means no range is recorded yet.
pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute_batch("ALTER TABLE object_segments ADD COLUMN column_bounds TEXT;")
        .map_err(|error| {
            StatsError::Internal(format!(
                "apply catalog migration 0007_column_bounds: {error}"
            ))
        })
}
