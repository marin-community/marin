use rusqlite::Connection;

use crate::errors::StatsError;

/// Drop columns that were only ever written: `table_specs.state` and the
/// `table_migrations` bookkeeping nobody reads back.
pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute_batch(
        "ALTER TABLE table_specs DROP COLUMN state;
         ALTER TABLE table_migrations DROP COLUMN migration_id;
         ALTER TABLE table_migrations DROP COLUMN source_generation;
         ALTER TABLE table_migrations DROP COLUMN phase_updated_at_ms;",
    )
    .map_err(|error| {
        StatsError::Internal(format!(
            "apply catalog migration 0006_drop_recorded_only_columns: {error}"
        ))
    })
}
