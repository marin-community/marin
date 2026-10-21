use rusqlite::Connection;

use crate::errors::StatsError;

pub(super) fn apply(conn: &Connection) -> Result<(), StatsError> {
    conn.execute_batch(
        r#"
        ALTER TABLE object_segments ADD COLUMN artifacts_json TEXT;
        "#,
    )
    .map_err(|error| {
        StatsError::Internal(format!(
            "apply catalog migration 0004_segment_artifacts: {error}"
        ))
    })
}
