// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Legacy segment rows, per-table storage policy, and eviction selection.
//!
//! For legacy (local-L0) tables these rows are the authority on which sealed
//! Parquet files exist; for object tables the same rows are written only as
//! part of the derived projection. Aggregated stats and age/size eviction
//! candidates come from here.

use rusqlite::{Connection, OptionalExtension};

use super::*;
use crate::errors::StatsError;
use crate::store::policy::StoragePolicy;
use crate::store::types::{NamespaceStats, SegmentRow};

/// Decode one row from the ordered segment query projection.
pub(super) fn row_to_segment(row: &rusqlite::Row) -> rusqlite::Result<SegmentRow> {
    use crate::store::types::SegmentLocation;
    let loc: String = row.get(10)?;
    let partition_json: Option<String> = row.get(11)?;
    // Partition metadata is an optimization. Treat an unknown future encoding
    // as unpartitioned so query planning scans the segment conservatively.
    let partition = partition_json.and_then(|value| serde_json::from_str(&value).ok());
    Ok(SegmentRow {
        namespace: row.get(0)?,
        path: row.get(1)?,
        level: row.get(2)?,
        min_seq: row.get(3)?,
        max_seq: row.get(4)?,
        row_count: row.get(5)?,
        byte_size: row.get(6)?,
        created_at_ms: row.get(7)?,
        min_key_value: row.get(8)?,
        max_key_value: row.get(9)?,
        partition,
        location: SegmentLocation::parse_str(&loc).unwrap_or(SegmentLocation::Local),
    })
}

/// Insert-or-replace one `segments` row on `conn`. Shared by `upsert_segment`
/// (plain connection) and `replace_segments` (inside a transaction; a
/// `&Transaction` deref-coerces to `&Connection`).
pub(super) fn upsert_segment_in(conn: &Connection, row: &SegmentRow) -> Result<(), StatsError> {
    let partition_json = row
        .partition
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| StatsError::Internal(format!("serialize segment partition: {error}")))?;
    conn.execute(
        r#"
        INSERT INTO segments
            (namespace, path, level, min_seq, max_seq, row_count, byte_size,
             created_at_ms, min_key_value, max_key_value, location, partition_json)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
        ON CONFLICT (namespace, path) DO UPDATE SET
            level         = excluded.level,
            min_seq       = excluded.min_seq,
            max_seq       = excluded.max_seq,
            row_count     = excluded.row_count,
            byte_size     = excluded.byte_size,
            created_at_ms = excluded.created_at_ms,
            min_key_value = excluded.min_key_value,
            max_key_value = excluded.max_key_value,
            location      = excluded.location,
            partition_json = excluded.partition_json
        "#,
        rusqlite::params![
            row.namespace,
            row.path,
            row.level,
            row.min_seq,
            row.max_seq,
            row.row_count,
            row.byte_size,
            row.created_at_ms,
            row.min_key_value,
            row.max_key_value,
            row.location.as_str(),
            partition_json,
        ],
    )
    .map_err(sqlite_err)?;
    Ok(())
}

pub(super) fn remove_segments_in(
    conn: &Connection,
    namespace: &str,
    paths: &[String],
) -> Result<(), StatsError> {
    for path in paths {
        conn.execute(
            "DELETE FROM segments WHERE namespace = ?1 AND path = ?2",
            rusqlite::params![namespace, path],
        )
        .map_err(sqlite_err)?;
        conn.execute(
            "DELETE FROM object_segments WHERE namespace = ?1 AND path = ?2",
            rusqlite::params![namespace, path],
        )
        .map_err(sqlite_err)?;
    }
    Ok(())
}

impl Catalog {
    pub fn get_policy(&self, name: &str) -> Result<StoragePolicy, StatsError> {
        let inner = self.inner.lock().unwrap();
        let row = inner.conn.query_row(
            "SELECT max_segments, max_bytes, max_age_seconds FROM storage_policies WHERE namespace = ?1",
            [name],
            |row| {
                Ok(StoragePolicy {
                    max_segments: row.get::<_, Option<i32>>(0)?,
                    max_bytes: row.get::<_, Option<i64>>(1)?,
                    max_age_seconds: row.get::<_, Option<i64>>(2)?,
                })
            },
        );
        match row {
            Ok(p) => Ok(p),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(StoragePolicy::default()),
            Err(e) => Err(sqlite_err(e)),
        }
    }
    /// Persist `policy` (or delete the row if every field is `None`), keeping
    /// the live value in sync.
    pub fn upsert_policy(&self, name: &str, policy: &StoragePolicy) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        inner.upsert_policy_locked(name, policy)?;
        if let Some(ns) = inner.live.get_mut(name) {
            ns.policy = policy.clone();
        }
        Ok(())
    }

    // ----- segments table -----------------------------------------------
    /// Segment rows for `name` ordered by `min_seq`. All levels.
    pub fn list_segments(&self, name: &str) -> Result<Vec<SegmentRow>, StatsError> {
        self.list_segments_min_level(name, 0)
    }
    /// Segment rows for `name` with `level >= min_level`, ordered by `min_seq`.
    ///
    /// The sync/reconcile paths pass `min_level = 1` (L0 is local-only and never
    /// offloaded).
    pub fn list_segments_min_level(
        &self,
        name: &str,
        min_level: i32,
    ) -> Result<Vec<SegmentRow>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut stmt = inner
            .conn
            .prepare(
                "SELECT namespace, path, level, min_seq, max_seq, row_count, byte_size, \
                 created_at_ms, min_key_value, max_key_value, location, partition_json \
                 FROM segments WHERE namespace = ?1 AND level >= ?2 ORDER BY min_seq",
            )
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map(rusqlite::params![name, min_level], row_to_segment)
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(sqlite_err)?);
        }
        Ok(out)
    }
    /// Oldest evictable segment in `name` (`level >= 1 AND location = BOTH`,
    /// smallest `min_seq`), or `None`.
    pub fn select_eviction_candidate(&self, name: &str) -> Result<Option<SegmentRow>, StatsError> {
        use crate::store::types::SegmentLocation;
        let inner = self.inner.lock().unwrap();
        let row = inner
            .conn
            .query_row(
                "SELECT namespace, path, level, min_seq, max_seq, row_count, byte_size, \
                 created_at_ms, min_key_value, max_key_value, location, partition_json \
                 FROM segments WHERE namespace = ?1 AND level >= 1 AND location = ?2 \
                 ORDER BY min_seq ASC LIMIT 1",
                rusqlite::params![name, SegmentLocation::Both.as_str()],
                row_to_segment,
            )
            .optional()
            .map_err(sqlite_err)?;
        Ok(row)
    }
    /// Oldest-by-`created_at_ms` evictable segment past `cutoff_ms`
    /// (`level >= 1 AND location = BOTH AND created_at_ms < cutoff`), or `None`.
    ///
    /// Ordering by `created_at_ms` (not `min_seq`) matters because compaction
    /// outputs inherit their inputs' `min_seq` but get a fresh `created_at_ms`,
    /// so a low-`min_seq` segment can be the youngest.
    pub fn select_aged_eviction_candidate(
        &self,
        name: &str,
        cutoff_ms: i64,
    ) -> Result<Option<SegmentRow>, StatsError> {
        use crate::store::types::SegmentLocation;
        let inner = self.inner.lock().unwrap();
        let row = inner
            .conn
            .query_row(
                "SELECT namespace, path, level, min_seq, max_seq, row_count, byte_size, \
                 created_at_ms, min_key_value, max_key_value, location, partition_json \
                 FROM segments WHERE namespace = ?1 AND level >= 1 AND location = ?2 \
                 AND created_at_ms < ?3 ORDER BY created_at_ms ASC LIMIT 1",
                rusqlite::params![name, SegmentLocation::Both.as_str(), cutoff_ms],
                row_to_segment,
            )
            .optional()
            .map_err(sqlite_err)?;
        Ok(row)
    }
    /// Set `created_at_ms` for one segment row, serving the flag-gated
    /// `--debug-admin` `/debug/backdate` route (age tests, no sleep).
    pub fn set_created_at_ms(
        &self,
        namespace: &str,
        path: &str,
        created_at_ms: i64,
    ) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute(
                "UPDATE segments SET created_at_ms = ?1 WHERE namespace = ?2 AND path = ?3",
                rusqlite::params![created_at_ms, namespace, path],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }
    /// Insert or replace the segments-table row for `(namespace, path)`.
    ///
    /// Called by the per-namespace flush task after the parquet file is renamed
    /// into place and by boot adoption / compaction.
    pub fn upsert_segment(&self, row: &SegmentRow) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        upsert_segment_in(&inner.conn, row)
    }
    /// Insert or replace a set of segment rows in one durable transaction.
    pub fn upsert_segments(&self, rows: &[SegmentRow]) -> Result<(), StatsError> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.lock().unwrap();
        let tx = inner.conn.transaction().map_err(sqlite_err)?;
        for row in rows {
            upsert_segment_in(&tx, row)?;
        }
        tx.commit().map_err(sqlite_err)
    }
    /// Atomically swap `removed_paths` for `added` rows in one transaction.
    ///
    /// Compaction collapses N inputs at level n into one level-(n+1) output. The
    /// whole swap must be visible-or-not to `list_segments` — never half — so the
    /// deletes + upserts run inside a single sqlite transaction.
    pub fn replace_segments(
        &self,
        namespace: &str,
        removed_paths: &[String],
        added: &[SegmentRow],
    ) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let tx = inner.conn.transaction().map_err(sqlite_err)?;
        remove_segments_in(&tx, namespace, removed_paths)?;
        for seg in added {
            upsert_segment_in(&tx, seg)?;
        }
        tx.commit().map_err(sqlite_err)?;
        Ok(())
    }
    /// Update one segment's `location` (after upload completes / eviction).
    pub fn set_location(
        &self,
        namespace: &str,
        path: &str,
        location: crate::store::types::SegmentLocation,
    ) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute(
                "UPDATE segments SET location = ?1 WHERE namespace = ?2 AND path = ?3",
                rusqlite::params![location.as_str(), namespace, path],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }
    /// Update one segment's `byte_size` (after an in-place layout rewrite, which
    /// re-encodes the same rows and leaves every other field correct).
    pub fn set_byte_size(
        &self,
        namespace: &str,
        path: &str,
        byte_size: i64,
    ) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute(
                "UPDATE segments SET byte_size = ?1 WHERE namespace = ?2 AND path = ?3",
                rusqlite::params![byte_size, namespace, path],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }
    /// Drop one segment row. Idempotent.
    pub fn remove_segment(&self, namespace: &str, path: &str) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute(
                "DELETE FROM segments WHERE namespace = ?1 AND path = ?2",
                rusqlite::params![namespace, path],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn expire_migration_observation(&self, namespace: &str) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let updated = transaction
            .execute(
                "UPDATE table_migrations SET observation_deadline_ms = 0 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        if updated == 1 {
            transaction
                .execute(
                    "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                     WHERE namespace = ?1",
                    [namespace],
                )
                .map_err(sqlite_err)?;
        }
        transaction.commit().map_err(sqlite_err)?;
        Ok(())
    }
    /// Single-namespace aggregate over the segments table.
    pub fn aggregate_namespace_stats(&self, name: &str) -> Result<NamespaceStats, StatsError> {
        let inner = self.inner.lock().unwrap();
        let stats = inner
            .conn
            .query_row(
                r#"
                SELECT
                    COALESCE(SUM(row_count), 0),
                    COALESCE(SUM(byte_size), 0),
                    -- Seq window excludes empty segments (row_count > 0 filter),
                    -- matching the engine stats(). The RPC-visible NamespaceInfo
                    -- is fed by that filtered stats(), so this CASE filter is what
                    -- keeps NamespaceInfo correct on a dir with 0-row segments.
                    COALESCE(MIN(CASE WHEN row_count > 0 THEN min_seq END), 0),
                    COALESCE(MAX(CASE WHEN row_count > 0 THEN max_seq END), 0),
                    COUNT(*)
                FROM segments
                WHERE namespace = ?1
                "#,
                [name],
                |row| {
                    Ok(NamespaceStats {
                        row_count: row.get(0)?,
                        byte_size: row.get(1)?,
                        min_seq: row.get(2)?,
                        max_seq: row.get(3)?,
                        segment_count: row.get(4)?,
                    })
                },
            )
            .map_err(sqlite_err)?;
        Ok(stats)
    }
}
