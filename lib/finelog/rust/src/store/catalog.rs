//! Catalog: single source of truth for namespace state.
//!
//! Backed by a `rusqlite` sidecar at `{data_dir}/_finelog_catalog.sqlite`.
//! `data_dir = None` selects an in-memory sqlite.
//!
//! Three coupled pieces of state under one mutex:
//! - the live `RegisteredNamespace` registry (`live`) + registration order
//!   (`registered_at`),
//! - the `dropping` reservation set (fences concurrent register during a drop),
//! - the sqlite connection (`namespaces`, `storage_policies`, `segments`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, OptionalExtension};

use crate::errors::StatsError;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_json, schema_to_json, Schema};
use crate::store::types::{NamespaceStats, SegmentRow};

/// Sidecar filename.
pub const CATALOG_DB_FILENAME: &str = "_finelog_catalog.sqlite";

/// A live namespace value.
#[derive(Debug, Clone)]
pub struct RegisteredNamespace {
    pub name: String,
    pub schema: Schema,
    pub policy: StoragePolicy,
}

struct CatalogInner {
    conn: Connection,
    live: BTreeMap<String, RegisteredNamespace>,
    /// Monotonic insertion ordinal per name; renders `list_namespaces` in
    /// registration order.
    registered_at: HashMap<String, u64>,
    next_ordinal: u64,
    dropping: HashSet<String>,
}

/// Single source of truth for namespace state, persistent and live.
pub struct Catalog {
    inner: Mutex<CatalogInner>,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn sqlite_err(e: rusqlite::Error) -> StatsError {
    StatsError::Internal(format!("catalog sqlite error: {e}"))
}

/// Map one selected `segments` row (in the canonical 11-column SELECT order) to
/// a `SegmentRow`. Shared by every segment-row reader so the column order stays
/// in lockstep with the SELECT lists.
fn row_to_segment(row: &rusqlite::Row) -> rusqlite::Result<SegmentRow> {
    use crate::store::types::SegmentLocation;
    let loc: String = row.get(10)?;
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
        location: SegmentLocation::parse_str(&loc).unwrap_or(SegmentLocation::Local),
    })
}

/// Insert-or-replace one `segments` row on `conn`. Shared by `upsert_segment`
/// (plain connection) and `replace_segments` (inside a transaction; a
/// `&Transaction` deref-coerces to `&Connection`).
fn upsert_segment_in(conn: &Connection, row: &SegmentRow) -> Result<(), StatsError> {
    conn.execute(
        r#"
        INSERT INTO segments
            (namespace, path, level, min_seq, max_seq, row_count, byte_size,
             created_at_ms, min_key_value, max_key_value, location)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        ON CONFLICT (namespace, path) DO UPDATE SET
            level         = excluded.level,
            min_seq       = excluded.min_seq,
            max_seq       = excluded.max_seq,
            row_count     = excluded.row_count,
            byte_size     = excluded.byte_size,
            created_at_ms = excluded.created_at_ms,
            min_key_value = excluded.min_key_value,
            max_key_value = excluded.max_key_value,
            location      = excluded.location
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
        ],
    )
    .map_err(sqlite_err)?;
    Ok(())
}

impl Catalog {
    /// Open the catalog. `data_dir = None` -> in-memory; otherwise the sidecar
    /// lives at `{data_dir}/_finelog_catalog.sqlite`. Creates the three tables
    /// idempotently.
    pub fn open(data_dir: Option<&Path>) -> Result<Catalog, StatsError> {
        let conn = match data_dir {
            None => Connection::open_in_memory().map_err(sqlite_err)?,
            Some(dir) => Connection::open(dir.join(CATALOG_DB_FILENAME)).map_err(sqlite_err)?,
        };
        Self::create_tables(&conn)?;
        Ok(Catalog {
            inner: Mutex::new(CatalogInner {
                conn,
                live: BTreeMap::new(),
                registered_at: HashMap::new(),
                next_ordinal: 0,
                dropping: HashSet::new(),
            }),
        })
    }

    fn create_tables(conn: &Connection) -> Result<(), StatsError> {
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS namespaces (
                namespace        TEXT PRIMARY KEY,
                schema_json      TEXT NOT NULL,
                registered_at_ms INTEGER NOT NULL,
                last_modified_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS storage_policies (
                namespace        TEXT PRIMARY KEY,
                max_segments     INTEGER,
                max_bytes        INTEGER,
                max_age_seconds  INTEGER
            );
            CREATE TABLE IF NOT EXISTS segments (
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
            CREATE INDEX IF NOT EXISTS segments_ns_level_minseq ON segments (namespace, level, min_seq);
            CREATE TABLE IF NOT EXISTS forward_state (
                target    TEXT    NOT NULL,
                namespace TEXT    NOT NULL,
                cursor    INTEGER NOT NULL,
                PRIMARY KEY (target, namespace)
            );
            "#,
        )
        .map_err(sqlite_err)
    }

    // ----- forward watermark ---------------------------------------------

    /// The seq in `namespace` below which nothing will be sent to `target` again, or
    /// `None` if this store has never forwarded that namespace there. A high-water mark
    /// of what is settled, which is not the same as what `target` holds: a sender may
    /// give a row up rather than deliver it.
    ///
    /// Keyed by `(target, namespace)` so each table advances independently and repointing
    /// a forwarder reseeds instead of replaying one store's seq space into another's.
    pub fn forward_cursor(&self, target: &str, namespace: &str) -> Result<Option<i64>, StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .query_row(
                "SELECT cursor FROM forward_state WHERE target = ?1 AND namespace = ?2",
                [target, namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)
    }

    /// Record `cursor` as settled for `(target, namespace)`. Callers write it only once
    /// the rows below it can never be sent again, so a crash mid-batch re-forwards that
    /// batch rather than losing it.
    pub fn set_forward_cursor(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute(
                "INSERT INTO forward_state (target, namespace, cursor) VALUES (?1, ?2, ?3)
                 ON CONFLICT(target, namespace) DO UPDATE SET cursor = excluded.cursor",
                rusqlite::params![target, namespace, cursor],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }

    // ----- live namespace registry --------------------------------------

    pub fn contains(&self, name: &str) -> bool {
        self.inner.lock().unwrap().live.contains_key(name)
    }

    pub fn get_live(&self, name: &str) -> Option<RegisteredNamespace> {
        self.inner.lock().unwrap().live.get(name).cloned()
    }

    pub fn require_live(&self, name: &str) -> Result<RegisteredNamespace, StatsError> {
        self.get_live(name).ok_or_else(|| {
            StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
        })
    }

    pub fn is_dropping(&self, name: &str) -> bool {
        self.inner.lock().unwrap().dropping.contains(name)
    }

    /// Live namespaces in registration order.
    pub fn snapshot_live(&self) -> Vec<RegisteredNamespace> {
        let inner = self.inner.lock().unwrap();
        let mut entries: Vec<&RegisteredNamespace> = inner.live.values().collect();
        entries.sort_by_key(|ns| inner.registered_at.get(&ns.name).copied().unwrap_or(0));
        entries.into_iter().cloned().collect()
    }

    /// Publish a freshly-built namespace (rehydrate path).
    pub fn insert_live(&self, ns: RegisteredNamespace) {
        let mut inner = self.inner.lock().unwrap();
        inner.publish_locked(ns);
    }

    /// Atomically register `name` or evolve the existing namespace.
    ///
    /// The whole decision-and-publish runs under a SINGLE lock so it cannot
    /// interleave with `begin_drop`/`finish_drop`. Releasing the lock between
    /// the drop-fence check and publish is unsafe: because RPC handlers dispatch blocking
    /// `Store` calls onto a multi-threaded `spawn_blocking` pool sharing one
    /// `Arc<Store>`, a concurrent register+drop of the same name could resurrect
    /// a dropped namespace with no persisted row.
    ///
    /// On a fresh registration, persists `stored_schema` + `policy` and
    /// publishes, returning `(stored_schema, policy)`. On an existing namespace,
    /// `merge` computes the effective schema from the existing one (a PURE
    /// function — it must not call back into the catalog, since the lock is
    /// held); the effective schema is persisted only if it changed, and an
    /// empty `policy` preserves the existing policy.
    ///
    /// Raises `InvalidNamespace` if a drop is in flight.
    pub fn register_or_evolve(
        &self,
        name: &str,
        stored_schema: Schema,
        policy: StoragePolicy,
        merge: impl FnOnce(&Schema) -> Result<Schema, StatsError>,
    ) -> Result<(Schema, StoragePolicy), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.dropping.contains(name) {
            return Err(StatsError::InvalidNamespace(format!(
                "namespace {name:?} is currently being dropped; retry once drop_table completes"
            )));
        }

        if let Some(existing) = inner.live.get(name).cloned() {
            // `merge` raises SchemaConflict on a non-additive change.
            let effective = merge(&existing.schema)?;
            if effective != existing.schema {
                inner.upsert_locked(name, &effective)?;
            }
            let effective_policy = if policy.is_empty() {
                existing.policy.clone()
            } else {
                inner.upsert_policy_locked(name, &policy)?;
                policy
            };
            inner.publish_locked(RegisteredNamespace {
                name: name.to_string(),
                schema: effective.clone(),
                policy: effective_policy.clone(),
            });
            return Ok((effective, effective_policy));
        }

        inner.upsert_locked(name, &stored_schema)?;
        inner.upsert_policy_locked(name, &policy)?;
        inner.publish_locked(RegisteredNamespace {
            name: name.to_string(),
            schema: stored_schema.clone(),
            policy: policy.clone(),
        });
        Ok((stored_schema, policy))
    }

    /// Pop `name` from the registry and reserve it in `dropping`.
    pub fn begin_drop(&self, name: &str) -> Result<RegisteredNamespace, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let ns = inner.live.remove(name).ok_or_else(|| {
            StatsError::NamespaceNotFound(format!("namespace {name:?} is not registered"))
        })?;
        inner.registered_at.remove(name);
        inner.dropping.insert(name.to_string());
        Ok(ns)
    }

    pub fn finish_drop(&self, name: &str) {
        self.inner.lock().unwrap().dropping.remove(name);
    }

    // ----- namespaces table ---------------------------------------------

    /// All persisted `(name, schema)` rows (used by rehydrate).
    pub fn list_all(&self) -> Result<Vec<(String, Schema)>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut stmt = inner
            .conn
            .prepare("SELECT namespace, schema_json FROM namespaces")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(sqlite_err)?;
        let mut out = Vec::new();
        for r in rows {
            let (name, json) = r.map_err(sqlite_err)?;
            out.push((name, schema_from_json(&json)?));
        }
        Ok(out)
    }

    /// Remove the namespace, its segment rows, and its policy row. Idempotent.
    pub fn delete(&self, name: &str) -> Result<(), StatsError> {
        let inner = self.inner.lock().unwrap();
        inner
            .conn
            .execute("DELETE FROM segments WHERE namespace = ?1", [name])
            .map_err(sqlite_err)?;
        inner
            .conn
            .execute("DELETE FROM storage_policies WHERE namespace = ?1", [name])
            .map_err(sqlite_err)?;
        inner
            .conn
            .execute("DELETE FROM namespaces WHERE namespace = ?1", [name])
            .map_err(sqlite_err)?;
        Ok(())
    }

    /// Insert or evolve the row for `name`, keeping the live value in sync.
    /// `registered_at_ms` is preserved on update; `last_modified_ms` is bumped.
    pub fn upsert(&self, name: &str, schema: &Schema) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        inner.upsert_locked(name, schema)?;
        if let Some(ns) = inner.live.get_mut(name) {
            ns.schema = schema.clone();
        }
        Ok(())
    }

    // ----- storage_policies table ---------------------------------------

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
                 created_at_ms, min_key_value, max_key_value, location \
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
                 created_at_ms, min_key_value, max_key_value, location \
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
                 created_at_ms, min_key_value, max_key_value, location \
                 FROM segments WHERE namespace = ?1 AND level >= 1 AND location = ?2 \
                 AND created_at_ms < ?3 ORDER BY created_at_ms ASC LIMIT 1",
                rusqlite::params![name, SegmentLocation::Both.as_str(), cutoff_ms],
                row_to_segment,
            )
            .optional()
            .map_err(sqlite_err)?;
        Ok(row)
    }

    /// Set `created_at_ms` for one segment row. Used only by the test-only
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
        for path in removed_paths {
            tx.execute(
                "DELETE FROM segments WHERE namespace = ?1 AND path = ?2",
                rusqlite::params![namespace, path],
            )
            .map_err(sqlite_err)?;
        }
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

impl CatalogInner {
    /// Persist the `namespaces` row for `name` (no live-registry update — the
    /// caller publishes). `registered_at_ms` is preserved on update;
    /// `last_modified_ms` is bumped. Operates on the held guard so it composes
    /// inside a single `register_or_evolve` critical section.
    fn upsert_locked(&mut self, name: &str, schema: &Schema) -> Result<(), StatsError> {
        let now = now_ms();
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT registered_at_ms FROM namespaces WHERE namespace = ?1",
                [name],
                |row| row.get(0),
            )
            .ok();
        let registered_at = existing.unwrap_or(now);
        self.conn
            .execute(
                r#"
                INSERT INTO namespaces (namespace, schema_json, registered_at_ms, last_modified_ms)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT (namespace) DO UPDATE
                  SET schema_json = excluded.schema_json,
                      last_modified_ms = excluded.last_modified_ms
                "#,
                rusqlite::params![name, schema_to_json(schema), registered_at, now],
            )
            .map_err(sqlite_err)?;
        Ok(())
    }

    /// Persist `policy` for `name`, or delete the row when every field is
    /// `None`. No live-registry update (the caller publishes).
    fn upsert_policy_locked(
        &mut self,
        name: &str,
        policy: &StoragePolicy,
    ) -> Result<(), StatsError> {
        if policy.is_empty() {
            self.conn
                .execute("DELETE FROM storage_policies WHERE namespace = ?1", [name])
                .map_err(sqlite_err)?;
        } else {
            self.conn
                .execute(
                    r#"
                    INSERT INTO storage_policies (namespace, max_segments, max_bytes, max_age_seconds)
                    VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT (namespace) DO UPDATE
                      SET max_segments    = excluded.max_segments,
                          max_bytes       = excluded.max_bytes,
                          max_age_seconds = excluded.max_age_seconds
                    "#,
                    rusqlite::params![name, policy.max_segments, policy.max_bytes, policy.max_age_seconds],
                )
                .map_err(sqlite_err)?;
        }
        Ok(())
    }

    fn publish_locked(&mut self, ns: RegisteredNamespace) {
        let name = ns.name.clone();
        self.live.insert(name.clone(), ns);
        self.registered_at.entry(name).or_insert_with(|| {
            let o = self.next_ordinal;
            self.next_ordinal += 1;
            o
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::schema::{with_implicit_seq, Column};

    fn worker_stored() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ))
    }

    #[test]
    fn open_in_memory_and_register_fresh() {
        let cat = Catalog::open(None).unwrap();
        let (schema, policy) = cat
            .register_or_evolve("a", worker_stored(), StoragePolicy::default(), |_| {
                panic!("fresh register should not call merge")
            })
            .unwrap();
        assert_eq!(schema, worker_stored());
        assert!(policy.is_empty());
        assert!(cat.contains("a"));
    }

    #[test]
    fn re_evolve_merges_existing() {
        let cat = Catalog::open(None).unwrap();
        cat.register_or_evolve(
            "a",
            worker_stored(),
            StoragePolicy::default(),
            |_| unreachable!(),
        )
        .unwrap();
        let (schema, _) = cat
            .register_or_evolve("a", worker_stored(), StoragePolicy::default(), |existing| {
                Ok(existing.clone())
            })
            .unwrap();
        assert_eq!(schema, worker_stored());
    }

    #[test]
    fn upsert_schema_round_trips_through_json() {
        let cat = Catalog::open(None).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
        let all = cat.list_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "a");
        assert_eq!(all[0].1, worker_stored());
    }

    #[test]
    fn upsert_preserves_registered_at_and_bumps_last_modified() {
        let cat = Catalog::open(None).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
        let inner = cat.inner.lock().unwrap();
        let (reg1, mod1): (i64, i64) = inner
            .conn
            .query_row(
                "SELECT registered_at_ms, last_modified_ms FROM namespaces WHERE namespace='a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        drop(inner);
        std::thread::sleep(std::time::Duration::from_millis(2));
        cat.upsert("a", &worker_stored()).unwrap();
        let inner = cat.inner.lock().unwrap();
        let (reg2, mod2): (i64, i64) = inner
            .conn
            .query_row(
                "SELECT registered_at_ms, last_modified_ms FROM namespaces WHERE namespace='a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(reg1, reg2, "registered_at preserved");
        assert!(mod2 >= mod1, "last_modified bumped");
    }

    #[test]
    fn aggregate_stats_empty_when_no_segments() {
        let cat = Catalog::open(None).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
        assert_eq!(
            cat.aggregate_namespace_stats("a").unwrap(),
            NamespaceStats::empty()
        );
        assert!(cat.list_segments("a").unwrap().is_empty());
    }

    #[test]
    fn begin_drop_fences_register() {
        let cat = Catalog::open(None).unwrap();
        cat.register_or_evolve(
            "a",
            worker_stored(),
            StoragePolicy::default(),
            |_| unreachable!(),
        )
        .unwrap();
        cat.begin_drop("a").unwrap();
        assert!(cat.is_dropping("a"));
        let err = cat.register_or_evolve(
            "a",
            worker_stored(),
            StoragePolicy::default(),
            |_| unreachable!(),
        );
        assert!(matches!(err, Err(StatsError::InvalidNamespace(_))));
        cat.finish_drop("a");
        assert!(!cat.is_dropping("a"));
    }

    #[test]
    fn snapshot_live_returns_registration_order() {
        let cat = Catalog::open(None).unwrap();
        for name in ["zeta", "alpha", "mid"] {
            cat.register_or_evolve(
                name,
                worker_stored(),
                StoragePolicy::default(),
                |_| unreachable!(),
            )
            .unwrap();
        }
        let order: Vec<String> = cat.snapshot_live().into_iter().map(|ns| ns.name).collect();
        assert_eq!(order, vec!["zeta", "alpha", "mid"]);
    }

    #[test]
    fn upsert_policy_empty_deletes_row() {
        let cat = Catalog::open(None).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
        cat.upsert_policy(
            "a",
            &StoragePolicy {
                max_segments: Some(7),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(cat.get_policy("a").unwrap().max_segments, Some(7));
        cat.upsert_policy("a", &StoragePolicy::default()).unwrap();
        assert!(cat.get_policy("a").unwrap().is_empty());
    }

    #[test]
    fn on_disk_catalog_persists_across_reopen() {
        let dir = tempdir();
        {
            let cat = Catalog::open(Some(&dir)).unwrap();
            cat.upsert("a", &worker_stored()).unwrap();
        }
        let cat = Catalog::open(Some(&dir)).unwrap();
        let all = cat.list_all().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "a");
        std::fs::remove_dir_all(&dir).ok();
    }

    fn tempdir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_catalog_test_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
