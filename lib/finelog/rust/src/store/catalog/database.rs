//! SQLite catalog: single source of truth for namespace state.
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
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    ForwardCursor, MigrationPhase, NamespaceCatalog, ObjectRef, TableMigrationStatus,
    TableSpec as ProtoTableSpec,
};
use crate::store::object_store::ObjectId;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_json, schema_to_json, Schema};
use crate::store::table_spec::{
    canonical_json_bytes, classify_definition_change, migration_phase_for_state,
    rollback_window_ms, table_spec_from_json, DefinitionChange,
};
use crate::store::table_state::{ArtifactReferences, SegmentDescriptor, TableRevision};
use crate::store::types::{NamespaceStats, SegmentRow};

/// Sidecar filename.
pub const CATALOG_DB_FILENAME: &str = "_finelog_catalog.sqlite";

/// Every table keyed by `namespace`, ordered so the `namespaces` row that
/// defines the namespace is removed last.
const NAMESPACE_OWNED_TABLES: [&str; 8] = [
    "segments",
    "storage_policies",
    "table_specs",
    "object_segments",
    "table_migrations",
    "table_heads",
    "forward_state",
    "namespaces",
];

/// A live namespace value.
#[derive(Debug, Clone)]
pub struct RegisteredNamespace {
    pub name: String,
    pub schema: Schema,
    pub policy: StoragePolicy,
}

#[derive(Debug, Clone)]
pub struct TableSpecStatus {
    pub active: Option<ProtoTableSpec>,
    pub desired: Option<ProtoTableSpec>,
    pub phase: MigrationPhase,
    pub catalog_generation: u64,
    pub migration: Option<TableMigrationStatus>,
}

#[derive(Debug, Clone)]
pub struct ObjectSegmentRecord {
    pub path: String,
    pub table_spec_version: u64,
    pub source: ObjectRef,
    pub artifacts: ArtifactReferences,
    pub migration_backfill: bool,
    pub migration_source_id: Option<String>,
    pub migration_source_rows: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct PublishedObjectSegment {
    pub row: SegmentRow,
    pub table_spec_version: u64,
    pub source: ObjectRef,
    pub migration_backfill: bool,
    pub migration_source_id: Option<String>,
    pub migration_source_rows: Option<i64>,
}

struct MigrationStatusRow {
    migration_id: String,
    from_version: i64,
    to_version: i64,
    phase: String,
    fence_seq: i64,
    source_generation: i64,
    rows_total: i64,
    rows_completed: i64,
    observation_deadline_ms: i64,
}

struct MigrationCheckpoint {
    to_version: i64,
    rows_total: i64,
    phase: String,
}

impl TableSpecStatus {
    /// Return the query-visible TableSpec version, or zero for a legacy table.
    pub fn active_version(&self) -> u64 {
        self.active
            .as_ref()
            .and_then(|spec| spec.version)
            .unwrap_or(0)
    }

    /// Return the migration target version, or zero when no transition is active.
    pub fn desired_version(&self) -> u64 {
        self.desired
            .as_ref()
            .and_then(|spec| spec.version)
            .unwrap_or(0)
    }
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

/// Encode a segment's artifact references for the `object_segments` row.
///
/// A segment with no artifacts stores SQL NULL rather than an empty document.
fn artifacts_json(artifacts: &ArtifactReferences) -> Result<Option<String>, StatsError> {
    if artifacts.is_empty() && artifacts.binding == Default::default() {
        return Ok(None);
    }
    serde_json::to_string(artifacts)
        .map(Some)
        .map_err(|error| StatsError::Internal(format!("serialize segment artifacts: {error}")))
}

fn parse_artifacts(json: Option<&str>) -> Result<ArtifactReferences, StatsError> {
    let Some(json) = json else {
        return Ok(ArtifactReferences::default());
    };
    serde_json::from_str(json)
        .map_err(|error| StatsError::Internal(format!("decode segment artifacts: {error}")))
}

fn sqlite_err(e: rusqlite::Error) -> StatsError {
    StatsError::Internal(format!("catalog sqlite error: {e}"))
}

fn table_spec_for_version(
    conn: &Connection,
    namespace: &str,
    version: u64,
) -> Result<Option<ProtoTableSpec>, StatsError> {
    if version == 0 {
        return Ok(None);
    }
    let json: Option<String> = conn
        .query_row(
            "SELECT spec_json FROM table_specs WHERE namespace = ?1 AND version = ?2",
            rusqlite::params![namespace, version],
            |row| row.get(0),
        )
        .optional()
        .map_err(sqlite_err)?;
    json.as_deref().map(table_spec_from_json).transpose()
}

fn table_spec_status_in(conn: &Connection, namespace: &str) -> Result<TableSpecStatus, StatsError> {
    let head: Option<(i64, i64, Option<i64>)> = conn
        .query_row(
            "SELECT catalog_generation, active_table_spec_version, desired_table_spec_version
             FROM table_heads WHERE namespace = ?1",
            [namespace],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()
        .map_err(sqlite_err)?;
    let Some((generation, active_version, desired_version)) = head else {
        return Ok(TableSpecStatus {
            active: None,
            desired: None,
            phase: MigrationPhase::MIGRATION_PHASE_UNSPECIFIED,
            catalog_generation: 0,
            migration: None,
        });
    };
    let active = table_spec_for_version(conn, namespace, active_version as u64)?;
    let desired = desired_version
        .map(|version| table_spec_for_version(conn, namespace, version as u64))
        .transpose()?
        .flatten();
    let migration = migration_status_in(conn, namespace)?;
    let phase = migration
        .as_ref()
        .and_then(|migration| migration.phase)
        .and_then(|phase| phase.as_known())
        .unwrap_or_else(|| migration_phase_for_state(desired.is_some()));
    Ok(TableSpecStatus {
        active,
        phase,
        desired,
        catalog_generation: generation as u64,
        migration,
    })
}

fn migration_status_in(
    conn: &Connection,
    namespace: &str,
) -> Result<Option<TableMigrationStatus>, StatsError> {
    let row: Option<MigrationStatusRow> = conn
        .query_row(
            "SELECT migration_id, from_version, to_version, phase, fence_seq,
                    source_generation, rows_total, rows_completed, observation_deadline_ms
             FROM table_migrations WHERE namespace = ?1",
            [namespace],
            |row| {
                Ok(MigrationStatusRow {
                    migration_id: row.get(0)?,
                    from_version: row.get(1)?,
                    to_version: row.get(2)?,
                    phase: row.get(3)?,
                    fence_seq: row.get(4)?,
                    source_generation: row.get(5)?,
                    rows_total: row.get(6)?,
                    rows_completed: row.get(7)?,
                    observation_deadline_ms: row.get(8)?,
                })
            },
        )
        .optional()
        .map_err(sqlite_err)?;
    row.map(|row| {
        Ok(TableMigrationStatus {
            migration_id: Some(row.migration_id),
            from_version: Some(row.from_version as u64),
            to_version: Some(row.to_version as u64),
            phase: Some(migration_phase_from_str(&row.phase)?.into()),
            fence_seq: Some(row.fence_seq),
            source_generation: Some(row.source_generation as u64),
            rows_total: Some(row.rows_total),
            rows_completed: Some(row.rows_completed),
            observation_deadline_ms: Some(row.observation_deadline_ms),
            ..Default::default()
        })
    })
    .transpose()
}

fn migration_phase_str(phase: MigrationPhase) -> &'static str {
    match phase {
        MigrationPhase::MIGRATION_PHASE_DUAL_WRITE => "DUAL_WRITE",
        MigrationPhase::MIGRATION_PHASE_BACKFILL => "BACKFILL",
        MigrationPhase::MIGRATION_PHASE_VERIFY => "VERIFY",
        MigrationPhase::MIGRATION_PHASE_ACTIVATED => "ACTIVATED",
        MigrationPhase::MIGRATION_PHASE_OBSERVING => "OBSERVING",
        MigrationPhase::MIGRATION_PHASE_RETIRED => "RETIRED",
        MigrationPhase::MIGRATION_PHASE_UNSPECIFIED => "UNSPECIFIED",
    }
}

fn migration_phase_from_str(value: &str) -> Result<MigrationPhase, StatsError> {
    match value {
        "DUAL_WRITE" => Ok(MigrationPhase::MIGRATION_PHASE_DUAL_WRITE),
        "BACKFILL" => Ok(MigrationPhase::MIGRATION_PHASE_BACKFILL),
        "VERIFY" => Ok(MigrationPhase::MIGRATION_PHASE_VERIFY),
        "ACTIVATED" => Ok(MigrationPhase::MIGRATION_PHASE_ACTIVATED),
        "OBSERVING" => Ok(MigrationPhase::MIGRATION_PHASE_OBSERVING),
        "RETIRED" => Ok(MigrationPhase::MIGRATION_PHASE_RETIRED),
        _ => Err(StatsError::Internal(format!(
            "unknown table migration phase {value:?}"
        ))),
    }
}

/// Decode one row from the ordered segment query projection.
fn row_to_segment(row: &rusqlite::Row) -> rusqlite::Result<SegmentRow> {
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
fn upsert_segment_in(conn: &Connection, row: &SegmentRow) -> Result<(), StatsError> {
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

fn remove_segments_in(
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
    /// Open the catalog. `data_dir = None` uses memory; otherwise the sidecar
    /// lives at `{data_dir}/_finelog_catalog.sqlite`. Initializes its schema
    /// idempotently.
    pub fn open(data_dir: Option<&Path>) -> Result<Catalog, StatsError> {
        let mut conn = match data_dir {
            None => Connection::open_in_memory().map_err(sqlite_err)?,
            Some(dir) => Connection::open(dir.join(CATALOG_DB_FILENAME)).map_err(sqlite_err)?,
        };
        let journal_mode: String = if data_dir.is_some() {
            // WAL requires shared memory and is not safe on the network-backed
            // filesystems used by Kubernetes finelog. PERSIST keeps rollback
            // journaling while avoiding a journal-file delete on every commit.
            conn.query_row("PRAGMA journal_mode = PERSIST", [], |row| row.get(0))
                .map_err(sqlite_err)?
        } else {
            conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .map_err(sqlite_err)?
        };
        conn.execute_batch("PRAGMA synchronous = FULL;")
            .map_err(sqlite_err)?;
        let synchronous: i64 = conn
            .query_row("PRAGMA synchronous", [], |row| row.get(0))
            .map_err(sqlite_err)?;
        super::migrations::migrate(&mut conn)?;
        tracing::info!(
            persistent = data_dir.is_some(),
            journal_mode,
            synchronous,
            "finelog catalog sqlite ready"
        );
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

    pub fn table_spec_status(&self, namespace: &str) -> Result<TableSpecStatus, StatsError> {
        let inner = self.inner.lock().unwrap();
        table_spec_status_in(&inner.conn, namespace)
    }

    /// Rebuild the complete local projection from a verified remote catalog.
    pub fn replace_with_published_snapshot(
        &self,
        namespace: &str,
        schema: Schema,
        policy: StoragePolicy,
        snapshot: &NamespaceCatalog,
        segments: &[PublishedObjectSegment],
    ) -> Result<(), StatsError> {
        let remote_generation = snapshot.catalog_generation.unwrap_or(0);
        if remote_generation == 0 {
            return Err(StatsError::Internal(format!(
                "published table state for {namespace:?} has no revision"
            )));
        }
        let mut inner = self.inner.lock().unwrap();
        let local_generation: Option<i64> = inner
            .conn
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        if local_generation == Some(remote_generation as i64) {
            return Ok(());
        }

        inner.upsert_locked(namespace, &schema)?;
        inner.upsert_policy_locked(namespace, &policy)?;
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        for table in [
            "segments",
            "object_segments",
            "table_specs",
            "table_migrations",
            "forward_state",
        ] {
            transaction
                .execute(
                    &format!("DELETE FROM {table} WHERE namespace = ?1"),
                    [namespace],
                )
                .map_err(sqlite_err)?;
        }
        for spec in &snapshot.retained_table_specs {
            let version = spec.version.unwrap_or(0);
            let spec_bytes = canonical_json_bytes(spec)?;
            let state = if version == snapshot.desired_table_spec_version.unwrap_or(0) {
                "DESIRED"
            } else if version == snapshot.active_table_spec_version.unwrap_or(0) {
                "ACTIVE"
            } else {
                "RETAINED"
            };
            transaction
                .execute(
                    "INSERT INTO table_specs (namespace, version, spec_json, spec_hash, state)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![
                        namespace,
                        version as i64,
                        String::from_utf8(spec_bytes.clone()).expect("JSON is UTF-8"),
                        Sha256::digest(&spec_bytes).as_slice(),
                        state,
                    ],
                )
                .map_err(sqlite_err)?;
        }
        // A published state whose version-0 import has already activated proves
        // the table's history lives in objects, so recovery re-establishes the
        // adoption block even when the local catalog was lost. A block already
        // recorded locally survives, because `excluded` never lowers it.
        let imported_from_version_zero = snapshot.migration.as_option().is_some_and(|migration| {
            migration.from_version.unwrap_or(0) == 0
                && matches!(
                    migration.phase.and_then(|phase| phase.as_known()),
                    Some(
                        MigrationPhase::MIGRATION_PHASE_OBSERVING
                            | MigrationPhase::MIGRATION_PHASE_RETIRED
                    )
                )
        });
        transaction
            .execute(
                "INSERT INTO table_heads
                    (namespace, catalog_generation, active_table_spec_version,
                     desired_table_spec_version, filesystem_adoption_disabled)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(namespace) DO UPDATE SET
                    catalog_generation = excluded.catalog_generation,
                    active_table_spec_version = excluded.active_table_spec_version,
                    desired_table_spec_version = excluded.desired_table_spec_version,
                    filesystem_adoption_disabled = MAX(
                        table_heads.filesystem_adoption_disabled,
                        excluded.filesystem_adoption_disabled)",
                rusqlite::params![
                    namespace,
                    remote_generation as i64,
                    snapshot.active_table_spec_version.unwrap_or(0) as i64,
                    (snapshot.desired_table_spec_version.unwrap_or(0) > 0)
                        .then_some(snapshot.desired_table_spec_version.unwrap_or(0) as i64),
                    imported_from_version_zero,
                ],
            )
            .map_err(sqlite_err)?;
        for segment in segments {
            upsert_segment_in(&transaction, &segment.row)?;
            if segment
                .source
                .object_id
                .as_deref()
                .is_none_or(|id| ObjectId::parse(id).is_err())
            {
                continue;
            }
            let source_json = serde_json::to_string(&segment.source).map_err(|error| {
                StatsError::Internal(format!("serialize published object source: {error}"))
            })?;
            transaction
                .execute(
                    "INSERT OR REPLACE INTO object_segments
                        (namespace, path, table_spec_version, source_json, migration_backfill,
                         migration_source_id, migration_source_rows)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    rusqlite::params![
                        namespace,
                        segment.row.path,
                        segment.table_spec_version as i64,
                        source_json,
                        segment.migration_backfill,
                        segment.migration_source_id,
                        segment.migration_source_rows,
                    ],
                )
                .map_err(sqlite_err)?;
        }
        if let Some(migration) = snapshot.migration.as_option() {
            let phase = migration
                .phase
                .and_then(|phase| phase.as_known())
                .unwrap_or(MigrationPhase::MIGRATION_PHASE_UNSPECIFIED);
            transaction
                .execute(
                    "INSERT INTO table_migrations
                        (namespace, migration_id, from_version, to_version, phase,
                         fence_seq, source_generation, rows_total, rows_completed,
                         phase_updated_at_ms, observation_deadline_ms)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    rusqlite::params![
                        namespace,
                        migration.migration_id.as_deref().unwrap_or("published"),
                        migration.from_version.unwrap_or(0) as i64,
                        migration.to_version.unwrap_or(0) as i64,
                        migration_phase_str(phase),
                        migration.fence_seq.unwrap_or(-1),
                        migration.source_generation.unwrap_or(0) as i64,
                        migration.rows_total.unwrap_or(0),
                        migration.rows_completed.unwrap_or(0),
                        now_ms(),
                        migration.observation_deadline_ms.unwrap_or(0),
                    ],
                )
                .map_err(sqlite_err)?;
        }
        for cursor in &snapshot.forward_cursors {
            transaction
                .execute(
                    "INSERT OR REPLACE INTO forward_state (target, namespace, cursor)
                     VALUES (?1, ?2, ?3)",
                    rusqlite::params![
                        cursor.target.as_deref().unwrap_or(""),
                        namespace,
                        cursor.cursor.unwrap_or(-1),
                    ],
                )
                .map_err(sqlite_err)?;
        }
        transaction.commit().map_err(sqlite_err)?;
        inner.publish_locked(RegisteredNamespace {
            name: namespace.to_string(),
            schema,
            policy,
        });
        Ok(())
    }

    /// Check that `spec` is a legal next definition for `namespace` without
    /// committing anything, so registration rejects an impossible version
    /// before it builds an engine.
    pub fn validate_table_spec_registration(
        &self,
        namespace: &str,
        spec: &ProtoTableSpec,
        expected_hash: &[u8; 32],
        has_rows: bool,
    ) -> Result<(), StatsError> {
        let version = spec.version.unwrap_or(0);
        let version_i64 = i64::try_from(version).map_err(|_| {
            StatsError::SchemaValidation(format!(
                "table_spec.version {version} exceeds the supported range"
            ))
        })?;
        let inner = self.inner.lock().unwrap();
        let existing_hash: Option<Vec<u8>> = inner
            .conn
            .query_row(
                "SELECT spec_hash FROM table_specs WHERE namespace = ?1 AND version = ?2",
                rusqlite::params![namespace, version_i64],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        if let Some(existing_hash) = existing_hash {
            return if existing_hash.as_slice() == expected_hash {
                Ok(())
            } else {
                Err(StatsError::SchemaConflict(format!(
                    "table_spec version {version} is already registered with different contents"
                )))
            };
        }
        let status = table_spec_status_in(&inner.conn, namespace)?;
        if status.desired.is_some() {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {namespace:?} already has a table specification transition in progress"
            )));
        }
        if status.phase == MigrationPhase::MIGRATION_PHASE_OBSERVING {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {namespace:?} is still in the rollback observation window"
            )));
        }
        let highest: i64 = inner
            .conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM table_specs WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        let expected = highest as u64 + 1;
        if version != expected {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec version {version} rejected; expected {expected}"
            )));
        }
        classify_definition_change(status.active.as_ref(), spec, has_rows)?;
        Ok(())
    }

    pub fn register_table_spec(
        &self,
        namespace: &str,
        spec: &ProtoTableSpec,
        expected_hash: &[u8; 32],
        has_rows: bool,
    ) -> Result<TableSpecStatus, StatsError> {
        let version = spec.version.unwrap_or(0);
        let version_i64 = i64::try_from(version).map_err(|_| {
            StatsError::SchemaValidation(format!(
                "table_spec.version {version} exceeds the supported range"
            ))
        })?;
        let spec_bytes = canonical_json_bytes(spec)?;
        let spec_json = String::from_utf8(spec_bytes).expect("JSON serialization is UTF-8");
        let mut inner = self.inner.lock().unwrap();

        let existing_hash: Option<Vec<u8>> = inner
            .conn
            .query_row(
                "SELECT spec_hash FROM table_specs WHERE namespace = ?1 AND version = ?2",
                rusqlite::params![namespace, version_i64],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        if let Some(existing_hash) = existing_hash {
            if existing_hash.as_slice() != expected_hash {
                return Err(StatsError::SchemaConflict(format!(
                    "table_spec version {version} is already registered with different contents"
                )));
            }
            return table_spec_status_in(&inner.conn, namespace);
        }

        let status = table_spec_status_in(&inner.conn, namespace)?;
        if status.desired.is_some() {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {namespace:?} already has a table specification transition in progress"
            )));
        }
        let highest: i64 = inner
            .conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM table_specs WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        let expected = highest as u64 + 1;
        if version != expected {
            return Err(StatsError::SchemaConflict(format!(
                "table_spec version {version} rejected; expected {expected}"
            )));
        }

        // A metadata-only change activates in this commit. A physical rewrite
        // over rows that already exist records the pending transition instead,
        // and background maintenance backfills and activates it.
        let migrate = matches!(
            classify_definition_change(status.active.as_ref(), spec, has_rows)?,
            DefinitionChange::CompatibleRewrite
        );
        let next_generation = status.catalog_generation + 1;
        let active_version = status.active_version();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        transaction
            .execute(
                "INSERT INTO table_specs (namespace, version, spec_json, spec_hash, state)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    namespace,
                    version_i64,
                    spec_json,
                    expected_hash.as_slice(),
                    if migrate { "DESIRED" } else { "ACTIVE" },
                ],
            )
            .map_err(sqlite_err)?;
        if migrate {
            let (rows_total, fence_seq): (i64, i64) = transaction
                .query_row(
                    "SELECT COALESCE(SUM(row_count), 0), COALESCE(MAX(max_seq), 0)
                     FROM segments WHERE namespace = ?1",
                    [namespace],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(sqlite_err)?;
            transaction
                .execute(
                    "INSERT OR REPLACE INTO table_migrations
                        (namespace, migration_id, from_version, to_version, phase,
                         fence_seq, source_generation, rows_total, rows_completed,
                         phase_updated_at_ms, observation_deadline_ms)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9, 0)",
                    rusqlite::params![
                        namespace,
                        format!("{namespace}-{active_version}-{version}-{next_generation}"),
                        active_version as i64,
                        version_i64,
                        migration_phase_str(MigrationPhase::MIGRATION_PHASE_DUAL_WRITE),
                        fence_seq,
                        status.catalog_generation as i64,
                        rows_total,
                        now_ms(),
                    ],
                )
                .map_err(sqlite_err)?;
        }

        if !migrate && active_version > 0 {
            transaction
                .execute(
                    "UPDATE table_specs SET state = 'RETAINED'
                     WHERE namespace = ?1 AND version = ?2",
                    rusqlite::params![namespace, active_version as i64],
                )
                .map_err(sqlite_err)?;
            // A metadata-only change leaves every object's physical layout
            // valid, so the same commit that activates the new version carries
            // the existing segments onto it. Nothing is rewritten and nothing
            // leaves the query view.
            transaction
                .execute(
                    "UPDATE object_segments SET table_spec_version = ?3
                     WHERE namespace = ?1 AND table_spec_version = ?2",
                    rusqlite::params![namespace, active_version as i64, version_i64],
                )
                .map_err(sqlite_err)?;
        }
        transaction
            .execute(
                "INSERT INTO table_heads
                    (namespace, catalog_generation, active_table_spec_version,
                     desired_table_spec_version)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(namespace) DO UPDATE SET
                    catalog_generation = excluded.catalog_generation,
                    active_table_spec_version = excluded.active_table_spec_version,
                    desired_table_spec_version = excluded.desired_table_spec_version",
                rusqlite::params![
                    namespace,
                    next_generation as i64,
                    if migrate { active_version } else { version } as i64,
                    migrate.then_some(version_i64),
                ],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        table_spec_status_in(&inner.conn, namespace)
    }

    pub fn activate_desired_table_spec(
        &self,
        namespace: &str,
    ) -> Result<TableSpecStatus, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = table_spec_status_in(&inner.conn, namespace)?;
        let desired = status.desired_version();
        if desired == 0 {
            return Ok(status);
        }
        let active = status.active_version();
        let next_generation = status.catalog_generation + 1;
        // The window the prior definition stays rollbackable for is the new
        // definition's own rollback window, not its query-time bound.
        let observation_ms = status
            .desired
            .as_ref()
            .map(rollback_window_ms)
            .unwrap_or(crate::store::table_spec::DEFAULT_ROLLBACK_WINDOW_MS);
        let activated_at_ms = now_ms();
        let observation_deadline_ms =
            activated_at_ms.saturating_add(i64::try_from(observation_ms).unwrap_or(i64::MAX));
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        if active > 0 {
            transaction
                .execute(
                    "UPDATE table_specs SET state = 'RETAINED'
                     WHERE namespace = ?1 AND version = ?2",
                    rusqlite::params![namespace, active as i64],
                )
                .map_err(sqlite_err)?;
        }
        transaction
            .execute(
                "UPDATE table_specs SET state = 'ACTIVE'
                 WHERE namespace = ?1 AND version = ?2",
                rusqlite::params![namespace, desired as i64],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_migrations SET phase = ?2, phase_updated_at_ms = ?3,
                    observation_deadline_ms = ?4
                 WHERE namespace = ?1",
                rusqlite::params![
                    namespace,
                    migration_phase_str(MigrationPhase::MIGRATION_PHASE_OBSERVING),
                    activated_at_ms,
                    observation_deadline_ms,
                ],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = ?2,
                    active_table_spec_version = ?3,
                    desired_table_spec_version = NULL
                 WHERE namespace = ?1",
                rusqlite::params![namespace, next_generation as i64, desired as i64],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        table_spec_status_in(&inner.conn, namespace)
    }

    /// Abort the one in-flight migration and restore its source version.
    ///
    /// Backfill-only target objects become unreachable; writes accepted during
    /// the migration are reassigned to the source version so abort never drops
    /// rows. Published snapshots retain remote bytes until catalog GC.
    pub fn abort_table_migration(&self, namespace: &str) -> Result<TableSpecStatus, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = table_spec_status_in(&inner.conn, namespace)?;
        let migration = status.migration.as_ref().ok_or_else(|| {
            StatsError::SchemaConflict(format!(
                "namespace {namespace:?} has no table migration to abort"
            ))
        })?;
        if status.phase == MigrationPhase::MIGRATION_PHASE_RETIRED {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {namespace:?} migration is already retired"
            )));
        }
        let from_version = migration.from_version.unwrap_or(0);
        let to_version = migration.to_version.unwrap_or(0);
        let next_generation = status.catalog_generation + 1;
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let backfill_paths = {
            let mut statement = transaction
                .prepare(
                    "SELECT path FROM object_segments
                     WHERE namespace = ?1 AND table_spec_version = ?2
                       AND migration_backfill = 1",
                )
                .map_err(sqlite_err)?;
            let rows = statement
                .query_map(rusqlite::params![namespace, to_version as i64], |row| {
                    row.get::<_, String>(0)
                })
                .map_err(sqlite_err)?;
            let mut paths = Vec::new();
            for row in rows {
                paths.push(row.map_err(sqlite_err)?);
            }
            paths
        };
        remove_segments_in(&transaction, namespace, &backfill_paths)?;
        transaction
            .execute(
                "UPDATE object_segments
                 SET table_spec_version = ?3,
                     migration_backfill = 0,
                     migration_source_id = NULL,
                     migration_source_rows = NULL
                 WHERE namespace = ?1 AND table_spec_version = ?2",
                rusqlite::params![namespace, to_version as i64, from_version as i64],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "DELETE FROM table_specs WHERE namespace = ?1 AND version = ?2",
                rusqlite::params![namespace, to_version as i64],
            )
            .map_err(sqlite_err)?;
        if from_version > 0 {
            transaction
                .execute(
                    "UPDATE table_specs SET state = 'ACTIVE'
                     WHERE namespace = ?1 AND version = ?2",
                    rusqlite::params![namespace, from_version as i64],
                )
                .map_err(sqlite_err)?;
        }
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = ?2,
                    active_table_spec_version = ?3
                 WHERE namespace = ?1",
                rusqlite::params![namespace, next_generation as i64, from_version as i64],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_heads SET desired_table_spec_version = NULL WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "DELETE FROM table_migrations WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        table_spec_status_in(&inner.conn, namespace)
    }

    pub fn update_migration_phase(
        &self,
        namespace: &str,
        expected: MigrationPhase,
        phase: MigrationPhase,
    ) -> Result<TableSpecStatus, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = table_spec_status_in(&inner.conn, namespace)?;
        if status.phase == phase {
            return Ok(status);
        }
        if status.phase != expected {
            return Err(StatsError::SchemaConflict(format!(
                "namespace {namespace:?} migration is {:?}, expected {:?}",
                status.phase, expected
            )));
        }
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_migrations SET phase = ?2, phase_updated_at_ms = ?3
                 WHERE namespace = ?1",
                rusqlite::params![namespace, migration_phase_str(phase), now_ms()],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        table_spec_status_in(&inner.conn, namespace)
    }

    pub fn retire_observed_migration(
        &self,
        namespace: &str,
    ) -> Result<TableSpecStatus, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = table_spec_status_in(&inner.conn, namespace)?;
        if status.phase != MigrationPhase::MIGRATION_PHASE_OBSERVING {
            return Ok(status);
        }
        let observation_deadline_ms: i64 = inner
            .conn
            .query_row(
                "SELECT observation_deadline_ms FROM table_migrations WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        if now_ms() < observation_deadline_ms {
            return Ok(status);
        }
        let migration = status.migration.as_ref().ok_or_else(|| {
            StatsError::Internal(format!(
                "observing namespace {namespace:?} has no migration state"
            ))
        })?;
        let from_version = migration.from_version.unwrap_or(0);
        let fence_seq = migration.fence_seq.unwrap_or(i64::MAX);
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let retired_paths = {
            let (sql, version): (&str, Option<i64>) = if from_version == 0 {
                (
                    "SELECT segments.path FROM segments
                     LEFT JOIN object_segments
                       ON object_segments.namespace = segments.namespace
                      AND object_segments.path = segments.path
                     WHERE segments.namespace = ?1
                       AND segments.max_seq <= ?2
                       AND object_segments.path IS NULL",
                    None,
                )
            } else {
                (
                    "SELECT path FROM object_segments
                     WHERE namespace = ?1 AND table_spec_version = ?2",
                    Some(from_version as i64),
                )
            };
            let mut statement = transaction.prepare(sql).map_err(sqlite_err)?;
            let rows = statement
                .query_map(
                    rusqlite::params![namespace, version.unwrap_or(fence_seq)],
                    |row| row.get::<_, String>(0),
                )
                .map_err(sqlite_err)?;
            let mut paths = Vec::new();
            for row in rows {
                paths.push(row.map_err(sqlite_err)?);
            }
            paths
        };
        remove_segments_in(&transaction, namespace, &retired_paths)?;
        transaction
            .execute(
                "UPDATE object_segments
                 SET migration_backfill = 0,
                     migration_source_id = NULL,
                     migration_source_rows = NULL
                 WHERE namespace = ?1 AND table_spec_version = ?2",
                rusqlite::params![namespace, status.active_version() as i64],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_migrations SET phase = ?2, phase_updated_at_ms = ?3
                 WHERE namespace = ?1",
                rusqlite::params![
                    namespace,
                    migration_phase_str(MigrationPhase::MIGRATION_PHASE_RETIRED),
                    now_ms(),
                ],
            )
            .map_err(sqlite_err)?;
        // Retiring a version-0 source completes the table's legacy import. Its
        // history now lives in immutable objects the table state references, so
        // the directory it was imported from is no longer a load source.
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1,
                    filesystem_adoption_disabled =
                        CASE WHEN ?2 THEN 1 ELSE filesystem_adoption_disabled END
                 WHERE namespace = ?1",
                rusqlite::params![namespace, from_version == 0],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        table_spec_status_in(&inner.conn, namespace)
    }

    /// Whether `namespace` has completed its version-0 import and must no
    /// longer be rebuilt from the parquet files on disk.
    pub fn filesystem_adoption_disabled(&self, namespace: &str) -> Result<bool, StatsError> {
        let inner = self.inner.lock().unwrap();
        let disabled: Option<bool> = inner
            .conn
            .query_row(
                "SELECT filesystem_adoption_disabled FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        Ok(disabled.unwrap_or(false))
    }

    pub fn retained_table_specs(&self, namespace: &str) -> Result<Vec<ProtoTableSpec>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut statement = inner
            .conn
            .prepare("SELECT spec_json FROM table_specs WHERE namespace = ?1 ORDER BY version")
            .map_err(sqlite_err)?;
        let rows = statement
            .query_map([namespace], |row| row.get::<_, String>(0))
            .map_err(sqlite_err)?;
        let mut specs = Vec::new();
        for row in rows {
            specs.push(table_spec_from_json(&row.map_err(sqlite_err)?)?);
        }
        Ok(specs)
    }

    /// Commit every object produced by one sealed buffer in one generation.
    pub fn commit_object_segments(
        &self,
        segments: &[SegmentDescriptor],
        table_spec_version: u64,
        migration_backfill: bool,
    ) -> Result<TableRevision, StatsError> {
        if segments.is_empty() {
            return Err(StatsError::Internal(
                "an object segment commit must carry at least one segment".to_string(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let namespace = &segments[0].row.namespace;
        for descriptor in segments {
            let SegmentDescriptor {
                row,
                source,
                artifacts,
            } = descriptor;
            if &row.namespace != namespace {
                return Err(StatsError::Internal(
                    "one object commit cannot span namespaces".to_string(),
                ));
            }
            let source_json = serde_json::to_string(source).map_err(|error| {
                StatsError::Internal(format!("serialize object segment source: {error}"))
            })?;
            upsert_segment_in(&transaction, row)?;
            transaction
                .execute(
                    "INSERT INTO object_segments
                        (namespace, path, table_spec_version, source_json, artifacts_json,
                         migration_backfill, migration_source_id, migration_source_rows)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL)
                     ON CONFLICT(namespace, path) DO UPDATE SET
                        table_spec_version = excluded.table_spec_version,
                        source_json = excluded.source_json,
                        artifacts_json = excluded.artifacts_json,
                        migration_backfill = excluded.migration_backfill,
                        migration_source_id = NULL,
                        migration_source_rows = NULL",
                    rusqlite::params![
                        row.namespace,
                        row.path,
                        table_spec_version as i64,
                        source_json,
                        artifacts_json(artifacts)?,
                        migration_backfill,
                    ],
                )
                .map_err(sqlite_err)?;
        }
        let changed = transaction
            .execute(
                "UPDATE table_heads
                 SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        if changed != 1 {
            return Err(StatsError::Internal(format!(
                "object segment committed for {:?} without a namespace head",
                namespace
            )));
        }
        let generation: i64 = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |result| result.get(0),
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation as u64))
    }

    /// Replace immutable objects atomically and advance one catalog generation.
    pub fn replace_object_segments(
        &self,
        namespace: &str,
        removed_paths: &[String],
        segments: &[SegmentDescriptor],
        table_spec_version: u64,
        migration_backfill: bool,
    ) -> Result<TableRevision, StatsError> {
        if removed_paths.is_empty() || segments.is_empty() {
            return Err(StatsError::Internal(
                "object replacement inputs and outputs must be non-empty".to_string(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        remove_segments_in(&transaction, namespace, removed_paths)?;
        for descriptor in segments {
            let SegmentDescriptor {
                row,
                source,
                artifacts,
            } = descriptor;
            if row.namespace != namespace {
                return Err(StatsError::Internal(
                    "one object replacement cannot span namespaces".to_string(),
                ));
            }
            upsert_segment_in(&transaction, row)?;
            transaction
                .execute(
                    "INSERT INTO object_segments
                        (namespace, path, table_spec_version, source_json, artifacts_json,
                         migration_backfill, migration_source_id, migration_source_rows)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL)",
                    rusqlite::params![
                        namespace,
                        row.path,
                        table_spec_version as i64,
                        serde_json::to_string(source).map_err(|error| {
                            StatsError::Internal(format!("serialize replacement source: {error}"))
                        })?,
                        artifacts_json(artifacts)?,
                        migration_backfill,
                    ],
                )
                .map_err(sqlite_err)?;
        }
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        let generation: i64 = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation as u64))
    }

    /// Commit every output from one migrated source and checkpoint it once.
    pub fn commit_migration_segments(
        &self,
        segments: &[SegmentDescriptor],
        table_spec_version: u64,
        migration_source_id: &str,
        migration_source_rows: i64,
    ) -> Result<TableRevision, StatsError> {
        if segments.is_empty() || migration_source_rows <= 0 {
            return Err(StatsError::Internal(
                "a migration checkpoint must carry at least one segment and one source row"
                    .to_string(),
            ));
        }
        let namespace = &segments[0].row.namespace;
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let migration: MigrationCheckpoint = transaction
            .query_row(
                "SELECT to_version, rows_total, phase FROM table_migrations
                 WHERE namespace = ?1",
                [namespace],
                |result| {
                    Ok(MigrationCheckpoint {
                        to_version: result.get(0)?,
                        rows_total: result.get(1)?,
                        phase: result.get(2)?,
                    })
                },
            )
            .map_err(sqlite_err)?;
        if migration.to_version as u64 != table_spec_version {
            return Err(StatsError::SchemaConflict(format!(
                "migration for {:?} targets version {}, not {table_spec_version}",
                namespace, migration.to_version
            )));
        }
        let phase = migration_phase_from_str(&migration.phase)?;
        if !matches!(
            phase,
            MigrationPhase::MIGRATION_PHASE_DUAL_WRITE | MigrationPhase::MIGRATION_PHASE_BACKFILL
        ) {
            return Err(StatsError::SchemaConflict(format!(
                "migration for {:?} cannot accept backfill in phase {:?}",
                namespace, phase
            )));
        }
        for descriptor in segments {
            let SegmentDescriptor {
                row,
                source,
                artifacts,
            } = descriptor;
            if &row.namespace != namespace {
                return Err(StatsError::Internal(
                    "one migration checkpoint cannot span namespaces".to_string(),
                ));
            }
            let source_json = serde_json::to_string(source).map_err(|error| {
                StatsError::Internal(format!("serialize migrated segment source: {error}"))
            })?;
            upsert_segment_in(&transaction, row)?;
            transaction
                .execute(
                    "INSERT INTO object_segments
                        (namespace, path, table_spec_version, source_json, artifacts_json,
                         migration_backfill, migration_source_id, migration_source_rows)
                     VALUES (?1, ?2, ?3, ?4, ?5, 1, ?6, ?7)",
                    rusqlite::params![
                        row.namespace,
                        row.path,
                        table_spec_version as i64,
                        source_json,
                        artifacts_json(artifacts)?,
                        migration_source_id,
                        migration_source_rows,
                    ],
                )
                .map_err(sqlite_err)?;
        }
        let changed = transaction
            .execute(
                "UPDATE table_migrations
                 SET phase = ?2,
                     rows_completed = rows_completed + ?3,
                     phase_updated_at_ms = ?4
                 WHERE namespace = ?1 AND rows_completed + ?3 <= rows_total",
                rusqlite::params![
                    namespace,
                    migration_phase_str(MigrationPhase::MIGRATION_PHASE_BACKFILL),
                    migration_source_rows,
                    now_ms(),
                ],
            )
            .map_err(sqlite_err)?;
        if changed != 1 {
            return Err(StatsError::SchemaConflict(format!(
                "migration progress for {:?} exceeds its frozen row total {}",
                namespace, migration.rows_total
            )));
        }
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        let generation: i64 = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |result| result.get(0),
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation as u64))
    }

    /// Advertise the artifacts an index build produced for one object segment.
    ///
    /// This is an ordinary durable state transition: the artifact objects are
    /// already immutable, and this revision is what makes them live.
    pub fn set_segment_artifacts(
        &self,
        namespace: &str,
        path: &str,
        artifacts: &ArtifactReferences,
    ) -> Result<TableRevision, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let changed = transaction
            .execute(
                "UPDATE object_segments SET artifacts_json = ?3
                 WHERE namespace = ?1 AND path = ?2",
                rusqlite::params![namespace, path, artifacts_json(artifacts)?],
            )
            .map_err(sqlite_err)?;
        if changed != 1 {
            return Err(StatsError::SchemaConflict(format!(
                "object segment {path:?} in {namespace:?} is no longer live"
            )));
        }
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        let generation: i64 = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation as u64))
    }

    pub fn object_segments(&self, namespace: &str) -> Result<Vec<ObjectSegmentRecord>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut statement = inner
            .conn
            .prepare(
                "SELECT path, table_spec_version, source_json, migration_backfill,
                        migration_source_id, migration_source_rows, artifacts_json
                 FROM object_segments WHERE namespace = ?1 ORDER BY path",
            )
            .map_err(sqlite_err)?;
        let mut rows = statement.query([namespace]).map_err(sqlite_err)?;
        let mut records = Vec::new();
        while let Some(row) = rows.next().map_err(sqlite_err)? {
            let source_json: String = row.get(2).map_err(sqlite_err)?;
            let source = serde_json::from_str(&source_json).map_err(|error| {
                StatsError::Internal(format!("decode object segment source: {error}"))
            })?;
            let artifacts_json: Option<String> = row.get(6).map_err(sqlite_err)?;
            records.push(ObjectSegmentRecord {
                path: row.get(0).map_err(sqlite_err)?,
                table_spec_version: row.get::<_, i64>(1).map_err(sqlite_err)? as u64,
                source,
                artifacts: parse_artifacts(artifacts_json.as_deref())?,
                migration_backfill: row.get(3).map_err(sqlite_err)?,
                migration_source_id: row.get(4).map_err(sqlite_err)?,
                migration_source_rows: row.get(5).map_err(sqlite_err)?,
            });
        }
        Ok(records)
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

    pub fn forward_cursors(&self, namespace: &str) -> Result<Vec<ForwardCursor>, StatsError> {
        let inner = self.inner.lock().unwrap();
        let mut statement = inner
            .conn
            .prepare(
                "SELECT target, cursor FROM forward_state
                 WHERE namespace = ?1 ORDER BY target",
            )
            .map_err(sqlite_err)?;
        let rows = statement
            .query_map([namespace], |row| {
                Ok(ForwardCursor {
                    target: Some(row.get(0)?),
                    cursor: Some(row.get(1)?),
                    ..Default::default()
                })
            })
            .map_err(sqlite_err)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_err)
    }

    /// Record `cursor` as settled for `(target, namespace)`. Callers write it only once
    /// the rows below it can never be sent again, so a crash mid-batch re-forwards that
    /// batch rather than losing it.
    ///
    /// Returns the table's revision after the write. A namespace without a
    /// versioned head keeps revision zero and publishes nothing.
    pub fn set_forward_cursor(
        &self,
        target: &str,
        namespace: &str,
        cursor: i64,
    ) -> Result<TableRevision, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        transaction
            .execute(
                "INSERT INTO forward_state (target, namespace, cursor) VALUES (?1, ?2, ?3)
                 ON CONFLICT(target, namespace) DO UPDATE SET cursor = excluded.cursor",
                rusqlite::params![target, namespace, cursor],
            )
            .map_err(sqlite_err)?;
        transaction
            .execute(
                "UPDATE table_heads SET catalog_generation = catalog_generation + 1
                 WHERE namespace = ?1",
                [namespace],
            )
            .map_err(sqlite_err)?;
        let generation: Option<i64> = transaction
            .query_row(
                "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(TableRevision::new(generation.unwrap_or(0) as u64))
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
            // `merge` raises SchemaConflict on a column-type change.
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

    /// Remove every row `name` owns in one transaction, so a failure part way
    /// through leaves the namespace whole rather than half deleted. Idempotent.
    pub fn delete(&self, name: &str) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        for table in NAMESPACE_OWNED_TABLES {
            transaction
                .execute(&format!("DELETE FROM {table} WHERE namespace = ?1"), [name])
                .map_err(sqlite_err)?;
        }
        transaction.commit().map_err(sqlite_err)?;
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

    /// Remove a set of namespace segment rows in one durable transaction.
    pub fn remove_segments(&self, namespace: &str, paths: &[String]) -> Result<(), StatsError> {
        if paths.is_empty() {
            return Ok(());
        }
        let mut inner = self.inner.lock().unwrap();
        let tx = inner.conn.transaction().map_err(sqlite_err)?;
        remove_segments_in(&tx, namespace, paths)?;
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
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    use buffa::MessageField;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::partition_policy::SegmentPartition;
    use crate::proto::finelog::stats::{
        ColumnType, OperatingPolicy, SourceLayout, TableSpec as ProtoTableSpec,
    };
    use crate::store::schema::{schema_to_proto_owned, with_implicit_seq, Column};

    fn worker_stored() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "",
        ))
    }

    fn table_spec(version: u64, target_object_bytes: u64) -> ProtoTableSpec {
        ProtoTableSpec {
            version: Some(version),
            logical_schema: MessageField::some(schema_to_proto_owned(&worker_stored())),
            source_layout: MessageField::some(SourceLayout {
                target_object_bytes: Some(target_object_bytes),
                ..Default::default()
            }),
            operating_policy: MessageField::some(OperatingPolicy::default()),
            ..Default::default()
        }
    }

    fn spec_hash(spec: &ProtoTableSpec) -> [u8; 32] {
        Sha256::digest(canonical_json_bytes(spec).unwrap()).into()
    }

    #[test]
    fn table_spec_registration_is_monotonic_and_idempotent() {
        let catalog = Catalog::open(None).unwrap();
        let v1 = table_spec(1, 128);
        let status = catalog
            .register_table_spec("a", &v1, &spec_hash(&v1), false)
            .unwrap();
        assert_eq!(status.active_version(), 1);
        assert_eq!(status.catalog_generation, 1);

        let repeated = catalog
            .register_table_spec("a", &v1, &spec_hash(&v1), false)
            .unwrap();
        assert_eq!(repeated.catalog_generation, 1);

        let conflicting_v1 = table_spec(1, 256);
        assert!(matches!(
            catalog.register_table_spec("a", &conflicting_v1, &spec_hash(&conflicting_v1), false,),
            Err(StatsError::SchemaConflict(_))
        ));
        let v3 = table_spec(3, 128);
        assert!(matches!(
            catalog.register_table_spec("a", &v3, &spec_hash(&v3), false),
            Err(StatsError::SchemaConflict(_))
        ));
    }

    #[test]
    fn source_layout_change_queues_activation_and_supports_abort() {
        let catalog = Catalog::open(None).unwrap();
        let v1 = table_spec(1, 128);
        catalog
            .register_table_spec("a", &v1, &spec_hash(&v1), false)
            .unwrap();
        let v2 = table_spec(2, 256);
        let pending = catalog
            .register_table_spec("a", &v2, &spec_hash(&v2), true)
            .unwrap();
        assert_eq!(pending.active_version(), 1);
        assert_eq!(pending.desired_version(), 2);
        assert_eq!(pending.phase, MigrationPhase::MIGRATION_PHASE_DUAL_WRITE);

        let activated = catalog.activate_desired_table_spec("a").unwrap();
        assert_eq!(activated.active_version(), 2);
        assert_eq!(activated.desired_version(), 0);
        assert!(activated
            .migration
            .as_ref()
            .and_then(|migration| migration.observation_deadline_ms)
            .is_some_and(|deadline| deadline > now_ms()));
        let aborted = catalog.abort_table_migration("a").unwrap();
        assert_eq!(aborted.active_version(), 1);
        assert!(aborted.migration.is_none());
        assert!(aborted.catalog_generation > activated.catalog_generation);
    }

    #[test]
    fn table_spec_state_persists_across_catalog_reopen() {
        let dir = tempdir();
        let v1 = table_spec(1, 128);
        {
            let catalog = Catalog::open(Some(&dir)).unwrap();
            catalog
                .register_table_spec("a", &v1, &spec_hash(&v1), false)
                .unwrap();
        }
        let catalog = Catalog::open(Some(&dir)).unwrap();
        let status = catalog.table_spec_status("a").unwrap();
        assert_eq!(status.active.as_ref(), Some(&v1));
        assert_eq!(status.catalog_generation, 1);
        std::fs::remove_dir_all(&dir).ok();
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
    fn a_failing_delete_leaves_every_namespace_row_in_place() {
        let cat = Catalog::open(None).unwrap();
        cat.upsert("a", &worker_stored()).unwrap();
        let v1 = table_spec(1, 128);
        cat.register_table_spec("a", &v1, &spec_hash(&v1), false)
            .unwrap();
        cat.upsert_policy(
            "a",
            &StoragePolicy {
                max_segments: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        cat.upsert_segment(&SegmentRow {
            namespace: "a".to_string(),
            path: "a.parquet".to_string(),
            level: 1,
            min_seq: 1,
            max_seq: 3,
            row_count: 3,
            byte_size: 100,
            created_at_ms: 1,
            min_key_value: None,
            max_key_value: None,
            partition: None,
            location: crate::store::types::SegmentLocation::Local,
        })
        .unwrap();
        cat.set_forward_cursor("hub", "a", 3).unwrap();

        // Fails the last of the delete's statement groups, after the groups that
        // clear the dependent tables have already run.
        cat.inner
            .lock()
            .unwrap()
            .conn
            .execute_batch(
                "CREATE TRIGGER pin_namespace BEFORE DELETE ON namespaces
                 BEGIN SELECT RAISE(ABORT, 'namespace pinned'); END",
            )
            .unwrap();

        assert!(cat.delete("a").is_err());

        assert_eq!(cat.list_segments("a").unwrap().len(), 1);
        assert_eq!(cat.get_policy("a").unwrap().max_segments, Some(3));
        assert_eq!(cat.table_spec_status("a").unwrap().active_version(), 1);
        assert_eq!(cat.forward_cursor("hub", "a").unwrap(), Some(3));
        assert!(cat.list_all().unwrap().iter().any(|(name, _)| name == "a"));
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

    #[test]
    fn segment_partition_persists_across_catalog_reopen() {
        let dir = tempdir();
        let partition = SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("name_bucket".to_string(), "6".to_string())]),
        };
        {
            let catalog = Catalog::open(Some(&dir)).unwrap();
            catalog.upsert("a", &worker_stored()).unwrap();
            catalog
                .upsert_segment(&SegmentRow {
                    namespace: "a".to_string(),
                    path: dir.join("a.parquet").to_string_lossy().into_owned(),
                    level: 1,
                    min_seq: 1,
                    max_seq: 3,
                    row_count: 3,
                    byte_size: 100,
                    created_at_ms: 1,
                    min_key_value: None,
                    max_key_value: None,
                    partition: Some(partition.clone()),
                    location: crate::store::types::SegmentLocation::Local,
                })
                .unwrap();
        }
        let catalog = Catalog::open(Some(&dir)).unwrap();
        assert_eq!(
            catalog.list_segments("a").unwrap()[0].partition,
            Some(partition)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn opening_an_old_catalog_adds_partition_metadata_without_losing_segments() {
        let dir = tempdir();
        let path = dir.join(CATALOG_DB_FILENAME);
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch(
                r#"
                CREATE TABLE segments (
                    namespace TEXT NOT NULL,
                    path TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    min_seq INTEGER NOT NULL,
                    max_seq INTEGER NOT NULL,
                    row_count INTEGER NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    min_key_value TEXT,
                    max_key_value TEXT,
                    location TEXT NOT NULL,
                    PRIMARY KEY (namespace, path)
                );
                INSERT INTO segments VALUES
                    ('a', '/old.parquet', 1, 1, 3, 3, 100, 1, NULL, NULL, 'LOCAL');
                "#,
            )
            .unwrap();
        drop(connection);

        let catalog = Catalog::open(Some(&dir)).unwrap();
        let old = catalog.list_segments("a").unwrap();
        assert_eq!(old.len(), 1);
        assert_eq!(old[0].path, "/old.parquet");
        assert_eq!(old[0].partition, None);

        let partition = SegmentPartition {
            spec_id: 1,
            values: BTreeMap::from([("name_bucket".to_string(), "6".to_string())]),
        };
        catalog
            .upsert_segment(&SegmentRow {
                namespace: "a".to_string(),
                path: "/new.parquet".to_string(),
                level: 1,
                min_seq: 4,
                max_seq: 4,
                row_count: 1,
                byte_size: 50,
                created_at_ms: 2,
                min_key_value: None,
                max_key_value: None,
                partition: Some(partition.clone()),
                location: crate::store::types::SegmentLocation::Local,
            })
            .unwrap();
        assert_eq!(
            catalog.list_segments("a").unwrap()[1].partition,
            Some(partition)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn tempdir() -> std::path::PathBuf {
        static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ordinal = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        p.push(format!("finelog_catalog_test_{nanos}_{ordinal}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
