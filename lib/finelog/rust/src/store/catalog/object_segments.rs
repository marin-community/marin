// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! The local projection of an object-backed table's published state.
//!
//! For object tables the remote `TableState` is authoritative; these rows are
//! a derived index (`object_segments` joined onto `segments`) that query
//! planning reads. Commits land the rows the controller has already made
//! durable remotely, and [`Catalog::replace_with_published_snapshot`] rebuilds
//! the whole projection from a verified remote state after recovery.

use rusqlite::OptionalExtension;
use sha2::{Digest, Sha256};

use super::segments::{remove_segments_in, upsert_segment_in};
use super::table_specs::{migration_phase_from_str, migration_phase_str};
use super::*;
use crate::errors::StatsError;
use crate::proto::finelog::stats::{MigrationPhase, NamespaceCatalog, ObjectRef};
use crate::store::object_store::ObjectId;
use crate::store::policy::StoragePolicy;
use crate::store::schema::Schema;
use crate::store::table_spec::canonical_json_bytes;
use crate::store::table_state::{ArtifactReferences, SegmentDescriptor, TableRevision};
use crate::store::types::SegmentRow;

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
    pub artifacts: ArtifactReferences,
    pub migration_backfill: bool,
    pub migration_source_id: Option<String>,
    pub migration_source_rows: Option<i64>,
}

/// Encode a segment's artifact references for the `object_segments` row.
///
/// A segment with no artifacts stores SQL NULL rather than an empty document.
pub(super) fn artifacts_json(artifacts: &ArtifactReferences) -> Result<Option<String>, StatsError> {
    if artifacts.is_empty() && artifacts.binding == Default::default() {
        return Ok(None);
    }
    serde_json::to_string(artifacts)
        .map(Some)
        .map_err(|error| StatsError::Internal(format!("serialize segment artifacts: {error}")))
}

pub(super) fn parse_artifacts(json: Option<&str>) -> Result<ArtifactReferences, StatsError> {
    let Some(json) = json else {
        return Ok(ArtifactReferences::default());
    };
    serde_json::from_str(json)
        .map_err(|error| StatsError::Internal(format!("decode segment artifacts: {error}")))
}

/// Advance the namespace's catalog generation and return the new revision.
/// Fails when the namespace has no head row to advance.
fn advance_generation_in(
    transaction: &rusqlite::Transaction<'_>,
    namespace: &str,
) -> Result<TableRevision, StatsError> {
    let changed = transaction
        .execute(
            "UPDATE table_heads SET catalog_generation = catalog_generation + 1
             WHERE namespace = ?1",
            [namespace],
        )
        .map_err(sqlite_err)?;
    if changed != 1 {
        return Err(StatsError::Internal(format!(
            "object segments committed for {namespace:?} without a namespace head"
        )));
    }
    let generation: i64 = transaction
        .query_row(
            "SELECT catalog_generation FROM table_heads WHERE namespace = ?1",
            [namespace],
            |row| row.get(0),
        )
        .map_err(sqlite_err)?;
    Ok(TableRevision::new(generation as u64))
}

impl Catalog {
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
            let spec_bytes = canonical_json_bytes(spec)?;
            transaction
                .execute(
                    "INSERT INTO table_specs (namespace, version, spec_json, spec_hash)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![
                        namespace,
                        spec.version.unwrap_or(0) as i64,
                        String::from_utf8(spec_bytes.clone()).expect("JSON is UTF-8"),
                        Sha256::digest(&spec_bytes).as_slice(),
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
                        (namespace, path, table_spec_version, source_json, artifacts_json,
                         migration_backfill, migration_source_id, migration_source_rows)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        namespace,
                        segment.row.path,
                        segment.table_spec_version as i64,
                        source_json,
                        artifacts_json(&segment.artifacts)?,
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
                        (namespace, from_version, to_version, phase,
                         fence_seq, rows_total, rows_completed, observation_deadline_ms)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        namespace,
                        migration.from_version.unwrap_or(0) as i64,
                        migration.to_version.unwrap_or(0) as i64,
                        migration_phase_str(phase),
                        migration.fence_seq.unwrap_or(-1),
                        migration.rows_total.unwrap_or(0),
                        migration.rows_completed.unwrap_or(0),
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
        let revision = advance_generation_in(&transaction, namespace)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(revision)
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
        let revision = advance_generation_in(&transaction, namespace)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(revision)
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
        let (to_version, phase): (i64, String) = transaction
            .query_row(
                "SELECT to_version, phase FROM table_migrations WHERE namespace = ?1",
                [namespace],
                |result| Ok((result.get(0)?, result.get(1)?)),
            )
            .map_err(sqlite_err)?;
        if to_version as u64 != table_spec_version {
            return Err(StatsError::SchemaConflict(format!(
                "migration for {namespace:?} targets version {to_version}, not {table_spec_version}"
            )));
        }
        let phase = migration_phase_from_str(&phase)?;
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
        // The total is the universe as it was last measured, so progress can
        // pass it: a source rewritten before an eviction removed it from that
        // universe is done, and the total rises to say so rather than rejecting
        // the checkpoint. Committing one source twice is prevented by its
        // content-addressed outputs, whose insert above conflicts.
        let changed = transaction
            .execute(
                "UPDATE table_migrations
                 SET phase = ?2,
                     rows_completed = rows_completed + ?3,
                     rows_total = MAX(rows_total, rows_completed + ?3)
                 WHERE namespace = ?1",
                rusqlite::params![
                    namespace,
                    migration_phase_str(MigrationPhase::MIGRATION_PHASE_BACKFILL),
                    migration_source_rows,
                ],
            )
            .map_err(sqlite_err)?;
        if changed != 1 {
            return Err(StatsError::SchemaConflict(format!(
                "migration for {namespace:?} has no progress row to checkpoint"
            )));
        }
        let revision = advance_generation_in(&transaction, namespace)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(revision)
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
        let revision = advance_generation_in(&transaction, namespace)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(revision)
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
}
