// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Table specifications and the migration lifecycle.
//!
//! Owns the durable answer to "which definition version does this table run,
//! and where is its transition": registration classifies a new version against
//! the active one, activation/abort/retire move the migration phase, and
//! [`SpecLifecycle`] is the read surface the runtime resolves its operating
//! policy from.

use rusqlite::{Connection, OptionalExtension};

use super::segments::remove_segments_in;
use super::*;
use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    MigrationPhase, TableMigrationStatus, TableSpec as ProtoTableSpec,
};
use crate::store::table_spec::{
    canonical_json_bytes, classify_definition_change, migration_phase_for_state,
    rollback_window_ms, table_spec_from_json, DefinitionChange,
};

/// Where one table stands in its specification lifecycle: the version queries
/// run against (`active`), the version a registration asked for (`desired`,
/// equal to or ahead of active), and the phase and progress of the transition
/// between them. [`SpecLifecycle::operative`] picks the spec the runtime
/// operates under.
#[derive(Debug, Clone)]
pub struct SpecLifecycle {
    pub active: Option<ProtoTableSpec>,
    pub desired: Option<ProtoTableSpec>,
    pub phase: MigrationPhase,
    pub catalog_generation: u64,
    pub migration: Option<TableMigrationStatus>,
}

impl SpecLifecycle {
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

    /// The specification the runtime operates under: the migration target while
    /// a transition is active, else the active specification. `None` for a
    /// legacy table with no specification.
    pub fn operative(&self) -> Option<&ProtoTableSpec> {
        self.desired.as_ref().or(self.active.as_ref())
    }
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

pub(super) struct MigrationCheckpoint {
    pub(super) to_version: i64,
    pub(super) phase: String,
}

pub(super) fn table_spec_for_version(
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

pub(super) fn spec_lifecycle_in(
    conn: &Connection,
    namespace: &str,
) -> Result<SpecLifecycle, StatsError> {
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
        return Ok(SpecLifecycle {
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
    Ok(SpecLifecycle {
        active,
        phase,
        desired,
        catalog_generation: generation as u64,
        migration,
    })
}

pub(super) fn migration_status_in(
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

pub(super) fn migration_phase_str(phase: MigrationPhase) -> &'static str {
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

pub(super) fn migration_phase_from_str(value: &str) -> Result<MigrationPhase, StatsError> {
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

/// The rows a migration out of `from_version` is responsible for rewriting.
///
/// A migration rewrites the rows the table serves, and reads them from where
/// the table already reads them. An object-backed source qualifies whatever its
/// cache location says, because it resolves by object reference and the object
/// store fetches it. A legacy source qualifies only while the catalog says a
/// local copy exists: once it has been evicted to `REMOTE` its bytes survive
/// only in the legacy GCS archive, which a migration never reads, rewrites, or
/// moves. Such a segment is already outside the live query view, stays in the
/// archive for history queries, and belongs to neither the frozen total nor the
/// backfill universe.
///
/// `fence_seq` bounds the universe to rows that predate the transition; rows
/// above it are written in the target layout already.
pub(super) fn migratable_source_rows_in(
    conn: &Connection,
    namespace: &str,
    from_version: i64,
    fence_seq: i64,
) -> Result<i64, StatsError> {
    conn.query_row(
        "SELECT COALESCE(SUM(segments.row_count), 0) FROM segments
         LEFT JOIN object_segments
           ON object_segments.namespace = segments.namespace
          AND object_segments.path = segments.path
         WHERE segments.namespace = ?1
           AND segments.max_seq <= ?2
           AND CASE WHEN object_segments.path IS NULL
                    THEN ?3 = 0 AND segments.location <> 'REMOTE'
                    ELSE object_segments.table_spec_version = ?3 END",
        rusqlite::params![namespace, fence_seq, from_version],
        |row| row.get(0),
    )
    .map_err(sqlite_err)
}

impl Catalog {
    pub fn spec_lifecycle(&self, namespace: &str) -> Result<SpecLifecycle, StatsError> {
        let inner = self.inner.lock().unwrap();
        spec_lifecycle_in(&inner.conn, namespace)
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
        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
    /// Record `spec` as the namespace's next definition version.
    ///
    /// A metadata-only change activates in this commit. A physical rewrite over
    /// existing rows records a pending transition instead, frozen against two
    /// numbers:
    ///
    /// - `fence_seq` is `MAX(max_seq)` over every segment, archive-only ones
    ///   included. It separates rows the backfill owes from rows later writes
    ///   produce directly in the target layout. Counting a segment the backfill
    ///   will not touch can only raise the fence, which classifies more rows as
    ///   pre-fence — the safe direction, since a pre-fence row that no longer
    ///   exists locally is simply not a source. A fence below a genuine source's
    ///   `max_seq` would be the unsafe direction: that source would be treated
    ///   as already migrated.
    /// - `rows_total` is the backfill universe, which counts only the rows the
    ///   migration can rewrite (see [`migratable_source_rows_in`]). Archived
    ///   rows are excluded, so a table whose history was long ago evicted to the
    ///   legacy archive still reaches an activatable total.
    pub fn register_table_spec(
        &self,
        namespace: &str,
        spec: &ProtoTableSpec,
        expected_hash: &[u8; 32],
        has_rows: bool,
    ) -> Result<SpecLifecycle, StatsError> {
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
            return spec_lifecycle_in(&inner.conn, namespace);
        }

        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
            let fence_seq: i64 = transaction
                .query_row(
                    "SELECT COALESCE(MAX(max_seq), 0) FROM segments WHERE namespace = ?1",
                    [namespace],
                    |row| row.get(0),
                )
                .map_err(sqlite_err)?;
            let rows_total = migratable_source_rows_in(
                &transaction,
                namespace,
                active_version as i64,
                fence_seq,
            )?;
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
        spec_lifecycle_in(&inner.conn, namespace)
    }
    pub fn activate_desired_table_spec(
        &self,
        namespace: &str,
    ) -> Result<SpecLifecycle, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
        spec_lifecycle_in(&inner.conn, namespace)
    }
    /// Abort the one in-flight migration and restore its source version.
    ///
    /// Backfill-only target objects become unreachable; writes accepted during
    /// the migration are reassigned to the source version so abort never drops
    /// rows. Published snapshots retain remote bytes until catalog GC.
    pub fn abort_table_migration(&self, namespace: &str) -> Result<SpecLifecycle, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
        spec_lifecycle_in(&inner.conn, namespace)
    }
    pub fn update_migration_phase(
        &self,
        namespace: &str,
        expected: MigrationPhase,
        phase: MigrationPhase,
    ) -> Result<SpecLifecycle, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
        spec_lifecycle_in(&inner.conn, namespace)
    }
    /// Restate the pending migration's row total as the universe it can still
    /// rewrite.
    ///
    /// The universe shrinks whenever a source leaves it: a legacy segment
    /// evicted to the archive after registration is no longer rewritable, and a
    /// total that kept counting it would report a migration as perpetually
    /// unfinished. Restating is idempotent — a repeated tick, or a tick after a
    /// crash, derives the same total from the same rows — where deducting per
    /// skip would compound. The total never drops below the progress already
    /// made, so a source rewritten and then evicted leaves the pair coherent.
    ///
    /// This is progress reporting. It moves no segment and publishes no state,
    /// so it does not advance the catalog generation, and whether a backfill is
    /// finished is decided by its sources rather than by these counters.
    pub fn refresh_migration_rows_total(&self, namespace: &str) -> Result<(), StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let transaction = inner.conn.transaction().map_err(sqlite_err)?;
        let (from_version, fence_seq, rows_completed): (i64, i64, i64) = transaction
            .query_row(
                "SELECT from_version, fence_seq, rows_completed FROM table_migrations
                 WHERE namespace = ?1",
                [namespace],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .map_err(sqlite_err)?;
        let rows_total =
            migratable_source_rows_in(&transaction, namespace, from_version, fence_seq)?
                .max(rows_completed);
        transaction
            .execute(
                "UPDATE table_migrations SET rows_total = ?2 WHERE namespace = ?1",
                rusqlite::params![namespace, rows_total],
            )
            .map_err(sqlite_err)?;
        transaction.commit().map_err(sqlite_err)?;
        Ok(())
    }
    /// Drop the transition's source rows and close the rollback window.
    ///
    /// A version-0 import retires every pre-fence legacy catalog row, including
    /// the archive-only ones the backfill skipped. Retirement is a catalog
    /// operation: it deletes no file, local or remote, so an archived segment
    /// keeps its bytes in the legacy archive exactly as a retired local
    /// segment's uploaded copy does. `finelog gcs-query` reads that archive by
    /// listing it, so archived history stays queryable with or without a
    /// catalog row, while the imported table stops carrying rows it can no
    /// longer serve or rewrite.
    pub fn retire_observed_migration(&self, namespace: &str) -> Result<SpecLifecycle, StatsError> {
        let mut inner = self.inner.lock().unwrap();
        let status = spec_lifecycle_in(&inner.conn, namespace)?;
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
        spec_lifecycle_in(&inner.conn, namespace)
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
}
