//! The central dispatch from a unit of table maintenance to the module that
//! owns it.
//!
//! The maintenance scheduler owns cadence and decides *when* a table owes work.
//! It never reaches into a table's internals: it names a [`TableWork`] and the
//! manager dispatches it here. Every arm below is one call into a work module,
//! and the contexts those modules receive are built in exactly one place.
//!
//! [`TableWork::Cycle`] is the composite the scheduler asks for. Its order is
//! the contract: a table flushes before it compacts, a migration owns the cycle
//! while it runs, the archive is made durable before anything is evicted from
//! the local cache, and the lowest-priority layout and index work runs last.

use std::sync::Arc;
use std::time::Instant;

use crate::errors::StatsError;
use crate::indices::SegmentIndexConfig;
use crate::maintenance::{
    OBJECT_GC_INTERVAL, OBJECT_ORPHAN_GRACE, OBJECT_ORPHAN_SWEEP_INTERVAL, REWRITE_LAYOUT_BUDGET,
};
use crate::policies::{physical_partition_policy_for, segment_indexes_enabled_for};
use crate::store::catalog::SpecLifecycle;
use crate::store::compaction::local_driver::{self, LocalCompaction};
use crate::store::compaction::object_driver::{self, ObjectCompaction};
use crate::store::legacy::archive::{self, LegacyArchive};
use crate::store::legacy::layout::{self, LocalLayout};
use crate::store::state_store::object::StateGcPolicy;
use crate::store::table::index_artifacts::{self, IndexBackfill, INDEX_BUNDLES_PER_TICK};
use crate::store::table::key_bounds;
use crate::store::table::runtime::TableRuntime;
use crate::store::table::spec_migration::{self, SpecMigration};

/// One unit of table maintenance, each owned by exactly one module.
#[derive(Clone, Copy, Debug)]
pub enum TableWork {
    /// Seal the ingest buffer and make its rows durable.
    Flush,
    /// Advance an automatic table-specification transition.
    SpecMigration,
    /// Compact one planner-issued run. `force_compact_l0` makes an L0 run
    /// eligible regardless of the size threshold.
    Compaction { force_compact_l0: bool },
    /// Recover missing key bounds from immutable object footers.
    KeyBounds,
    /// Build the derived index artifacts segments still owe, or remove them from
    /// a table whose policy disables them.
    IndexArtifacts,
    /// Collect superseded state documents and unreferenced objects.
    ObjectCollection,
    /// Upload legacy segments to the archive and delete its orphans.
    LegacyArchive,
    /// Trim the local cache to the table's retention policy.
    Eviction,
    /// Converge legacy physical placement.
    Placement,
    /// Relocate archived segments to their current physical key.
    ArchivePlacement,
    /// Re-encode segments whose row-group layout predates the writer policy.
    EncodingRewrite,
    /// The ordered cycle the scheduler asks for.
    Cycle { force_compact_l0: bool },
}

/// Whether the scheduler should run another cycle on its prompt cadence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WorkOutcome {
    #[default]
    Complete,
    MoreWork,
}

impl WorkOutcome {
    pub fn has_more_work(self) -> bool {
        self == Self::MoreWork
    }

    fn from_pending(pending: bool) -> Self {
        if pending {
            Self::MoreWork
        } else {
            Self::Complete
        }
    }
}

/// Dispatch one unit of work against `runtime`.
///
/// An in-memory table owes no maintenance at all: it has no segments, no cache,
/// and nothing to make durable.
pub async fn run(runtime: &Arc<TableRuntime>, work: TableWork) -> Result<WorkOutcome, StatsError> {
    if !runtime.is_disk_backed() {
        return Ok(WorkOutcome::Complete);
    }
    match work {
        TableWork::Cycle { force_compact_l0 } => cycle(runtime, force_compact_l0).await,
        one => run_one(runtime, one).await,
    }
}

/// Dispatch one indivisible unit of work. [`TableWork::Cycle`] is the ordered
/// composition of these and is handled by [`cycle`].
async fn run_one(runtime: &Arc<TableRuntime>, work: TableWork) -> Result<WorkOutcome, StatsError> {
    match work {
        TableWork::Flush => {
            runtime.flush().await?;
            Ok(WorkOutcome::Complete)
        }
        TableWork::SpecMigration => {
            let activated = |status: &SpecLifecycle| -> Result<(), StatsError> {
                runtime.activate_query_version(status.active_version())?;
                runtime.update_table_spec(status);
                Ok(())
            };
            let owns_cycle = spec_migration::advance(&SpecMigration {
                table: runtime.name(),
                table_dir: table_dir(runtime),
                format: &runtime.format,
                index_config: index_config(runtime),
                catalog: &runtime.catalog,
                controller: &runtime.controller,
                segments: &runtime.segments,
                query_visibility: &runtime.query_visibility,
                flush_gate: &runtime.object_flush_lock,
                max_merge_arrow_bytes: runtime.compaction_config.max_merge_arrow_bytes,
                migration_batch_sources: runtime.compaction_config.migration_batch_sources,
                blocked: &runtime.migration_block,
                identities: &runtime.migration_identities,
                on_activated: &activated,
            })
            .await?;
            Ok(WorkOutcome::from_pending(owns_cycle))
        }
        TableWork::Compaction { force_compact_l0 } => compact(runtime, force_compact_l0).await,
        TableWork::KeyBounds => Ok(WorkOutcome::from_pending(
            key_bounds::maintain(runtime).await?,
        )),
        TableWork::IndexArtifacts => {
            let tracker = &runtime.layout_tracker;
            let layout_is_current = |path: &str| tracker.is_current(path);
            index_artifacts::maintain(
                index_backfill(runtime, &layout_is_current),
                INDEX_BUNDLES_PER_TICK,
            )
            .await;
            Ok(WorkOutcome::Complete)
        }
        TableWork::ObjectCollection => {
            collect_objects(runtime).await?;
            Ok(WorkOutcome::Complete)
        }
        TableWork::LegacyArchive => {
            sync_archive(runtime).await?;
            Ok(WorkOutcome::Complete)
        }
        TableWork::Eviction => {
            let runtime = Arc::clone(runtime);
            tokio::task::spawn_blocking(move || {
                archive::evict_to_policy(
                    runtime.name(),
                    &runtime.catalog,
                    &runtime.segments,
                    &runtime.query_visibility,
                    &runtime.storage_policy.lock().unwrap().clone(),
                    &runtime.compaction_config,
                )
            })
            .await
            .map_err(|error| {
                StatsError::Internal(format!("maintenance evict task panicked: {error}"))
            })??;
            Ok(WorkOutcome::Complete)
        }
        TableWork::Placement => {
            let runtime = Arc::clone(runtime);
            let pending = tokio::task::spawn_blocking(move || {
                layout::advance_partitioning(&local_layout(&runtime))
            })
            .await
            .map_err(|error| {
                StatsError::Internal(format!("maintenance placement task panicked: {error}"))
            })??;
            Ok(WorkOutcome::from_pending(pending))
        }
        TableWork::ArchivePlacement => {
            layout::advance_archive_placement(&local_layout(runtime)).await?;
            Ok(WorkOutcome::Complete)
        }
        TableWork::EncodingRewrite => {
            let runtime = Arc::clone(runtime);
            tokio::task::spawn_blocking(move || {
                layout::rewrite_stale_encodings(&local_layout(&runtime), REWRITE_LAYOUT_BUDGET)
            })
            .await
            .map_err(|error| {
                StatsError::Internal(format!("maintenance rewrite task panicked: {error}"))
            })?;
            Ok(WorkOutcome::Complete)
        }
        TableWork::Cycle { .. } => unreachable!("a cycle is composed, not dispatched"),
    }
}

/// Run one full maintenance cycle, serialized against other cycles.
///
/// An object-backed table publishes pending state, compacts immutable objects,
/// collects, and maintains indexes. A legacy table converges its physical
/// placement, compacts local segments, synchronizes the archive, evicts, and
/// performs index and encoding maintenance.
async fn cycle(
    runtime: &Arc<TableRuntime>,
    force_compact_l0: bool,
) -> Result<WorkOutcome, StatsError> {
    let _cycle_guard = runtime.maint_lock.lock().await;
    run_one(runtime, TableWork::Flush).await?;
    if run_one(runtime, TableWork::SpecMigration)
        .await?
        .has_more_work()
    {
        // The migration owns the table's maintenance while it runs: placement
        // and legacy compaction would destroy the sources it rewrites. Object
        // compaction stays on: it folds only ordinary-stream objects at the
        // target version — the dual-write flush L0s that would otherwise stack
        // up for the whole backfill (see `compact_once`). Cache eviction and
        // superseded-state collection also stay on: each backfill commit
        // publishes a fresh full catalog snapshot, so without them the
        // superseded snapshots pile up locally and remotely for the whole
        // backfill (a full disk on the marin hub). Both only touch synced
        // cache copies and states older than the retention window, never the
        // sources or coverage the migration reads. The cycle reports pending
        // so the scheduler re-polls immediately and routes the next tick
        // through the dedicated migration slot instead of the shared queue.
        if runtime.policy().object_backed() {
            run_one(runtime, TableWork::Compaction { force_compact_l0 }).await?;
            runtime.controller.gc_objects().await?;
            run_one(runtime, TableWork::ObjectCollection).await?;
        }
        return Ok(WorkOutcome::MoreWork);
    }
    if runtime.policy().object_backed() {
        runtime.controller.publish_owed().await?;
        // Report pending while compaction keeps finding runs: an L0 backlog
        // then drains at the fast re-poll cadence through the dedicated slot
        // instead of one run per shared-queue visit, mirroring how a legacy
        // table drains its whole backlog in one cycle.
        let bounds_pending = run_one(runtime, TableWork::KeyBounds)
            .await?
            .has_more_work();
        let compacted = run_one(runtime, TableWork::Compaction { force_compact_l0 })
            .await?
            .has_more_work();
        runtime.controller.gc_objects().await?;
        run_one(runtime, TableWork::ObjectCollection).await?;
        run_one(runtime, TableWork::IndexArtifacts).await?;
        return Ok(WorkOutcome::from_pending(bounds_pending || compacted));
    }

    let placement_pending = run_one(runtime, TableWork::Placement)
        .await?
        .has_more_work();
    run_one(runtime, TableWork::Compaction { force_compact_l0 }).await?;
    run_one(runtime, TableWork::LegacyArchive).await?;
    run_one(runtime, TableWork::ObjectCollection).await?;
    // Relocate archived segments after local outputs are durable.
    run_one(runtime, TableWork::ArchivePlacement).await?;
    run_one(runtime, TableWork::Eviction).await?;
    // Derived indexes are maintained last and in bounded batches.
    run_one(runtime, TableWork::IndexArtifacts).await?;
    // Re-encoding is the lowest priority: the terminal level is never
    // re-compacted, so without it a table carries whatever layout it was written
    // with until eviction ages it out.
    run_one(runtime, TableWork::EncodingRewrite).await?;
    Ok(WorkOutcome::from_pending(placement_pending))
}

/// Compact one run, or — for a legacy table — drain the planner's backlog.
async fn compact(
    runtime: &Arc<TableRuntime>,
    force_compact_l0: bool,
) -> Result<WorkOutcome, StatsError> {
    if runtime.policy().object_backed() {
        let compacted = object_driver::compact_once(
            ObjectCompaction {
                table: runtime.name(),
                table_dir: table_dir(runtime),
                format: &runtime.format,
                index_config: index_config(runtime),
                catalog: &runtime.catalog,
                controller: &runtime.controller,
                segments: &runtime.segments,
                query_visibility: &runtime.query_visibility,
                config: &runtime.compaction_config,
                target_object_bytes: runtime.policy().target_object_bytes,
            },
            force_compact_l0,
        )
        .await?;
        return Ok(WorkOutcome::from_pending(compacted.has_pending_work()));
    }
    // The legacy path decodes Parquet and takes the query-visibility write lock,
    // so the whole drain runs on the blocking pool. It checks the stop latch
    // between jobs: a stop signalled mid-backlog (a re-register replacing this
    // runtime, or shutdown) ends the drain promptly so `stop_and_join` joins
    // inside its timeout. Otherwise a long drain outlives the timeout, the task
    // is aborted, and its detached blocking compaction keeps unlinking inputs
    // while the replacement runtime adopts the same directory — the race that
    // plants a phantom segment.
    let runtime = Arc::clone(runtime);
    let single_job = layout::partitioning_is_pending(&local_layout(&runtime));
    tokio::task::spawn_blocking(move || -> Result<WorkOutcome, StatsError> {
        let compaction = local_compaction(&runtime);
        if force_compact_l0 {
            local_driver::promote_all_l0(&compaction)?;
        }
        let mut compacted = false;
        while !runtime.is_stopped() {
            if !local_driver::compact_once(&compaction)? {
                break;
            }
            compacted = true;
            // While a rebuild is active, one ordinary job is enough to keep live
            // L0 bounded. Spend the remaining CPU on releasing legacy inputs;
            // partition-local L1 consolidation can catch up after the source
            // backlog is gone.
            if single_job {
                break;
            }
        }
        Ok(WorkOutcome::from_pending(compacted))
    })
    .await
    .map_err(|error| StatsError::Internal(format!("maintenance compact task panicked: {error}")))?
}

/// Make the legacy archive match the catalog.
pub async fn sync_archive(runtime: &Arc<TableRuntime>) -> Result<(), StatsError> {
    let Some(remote) = runtime.controller.legacy_store() else {
        return Ok(());
    };
    archive::sync(LegacyArchive {
        table: runtime.name(),
        table_dir: table_dir(runtime),
        catalog: &runtime.catalog,
        segments: &runtime.segments,
        remote: &remote,
        // The legacy archive is outside the object-backed catalog's MVCC
        // lifetime, so an object-backed table retains archive objects when their
        // local migration source is replaced.
        retain_orphans: runtime.policy().object_backed(),
    })
    .await
}

/// Collect superseded state documents and the objects no retained state names.
async fn collect_objects(runtime: &Arc<TableRuntime>) -> Result<(), StatsError> {
    if !runtime.controller.is_object_backed() {
        return Ok(());
    }
    let due = {
        let mut last = runtime.last_object_gc.lock().unwrap();
        let due = last.is_none_or(|instant| instant.elapsed() >= OBJECT_GC_INTERVAL);
        if due {
            *last = Some(Instant::now());
        }
        due
    };
    if !due {
        return Ok(());
    }
    // Retired objects answer two independent readers: a query holding a pinned
    // snapshot, and a rollback to the definition they belong to. Collection waits
    // for whichever window is longer.
    let policy = runtime.policy();
    let state_retention_ms = policy.max_query_time_ms.max(policy.rollback_window_ms);
    let orphan_grace_ms = u64::try_from(OBJECT_ORPHAN_GRACE.as_millis()).unwrap_or(u64::MAX);
    let sweep_orphans = {
        let mut last = runtime.last_orphan_sweep.lock().unwrap();
        let due = last.is_none_or(|instant| instant.elapsed() >= OBJECT_ORPHAN_SWEEP_INTERVAL);
        if due {
            *last = Some(Instant::now());
        }
        due
    };
    let removed = runtime
        .controller
        .gc_published(
            crate::store::table::now_ms(),
            StateGcPolicy {
                pin_retention_ms: policy.max_query_time_ms,
                state_retention_ms,
                orphan_grace_ms,
                sweep_orphans,
            },
        )
        .await?;
    if removed > 0 {
        tracing::info!(namespace = %runtime.name(), removed, "removed obsolete table objects");
    }
    Ok(())
}

/// Whether the table still owes legacy placement work, so the scheduler can poll
/// it at the faster migration cadence.
pub fn placement_is_pending(runtime: &Arc<TableRuntime>) -> bool {
    if !runtime.is_disk_backed() {
        return false;
    }
    layout::partitioning_is_pending(&local_layout(runtime))
}

fn table_dir(runtime: &TableRuntime) -> &std::path::Path {
    runtime
        .data_dir
        .as_deref()
        .expect("maintenance runs only for disk-backed tables")
}

fn index_config(runtime: &TableRuntime) -> SegmentIndexConfig {
    runtime.format.index_config(runtime.name())
}

/// Everything the index backfill for `runtime` reads and commits through.
pub(crate) fn index_backfill<'a>(
    runtime: &'a TableRuntime,
    layout_is_current: &'a (dyn Fn(&str) -> bool + Send + Sync),
) -> IndexBackfill<'a> {
    IndexBackfill {
        table: runtime.name(),
        catalog: &runtime.catalog,
        controller: &runtime.controller,
        segments: &runtime.segments,
        registry: &runtime.indices,
        limits: &runtime.limits,
        config: index_config(runtime),
        indexes_enabled: segment_indexes_enabled_for(runtime.name()),
        layout_is_current,
        skips: &runtime.index_skips,
    }
}

pub(crate) fn local_compaction(runtime: &TableRuntime) -> LocalCompaction<'_> {
    LocalCompaction {
        table: runtime.name(),
        table_dir: table_dir(runtime),
        format: &runtime.format,
        index_config: index_config(runtime),
        catalog: &runtime.catalog,
        segments: &runtime.segments,
        query_visibility: &runtime.query_visibility,
        config: &runtime.compaction_config,
        partition_policy: physical_partition_policy_for(runtime.name()),
    }
}

pub(crate) fn local_layout(runtime: &TableRuntime) -> LocalLayout<'_> {
    LocalLayout {
        compaction: local_compaction(runtime),
        limits: &runtime.limits,
        tracker: &runtime.layout_tracker,
        stopped: &runtime.stopped,
        remote: runtime.controller.legacy_store(),
    }
}
