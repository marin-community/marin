//! Driving compaction for an object-backed table.
//!
//! One step plans a run with the pure leveled planner, takes a maintenance
//! lease over its exact inputs, localizes them, runs the shared executor into a
//! staging directory, uploads each output as an immutable object, and commits
//! the replacement under the lease.
//!
//! The heavy work runs outside the table controller: the controller serializes
//! only the lease and the short commit, so ordinary flushes keep committing
//! while a long compaction runs. The controller rebases the replacement onto
//! whatever state is current and rejects it only on a real conflict — a retired
//! input, a moved definition version, or a fenced writer — in which case the
//! uploaded outputs are simply left unreferenced for object GC to collect.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::errors::StatsError;
use crate::indices::SegmentIndexConfig;
use crate::store::catalog::Catalog;
use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::compaction::executor::{
    run_job_with_partition_policy, run_merge_thread, CompactionExecution, OutputPolicy,
};
use crate::store::compaction::planner::{plan, plan_forced_l0, UnpartitionedRunPolicy};
use crate::store::compaction::staging::StagingDir;
use crate::store::object_store::OBJECTS_PREFIX;
use crate::store::table::controller::{MaintenanceLease, TableController};
use crate::store::table::index_artifacts::publish_segment_artifacts;
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::SegmentView;
use crate::store::table_state::{CommitError, SegmentDescriptor};
use crate::store::types::{segment_to_row, LocalSegment, SegmentLocation, SegmentRow};

/// Everything one object compaction reads, writes, and commits through.
pub struct ObjectCompaction<'a> {
    pub table: &'a str,
    pub table_dir: &'a Path,
    pub format: &'a SegmentFormat,
    pub index_config: SegmentIndexConfig,
    pub catalog: &'a Catalog,
    pub controller: &'a TableController,
    pub segments: &'a SegmentView,
    pub query_visibility: &'a Arc<RwLock<()>>,
    /// The process-wide leveled tuning, before the table's own object-size
    /// target is applied.
    pub config: &'a CompactionConfig,
    /// The object size this table's specification asks every level to promote at.
    pub target_object_bytes: i64,
}

/// One planned compaction with the lifecycle state its commit is fenced to.
struct LeasedCompaction<'context, 'resources> {
    resources: &'context ObjectCompaction<'resources>,
    lease: MaintenanceLease,
    table_spec_version: u64,
    migration_backfill: bool,
}

/// Result of one object-compaction attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactionOutcome {
    /// No stream was eligible, or no readable input was available.
    Idle,
    /// The replacement committed and more debt may remain.
    Committed,
    /// Planning succeeded but a concurrent lifecycle or input change won.
    Conflicted,
}

impl CompactionOutcome {
    /// Whether the scheduler should retry on its fast maintenance cadence.
    pub fn has_pending_work(self) -> bool {
        self != Self::Idle
    }
}

/// Compact one planner-issued run of immutable objects and commit the
/// replacement under a maintenance lease.
///
/// `force_compact_l0` makes an L0 run eligible regardless of the size threshold
/// the planner would otherwise apply.
///
/// Returns the attempt outcome so the scheduler can distinguish a quiet table
/// from a conflict that should be replanned promptly.
pub async fn compact_once(
    compaction: ObjectCompaction<'_>,
    force_compact_l0: bool,
) -> Result<CompactionOutcome, StatsError> {
    let lifecycle = compaction.catalog.spec_lifecycle(compaction.table)?;
    // While a spec migration is pending, dual-write flushes commit ordinary L0
    // objects at the target version; compact those so a many-hour backfill does
    // not stack hundreds of small objects into the query view.
    let migration_pending = lifecycle.desired_version() != 0;
    let table_spec_version = if migration_pending {
        lifecycle.desired_version()
    } else {
        lifecycle.active_version()
    };
    if table_spec_version == 0 {
        return Ok(CompactionOutcome::Idle);
    }
    let object_records: HashMap<_, _> = compaction
        .catalog
        .object_segments(compaction.table)?
        .into_iter()
        .filter(|record| record.table_spec_version == table_spec_version)
        .map(|record| (record.path.clone(), record))
        .collect();
    // A backfilled run and an ordinary run are separate compaction streams: a
    // checkpointed migration output must not be merged with a segment the
    // migration has not accounted for. While the transition is pending the
    // backfilled stream is untouchable outright: a replacement re-inserts its
    // outputs without `migration_source_id`, which would erase the coverage
    // checkpoints the backfill reads and make it rewrite those sources again.
    let mut rows_by_class: BTreeMap<bool, Vec<SegmentRow>> = BTreeMap::new();
    for row in compaction.catalog.list_segments(compaction.table)? {
        if let Some(record) = object_records.get(&row.path) {
            if migration_pending && record.migration_backfill {
                continue;
            }
            rows_by_class
                .entry(record.migration_backfill)
                .or_default()
                .push(row);
        }
    }
    let config = CompactionConfig {
        level_targets: vec![compaction.target_object_bytes; compaction.config.level_targets.len()],
        ..compaction.config.clone()
    };
    let Some((migration_backfill, job)) = rows_by_class.into_iter().find_map(|(backfill, rows)| {
        plan(&config, &rows, UnpartitionedRunPolicy::SparseStream)
            .or_else(|| force_compact_l0.then(|| plan_forced_l0(&rows)).flatten())
            .map(|job| (backfill, job))
    }) else {
        return Ok(CompactionOutcome::Idle);
    };

    // The lease pins the writer fence and definition version. Merging,
    // encoding, and uploading then run outside the controller; only the
    // replacement commit is serialized against concurrent flushes.
    let lease = compaction.controller.begin_compaction_for(&lifecycle)?;
    let localize_started = Instant::now();
    for input in &job.inputs {
        let record = object_records.get(&input.path).ok_or_else(|| {
            StatsError::Internal(format!("object compaction lost input {}", input.path))
        })?;
        let localized = compaction.controller.localize(&record.source).await?;
        if localized != Path::new(&input.path) {
            return Err(StatsError::Internal(format!(
                "object {} localized to {} rather than its catalog path",
                input.path,
                localized.display()
            )));
        }
    }
    let localize_ms = localize_started.elapsed().as_millis() as u64;
    if localize_ms > 1_000 {
        // Uncached inputs are fetched from remote storage one at a time; when
        // that dominates a cycle it explains a table's stalled maintenance.
        tracing::info!(
            namespace = %compaction.table,
            inputs = job.inputs.len(),
            localize_ms,
            "object compaction localized inputs"
        );
    }

    let staging = StagingDir::create(compaction.table_dir)?;
    let leased = LeasedCompaction {
        resources: &compaction,
        lease,
        table_spec_version,
        migration_backfill,
    };
    let outcome = run(&leased, &staging, &job).await;
    drop(staging);
    outcome
}

/// Execute one object compaction inside `staging` and commit its result.
async fn run(
    leased: &LeasedCompaction<'_, '_>,
    staging: &StagingDir,
    job: &CompactionJob,
) -> Result<CompactionOutcome, StatsError> {
    let compaction = leased.resources;
    let index_config = compaction.index_config.clone();
    let arrow_schema = Arc::clone(compaction.format.arrow_schema());
    let sort_columns = compaction.format.sort_columns().to_vec();
    let key_column = compaction.format.key_column().to_string();
    let max_row_group_rows = compaction.format.max_row_group_rows();
    let max_merge_arrow_bytes = compaction.config.max_merge_arrow_bytes;
    let key_bounds: HashMap<String, (Option<String>, Option<String>)> = job
        .inputs
        .iter()
        .map(|row| (row.path.clone(), compaction.segments.key_bounds(&row.path)))
        .collect();
    let job_for_run = job.clone();
    let staging_dir = staging.path().to_path_buf();
    // Same dedicated merge thread as a migration rewrite, mildly deprioritized:
    // an ordinary compaction merge at blocking-pool priority competes with
    // query threads for cores, and a backlog of large merges makes dashboards
    // unusable. nice(10) yields to busy serving threads but still finishes
    // (nice(19) would starve outright on a saturated box).
    let merge_started = Instant::now();
    let swap = run_merge_thread("finelog-compact", 10, move || {
        run_job_with_partition_policy(
            &job_for_run,
            &staging_dir,
            &arrow_schema,
            CompactionExecution {
                layout: crate::store::compaction::executor::CompactionLayout {
                    sort_columns: &sort_columns,
                    key_column: &key_column,
                    max_row_group_rows,
                },
                index_config: &index_config,
                partition_policy: None,
                max_merge_arrow_bytes,
                output: OutputPolicy::PromoteWhenUnchanged,
            },
            move |path| key_bounds.get(path).cloned().unwrap_or((None, None)),
        )
    })
    .await??;
    let merge_ms = merge_started.elapsed().as_millis() as u64;

    // A single-input run is a level promotion. An immutable object is never
    // renamed, so the promotion re-advertises the same source and artifacts at
    // the higher level instead of rewriting anything.
    if swap.bump_rename.is_some() {
        return commit_level_bump(leased, &swap.removed, swap.added).await;
    }
    if swap.added.is_empty() {
        tracing::warn!(
            namespace = %compaction.table,
            dropped = ?swap.removed,
            "object compaction found no readable input; leaving the run for the next tick"
        );
        return Ok(CompactionOutcome::Idle);
    }

    let upload_started = Instant::now();
    let mut outputs = Vec::with_capacity(swap.added.len());
    let mut published = Vec::with_capacity(swap.added.len());
    for staged in swap.added {
        let staged_path = PathBuf::from(&staged.path);
        let stored = compaction
            .controller
            .write_staged_object(OBJECTS_PREFIX, "parquet", &staged_path)
            .await?;
        let (references, local) = publish_segment_artifacts(
            compaction.controller,
            compaction.table,
            &staged_path,
            &stored,
        )
        .await?;
        let segment = LocalSegment {
            path: stored.path.to_string_lossy().into_owned(),
            size_bytes: stored.byte_size,
            location: SegmentLocation::Both,
            artifacts: local,
            ..staged
        };
        outputs.push(SegmentDescriptor {
            row: segment_to_row(compaction.table, &segment),
            source: stored.source,
            artifacts: references,
        });
        published.push(segment);
    }
    tracing::info!(
        namespace = %compaction.table,
        inputs = job.inputs.len(),
        merge_ms,
        upload_ms = upload_started.elapsed().as_millis() as u64,
        "object compaction rewrite finished"
    );
    commit_replacement(leased, &swap.removed, outputs, published).await
}

/// Promote a single input to the next level without rewriting its object.
async fn commit_level_bump(
    leased: &LeasedCompaction<'_, '_>,
    removed: &[String],
    added: Vec<LocalSegment>,
) -> Result<CompactionOutcome, StatsError> {
    let compaction = leased.resources;
    let records: HashMap<_, _> = compaction
        .catalog
        .object_segments(compaction.table)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let mut outputs = Vec::with_capacity(added.len());
    let mut published = Vec::with_capacity(added.len());
    for (staged, input) in added.into_iter().zip(removed) {
        let record = records
            .get(input)
            .ok_or_else(|| StatsError::Internal(format!("level bump lost input {input}")))?;
        let segment = LocalSegment {
            path: input.clone(),
            artifacts: compaction.segments.artifacts(input),
            ..staged
        };
        outputs.push(SegmentDescriptor {
            row: segment_to_row(compaction.table, &segment),
            source: record.source.clone(),
            artifacts: record.artifacts.clone(),
        });
        published.push(segment);
    }
    commit_replacement(leased, removed, outputs, published).await
}

/// Commit one leased replacement and swap the local query view.
async fn commit_replacement(
    leased: &LeasedCompaction<'_, '_>,
    removed: &[String],
    outputs: Vec<SegmentDescriptor>,
    published: Vec<LocalSegment>,
) -> Result<CompactionOutcome, StatsError> {
    let compaction = leased.resources;
    let removed_paths = removed.to_vec();
    let committed = match compaction
        .controller
        .commit_maintenance(&leased.lease, || {
            let live: HashSet<String> = compaction
                .catalog
                .object_segments(compaction.table)?
                .into_iter()
                .map(|record| record.path)
                .collect();
            if let Some(retired) = removed_paths.iter().find(|path| !live.contains(*path)) {
                return Err(StatsError::SchemaConflict(format!(
                    "compaction input {retired} is no longer live"
                )));
            }
            let revision = compaction.catalog.replace_object_segments(
                compaction.table,
                &removed_paths,
                &outputs,
                leased.table_spec_version,
                leased.migration_backfill,
            )?;
            Ok((revision, ()))
        })
        .await
    {
        Ok(committed) => Some(committed.token.revision()),
        Err(error) if is_lease_conflict(&error) => {
            tracing::info!(
                namespace = %compaction.table,
                inputs = removed_paths.len(),
                outputs = published.len(),
                %error,
                "compaction lease lost a real conflict; abandoning the uploaded outputs"
            );
            return Ok(CompactionOutcome::Conflicted);
        }
        Err(error) if !error.is_committed() => return Err(error.into()),
        // Durable locally but not published; the maintenance loop owes HEAD that
        // revision. The local view still follows the committed rows.
        Err(error) => {
            tracing::warn!(namespace = %compaction.table, %error, "compaction commit awaits publication");
            None
        }
    };
    let _visibility_guard = compaction.query_visibility.write().await;
    let rows = published
        .iter()
        .map(|segment| segment.row_count)
        .sum::<i64>();
    let outputs = published.len();
    compaction.segments.replace(&removed_paths, published);
    tracing::info!(
        namespace = %compaction.table,
        inputs = removed_paths.len(),
        outputs,
        rows,
        catalog_generation = committed.map(|revision| revision.get()),
        "object-backed compaction committed"
    );
    Ok(CompactionOutcome::Committed)
}

/// Whether a leased commit lost a real conflict rather than hitting a transient
/// failure.
///
/// A conflict means the replacement can never apply: an input was retired, the
/// definition version moved, or another writer owns the table. Its outputs are
/// already immutable objects, so abandoning them leaves them unreferenced and
/// collectible instead of failing the table.
fn is_lease_conflict(error: &CommitError) -> bool {
    matches!(
        error,
        CommitError::Fenced(_) | CommitError::NotCommitted(StatsError::SchemaConflict(_))
    )
}
