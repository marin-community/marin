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
use crate::store::compaction::planner::plan;
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

/// Compact one planner-issued run of immutable objects and commit the
/// replacement under a maintenance lease.
///
/// `force_compact_l0` makes an L0 run eligible regardless of the size threshold
/// the planner would otherwise apply.
///
/// Returns `true` when a run was replaced and `false` when nothing is eligible
/// or the commit lost a conflict.
pub async fn compact_once(
    compaction: ObjectCompaction<'_>,
    force_compact_l0: bool,
) -> Result<bool, StatsError> {
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
        return Ok(false);
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
        plan(&config, &rows)
            .or_else(|| force_compact_l0.then(|| l0_promotion_job(&rows)).flatten())
            .map(|job| (backfill, job))
    }) else {
        return Ok(false);
    };

    // The lease pins the writer fence and definition version. Merging,
    // encoding, and uploading then run outside the controller; only the
    // replacement commit is serialized against concurrent flushes.
    let lease = compaction.controller.begin_compaction()?;
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
    let outcome = run(
        &compaction,
        &staging,
        &job,
        &lease,
        table_spec_version,
        migration_backfill,
    )
    .await;
    drop(staging);
    outcome
}

/// Execute one object compaction inside `staging` and commit its result.
async fn run(
    compaction: &ObjectCompaction<'_>,
    staging: &StagingDir,
    job: &CompactionJob,
    lease: &MaintenanceLease,
    table_spec_version: u64,
    migration_backfill: bool,
) -> Result<bool, StatsError> {
    let index_config = compaction.index_config.clone();
    let arrow_schema = Arc::clone(compaction.format.arrow_schema());
    let sort_columns = compaction.format.sort_columns().to_vec();
    let key_column = compaction.format.key_column().to_string();
    let max_row_group_rows = compaction.format.max_row_group_rows();
    let max_merge_arrow_bytes = compaction.config.max_merge_arrow_bytes;
    let key_bounds: HashMap<String, (Option<i64>, Option<i64>)> = job
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
            move |path| key_bounds.get(path).copied().unwrap_or((None, None)),
        )
    })
    .await??;
    let merge_ms = merge_started.elapsed().as_millis() as u64;

    // A single-input run is a level promotion. An immutable object is never
    // renamed, so the promotion re-advertises the same source and artifacts at
    // the higher level instead of rewriting anything.
    if swap.bump_rename.is_some() {
        return commit_level_bump(
            compaction,
            &swap.removed,
            swap.added,
            lease,
            table_spec_version,
            migration_backfill,
        )
        .await;
    }
    if swap.added.is_empty() {
        tracing::warn!(
            namespace = %compaction.table,
            dropped = ?swap.removed,
            "object compaction found no readable input; leaving the run for the next tick"
        );
        return Ok(false);
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
    commit_replacement(
        compaction,
        &swap.removed,
        outputs,
        published,
        lease,
        table_spec_version,
        migration_backfill,
    )
    .await
}

/// Promote a single input to the next level without rewriting its object.
async fn commit_level_bump(
    compaction: &ObjectCompaction<'_>,
    removed: &[String],
    added: Vec<LocalSegment>,
    lease: &MaintenanceLease,
    table_spec_version: u64,
    migration_backfill: bool,
) -> Result<bool, StatsError> {
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
    commit_replacement(
        compaction,
        removed,
        outputs,
        published,
        lease,
        table_spec_version,
        migration_backfill,
    )
    .await
}

/// Commit one leased replacement and swap the local query view.
async fn commit_replacement(
    compaction: &ObjectCompaction<'_>,
    removed: &[String],
    outputs: Vec<SegmentDescriptor>,
    published: Vec<LocalSegment>,
    lease: &MaintenanceLease,
    table_spec_version: u64,
    migration_backfill: bool,
) -> Result<bool, StatsError> {
    let removed_paths = removed.to_vec();
    let committed = match compaction
        .controller
        .commit_maintenance(lease, || {
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
                table_spec_version,
                migration_backfill,
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
            return Ok(false);
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
    Ok(true)
}

/// One job promoting every L0 segment in `rows` to L1.
///
/// The leveled policy only promotes a run that meets its byte or fanout target.
/// A caller that needs L0 stabilized now — a forced maintenance cycle — asks for
/// this instead.
fn l0_promotion_job(rows: &[SegmentRow]) -> Option<CompactionJob> {
    let mut inputs: Vec<SegmentRow> = rows.iter().filter(|row| row.level == 0).cloned().collect();
    if inputs.is_empty() {
        return None;
    }
    inputs.sort_by_key(|row| row.min_seq);
    let output_min_seq = inputs
        .iter()
        .map(|row| row.min_seq)
        .min()
        .expect("a non-empty run has a minimum seq");
    Some(CompactionJob {
        inputs,
        output_level: 1,
        output_min_seq,
    })
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
