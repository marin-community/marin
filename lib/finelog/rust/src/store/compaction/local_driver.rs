//! Driving compaction for a table whose segments are local files.
//!
//! The planner picks a run from the live view, the executor merges it into new
//! files beside the inputs, and [`commit_swap`] splices the result into the
//! view, the catalog, and the directory in the one order that keeps an in-flight
//! query safe.
//!
//! This path retires with legacy tables. An object-backed table never renames or
//! unlinks a file a reader may hold, so it uses
//! [`object_driver`](crate::store::compaction::object_driver) instead.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;

use crate::errors::StatsError;
use crate::indices::exact::NAMED_PROJECTION_MARKER;
use crate::indices::projection::covering_projection_paths;
use crate::indices::SegmentIndexConfig;
use crate::indices::{legacy_artifact_paths, remove_if_exists, remove_index_artifacts};
use crate::partition_policy::PhysicalPartitionPolicy;
use crate::store::catalog::Catalog;
use crate::store::compaction::config::{CompactionConfig, CompactionJob};
use crate::store::compaction::executor::{
    run_job_with_partition_policy, CompactionExecution, OutputPolicy, PlannedSwap,
};
use crate::store::compaction::planner::{plan, UnpartitionedRunPolicy};
use crate::store::legacy::archive::evict_segment;
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::SegmentView;
use crate::store::types::{segment_to_row, SegmentRow};

/// Everything one local compaction reads, writes, and commits through.
pub struct LocalCompaction<'a> {
    pub table: &'a str,
    pub table_dir: &'a Path,
    pub format: &'a SegmentFormat,
    pub index_config: SegmentIndexConfig,
    pub catalog: &'a Catalog,
    pub segments: &'a SegmentView,
    pub query_visibility: &'a Arc<RwLock<()>>,
    pub config: &'a CompactionConfig,
    pub partition_policy: Option<&'static dyn PhysicalPartitionPolicy>,
}

/// Run one planner-issued compaction job, returning `true` if a job ran.
///
/// The caller drains by looping while this returns `true`.
pub fn compact_once(compaction: &LocalCompaction<'_>) -> Result<bool, StatsError> {
    let rows = compaction.segments.rows();
    let Some(job) = plan(
        compaction.config,
        &rows,
        UnpartitionedRunPolicy::StrictAdjacency,
    ) else {
        return Ok(false);
    };
    run_job(compaction, &job)?;
    Ok(true)
}

/// Synthesize and apply a single L0->L1 merge of every L0 segment that fits
/// `max_merge_arrow_bytes` (all of them, at test data sizes).
///
/// Serves the flag-gated `/debug/maintain?compact=l0` route and tests that
/// need L1 state without tiny `level_targets`. No-op with no L0 segments.
pub fn promote_all_l0(compaction: &LocalCompaction<'_>) -> Result<(), StatsError> {
    let mut inputs: Vec<SegmentRow> = compaction
        .segments
        .segments()
        .iter()
        .filter(|segment| segment.level == 0)
        .map(|segment| segment_to_row(compaction.table, segment))
        .collect();
    if inputs.is_empty() {
        return Ok(());
    }
    inputs.sort_by_key(|row| row.min_seq);
    let output_min_seq = inputs
        .iter()
        .map(|row| row.min_seq)
        .min()
        .expect("non-empty");
    run_job(
        compaction,
        &CompactionJob {
            inputs,
            output_level: 1,
            output_min_seq,
        },
    )
}

/// Execute `job` (read+merge+write or rename) then commit the resulting swap.
///
/// The executor may consume only a prefix of `job.inputs` — as much as
/// `max_merge_arrow_bytes` admits — so the committed span comes from the swap,
/// not the job, and both counts are logged.
pub fn run_job(compaction: &LocalCompaction<'_>, job: &CompactionJob) -> Result<(), StatsError> {
    let started = Instant::now();
    tracing::info!(
        namespace = %compaction.table,
        planned_inputs = job.inputs.len(),
        output_level = job.output_level,
        input_bytes = job.inputs.iter().map(|row| row.byte_size).sum::<i64>(),
        input_rows = job.inputs.iter().map(|row| row.row_count).sum::<i64>(),
        "compaction job starting"
    );
    let swap = run_job_with_partition_policy(
        job,
        compaction.table_dir,
        compaction.format.arrow_schema(),
        CompactionExecution {
            layout: compaction.format.compaction_layout(),
            index_config: &compaction.index_config,
            partition_policy: compaction.partition_policy,
            max_merge_arrow_bytes: compaction.config.max_merge_arrow_bytes,
            output: OutputPolicy::PromoteWhenUnchanged,
        },
        |path| compaction.segments.key_bounds(path),
    )?;
    let merged_inputs = swap.removed.len();
    let input_arrow_bytes = swap.input_arrow_bytes;
    // A missing head input produces no output — the swap only names the stale
    // reference to drop. Route it through `evict_segment`, which is
    // location-aware (a BOTH segment collapses to REMOTE, preserving its durable
    // archive; a LOCAL-only row is removed) and tolerates the already absent
    // file. This unwedges compaction without deleting a segment that still has a
    // remote copy.
    if swap.added.is_empty() {
        for path in &swap.removed {
            evict_segment(
                compaction.table,
                compaction.catalog,
                compaction.segments,
                compaction.query_visibility,
                path,
            );
        }
        tracing::warn!(
            namespace = %compaction.table,
            dropped = ?swap.removed,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "dropped stale segment reference with no local file; compaction resumed"
        );
        return Ok(());
    }
    let output_bytes: i64 = swap.added.iter().map(|added| added.size_bytes).sum();
    let output_rows: i64 = swap.added.iter().map(|added| added.row_count).sum();
    let output_segments = swap.added.len();
    // A bump is a rename; a merge decodes its inputs into RAM. The distinction is
    // the whole memory story, so name it — along with the decoded size the
    // ceiling actually bounds, and how much of the planned job that ceiling let
    // this tick take.
    let kind = if swap.bump_rename.is_some() {
        "bump"
    } else {
        "merge"
    };
    commit_swap(compaction, swap)?;
    tracing::info!(
        namespace = %compaction.table,
        kind,
        output_level = job.output_level,
        output_segments,
        output_bytes,
        output_rows,
        merged_inputs,
        planned_inputs = job.inputs.len(),
        input_arrow_bytes,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "compaction job committed"
    );
    Ok(())
}

/// Splice the view + catalog: replace `swap.removed` paths with `swap.added`.
///
/// Takes the process-wide query-visibility WRITE lock (via `blocking_write`, so
/// it is only safe from a `spawn_blocking` / synchronous context) so in-flight
/// queries — which snapshot segment paths and open the Parquet files lazily —
/// have drained before any rename/unlink: renaming or unlinking a file under a
/// stale snapshot path surfaces as "No files found". A level-bump rename
/// (`swap.bump_rename`) runs FIRST, inside the held write lock, then the view +
/// catalog are spliced under the short view lock; merge inputs are unlinked last.
///
/// Lock order: query_visibility(write) -> view lock. The flush path takes the
/// flush lock and the view lock but NOT query_visibility, so there is no cycle.
pub fn commit_swap(compaction: &LocalCompaction<'_>, swap: PlannedSwap) -> Result<(), StatsError> {
    let PlannedSwap {
        removed,
        added,
        unlink_removed,
        bump_rename,
        ..
    } = swap;
    let _write_guard = compaction.query_visibility.blocking_write();
    // 1) Level-bump rename happens before the view mirrors the new path, so a
    //    drained reader never sees a half-renamed file. A failure here is
    //    propagated BEFORE any view/catalog mutation, so the swap aborts with
    //    nothing changed.
    if let Some((from, to)) = &bump_rename {
        rename_segment_with_artifacts(compaction.table, from, to)?;
    }
    assert!(
        !added.is_empty(),
        "commit_swap requires output segments; drops are handled by run_job"
    );
    let added_rows: Vec<SegmentRow> = added
        .iter()
        .map(|segment| segment_to_row(compaction.table, segment))
        .collect();
    compaction.segments.replace(&removed, added);
    // Atomic catalog splice. Propagate on failure: the view now points at paths
    // that exist on disk (the renamed bump target / the already-written merged
    // output), so a propagated error is a stats/boot-adoption metadata
    // inconsistency that self-heals at next boot adoption — never a
    // mid-scan-unlink hazard — and the merge inputs below are left intact because
    // we return before unlinking.
    compaction
        .catalog
        .replace_segments(compaction.table, &removed, &added_rows)?;
    // 2) Unlink merged inputs after the swap (level bumps already renamed).
    if unlink_removed {
        for path in &removed {
            if let Err(error) = std::fs::remove_file(path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!(namespace = %compaction.table, path = %path, %error, "failed to unlink merged input");
                }
            }
            // The merged output carries a fresh bundle; the inputs' derived
            // indexes are stale and unlinked with their Parquet.
            remove_index_artifacts(path);
        }
    }
    Ok(())
}

/// Move a segment file and carry the artifacts that can follow it.
///
/// The query path no longer reads legacy containers, so a source rename would
/// orphan them; they are deleted instead of carried forward. A bundle or
/// projection that cannot be moved is removed rather than left pointing at a
/// file that no longer exists.
fn rename_segment_with_artifacts(table: &str, from: &Path, to: &Path) -> Result<(), StatsError> {
    let destination_dir = to.parent().ok_or_else(|| {
        StatsError::Internal(format!(
            "segment destination has no parent: {}",
            to.display()
        ))
    })?;
    std::fs::create_dir_all(destination_dir).map_err(|error| {
        StatsError::Internal(format!(
            "create segment destination directory {}: {error}",
            destination_dir.display()
        ))
    })?;
    std::fs::rename(from, to).map_err(|error| {
        StatsError::Internal(format!(
            "level-bump rename {} -> {} failed: {error}",
            from.display(),
            to.display()
        ))
    })?;
    for legacy in legacy_artifact_paths(from) {
        remove_orphaned_artifact(table, &legacy, "legacy index");
    }
    let (bundle_from, bundle_to) = (
        crate::indices::format::bundle_path(from),
        crate::indices::format::bundle_path(to),
    );
    if bundle_from.exists() {
        if let Err(error) = std::fs::rename(&bundle_from, &bundle_to) {
            tracing::warn!(namespace = %table, from = %bundle_from.display(), %error, "failed to carry index bundle on level bump");
            remove_orphaned_artifact(table, &bundle_from, "index bundle");
        }
    }
    let (Some(from_name), Some(to_name)) = (from.file_name(), to.file_name()) else {
        return Ok(());
    };
    let prefix = format!("{}{NAMED_PROJECTION_MARKER}", from_name.to_string_lossy());
    match covering_projection_paths(from) {
        Ok(projections) => {
            for source in projections {
                let name = source.file_name().and_then(|name| name.to_str()).unwrap();
                let suffix = name.strip_prefix(&prefix).unwrap();
                let destination = destination_dir.join(format!(
                    "{}{NAMED_PROJECTION_MARKER}{suffix}",
                    to_name.to_string_lossy()
                ));
                if let Err(error) = std::fs::rename(&source, &destination) {
                    tracing::warn!(namespace = %table, from = %source.display(), %error, "failed to carry covering projection on level bump");
                    remove_orphaned_artifact(table, &source, "covering projection");
                }
            }
        }
        Err(error) => {
            tracing::warn!(namespace = %table, path = %from.display(), %error, "failed to enumerate covering projections on level bump")
        }
    }
    Ok(())
}

fn remove_orphaned_artifact(table: &str, path: &Path, kind: &str) {
    if let Err(error) = remove_if_exists(path) {
        tracing::warn!(
            namespace = table,
            path = %path.display(),
            index_artifact = kind,
            %error,
            "failed to remove orphaned segment index artifact"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::levanter_metrics_policy::levanter_metrics_schema;
    use crate::store::catalog::Catalog;
    use crate::store::schema::stored_form;
    use crate::store::segment::{discover_segments, read_segment_footer};
    use crate::store::table::maintenance::{self, TableWork};
    use crate::store::table::test_tables::*;

    #[tokio::test]
    async fn levanter_l0_stays_flat_and_compaction_writes_bucketed_l1() {
        let dir = tempdir();
        let table_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            catalog,
        );
        table
            .append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]))
            .unwrap();
        table.flush().await.unwrap();

        let l0 = discover_segments(&table_dir);
        assert_eq!(l0.len(), 1);
        assert_eq!(l0[0].parent(), Some(table_dir.as_path()));
        assert!(read_segment_footer(&l0[0], Some("timestamp_ms"))
            .unwrap()
            .partition
            .is_none());

        maintenance::run(
            &table,
            TableWork::Cycle {
                force_compact_l0: true,
            },
        )
        .await
        .unwrap();
        let l1 = discover_segments(&table_dir);
        assert_eq!(l1.len(), 2);
        for path in &l1 {
            assert_eq!(
                path.parent().unwrap().parent(),
                Some(table_dir.join("run_id").as_path())
            );
            let bucket: u32 = path
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .parse()
                .unwrap();
            assert!(bucket < 32);
            assert!(read_segment_footer(path, Some("timestamp_ms"))
                .unwrap()
                .partition
                .is_some());
        }
        assert_eq!(table.stats().row_count, 2);
        table.shutdown(Duration::from_secs(10)).await;
        let reopened = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(reopened.stats().row_count, 2);
        assert_eq!(reopened.query_snapshot().unwrap().paths.len(), 2);
        reopened.shutdown(Duration::from_secs(10)).await;

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn maintenance_drops_dangling_segment_reference_instead_of_wedging() {
        // Regression for the `iris.task` compaction wedge. A merge that consumed
        // and unlinked a segment can leave its view/catalog reference behind (a
        // duplicate entry the splice missed). The planner then hands back a job
        // whose head input file is gone; the old recovery tried to promote it by
        // rename, which failed on the absent source every `check_interval` and
        // wedged the table's compaction for good (14k L0 files and growing in
        // production). Maintenance must instead DROP the dangling reference and
        // keep compacting.
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            Arc::clone(&catalog),
        );

        // Three L0 segments (seq 1, 2, 3), each its own flush.
        write_one(&table).await;
        write_one(&table).await;
        write_one(&table).await;
        let before = discover_segments(&table_dir);
        assert_eq!(before.len(), 3, "three L0 segments on disk");

        // Delete the lowest-min_seq file while its view + catalog rows survive —
        // the dangling reference an already-consumed-and-unlinked input leaves,
        // and the head of the next planned run.
        let head = before.iter().min().unwrap().clone();
        std::fs::remove_file(&head).unwrap();

        // Before the fix this returned Err (rename of the absent head failed) and
        // every later tick replanned the identical doomed job.
        maintenance::run(
            &table,
            TableWork::Cycle {
                force_compact_l0: true,
            },
        )
        .await
        .expect("a dangling reference must not wedge maintenance");

        // The stale reference is gone from the catalog, and the two intact rows
        // survive (compacted forward, none lost).
        let rows = catalog.list_segments("iris.worker").unwrap();
        let head_str = head.to_string_lossy().to_string();
        assert!(
            rows.iter().all(|r| r.path != head_str),
            "the dangling reference was dropped from the catalog"
        );
        let total_rows: i64 = rows.iter().map(|r| r.row_count).sum();
        assert_eq!(total_rows, 2, "the two intact segments' rows survive");

        // Compaction is live again: a further tick runs without error.
        maintenance::run(
            &table,
            TableWork::Cycle {
                force_compact_l0: true,
            },
        )
        .await
        .expect("compaction stays live after the drop");
        std::fs::remove_dir_all(&dir).ok();
    }
}
