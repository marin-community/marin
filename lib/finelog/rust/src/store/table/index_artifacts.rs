//! Derived index artifacts for one table: publish, backfill, and removal.
//!
//! [`IndexRegistry`] owns the artifact formats; this module pairs it with the
//! table's controller so an artifact becomes *live* the same way any other fact
//! about a table does. A local table records the files a build produced beside
//! its Parquet. An object-backed table uploads each artifact as an immutable
//! object and commits its reference in a new table revision: adjacency to a
//! filename is never what makes an artifact live.
//!
//! An artifact that fails to build or upload is simply omitted. Queries fall
//! back to the source Parquet and a later backfill supplies it.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::errors::StatsError;
use crate::indices::{
    fixed_index_artifacts_exist, local_sidecar_artifacts, needs_rebuild, remove_index_artifacts,
    IndexBuildRequest, IndexRegistry, SegmentIndexConfig,
};
use crate::maintenance::MaintenanceLimits;
use crate::store::catalog::object_state_store::{INDICES_PREFIX, PROJECTIONS_PREFIX};
use crate::store::catalog::Catalog;
use crate::store::compaction::executor::read_segment_projected;
use crate::store::segment::segment_id;
use crate::store::table::controller::{TableController, WrittenObject};
use crate::store::table::segment_view::SegmentView;
use crate::store::table_state::{ArtifactReferences, LocalArtifacts, SourceBinding};
use crate::store::types::LocalSegment;

/// Segment index bundles rebuilt or removed per maintenance tick.
///
/// A single index build over a terminal-level segment is heavy (substantial CPU
/// and RAM), and the backfill is the lowest-priority maintenance work, so this
/// stays small enough never to starve compaction/sync/eviction. It is four rather
/// than one so a table whose bundles all need rebuilding converges in tens of
/// minutes instead of hours while queries safely use partial coverage.
pub const INDEX_BUNDLES_PER_TICK: usize = 4;

/// Upload the artifacts an executor built beside `staged` and return both their
/// durable references and the local files those references resolve to.
///
/// An artifact that fails to upload is omitted: the source segment commits
/// without it and backfill supplies it later.
pub async fn publish_segment_artifacts(
    controller: &TableController,
    table: &str,
    staged: &Path,
    stored: &WrittenObject,
) -> Result<(ArtifactReferences, LocalArtifacts), StatsError> {
    let built = local_sidecar_artifacts(staged);
    if built.is_empty() {
        return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
    }
    let binding = SourceBinding {
        segment_uuid: segment_id(&stored.path).map(|id| id.to_string()),
        row_count: built.binding.row_count,
    };
    let mut references = ArtifactReferences {
        binding: binding.clone(),
        ..Default::default()
    };
    let mut local = LocalArtifacts {
        binding,
        ..Default::default()
    };
    for (name, path) in &built.projections {
        match controller
            .write_staged_object(PROJECTIONS_PREFIX, "parquet", path)
            .await
        {
            Ok(uploaded) => {
                references.projections.insert(name.clone(), uploaded.source);
                local.projections.insert(name.clone(), uploaded.path);
            }
            Err(error) => {
                tracing::warn!(namespace = %table, projection = %name, %error, "covering projection upload failed; committing without it");
                return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
            }
        }
    }
    let Some(bundle) = built.bundle.as_ref() else {
        return Ok((ArtifactReferences::default(), LocalArtifacts::default()));
    };
    match controller
        .write_staged_object(INDICES_PREFIX, "fidx", bundle)
        .await
    {
        Ok(uploaded) => {
            references.bundle = Some(uploaded.source);
            local.bundle = Some(uploaded.path);
            Ok((references, local))
        }
        Err(error) => {
            tracing::warn!(namespace = %table, %error, "index bundle upload failed; committing the segment without it");
            Ok((ArtifactReferences::default(), LocalArtifacts::default()))
        }
    }
}

/// Everything the backfill reads and commits through.
pub struct IndexBackfill<'a> {
    pub table: &'a str,
    pub catalog: &'a Catalog,
    pub controller: &'a TableController,
    pub segments: &'a SegmentView,
    pub registry: &'a Arc<IndexRegistry>,
    pub limits: &'a MaintenanceLimits,
    pub config: SegmentIndexConfig,
    /// Whether the deployment's policy declares derived indexes for this table.
    /// A table whose policy turned them off has its stale artifacts removed
    /// instead of rebuilt.
    pub indexes_enabled: bool,
    /// Whether a segment already carries the current physical layout. A segment
    /// awaiting a rewrite is not worth indexing yet.
    pub layout_is_current: &'a (dyn Fn(&str) -> bool + Send + Sync),
    pub skips: &'a Mutex<BackfillSkips>,
}

/// Bring up to `max` segments' artifacts in line with the table's index policy.
///
/// A table whose policy declares indexes has its missing artifacts rebuilt; a
/// table whose policy disables them has stale artifacts removed. Returns how
/// many segments changed.
pub async fn maintain(backfill: IndexBackfill<'_>, max: usize) -> usize {
    if !backfill.indexes_enabled {
        return remove_disabled_artifacts(&backfill, max);
    }
    rebuild_missing_artifacts(&backfill, max).await
}

/// Rebuild complete artifacts for up to `max` L>=1 segments.
///
/// All index kinds share one projected source read.
async fn rebuild_missing_artifacts(backfill: &IndexBackfill<'_>, max: usize) -> usize {
    if max == 0 || backfill.config.is_empty() {
        return 0;
    }
    let Ok(_slot) = backfill.limits.index_backfill().try_lock() else {
        return 0;
    };
    let table = backfill.table;
    let mut built = 0;
    for candidate in candidates(backfill, max) {
        let path = PathBuf::from(&candidate.path);
        let registry = Arc::clone(backfill.registry);
        let config = backfill.config.clone();
        let build_path = path.clone();
        let artifacts = match tokio::task::spawn_blocking(move || {
            let projection = config.input_columns();
            let batches = read_segment_projected(&build_path, Some(&projection))?;
            registry
                .build(IndexBuildRequest {
                    source: &build_path,
                    batches: &batches,
                    config: &config,
                })
                .map_err(|error| StatsError::Internal(format!("build segment index: {error}")))
        })
        .await
        {
            Ok(Ok(artifacts)) => artifacts,
            Ok(Err(error)) => {
                tracing::warn!(namespace = %table, path = %candidate.path, %error, "index backfill build failed");
                continue;
            }
            Err(error) => {
                tracing::warn!(namespace = %table, path = %candidate.path, %error, "index backfill task panicked");
                continue;
            }
        };
        if !artifacts.is_empty() {
            if let Err(error) = commit_built_artifacts(backfill, &candidate.path).await {
                tracing::warn!(namespace = %table, path = %candidate.path, %error, "index backfill commit failed");
                continue;
            }
            built += 1;
            tracing::debug!(namespace = %table, path = %candidate.path, "backfilled segment index artifacts");
        }
        if needs_rebuild(
            &path,
            candidate.expected_rows,
            &local_sidecar_artifacts(&path),
            &backfill.config,
        ) {
            tracing::debug!(namespace = %table, path = %candidate.path, "segment cannot satisfy the current index policy; not retrying");
            backfill.skips.lock().unwrap().paths.insert(candidate.path);
        }
    }
    built
}

/// Publish the artifacts just built beside `path` and make them live.
async fn commit_built_artifacts(
    backfill: &IndexBackfill<'_>,
    path: &str,
) -> Result<(), StatsError> {
    let staged = PathBuf::from(path);
    if !backfill.controller.is_object_backed() {
        let local = local_sidecar_artifacts(&staged);
        if let Some(bundle) = local.bundle.as_ref() {
            backfill.registry.invalidate(bundle);
        }
        backfill.segments.update(path, |segment| {
            segment.artifacts = local;
        });
        return Ok(());
    }
    let record = backfill
        .catalog
        .object_segments(backfill.table)?
        .into_iter()
        .find(|record| record.path == path)
        .ok_or_else(|| {
            StatsError::Internal(format!("index backfill lost object segment {path}"))
        })?;
    let stored = WrittenObject {
        path: staged.clone(),
        source: record.source.clone(),
        byte_size: 0,
    };
    let (references, local) =
        publish_segment_artifacts(backfill.controller, backfill.table, &staged, &stored).await?;
    if references.is_empty() {
        return Ok(());
    }
    let owned_path = path.to_string();
    backfill
        .controller
        .commit(|| {
            let revision =
                backfill
                    .catalog
                    .set_segment_artifacts(backfill.table, &owned_path, &references)?;
            Ok((revision, ()))
        })
        .await?;
    if let Some(bundle) = local.bundle.as_ref() {
        backfill.registry.invalidate(bundle);
    }
    backfill.segments.update(path, |segment| {
        segment.artifacts = local;
    });
    Ok(())
}

/// Remove derived index files from a table whose policy disables them.
fn remove_disabled_artifacts(backfill: &IndexBackfill<'_>, max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    let Ok(_slot) = backfill.limits.index_backfill().try_lock() else {
        return 0;
    };
    let segments = backfill.segments.segments();
    let candidates = {
        let mut skips = backfill.skips.lock().unwrap();
        let live: HashSet<&str> = segments
            .iter()
            .map(|segment| segment.path.as_str())
            .collect();
        skips.reconcile(&["disabled"], &live);
        let mut candidates = Vec::new();
        for segment in segments.iter().rev() {
            if skips.paths.contains(&segment.path) {
                continue;
            }
            if fixed_index_artifacts_exist(Path::new(&segment.path)) {
                candidates.push(segment.path.clone());
                if candidates.len() >= max {
                    break;
                }
            } else {
                skips.paths.insert(segment.path.clone());
            }
        }
        candidates
    };
    let mut cleaned = 0;
    for path in candidates {
        remove_index_artifacts(&path);
        if fixed_index_artifacts_exist(Path::new(&path)) {
            continue;
        }
        if let Some(bundle) = local_sidecar_artifacts(Path::new(&path)).bundle.as_ref() {
            backfill.registry.invalidate(bundle);
        }
        backfill.segments.update(&path, |segment| {
            segment.artifacts = LocalArtifacts::default();
        });
        backfill.skips.lock().unwrap().paths.insert(path);
        cleaned += 1;
    }
    if cleaned > 0 {
        tracing::info!(
            namespace = %backfill.table,
            segments = cleaned,
            "removed disabled segment index artifacts"
        );
    }
    cleaned
}

struct BackfillCandidate {
    path: String,
    expected_rows: i64,
}

/// The newest-first segments whose artifacts do not satisfy the index policy.
fn candidates(backfill: &IndexBackfill<'_>, max: usize) -> Vec<BackfillCandidate> {
    let segments: Vec<LocalSegment> = backfill.segments.segments();
    let fingerprint = format!("{:?}", backfill.config.policy_fingerprint());
    let mut skips = backfill.skips.lock().unwrap();
    let live: HashSet<&str> = segments
        .iter()
        .map(|segment| segment.path.as_str())
        .collect();
    skips.reconcile(&[fingerprint.as_str()], &live);
    let mut candidates = segments
        .iter()
        .filter(|segment| !skips.paths.contains(&segment.path))
        .filter(|segment| {
            segment.level >= 1
                && (backfill.layout_is_current)(&segment.path)
                && needs_rebuild(
                    Path::new(&segment.path),
                    segment.row_count,
                    &segment.artifacts,
                    &backfill.config,
                )
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|segment| Reverse(segment.max_seq));
    candidates
        .into_iter()
        .take(max)
        .map(|segment| BackfillCandidate {
            path: segment.path.clone(),
            expected_rows: segment.row_count,
        })
        .collect()
}

/// Segments the backfill cannot bring up to date, and the indexed set that
/// verdict was reached under.
///
/// A trigram index covers only the columns a segment actually has: one written
/// before a column existed indexes nothing for it, and its bundle can never
/// satisfy the rebuild condition. Without this the backfill would re-read and
/// re-serialize that segment on every maintenance tick forever, at one segment
/// per tick starving every segment that can still make progress. Enabling
/// another index resets the verdict, since the new column may well be present.
#[derive(Default)]
pub struct BackfillSkips {
    indexed: Vec<String>,
    paths: HashSet<String>,
}

impl BackfillSkips {
    /// Drop the recorded verdicts when the indexed set changes, and forget
    /// segments that are no longer live (compacted away or evicted).
    fn reconcile(&mut self, indexed: &[&str], live: &HashSet<&str>) {
        if self.indexed.len() != indexed.len()
            || !self.indexed.iter().zip(indexed).all(|(a, b)| a == b)
        {
            self.indexed = indexed.iter().map(|column| column.to_string()).collect();
            self.paths.clear();
            return;
        }
        self.paths.retain(|path| live.contains(path.as_str()));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    use crate::indices::legacy_artifact_paths;
    use crate::levanter_metrics_policy::levanter_metrics_schema;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::catalog::object_state_store::{INDICES_PREFIX, PROJECTIONS_PREFIX};
    use crate::store::catalog::Catalog;
    use crate::store::schema::{stored_form, with_implicit_seq, AlignedBatch, Column, Schema};
    use crate::store::segment::discover_segments;
    use crate::store::table::maintenance::{self, TableWork};
    use crate::store::table::runtime::TableRuntime;
    use crate::store::table::test_tables::*;

    /// Bring up to `max` segments in line with the table's index policy through
    /// the same context the maintenance cycle builds.
    async fn maintain_artifacts(table: &Arc<TableRuntime>, max: usize) -> usize {
        let tracker = &table.layout_tracker;
        let layout_is_current = |path: &str| tracker.is_current(path);
        maintain(maintenance::index_backfill(table, &layout_is_current), max).await
    }

    async fn maintain_cycle(table: &Arc<TableRuntime>, force_compact_l0: bool) {
        maintenance::run(table, TableWork::Cycle { force_compact_l0 })
            .await
            .unwrap();
    }

    /// Log-form schema carrying the trigram-indexed `data` string column.
    fn data_schema() -> Schema {
        with_implicit_seq(Schema::new(
            vec![
                Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index(),
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            ],
            "timestamp_ms",
        ))
    }

    fn exact_data_schema() -> Schema {
        with_implicit_seq(
            Schema::new(
                vec![
                    Column::new("data", ColumnType::COLUMN_TYPE_STRING, false)
                        .with_exact_values(["log line 0 searchable text"])
                        .with_value_counts(),
                    Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                ],
                "timestamp_ms",
            )
            .with_covering_projection(crate::store::schema::CoveringProjection::new(
                "matching-lines",
                "data",
                ["log line 0 searchable text"],
                ["seq", "data", "timestamp_ms"],
            )),
        )
    }

    /// `n` rows of searchable `data` + monotonic `timestamp_ms` (non-seq columns
    /// in registered order, as `append_aligned_batch` expects).
    fn data_aligned(n: i64, first: i64) -> AlignedBatch {
        let data: Vec<String> = (0..n)
            .map(|i| format!("log line {} searchable text", first + i))
            .collect();
        let ts: Vec<i64> = (0..n).map(|i| 1000 + first + i).collect();
        AlignedBatch {
            arrays: vec![
                Arc::new(StringArray::from(data)),
                Arc::new(Int64Array::from(ts)),
            ],
            fields: vec![
                Field::new("data", DataType::Utf8, false),
                Field::new("timestamp_ms", DataType::Int64, false),
            ],
            num_rows: n as usize,
            byte_size: 48 * n,
        }
    }

    #[tokio::test]
    async fn levanter_index_cleanup_converges_across_restart() {
        let dir = tempdir();
        let table_dir = dir.join("levanter.metrics");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            catalog,
        );
        table.append_aligned_batch(&metrics_aligned(&["run-a", "run-b"]));
        table.flush().await.unwrap();
        maintain_cycle(&table, true).await;

        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 2);
        for segment in &segments {
            std::fs::write(crate::indices::format::bundle_path(segment), b"stale").unwrap();
        }
        let legacy = legacy_artifact_paths(&segments[0])[0].clone();
        std::fs::write(&legacy, b"stale").unwrap();
        let projection = crate::indices::exact::named_projection_path(&segments[0], "legacy");
        std::fs::write(&projection, b"stale").unwrap();
        table.shutdown(Duration::from_secs(10)).await;

        let cleanup = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        assert_eq!(maintain_artifacts(&cleanup, 1).await, 1);
        assert_eq!(
            segments
                .iter()
                .filter(|segment| crate::indices::format::bundle_path(segment).exists())
                .count(),
            1
        );
        cleanup.shutdown(Duration::from_secs(10)).await;

        let reopened = open_table(
            "levanter.metrics",
            stored_form(levanter_metrics_schema()),
            Some(table_dir.clone()),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );
        maintain_cycle(&reopened, false).await;
        for segment in &segments {
            assert!(!crate::indices::format::bundle_path(segment).exists());
        }
        assert!(!legacy.exists());
        assert!(!projection.exists());
        assert_eq!(reopened.stats().row_count, 2);
        reopened.shutdown(Duration::from_secs(10)).await;

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn low_cardinality_counts_are_automatic_for_string_columns() {
        let dir = tempdir();
        let table = open_table(
            "logs",
            data_schema(),
            Some(dir.join("logs")),
            Arc::new(Catalog::open(Some(&dir)).unwrap()),
        );

        assert!(table
            .format
            .index_config(table.name())
            .indexes
            .iter()
            .any(|index| {
                matches!(
                    index,
                    crate::indices::IndexSpec::AdaptiveValueCounts { column }
                        if column == "data"
                )
            }));
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn grouped_extrema_are_declared_instead_of_column_inferred() {
        let dir = tempdir();
        let schema = Schema::new(
            vec![
                Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
                Column::new("service", ColumnType::COLUMN_TYPE_STRING, false),
                Column::new(
                    "resource_attributes_json",
                    ColumnType::COLUMN_TYPE_STRING,
                    false,
                ),
            ],
            "timestamp_ms",
        );
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let undeclared = open_table(
            "same_columns_without_policy",
            with_implicit_seq(schema.clone()),
            Some(dir.join("same_columns_without_policy")),
            Arc::clone(&catalog),
        );
        assert!(!undeclared
            .format
            .index_config(undeclared.name())
            .indexes
            .iter()
            .any(|index| {
                matches!(
                    index,
                    crate::indices::IndexSpec::AdaptiveGroupExtrema { .. }
                )
            }));

        let config = crate::indices::group_extrema::GroupExtremaConfig::new(
            "service",
            "resource_attributes_json",
            "job_id",
            "timestamp_ms",
        );
        let declared = open_table(
            "declared_policy",
            with_implicit_seq(schema.with_grouped_extrema(config)),
            Some(dir.join("declared_policy")),
            catalog,
        );
        assert!(declared
            .format
            .index_config(declared.name())
            .indexes
            .iter()
            .any(|index| {
                matches!(
                    index,
                    crate::indices::IndexSpec::AdaptiveGroupExtrema { config }
                        if config.filter_column == "service"
                            && config.json_column == "resource_attributes_json"
                            && config.json_key == "job_id"
                            && config.extrema_column == "timestamp_ms"
                )
            }));
        std::fs::remove_dir_all(dir).ok();
    }

    #[tokio::test]
    async fn backfill_rebuilds_missing_index_bundle() {
        let dir = tempdir();
        let table_dir = dir.join("log.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table("log.test", data_schema(), Some(table_dir.clone()), catalog);

        // Two L0 flushes merged to one L1 — the merge builds the bundle.
        table.append_aligned_batch(&data_aligned(5, 0));
        table.flush().await.unwrap();
        let last = table.append_aligned_batch(&data_aligned(5, 5));
        table.flush().await.unwrap();
        table
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        // The maintenance cycle wraps the merge in spawn_blocking (commit_swap
        // takes the blocking query-visibility lock); a multi-input merge builds
        // the bundle.
        maintain_cycle(&table, true).await;

        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1, "two L0 merged into one L1");
        let bundle = crate::indices::format::bundle_path(&segments[0]);
        assert!(bundle.exists(), "the merge wrote an index bundle");

        // Simulate a segment compaction never indexed (single-input bump, or one
        // written before indexes existed): drop the bundle.
        std::fs::remove_file(&bundle).unwrap();
        assert!(!bundle.exists());

        // The backfill rebuilds exactly the one missing bundle, then idles.
        assert_eq!(maintain_artifacts(&table, 10).await, 1);
        assert!(bundle.exists(), "backfill rebuilt the bundle");
        assert_eq!(
            maintain_artifacts(&table, 10).await,
            0,
            "nothing left to do"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Object persistence is a property of the server: every table it serves
    /// has a remote available, and the legacy ones among them still keep their
    /// data in local segment files. A legacy table classified object-backed
    /// would look itself up among object segments it never wrote and fail every
    /// backfill commit, so both the classification and the commit are asserted.
    #[tokio::test]
    async fn a_legacy_table_backfills_locally_on_a_remote_configured_store() {
        let dir = tempdir();
        let table_dir = dir.join("log.test");
        let remote = dir.join("remote");
        std::fs::create_dir_all(&remote).unwrap();
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table_remote(
            "log.test",
            data_schema(),
            Some(table_dir.clone()),
            catalog,
            remote.to_str().unwrap(),
            crate::store::policy::StoragePolicy::default(),
        );

        assert!(
            table.controller().object_persistence_configured(),
            "the store has a remote configured"
        );
        assert!(
            !table.controller().is_object_backed(),
            "a table with no object-store specification keeps its data locally"
        );

        table.append_aligned_batch(&data_aligned(5, 0));
        table.flush().await.unwrap();
        let last = table.append_aligned_batch(&data_aligned(5, 5));
        table.flush().await.unwrap();
        table
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        maintain_cycle(&table, true).await;

        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1, "two L0 merged into one L1");
        let bundle = crate::indices::format::bundle_path(&segments[0]);
        std::fs::remove_file(&bundle).unwrap();

        assert_eq!(
            maintain_artifacts(&table, 10).await,
            1,
            "the legacy branch commits the rebuilt bundle"
        );
        assert!(bundle.exists());
        assert_eq!(maintain_artifacts(&table, 10).await, 0);

        table.shutdown(Duration::from_secs(10)).await;
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn exact_backfill_rebuilds_a_missing_filtered_projection() {
        let dir = tempdir();
        let table_dir = dir.join("telemetry.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "telemetry.test",
            exact_data_schema(),
            Some(table_dir.clone()),
            catalog,
        );
        table.append_aligned_batch(&data_aligned(5, 0));
        table.flush().await.unwrap();
        maintain_cycle(&table, true).await;

        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1);
        assert!(segments[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("seg_L1_"));
        let bundle = crate::indices::format::bundle_path(&segments[0]);
        let projection =
            crate::indices::exact::named_projection_path(&segments[0], "matching-lines");
        assert!(bundle.exists());
        assert!(projection.exists());
        std::fs::remove_file(&projection).unwrap();

        assert_eq!(maintain_artifacts(&table, 10).await, 1);
        assert!(projection.exists());
        assert_eq!(maintain_artifacts(&table, 10).await, 0);

        assert!(crate::indices::format::bundle_path(&segments[0]).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn l0_does_not_write_derived_index_artifacts() {
        let dir = tempdir();
        let table_dir = dir.join("telemetry.test");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table(
            "telemetry.test",
            exact_data_schema(),
            Some(table_dir.clone()),
            catalog,
        );
        table.append_aligned_batch(&data_aligned(5, 0));
        table.flush().await.unwrap();

        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1);
        assert!(segments[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("seg_L0_"));
        assert!(!crate::indices::format::bundle_path(&segments[0]).exists());
        assert!(
            !crate::indices::exact::named_projection_path(&segments[0], "matching-lines").exists()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn backfill_is_a_noop_without_the_indexed_column() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // worker_schema has no `data` column, so there is nothing to index.
        let table = open_table(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            catalog,
        );
        table.append_aligned_batch(&aligned(3));
        table.flush().await.unwrap();
        let last = table.append_aligned_batch(&aligned(3));
        table.flush().await.unwrap();
        table
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        maintain_cycle(&table, true).await;

        assert_eq!(maintain_artifacts(&table, 10).await, 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A column indexed after a segment was written is not in that segment to
    /// index, so its bundle can never satisfy the rebuild condition. The
    /// backfill must try once and drop it, or it re-reads that segment on every
    /// tick and never reaches the segments that can still be indexed.
    #[tokio::test]
    async fn backfill_gives_up_on_a_segment_predating_an_indexed_column() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());

        // Segments written under a schema with no `data` column at all.
        let before = open_table(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            catalog.clone(),
        );
        before.append_aligned_batch(&aligned(3));
        before.flush().await.unwrap();
        let last = before.append_aligned_batch(&aligned(3));
        before.flush().await.unwrap();
        before
            .await_persisted(last, Duration::from_secs(10))
            .await
            .unwrap();
        maintain_cycle(&before, true).await;
        let segments = discover_segments(&table_dir);
        assert_eq!(segments.len(), 1, "two L0 merged into one L1");
        assert!(
            crate::indices::format::bundle_path(&segments[0]).exists(),
            "the original string columns receive adaptive counts"
        );
        before.shutdown(Duration::from_secs(10)).await;

        // Reopen with `data` added and indexed, as `merge_schemas` would leave it.
        let mut columns = worker_schema().columns;
        columns
            .push(Column::new("data", ColumnType::COLUMN_TYPE_STRING, false).with_trigram_index());
        let after = open_table(
            "iris.worker",
            Schema::new(columns, "timestamp_ms"),
            Some(table_dir.clone()),
            catalog,
        );

        assert_eq!(
            maintain_artifacts(&after, 10).await,
            1,
            "available adaptive sections are rebuilt once"
        );
        let path = segments[0].to_string_lossy().to_string();
        assert!(
            after.index_skips.lock().unwrap().paths.contains(&path),
            "the segment is dropped from future ticks rather than retried",
        );

        // Indexing another column is a new question, so the verdict is dropped.
        after
            .index_skips
            .lock()
            .unwrap()
            .reconcile(&["data", "worker_id"], &HashSet::from([path.as_str()]));
        assert!(after.index_skips.lock().unwrap().paths.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }
}
