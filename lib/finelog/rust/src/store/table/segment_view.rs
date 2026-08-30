//! The live segment set one table serves reads from.
//!
//! A legacy table's view is the files currently on disk; an object-backed
//! table's view is the projection of its published state. Both are held here as
//! one ordered set behind a short lock, so a read, a compaction commit, and an
//! eviction all mutate the same place under the same invariant: no two entries
//! share a path.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::Path;

use crate::errors::StatsError;
use crate::indices::{local_sidecar_artifacts, SegmentArtifacts};
use crate::partition_policy::SegmentPartition;
use crate::store::catalog::{Catalog, ObjectSegmentRecord};
use crate::store::object_store::ObjectStore;
use crate::store::table::controller::local_artifacts;
use crate::store::table::query_view::SegmentObjectMap;
use crate::store::table_state::LocalArtifacts;
use crate::store::types::{segment_to_row, LocalSegment, SegmentRow};

/// A table's readable segments as one consistent observation: the files a scan
/// may open, their known key bounds and partitions, and the lowest `seq` any of
/// them holds.
///
/// An object-backed table plans this from its pinned `TableSnapshot`, so
/// `sources` names the immutable object behind every path and the paths exist
/// only after the scan localizes the ones it selected. A legacy table plans it
/// from files already on disk and carries no `sources`.
pub struct SegmentSnapshot {
    pub paths: Vec<String>,
    pub key_bounds: BTreeMap<String, (i64, i64)>,
    /// Exact per-segment `seq` bounds, so a `seq`-bounded scan selects only the
    /// segments whose disjoint ranges it overlaps.
    pub seq_bounds: BTreeMap<String, (i64, i64)>,
    pub partitions: BTreeMap<String, SegmentPartition>,
    pub min_seq: Option<i64>,
    /// What each snapshotted segment advertises, so a scan opens artifacts by
    /// reference instead of probing for files beside the Parquet.
    pub artifacts: SegmentArtifacts,
    /// The immutable objects each path resolves to, for the segments the scan
    /// selects. Empty for a legacy table.
    pub sources: SegmentObjectMap,
}

/// Aggregate accounting over the live segments, read under one lock.
pub struct SegmentTotals {
    pub count: usize,
    pub rows: i64,
    pub bytes: i64,
    /// Bounds over segments that actually hold rows.
    pub min_seq: Option<i64>,
    pub max_seq: Option<i64>,
}

/// One table's live segments, ordered by `min_seq`.
pub struct SegmentView {
    table: String,
    segments: std::sync::Mutex<VecDeque<LocalSegment>>,
}

impl SegmentView {
    pub fn new(table: &str, segments: VecDeque<LocalSegment>) -> Self {
        debug_assert_unique_paths(&segments);
        Self {
            table: table.to_string(),
            segments: std::sync::Mutex::new(segments),
        }
    }

    /// Replace the whole view, as a definition activation does.
    pub fn replace_all(&self, mut segments: VecDeque<LocalSegment>) {
        segments.make_contiguous().sort_by_key(|s| s.min_seq);
        debug_assert_unique_paths(&segments);
        *self.segments.lock().unwrap() = segments;
    }

    /// Every live segment, cloned, for callers that scan or filter them.
    pub fn segments(&self) -> Vec<LocalSegment> {
        self.segments.lock().unwrap().iter().cloned().collect()
    }

    /// Every live segment as a catalog row.
    pub fn rows(&self) -> Vec<SegmentRow> {
        self.segments
            .lock()
            .unwrap()
            .iter()
            .map(|segment| segment_to_row(&self.table, segment))
            .collect()
    }

    /// The first segment matching `select`, cloned.
    pub fn find(&self, select: impl Fn(&LocalSegment) -> bool) -> Option<LocalSegment> {
        self.segments
            .lock()
            .unwrap()
            .iter()
            .find(|segment| select(segment))
            .cloned()
    }

    /// The first non-`None` result of `select` over the live segments.
    pub fn find_map<T>(&self, select: impl FnMut(&LocalSegment) -> Option<T>) -> Option<T> {
        self.segments.lock().unwrap().iter().find_map(select)
    }

    /// Mutate the live entry for `path`, or return `None` when it is gone.
    ///
    /// Holding the lock across `mutate` is what lets a caller pair a file
    /// operation with the view update, so a segment evicted mid-rewrite is never
    /// resurrected.
    pub fn update<T>(&self, path: &str, mutate: impl FnOnce(&mut LocalSegment) -> T) -> Option<T> {
        let mut segments = self.segments.lock().unwrap();
        segments
            .iter_mut()
            .find(|segment| segment.path == path)
            .map(mutate)
    }

    /// Drop `removed` and add `added`, keeping the set ordered.
    pub fn replace(&self, removed: &[String], added: Vec<LocalSegment>) {
        let retired: std::collections::HashSet<&str> = removed.iter().map(String::as_str).collect();
        let mut segments = self.segments.lock().unwrap();
        segments.retain(|segment| !retired.contains(segment.path.as_str()));
        segments.extend(added);
        segments
            .make_contiguous()
            .sort_by(|left, right| (left.min_seq, &left.path).cmp(&(right.min_seq, &right.path)));
        debug_assert_unique_paths(&segments);
    }

    /// Add sealed segments, keeping the set ordered.
    pub fn extend(&self, added: Vec<LocalSegment>) {
        let mut segments = self.segments.lock().unwrap();
        segments.extend(added);
        segments.make_contiguous().sort_by_key(|s| s.min_seq);
        debug_assert_unique_paths(&segments);
    }

    /// Remove `path` and return the entry that left the view.
    pub fn remove(&self, path: &str) -> Option<LocalSegment> {
        let mut segments = self.segments.lock().unwrap();
        let index = segments.iter().position(|segment| segment.path == path)?;
        segments.remove(index)
    }

    /// Typed Int64 key bounds for one input segment. The catalog round-trip
    /// stringifies them, losing numeric ordering, so compaction reads them here.
    pub fn key_bounds(&self, path: &str) -> (Option<i64>, Option<i64>) {
        self.segments
            .lock()
            .unwrap()
            .iter()
            .find(|segment| segment.path == path)
            .map(|segment| (segment.min_key_value, segment.max_key_value))
            .unwrap_or((None, None))
    }

    /// The local artifacts `path` currently advertises.
    pub fn artifacts(&self, path: &str) -> LocalArtifacts {
        self.segments
            .lock()
            .unwrap()
            .iter()
            .find(|segment| segment.path == path)
            .map(|segment| segment.artifacts.clone())
            .unwrap_or_default()
    }

    pub fn totals(&self) -> SegmentTotals {
        let segments = self.segments.lock().unwrap();
        SegmentTotals {
            count: segments.len(),
            rows: segments.iter().map(|segment| segment.row_count).sum(),
            bytes: segments.iter().map(|segment| segment.size_bytes).sum(),
            min_seq: segments
                .iter()
                .filter(|segment| segment.row_count > 0)
                .map(|segment| segment.min_seq)
                .min(),
            max_seq: segments
                .iter()
                .filter(|segment| segment.row_count > 0)
                .map(|segment| segment.max_seq)
                .max(),
        }
    }

    /// The read view of the files currently on disk.
    pub fn snapshot(&self) -> SegmentSnapshot {
        let segments = self.segments.lock().unwrap();
        SegmentSnapshot {
            paths: segments.iter().map(|s| s.path.clone()).collect(),
            key_bounds: segments
                .iter()
                .filter_map(|segment| {
                    Some((
                        segment.path.clone(),
                        (segment.min_key_value?, segment.max_key_value?),
                    ))
                })
                .collect(),
            seq_bounds: segments
                .iter()
                .map(|segment| (segment.path.clone(), (segment.min_seq, segment.max_seq)))
                .collect(),
            partitions: segments
                .iter()
                .filter_map(|segment| Some((segment.path.clone(), segment.partition.clone()?)))
                .collect(),
            min_seq: segments.iter().map(|segment| segment.min_seq).min(),
            artifacts: segments
                .iter()
                .filter(|segment| !segment.artifacts.is_empty())
                .map(|segment| (segment.path.clone(), segment.artifacts.clone()))
                .collect(),
            sources: SegmentObjectMap::new(),
        }
    }
}

/// Debug-only invariant: no two entries share a path.
///
/// A same-path duplicate is a phantom reference — two entries for one seq range,
/// one of whose file a prior compaction already unlinked (#7361). It surfaces
/// duplicate rows in a query and wedges compaction when the planner picks the
/// dead entry. Compiled out of release builds; a cheap guard that trips tests
/// the instant any mutation reintroduces a duplicate.
pub fn debug_assert_unique_paths(segments: &VecDeque<LocalSegment>) {
    if !cfg!(debug_assertions) {
        return;
    }
    let mut seen = std::collections::HashSet::with_capacity(segments.len());
    for segment in segments {
        debug_assert!(
            seen.insert(segment.path.as_str()),
            "duplicate live-segment path: {}",
            segment.path
        );
    }
}

/// The segments visible at definition version `version`, projected from the
/// catalog rows the table's committed state owns.
///
/// A legacy segment is visible because its file is there; an object-backed
/// segment is visible because the table's state references it. Cache contents
/// never create or remove visibility.
pub fn visible_segments(
    catalog: &Catalog,
    table: &str,
    version: u64,
    store: Option<&dyn ObjectStore>,
) -> Result<VecDeque<LocalSegment>, StatsError> {
    let status = catalog.table_spec_status(table)?;
    let rollback_alias = status.migration.as_ref().and_then(|migration| {
        (status.active_version() == version && migration.from_version == Some(version))
            .then_some(migration.to_version.unwrap_or(0))
    });
    let object_records: HashMap<_, _> = catalog
        .object_segments(table)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let mut segments = VecDeque::new();
    for row in catalog.list_segments(table)? {
        let record = object_records.get(&row.path);
        let visible = match record {
            Some(record) => {
                record.table_spec_version == version
                    || (rollback_alias == Some(record.table_spec_version)
                        && !record.migration_backfill)
            }
            None => version == 0,
        };
        if !visible || (record.is_none() && !Path::new(&row.path).exists()) {
            continue;
        }
        let artifacts = segment_artifacts(store, record, Path::new(&row.path))?;
        segments.push_back(LocalSegment {
            path: row.path,
            size_bytes: row.byte_size,
            level: row.level,
            min_seq: row.min_seq,
            max_seq: row.max_seq,
            row_count: row.row_count,
            created_at_ms: row.created_at_ms,
            min_key_value: row.min_key_value.and_then(|value| value.parse().ok()),
            max_key_value: row.max_key_value.and_then(|value| value.parse().ok()),
            partition: row.partition,
            location: row.location,
            artifacts,
        });
    }
    segments.make_contiguous().sort_by_key(|s| s.min_seq);
    Ok(segments)
}

/// The local files one segment's artifacts resolve to.
///
/// An object-backed segment resolves each path from the object identity its
/// table state names, so an empty reference set means the segment advertises no
/// artifacts. A version-0 segment has no references at all; its sidecars come
/// from the local layout it was written with, and stop being consulted once the
/// table is imported to object storage.
pub fn segment_artifacts(
    store: Option<&dyn ObjectStore>,
    record: Option<&ObjectSegmentRecord>,
    parquet: &Path,
) -> Result<LocalArtifacts, StatsError> {
    let Some(record) = record else {
        return Ok(local_sidecar_artifacts(parquet));
    };
    let store = store.ok_or_else(|| {
        StatsError::Internal(format!(
            "object segment {} has no object store to resolve artifacts",
            parquet.display()
        ))
    })?;
    local_artifacts(store, &record.artifacts)
}
