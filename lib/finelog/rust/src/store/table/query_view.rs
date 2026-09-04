//! Planning a read from one pinned table state.
//!
//! A server read pins an `Arc<TableSnapshot>` per referenced table and plans
//! entirely from the metadata that state carries: the segments live under the
//! active definition version, their sequence and key bounds, their physical
//! partitions, and exact references to their source Parquet and derived
//! artifacts. Nothing here touches the object store or the local cache, so
//! planning costs the same on a cold cache as on a warm one.
//!
//! Localization happens later and only for the objects the plan selected. The
//! local filename of an object is derived from the object's own identity, so a
//! plan can name the file a scan will open before that file exists.

use std::collections::BTreeMap;

use crate::errors::StatsError;
use crate::partition_policy::SegmentPartition;
use crate::proto::finelog::stats::CatalogSegment;
use crate::store::catalog::{ObjectSegmentRecord, SpecLifecycle};
use crate::store::object_store::{ObjectReference, ObjectStore};
use crate::store::table_state::{ArtifactReferences, LocalArtifacts, TableSnapshot};

/// The immutable objects one planned segment resolves to.
#[derive(Clone, Debug)]
pub struct SegmentObjects {
    pub source: ObjectReference,
    pub artifacts: ArtifactReferences,
}

/// Every object a planned segment set may need, keyed by the local file the
/// segment resolves to.
pub type SegmentObjectMap = BTreeMap<String, SegmentObjects>;

/// One query-visible segment of a pinned table state.
pub struct PlannedSegment {
    /// The local file this segment's source object resolves to, derived from the
    /// object's identity rather than from the local directory.
    pub path: String,
    pub min_seq: i64,
    pub max_seq: i64,
    /// Encoded key bounds, when the segment's key column carries them.
    pub key_bounds: Option<(String, String)>,
    pub partition: Option<SegmentPartition>,
    /// Local files the segment's advertised artifacts resolve to.
    pub artifacts: LocalArtifacts,
    pub objects: SegmentObjects,
}

/// The segments a query reads from `snapshot`, ordered by sequence.
///
/// Visibility is the live set of the state's active definition version. The
/// published state already folds a migration's aliasing into that set, so a
/// reader never re-derives which versions alias which.
pub fn plan_visible_segments(
    snapshot: &TableSnapshot,
    store: &dyn ObjectStore,
) -> Result<Vec<PlannedSegment>, StatsError> {
    let catalog = snapshot.state().catalog();
    let active = catalog.active_table_spec_version.unwrap_or(0);
    let mut planned = Vec::new();
    for version in &catalog.version_segments {
        if version.table_spec_version.unwrap_or(0) != active {
            continue;
        }
        for segment in &version.live_segments {
            planned.push(plan_segment(segment, store)?);
        }
    }
    planned.sort_by_key(|segment| segment.min_seq);
    Ok(planned)
}

fn plan_segment(
    segment: &CatalogSegment,
    store: &dyn ObjectStore,
) -> Result<PlannedSegment, StatsError> {
    let source = segment.source.as_option().ok_or_else(|| {
        StatsError::Internal(format!(
            "published segment {:?} carries no source object",
            segment.segment_id
        ))
    })?;
    let reference = ObjectReference::try_from(source)?;
    let path = store.planned_local_path(&reference.id)?;
    let artifacts = ArtifactReferences::from_catalog_segment(segment);
    let partition = segment
        .partition_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|error| {
            StatsError::Internal(format!(
                "decode partition metadata for segment {:?}: {error}",
                segment.segment_id
            ))
        })?;
    Ok(PlannedSegment {
        path: path.to_string_lossy().into_owned(),
        min_seq: segment.min_seq.unwrap_or(0),
        // A missing bound must never prune: i64::MAX makes every range overlap.
        max_seq: segment.max_seq.unwrap_or(i64::MAX),
        key_bounds: key_bounds(segment),
        partition,
        artifacts: crate::store::table::segment_view::local_artifacts(store, &artifacts)?,
        objects: SegmentObjects {
            source: reference,
            artifacts,
        },
    })
}

fn key_bounds(segment: &CatalogSegment) -> Option<(String, String)> {
    Some((
        segment.min_key_value.clone()?,
        segment.max_key_value.clone()?,
    ))
}

/// Whether a committed object segment belongs to the version a query reads.
///
/// A segment is visible under the active version, under a desired version whose
/// rows were written after the migration fence, and under an in-flight migration
/// aliasing the source version onto the target.
pub fn object_segment_is_query_visible(
    status: &SpecLifecycle,
    record: &ObjectSegmentRecord,
) -> bool {
    record.table_spec_version == status.active_version()
        || (status.desired_version() == record.table_spec_version && !record.migration_backfill)
        || (status.migration.as_ref().is_some_and(|migration| {
            migration.from_version == Some(status.active_version())
                && migration.to_version == Some(record.table_spec_version)
                && !record.migration_backfill
        }))
}
