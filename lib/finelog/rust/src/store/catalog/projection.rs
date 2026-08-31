//! Projection from transactional catalog rows to the published table state.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use buffa::MessageField;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    CatalogSegment, MigrationPhase, NamespaceCatalog, ObjectRef, ProjectionArtifact,
    TableVersionSegments,
};
use crate::store::catalog::Catalog;
use crate::store::state_store::object::TABLE_STATE_FORMAT_VERSION;
use crate::store::table_state::ArtifactReferences;
use crate::store::types::{basename, segment_relative_key, SegmentLocation};

/// Build one immutable, self-contained query and recovery catalog value.
pub fn namespace_catalog(
    catalog: &Catalog,
    namespace: &str,
    namespace_dir: &Path,
) -> Result<NamespaceCatalog, StatsError> {
    let status = catalog.spec_lifecycle(namespace)?;
    if status.catalog_generation == 0 {
        return Err(StatsError::SchemaValidation(format!(
            "namespace {namespace:?} has no versioned table specification"
        )));
    }
    let object_segments: HashMap<_, _> = catalog
        .object_segments(namespace)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let mut by_version: BTreeMap<u64, Vec<CatalogSegment>> = BTreeMap::new();
    for row in catalog.list_segments(namespace)? {
        let (version, source) = match object_segments.get(&row.path) {
            Some(record) => (record.table_spec_version, record.source.clone()),
            None if row.location != SegmentLocation::Local => {
                let Some(relative_key) = segment_relative_key(namespace_dir, &row.path) else {
                    continue;
                };
                (
                    0,
                    ObjectRef {
                        object_id: Some(relative_key),
                        byte_size: u64::try_from(row.byte_size).ok(),
                        ..Default::default()
                    },
                )
            }
            None => continue,
        };
        let segment = CatalogSegment {
            segment_id: Some(basename(&row.path)),
            source: MessageField::some(source),
            level: Some(row.level),
            min_seq: Some(row.min_seq),
            max_seq: Some(row.max_seq),
            row_count: Some(row.row_count),
            created_at_ms: Some(row.created_at_ms),
            min_key_value: row.min_key_value,
            max_key_value: row.max_key_value,
            partition_json: row
                .partition
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|error| {
                    StatsError::Internal(format!(
                        "encode partition metadata for namespace {namespace:?}: {error}"
                    ))
                })?,
            table_spec_version: Some(version),
            migration_source_id: object_segments
                .get(&row.path)
                .and_then(|record| record.migration_source_id.clone()),
            migration_source_rows: object_segments
                .get(&row.path)
                .and_then(|record| record.migration_source_rows),
            migration_backfill: object_segments
                .get(&row.path)
                .map(|record| record.migration_backfill),
            ..artifact_fields(
                object_segments
                    .get(&row.path)
                    .map(|record| &record.artifacts),
            )
        };
        by_version.entry(version).or_default().push(segment.clone());
        let rollback_write_version = if status.desired_version() > 0 {
            Some(status.active_version())
        } else if status.phase == MigrationPhase::MIGRATION_PHASE_OBSERVING
            || (status.phase == MigrationPhase::MIGRATION_PHASE_RETIRED
                && status.migration.as_ref().is_some_and(|migration| {
                    migration.from_version == Some(status.active_version())
                }))
        {
            status
                .migration
                .as_ref()
                .and_then(|migration| migration.from_version)
        } else {
            None
        };
        if let Some(rollback_write_version) = rollback_write_version {
            if rollback_write_version != version
                && object_segments
                    .get(&row.path)
                    .is_some_and(|record| !record.migration_backfill)
            {
                by_version
                    .entry(rollback_write_version)
                    .or_default()
                    .push(segment);
            }
        }
    }
    by_version.entry(status.active_version()).or_default();
    if status.desired_version() > 0 {
        by_version.entry(status.desired_version()).or_default();
    }
    let stats = catalog.aggregate_namespace_stats(namespace)?;
    let active_segments = by_version
        .get(&status.active_version())
        .cloned()
        .unwrap_or_default();
    let direct_query_high_water = active_segments
        .iter()
        .filter(|segment| segment.level.unwrap_or(0) == 0)
        .filter_map(|segment| segment.min_seq)
        .min()
        .map(|first_unstable_seq| first_unstable_seq.saturating_sub(1))
        .unwrap_or(stats.max_seq);
    let direct_query_segments = active_segments
        .into_iter()
        .filter(|segment| {
            segment.level.unwrap_or(0) >= 1
                && segment.max_seq.unwrap_or(i64::MAX) <= direct_query_high_water
        })
        .collect();
    let version_segments: Vec<TableVersionSegments> = by_version
        .into_iter()
        .map(|(version, live_segments)| TableVersionSegments {
            table_spec_version: Some(version),
            live_segments,
            ..Default::default()
        })
        .collect();
    let desired_version = status.desired_version();
    let effective_spec = status.active.as_ref().or(status.desired.as_ref());
    let max_query_time_ms = effective_spec
        .map(crate::store::table_spec::max_query_time_ms)
        .unwrap_or(crate::store::table_spec::DEFAULT_MAX_QUERY_TIME_MS);
    let rollback_window_ms = effective_spec
        .map(crate::store::table_spec::rollback_window_ms)
        .unwrap_or(crate::store::table_spec::DEFAULT_ROLLBACK_WINDOW_MS);
    Ok(NamespaceCatalog {
        format_version: Some(TABLE_STATE_FORMAT_VERSION),
        namespace: Some(namespace.to_string()),
        catalog_generation: Some(status.catalog_generation),
        active_table_spec_version: Some(status.active_version()),
        desired_table_spec_version: Some(desired_version),
        retained_table_specs: catalog.retained_table_specs(namespace)?,
        persisted_high_water: Some(stats.max_seq),
        version_segments,
        migration: status.migration.into(),
        max_query_time_ms: Some(max_query_time_ms),
        rollback_window_ms: Some(rollback_window_ms),
        forward_cursors: catalog.forward_cursors(namespace)?,
        direct_query_segments,
        direct_query_high_water: Some(direct_query_high_water),
        ..Default::default()
    })
}

/// The artifact-reference fields of a published segment.
///
/// Membership is by reference: a reader opens exactly the bundle and projection
/// objects named here, and validates them against the recorded Parquet footer
/// UUID. A segment with no artifacts publishes none, and its scan reads the
/// source Parquet.
fn artifact_fields(artifacts: Option<&ArtifactReferences>) -> CatalogSegment {
    let Some(artifacts) = artifacts else {
        return CatalogSegment::default();
    };
    CatalogSegment {
        index_bundle: artifacts.bundle.clone().into(),
        projections: artifacts
            .projections
            .iter()
            .map(|(name, object)| ProjectionArtifact {
                name: Some(name.clone()),
                object: MessageField::some(object.clone()),
                ..Default::default()
            })
            .collect(),
        source_segment_uuid: artifacts.binding.segment_uuid.clone(),
        ..Default::default()
    }
}
