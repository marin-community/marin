//! Bounded recovery of missing object-segment key and column bounds.
//!
//! Segment objects are immutable, so a table can recover catalog pruning
//! metadata from cached or remote Parquet footers without downloading the object.
//! This backfills the key bounds of segments that predate them, and the bounds
//! of a `catalog_bounds` column for segments written before the table flagged
//! it; most of those segments are at the final level and are never rewritten.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::errors::StatsError;
use crate::query::file_scan::open_cached_parquet;
use crate::store::catalog::RecoveredBounds;
use crate::store::column_bounds;
use crate::store::object_store::ObjectReference;
use crate::store::segment::segment_metadata_from_parquet;
use crate::store::table::runtime::TableRuntime;
use crate::store::table_state::CommitError;

pub const SEGMENT_BOUNDS_PER_TICK: usize = 64;

pub async fn maintain(runtime: &TableRuntime) -> Result<bool, StatsError> {
    let lifecycle = runtime.catalog.spec_lifecycle(runtime.name())?;
    let active_version = lifecycle.active_version();
    let bounds_columns = runtime.format.bounds_columns();
    let records: HashMap<_, _> = runtime
        .catalog
        .object_segments(runtime.name())?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let candidates: Vec<_> = runtime
        .catalog
        .list_segments(runtime.name())?
        .into_iter()
        .filter(|row| {
            row.row_count > 0
                && records.get(&row.path).is_some_and(|record| {
                    record.table_spec_version == active_version
                        && (row.min_key_value.is_none()
                            || row.max_key_value.is_none()
                            || column_bounds::missing(&record.column_bounds, bounds_columns))
                })
        })
        .take(SEGMENT_BOUNDS_PER_TICK)
        .collect();
    if candidates.is_empty() {
        return Ok(false);
    }

    let store = runtime.controller.object_store().ok_or_else(|| {
        StatsError::Internal(format!(
            "object-backed table {:?} has no object store",
            runtime.name()
        ))
    })?;
    let mut recovered = Vec::new();
    let mut unresolved = false;
    for row in &candidates {
        let record = records
            .get(&row.path)
            .expect("candidate object record was selected above");
        let reference = ObjectReference::try_from(&record.source)?;
        let input = open_cached_parquet(Arc::clone(store), &reference).await?;
        let key = if row.min_key_value.is_some() && row.max_key_value.is_some() {
            None
        } else {
            let recovered_key = segment_metadata_from_parquet(
                input.metadata(),
                row.level,
                row.min_seq,
                Some(runtime.format.key_column()),
            )
            .and_then(|metadata| metadata.min_key_value.zip(metadata.max_key_value));
            if recovered_key.is_none() {
                tracing::warn!(
                    table = runtime.name(),
                    segment = %row.path,
                    "cannot recover object segment key bounds from footer"
                );
                unresolved = true;
            }
            recovered_key
        };
        let missing_columns: Vec<String> = bounds_columns
            .iter()
            .filter(|column| !record.column_bounds.contains_key(*column))
            .cloned()
            .collect();
        let columns = column_bounds::from_parquet(input.metadata(), &missing_columns);
        if columns.len() < missing_columns.len() {
            tracing::warn!(
                table = runtime.name(),
                segment = %row.path,
                "object segment footer lacks statistics for a catalog-bounds column"
            );
            unresolved = true;
        }
        if key.is_none() && columns.is_empty() {
            continue;
        }
        recovered.push(RecoveredBounds {
            path: row.path.clone(),
            key,
            columns,
        });
    }
    if recovered.is_empty() {
        return Ok(true);
    }

    let lease = runtime.controller.begin_compaction_for(&lifecycle)?;
    let paths: HashSet<_> = recovered
        .iter()
        .map(|segment| segment.path.clone())
        .collect();
    let committed = runtime.controller.commit_maintenance(&lease, || {
        let current: HashMap<_, _> = runtime
            .catalog
            .object_segments(runtime.name())?
            .into_iter()
            .map(|record| (record.path.clone(), record.table_spec_version))
            .collect();
        if let Some(path) = paths
            .iter()
            .find(|path| current.get(*path).copied() != Some(active_version))
        {
            return Err(StatsError::SchemaConflict(format!(
                "segment-bounds repair input {path:?} is no longer live"
            )));
        }
        let revision = runtime
            .catalog
            .update_object_segment_bounds(runtime.name(), &recovered)?;
        Ok((revision, ()))
    });
    match committed {
        Ok(_) => {}
        Err(CommitError::NotCommitted(StatsError::SchemaConflict(error))) => {
            tracing::info!(table = runtime.name(), %error, "segment-bounds repair will replan");
            return Ok(true);
        }
        Err(error) => return Err(error.into()),
    }
    for segment in &recovered {
        runtime.segments.update(&segment.path, |local| {
            if let Some((minimum, maximum)) = &segment.key {
                local.min_key_value = Some(minimum.clone());
                local.max_key_value = Some(maximum.clone());
            }
            local.column_bounds.extend(segment.columns.clone());
        });
    }
    tracing::info!(
        table = runtime.name(),
        segments = recovered.len(),
        "recovered object segment key and column bounds"
    );
    Ok(unresolved || candidates.len() == SEGMENT_BOUNDS_PER_TICK)
}
