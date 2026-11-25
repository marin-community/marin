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
use crate::store::catalog::{ObjectSegmentRecord, RecoveredBounds};
use crate::store::column_bounds;
use crate::store::object_store::ObjectReference;
use crate::store::segment::segment_metadata_from_parquet;
use crate::store::table::runtime::TableRuntime;
use crate::store::table_state::CommitError;
use crate::store::types::SegmentRow;

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
    let candidates = bounds_candidates(
        runtime.catalog.list_segments(runtime.name())?,
        &records,
        bounds_columns,
        active_version,
    );
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
        let missing_columns = column_bounds::missing_columns(&record.column_bounds, bounds_columns);
        // Every missing column gets an entry, if only `ColumnRange::Unknown`, so a
        // footer without statistics cannot hold a backfill slot forever.
        let columns = column_bounds::from_parquet(input.metadata(), &missing_columns);
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

/// The segments one tick repairs, at most `SEGMENT_BOUNDS_PER_TICK`.
///
/// L0 segments are skipped: a merge soon rewrites them and records bounds for
/// its output. The highest levels go first: the final level is never rewritten,
/// so its segments get bounds only from this backfill. Segments that still owe
/// column bounds go before those that owe only key bounds. A key whose footer
/// has no statistics stays a candidate on every tick, and must not take the
/// slots column repair needs.
fn bounds_candidates(
    mut segments: Vec<SegmentRow>,
    records: &HashMap<String, ObjectSegmentRecord>,
    bounds_columns: &[String],
    active_version: u64,
) -> Vec<SegmentRow> {
    let owes_columns = |record: &ObjectSegmentRecord| {
        !column_bounds::missing_columns(&record.column_bounds, bounds_columns).is_empty()
    };
    segments.retain(|row| {
        row.level > 0
            && row.row_count > 0
            && records.get(&row.path).is_some_and(|record| {
                record.table_spec_version == active_version
                    && (row.min_key_value.is_none()
                        || row.max_key_value.is_none()
                        || owes_columns(record))
            })
    });
    segments.sort_by_key(|row| {
        (
            !owes_columns(&records[&row.path]),
            std::cmp::Reverse(row.level),
        )
    });
    segments.truncate(SEGMENT_BOUNDS_PER_TICK);
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::column_bounds::{ColumnBounds, ColumnRange};
    use crate::store::types::SegmentLocation;

    fn row(path: &str, level: i32, key_bounds: bool) -> SegmentRow {
        let key = key_bounds.then(|| "k".to_string());
        SegmentRow {
            namespace: "iris.worker".to_string(),
            path: path.to_string(),
            level,
            min_seq: 1,
            max_seq: 1,
            row_count: 1,
            byte_size: 1,
            created_at_ms: 1,
            min_key_value: key.clone(),
            max_key_value: key,
            partition: None,
            location: SegmentLocation::Remote,
        }
    }

    fn record(path: &str, column_bounds: ColumnBounds) -> ObjectSegmentRecord {
        ObjectSegmentRecord {
            path: path.to_string(),
            table_spec_version: 1,
            source: Default::default(),
            artifacts: Default::default(),
            column_bounds,
            migration_backfill: false,
            migration_source_id: None,
            migration_source_rows: None,
        }
    }

    #[test]
    fn keys_without_statistics_do_not_starve_column_repair() {
        // More stuck keys than one tick's slots, at a higher level than the
        // segment that still owes column bounds.
        let columns = vec!["epoch_ms".to_string()];
        let recorded = ColumnBounds::from([("epoch_ms".to_string(), ColumnRange::new(1, 2))]);
        let mut segments = Vec::new();
        let mut records = HashMap::new();
        for index in 0..SEGMENT_BOUNDS_PER_TICK + 6 {
            let path = format!("stuck-key-{index}");
            segments.push(row(&path, 2, false));
            records.insert(path.clone(), record(&path, recorded.clone()));
        }
        segments.push(row("owes-columns", 1, true));
        records.insert(
            "owes-columns".to_string(),
            record("owes-columns", ColumnBounds::new()),
        );

        let candidates = bounds_candidates(segments, &records, &columns, 1);
        assert_eq!(candidates.len(), SEGMENT_BOUNDS_PER_TICK);
        assert_eq!(candidates[0].path, "owes-columns");
    }
}
