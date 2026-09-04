//! Bounded recovery of missing object-segment key bounds.
//!
//! Segment objects are immutable, so a table can recover catalog pruning
//! metadata from Parquet footers without rewriting their rows. Remote-only
//! objects are warmed first and revisited on a later maintenance tick; one
//! repair never turns into an unbounded synchronous table download.

use std::collections::{HashMap, HashSet};

use crate::errors::StatsError;
use crate::store::object_store::ObjectReference;
use crate::store::segment::read_segment_footer_at;
use crate::store::table::runtime::TableRuntime;
use crate::store::table_state::CommitError;

pub const KEY_BOUNDS_PER_TICK: usize = 64;

pub async fn maintain(runtime: &TableRuntime) -> Result<bool, StatsError> {
    let lifecycle = runtime.catalog.spec_lifecycle(runtime.name())?;
    let active_version = lifecycle.active_version();
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
                && (row.min_key_value.is_none() || row.max_key_value.is_none())
                && records
                    .get(&row.path)
                    .is_some_and(|record| record.table_spec_version == active_version)
        })
        .take(KEY_BOUNDS_PER_TICK)
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
    for row in &candidates {
        let record = records
            .get(&row.path)
            .expect("candidate object record was selected above");
        let reference = ObjectReference::try_from(&record.source)?;
        let path = match store.cached_path(&reference).await? {
            Some(path) => path,
            None if store.remote_scan_url(&reference.id).is_some() => {
                store.warm(&reference);
                continue;
            }
            None => store.local_path(&reference).await?,
        };
        let Some(metadata) = read_segment_footer_at(
            &path,
            row.level,
            row.min_seq,
            Some(runtime.format.key_column()),
        ) else {
            tracing::warn!(
                table = runtime.name(),
                segment = %row.path,
                "cannot recover object segment key bounds from footer"
            );
            continue;
        };
        let (Some(minimum), Some(maximum)) = (metadata.min_key_value, metadata.max_key_value)
        else {
            continue;
        };
        recovered.push((row.path.clone(), minimum, maximum));
    }
    let unresolved = recovered.len() < candidates.len();
    if recovered.is_empty() {
        return Ok(true);
    }

    let lease = runtime.controller.begin_compaction_for(&lifecycle)?;
    let paths: HashSet<_> = recovered.iter().map(|(path, _, _)| path.clone()).collect();
    let committed = runtime
        .controller
        .commit_maintenance(&lease, || {
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
                    "key-bounds repair input {path:?} is no longer live"
                )));
            }
            let revision = runtime
                .catalog
                .update_object_segment_key_bounds(runtime.name(), &recovered)?;
            Ok((revision, ()))
        })
        .await;
    match committed {
        Ok(_) => {}
        Err(CommitError::NotCommitted(StatsError::SchemaConflict(error))) => {
            tracing::info!(table = runtime.name(), %error, "key-bounds repair will replan");
            return Ok(true);
        }
        Err(error) if error.is_committed() => {
            tracing::warn!(table = runtime.name(), %error, "key-bounds repair awaits publication");
        }
        Err(error) => return Err(error.into()),
    }
    for (path, minimum, maximum) in &recovered {
        runtime.segments.update(path, |segment| {
            segment.min_key_value = Some(minimum.clone());
            segment.max_key_value = Some(maximum.clone());
        });
    }
    tracing::info!(
        table = runtime.name(),
        segments = recovered.len(),
        "recovered object segment key bounds"
    );
    Ok(unresolved || candidates.len() == KEY_BOUNDS_PER_TICK)
}
