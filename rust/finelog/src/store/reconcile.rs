//! Boot-time remote reconcile: adopt unknown remote parquet + drop redundant.
//!
//! Port of `DiskLogNamespace._reconcile_remote_segments`. Runs once at namespace
//! construction (before the maintenance task starts), when a remote dir is
//! configured.
//!
//! Two jobs:
//! 1. **Adoption** (wiped-catalog recovery): the bucket is the only durable
//!    record of L>=1 segments after the local catalog is lost. Each unknown
//!    remote parquet's footer is fetched to rebuild the catalog row as REMOTE
//!    (not added to the deque — queries don't see archived data).
//! 2. **Redundancy drop**: any segment whose `[min_seq, max_seq]` is fully
//!    covered by a strictly-higher level is dropped from both the catalog and
//!    the bucket. Otherwise a crash between a compaction commit and its remote
//!    delete leaves the input file in the bucket, and adoption would give it a
//!    permanent REMOTE row.

use std::collections::HashMap;

use crate::errors::StatsError;
use crate::store::catalog::Catalog;
use crate::store::remote::RemoteStore;
use crate::store::types::{parse_seg_filename, SegmentLocation, SegmentRow};

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Reconcile the remote bucket for `namespace` against the catalog at boot.
///
/// `local_dir` is the namespace's on-disk directory (adopted REMOTE rows point
/// their `path` at `{local_dir}/{basename}`, matching the Python convention so a
/// later download lands the file in place). `key_column` is the namespace's
/// ordering key for footer key-bound recovery.
pub async fn reconcile_remote_segments(
    catalog: &Catalog,
    remote: &RemoteStore,
    namespace: &str,
    local_dir: &std::path::Path,
    key_column: Option<&str>,
) -> Result<(), StatsError> {
    let objects = match remote.list_segment_objects(namespace).await {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!(namespace, error = %e, "remote reconcile list failed");
            return Ok(());
        }
    };

    // Catalog rows at L>=1 keyed by basename (the durable-archive pointers).
    let catalog_rows = catalog.list_segments_min_level(namespace, 1)?;
    let catalog_by_basename: HashMap<String, SegmentRow> = catalog_rows
        .into_iter()
        .map(|r| (basename(&r.path), r))
        .collect();

    // Footer-fetch every remote parquet not already known to the catalog.
    struct Footer {
        basename: String,
        level: i32,
        min_seq: i64,
        max_seq: i64,
        row_count: i64,
        byte_size: i64,
        min_key: Option<i64>,
        max_key: Option<i64>,
    }
    let mut footers: Vec<Footer> = Vec::new();
    for (name, size) in &objects {
        if catalog_by_basename.contains_key(name) {
            continue;
        }
        let Some((level, min_seq)) = parse_seg_filename(name) else {
            continue;
        };
        let Some((row_count, min_key, max_key)) =
            remote.read_footer(namespace, name, key_column).await
        else {
            tracing::warn!(namespace, %name, "failed reading remote parquet footer");
            continue;
        };
        let max_seq = min_seq + (row_count - 1).max(0);
        footers.push(Footer {
            basename: name.clone(),
            level,
            min_seq,
            max_seq,
            row_count,
            byte_size: *size as i64,
            min_key,
            max_key,
        });
    }

    // Union catalog + remote-only seq ranges; mark any segment fully spanned by
    // a strictly-higher level as redundant (transitivity makes a single pass
    // sufficient — Z covers Y, Y covers X => Z covers X).
    let mut all_known: HashMap<String, (i32, i64, i64)> = HashMap::new();
    for (name, row) in &catalog_by_basename {
        all_known.insert(name.clone(), (row.level, row.min_seq, row.max_seq));
    }
    for f in &footers {
        all_known.insert(f.basename.clone(), (f.level, f.min_seq, f.max_seq));
    }
    let mut by_level: HashMap<i32, Vec<(i64, i64)>> = HashMap::new();
    for (level, min_seq, max_seq) in all_known.values() {
        by_level
            .entry(*level)
            .or_default()
            .push((*min_seq, *max_seq));
    }
    let mut redundant: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, (level, min_seq, max_seq)) in &all_known {
        for (higher_level, ranges) in &by_level {
            if *higher_level <= *level {
                continue;
            }
            if ranges
                .iter()
                .any(|(h_min, h_max)| *h_min <= *min_seq && *h_max >= *max_seq)
            {
                redundant.insert(name.clone());
                break;
            }
        }
    }

    // Drop redundant catalog rows + delete their bucket files.
    let mut dropped = 0;
    for name in &redundant {
        if let Some(row) = catalog_by_basename.get(name) {
            catalog.remove_segment(namespace, &row.path)?;
        }
        remote.delete(namespace, name).await;
        dropped += 1;
    }

    // Adopt the surviving (non-redundant) remote-only footers as REMOTE rows.
    let now = now_ms();
    let mut adopted = 0;
    for f in &footers {
        if redundant.contains(&f.basename) {
            continue;
        }
        let local_path = local_dir.join(&f.basename);
        // Record the footer's actual num_rows (matching Python `_reconcile_remote_
        // segments`), not a seq-span recomputation, so an edge-case empty footer
        // adopts row_count=0 rather than 1.
        catalog.upsert_segment(&SegmentRow {
            namespace: namespace.to_string(),
            path: local_path.to_string_lossy().into_owned(),
            level: f.level,
            min_seq: f.min_seq,
            max_seq: f.max_seq,
            row_count: f.row_count,
            byte_size: f.byte_size,
            created_at_ms: now,
            min_key_value: f.min_key.map(|v| v.to_string()),
            max_key_value: f.max_key.map(|v| v.to_string()),
            location: SegmentLocation::Remote,
        })?;
        adopted += 1;
    }

    if adopted > 0 || dropped > 0 {
        tracing::info!(
            namespace,
            adopted,
            dropped_redundant = dropped,
            "reconciled remote"
        );
    }
    Ok(())
}

fn basename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}
