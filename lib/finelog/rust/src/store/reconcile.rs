//! Boot-time remote reconcile: adopt unknown remote parquet + drop redundant.
//!
//! Runs once at namespace construction (before the maintenance task starts),
//! when a remote dir is configured.
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

use std::collections::{HashMap, HashSet};

use futures::StreamExt;

use crate::errors::StatsError;
use crate::partition_policy::SegmentPartition;
use crate::store::catalog::Catalog;
use crate::store::remote::RemoteStore;
use crate::store::types::{parse_seg_filename, segment_relative_key, SegmentLocation, SegmentRow};

/// Bounded concurrency for the boot reconcile's remote footer reads. High enough
/// to hide cross-region round-trip latency (a sequential await chain costs O(N)
/// RTTs — minutes on a first-ever reconcile of a large archived namespace), low
/// enough to keep the object_store connection pool sane.
const RECONCILE_FOOTER_CONCURRENCY: usize = 64;

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Reconcile the remote bucket for `namespace` against the catalog at boot.
///
/// `local_dir` is the namespace's on-disk directory (adopted REMOTE rows point
/// their `path` at `{local_dir}/{relative key}` so a later download lands the file
/// in place). `key_column` is the namespace's ordering key for footer key-bound
/// recovery.
pub async fn reconcile_remote_segments(
    catalog: &Catalog,
    remote: &RemoteStore,
    namespace: &str,
    local_dir: &std::path::Path,
    key_column: Option<&str>,
) -> Result<(), StatsError> {
    let started = std::time::Instant::now();
    let list_started = std::time::Instant::now();
    let objects = match remote.list_segment_objects(namespace).await {
        Ok(o) => o,
        Err(e) => {
            tracing::warn!(namespace, error = %e, "remote reconcile list failed");
            return Ok(());
        }
    };
    let list_ms = list_started.elapsed().as_millis() as u64;

    // Catalog rows at L>=1 keyed by relative object key (the durable pointers).
    let catalog_rows = catalog.list_segments_min_level(namespace, 1)?;
    let mut catalog_by_key: HashMap<String, SegmentRow> = HashMap::new();
    for row in catalog_rows {
        let Some(key) = segment_relative_key(local_dir, &row.path) else {
            tracing::warn!(
                namespace,
                path = %row.path,
                "remote catalog segment is outside its namespace directory"
            );
            continue;
        };
        catalog_by_key.insert(key, row);
    }

    // A REMOTE row is only a pointer into this object listing. If its object is
    // absent after a successful strongly-consistent list, the row is already
    // unreadable and must not poison later maintenance. This also repairs a
    // layout move that crashed after deleting the old key: the destination is
    // adopted below instead of being discarded as a duplicate of a phantom.
    let object_keys: HashSet<&str> = objects.iter().map(|(key, _)| key.as_str()).collect();
    let missing_remote: Vec<(String, String)> = catalog_by_key
        .iter()
        .filter(|(key, row)| {
            row.location == SegmentLocation::Remote && !object_keys.contains(key.as_str())
        })
        .map(|(key, row)| (key.clone(), row.path.clone()))
        .collect();
    let missing_remote_paths: Vec<String> = missing_remote
        .iter()
        .map(|(_key, path)| path.clone())
        .collect();
    catalog.remove_segments(namespace, &missing_remote_paths)?;
    for (key, _path) in &missing_remote {
        catalog_by_key.remove(key);
    }
    if !missing_remote.is_empty() {
        tracing::warn!(
            namespace,
            segments = missing_remote.len(),
            "removed catalog pointers to missing remote segments"
        );
    }

    // Footer-fetch every remote parquet not already known to the catalog.
    struct Footer {
        relative_key: String,
        level: i32,
        min_seq: i64,
        max_seq: i64,
        row_count: i64,
        byte_size: i64,
        min_key: Option<i64>,
        max_key: Option<i64>,
        partition: Option<SegmentPartition>,
    }
    let pending: Vec<(String, u64)> = objects
        .iter()
        .filter(|(key, _)| !catalog_by_key.contains_key(key))
        .filter_map(|(key, size)| {
            let filename = std::path::Path::new(key).file_name()?.to_str()?;
            parse_seg_filename(filename).map(|_| (key.clone(), *size))
        })
        .collect();
    // Fetch footers CONCURRENTLY: these are latency-bound cross-region round
    // trips, so a sequential await would cost O(N) RTTs. `buffer_unordered`
    // caps in-flight requests; `read_footer` is a single ranged GET (size is
    // already known, no `head`).
    let footer_started = std::time::Instant::now();
    let footers: Vec<Footer> = futures::stream::iter(pending)
        .map(|(name, size)| async move {
            let footer = remote.read_footer(namespace, &name, size, key_column).await;
            (name, size, footer)
        })
        .buffer_unordered(RECONCILE_FOOTER_CONCURRENCY)
        .filter_map(|(name, size, footer)| async move {
            let Some(metadata) = footer else {
                tracing::warn!(namespace, %name, "failed reading remote parquet footer");
                return None;
            };
            Some(Footer {
                relative_key: name,
                level: metadata.level,
                min_seq: metadata.min_seq,
                max_seq: metadata.max_seq,
                row_count: metadata.row_count,
                byte_size: size as i64,
                min_key: metadata.min_key_value,
                max_key: metadata.max_key_value,
                partition: metadata.partition,
            })
        })
        .collect()
        .await;
    let footer_ms = footer_started.elapsed().as_millis() as u64;

    // Union catalog + remote-only seq ranges; mark any segment fully spanned by
    // a strictly-higher level as redundant (transitivity makes a single pass
    // sufficient — Z covers Y, Y covers X => Z covers X).
    let mut all_known: HashMap<String, (i32, i64, i64, Option<SegmentPartition>)> = HashMap::new();
    for (name, row) in &catalog_by_key {
        all_known.insert(
            name.clone(),
            (row.level, row.min_seq, row.max_seq, row.partition.clone()),
        );
    }
    for f in &footers {
        all_known.insert(
            f.relative_key.clone(),
            (f.level, f.min_seq, f.max_seq, f.partition.clone()),
        );
    }
    let mut by_level: HashMap<i32, Vec<(i64, i64, Option<SegmentPartition>)>> = HashMap::new();
    for (level, min_seq, max_seq, partition) in all_known.values() {
        by_level
            .entry(*level)
            .or_default()
            .push((*min_seq, *max_seq, partition.clone()));
    }
    let mut redundant: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, (level, min_seq, max_seq, partition)) in &all_known {
        // A path-only layout migration changes the object key without changing
        // the segment identity represented by its level/seq/partition tuple.
        // If the process crashes after uploading the new key but before deleting
        // the old one, retain the catalog-owned key and discard the duplicate
        // instead of adopting a permanent REMOTE row for it.
        if !catalog_by_key.contains_key(name)
            && catalog_by_key.values().any(|row| {
                row.level == *level
                    && row.min_seq == *min_seq
                    && row.max_seq == *max_seq
                    && row.partition == *partition
            })
        {
            redundant.insert(name.clone());
            continue;
        }
        for (higher_level, ranges) in &by_level {
            if *higher_level <= *level {
                continue;
            }
            if ranges.iter().any(|(h_min, h_max, higher_partition)| {
                higher_partition == partition && *h_min <= *min_seq && *h_max >= *max_seq
            }) {
                redundant.insert(name.clone());
                break;
            }
        }
    }

    // Drop redundant catalog rows + delete their bucket files.
    let catalog_update_started = std::time::Instant::now();
    let removed_paths: Vec<String> = redundant
        .iter()
        .filter_map(|name| catalog_by_key.get(name).map(|row| row.path.clone()))
        .collect();
    catalog.remove_segments(namespace, &removed_paths)?;
    let catalog_remove_ms = catalog_update_started.elapsed().as_millis() as u64;

    let remote_delete_started = std::time::Instant::now();
    let mut dropped = 0;
    for name in &redundant {
        remote.delete(namespace, name).await;
        dropped += 1;
    }
    let remote_delete_ms = remote_delete_started.elapsed().as_millis() as u64;

    // Adopt the surviving (non-redundant) remote-only footers as REMOTE rows.
    let now = now_ms();
    let mut adopted_rows = Vec::new();
    for f in &footers {
        if redundant.contains(&f.relative_key) {
            continue;
        }
        let local_path = local_dir.join(&f.relative_key);
        // Record the footer's actual num_rows, not a seq-span recomputation, so
        // an edge-case empty footer adopts row_count=0 rather than 1.
        adopted_rows.push(SegmentRow {
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
            partition: f.partition.clone(),
            location: SegmentLocation::Remote,
        });
    }
    let catalog_upsert_started = std::time::Instant::now();
    catalog.upsert_segments(&adopted_rows)?;
    let catalog_upsert_ms = catalog_upsert_started.elapsed().as_millis() as u64;
    let adopted = adopted_rows.len();

    tracing::info!(
        namespace,
        remote_objects = objects.len(),
        catalog_segments = catalog_by_key.len(),
        removed_missing = missing_remote.len(),
        footer_reads = footers.len(),
        adopted,
        dropped_redundant = dropped,
        list_ms,
        footer_ms,
        catalog_remove_ms,
        remote_delete_ms,
        catalog_upsert_ms,
        total_ms = started.elapsed().as_millis() as u64,
        "finelog remote reconcile complete"
    );
    Ok(())
}
