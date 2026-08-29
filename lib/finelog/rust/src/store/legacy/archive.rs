//! The legacy remote archive: upload, orphan cleanup, and cache eviction.
//!
//! A legacy table keeps its rows in local files and copies each compacted
//! segment to a flat remote key. Eviction then unlinks the local copy of a
//! segment that has a durable remote one, so the disk holds a working set rather
//! than the whole history.
//!
//! This whole path retires with legacy tables. An object-backed table writes
//! immutable objects on the flush path, so it has no upload step, and its cache
//! is managed by the object store rather than by the table.

use std::collections::HashSet;
use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::RwLock;

use crate::errors::StatsError;
use crate::indices::remove_index_artifacts;
use crate::store::catalog::Catalog;
use crate::store::compaction::config::CompactionConfig;
use crate::store::object_store::{ObjectId, ObjectPrefix, ObjectStore};
use crate::store::policy::StoragePolicy;
use crate::store::table::segment_view::SegmentView;
use crate::store::types::{segment_relative_key, SegmentLocation};

/// Everything the archive path reads and writes through.
pub struct LegacyArchive<'a> {
    pub table: &'a str,
    pub table_dir: &'a std::path::Path,
    pub catalog: &'a Catalog,
    pub segments: &'a SegmentView,
    pub query_visibility: &'a Arc<RwLock<()>>,
    pub remote: &'a Arc<dyn ObjectStore>,
    /// An object-backed table keeps its archive: the legacy objects are outside
    /// the object catalog's MVCC lifetime, so orphan deletion must not run while
    /// its version-0 history is still being imported.
    pub retain_orphans: bool,
}

/// Two-phase remote sync.
///
/// Phase 1: upload every L>=1 `LOCAL` catalog row (or adopt a row whose file is
/// already remote — crash recovery), flipping it to `BOTH`. If any upload fails,
/// `all_durable` is false.
///
/// Phase 2 (orphan delete): runs ONLY if `all_durable`. Delete remote files whose
/// relative key has no catalog row — those are compaction inputs whose row was
/// dropped at commit. The ordering is the data-safety invariant: by the time
/// phase 2 runs, the merged output subsuming those inputs is durable in the
/// bucket (uploaded in phase 1), so the durable copy is in place before any input
/// remote bytes are deleted. Skipping phase 2 on a failed upload means the only
/// remaining copies of an unmerged seq range (the inputs in the bucket) are
/// preserved.
pub async fn sync(archive: LegacyArchive<'_>) -> Result<(), StatsError> {
    let table = archive.table;
    // A TableSpec migration temporarily keeps legacy segments and object-backed
    // cache entries in the same SQLite `segments` table. Legacy sync must
    // continue making the former durable while backfill is incomplete, but
    // object objects already live under the canonical `_finelog` prefix and must
    // never be interpreted as legacy upload candidates or orphans.
    let object_paths: HashSet<String> = archive
        .catalog
        .object_segments(table)?
        .into_iter()
        .map(|record| record.path)
        .collect();
    let remote_keys: HashSet<String> =
        match archive.remote.list(&ObjectPrefix::table(table, "")?).await {
            Ok(objects) => objects
                .into_iter()
                .filter_map(|object| object.id.table_relative(table).map(str::to_string))
                .collect(),
            Err(error) => {
                tracing::warn!(namespace = %table, %error, "remote sync list failed");
                return Ok(());
            }
        };

    let rows = archive.catalog.list_segments_min_level(table, 1)?;
    let mut all_durable = true;
    for row in &rows {
        if row.location != SegmentLocation::Local || object_paths.contains(&row.path) {
            continue;
        }
        let Some(key) = segment_relative_key(archive.table_dir, &row.path) else {
            tracing::warn!(namespace = %table, path = %row.path, "catalog segment is outside its table directory");
            all_durable = false;
            continue;
        };
        if remote_keys.contains(&key) {
            // Uploaded but the catalog never flipped — adopt, no re-upload.
            mark_uploaded(&archive, &row.path)?;
            continue;
        }
        let bytes = match tokio::fs::read(&row.path).await {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::warn!(namespace = %table, path = %row.path, %error, "legacy upload read failed");
                all_durable = false;
                continue;
            }
        };
        if let Err(error) = archive
            .remote
            .write(&ObjectId::table(table, &key)?, Bytes::from(bytes))
            .await
        {
            tracing::warn!(namespace = %table, key = %key, %error, "legacy upload failed");
            all_durable = false;
            continue;
        }
        mark_uploaded(&archive, &row.path)?;
    }

    if !all_durable || archive.retain_orphans {
        return Ok(());
    }

    // Re-snapshot the L>=1 catalog rows (phase 1 may have added keys) and delete
    // only genuine orphans. min_level=1 is equivalent to scanning all levels
    // here because remote files are exclusively L>=1 (L0 is never uploaded), so
    // an L0 key can never appear in the remote set.
    let catalog_keys: HashSet<String> = archive
        .catalog
        .list_segments_min_level(table, 1)?
        .iter()
        .filter(|row| !object_paths.contains(&row.path))
        .filter_map(|row| segment_relative_key(archive.table_dir, &row.path))
        .collect();
    for key in remote_keys.difference(&catalog_keys) {
        if let Err(error) = archive.remote.delete(&ObjectId::table(table, key)?).await {
            tracing::warn!(namespace = %table, key = %key, %error, "legacy object delete failed");
            continue;
        }
        tracing::info!(namespace = %table, segment = %key, "deleted orphan remote segment");
    }
    Ok(())
}

/// Flip `path`'s location to `BOTH` after a successful upload, in both the live
/// view and the catalog.
fn mark_uploaded(archive: &LegacyArchive<'_>, path: &str) -> Result<(), StatsError> {
    archive.segments.update(path, |segment| {
        segment.location = SegmentLocation::Both;
    });
    archive
        .catalog
        .set_location(archive.table, path, SegmentLocation::Both)
}

/// Evict the table's oldest L>=1 copied segments until it is under the
/// count/byte caps, then age-trim.
///
/// Caps resolve from the per-table [`StoragePolicy`] first; unset fields fall
/// back to the cluster-wide [`CompactionConfig`]. Size/count trim is
/// FIFO-by-`min_seq` through `select_eviction_candidate` (BOTH only, so a
/// LOCAL-only segment is never destroyed by the offload path). The age trim
/// (when `max_age_seconds` is set) drops eligible BOTH segments older than
/// `now - max_age`, ordered by `created_at_ms`.
pub fn evict_to_policy(
    table: &str,
    catalog: &Catalog,
    segments: &SegmentView,
    query_visibility: &Arc<RwLock<()>>,
    policy: &StoragePolicy,
    config: &CompactionConfig,
) -> Result<(), StatsError> {
    let max_segments = policy
        .max_segments
        .map(|count| count as usize)
        .unwrap_or(config.max_segments_per_namespace);
    let max_bytes = policy.max_bytes.unwrap_or(config.max_bytes_per_namespace);

    // Size + count trim: FIFO-by-min_seq.
    loop {
        let totals = segments.totals();
        if totals.count <= max_segments && totals.bytes <= max_bytes {
            break;
        }
        let Some(row) = catalog.select_eviction_candidate(table)? else {
            // Over cap but nothing eligible (still L0, or not yet uploaded).
            break;
        };
        evict_segment(table, catalog, segments, query_visibility, &row.path);
    }

    // Age trim: independent of size; ordered by created_at_ms.
    let Some(max_age_ms) = policy.max_age_seconds.map(|seconds| seconds * 1000) else {
        return Ok(());
    };
    let cutoff_ms = crate::store::table::now_ms() - max_age_ms;
    while let Some(row) = catalog.select_aged_eviction_candidate(table, cutoff_ms)? {
        evict_segment(table, catalog, segments, query_visibility, &row.path);
    }
    Ok(())
}

/// Drop `path` from the live view and unlink the local file, returning the bytes
/// reclaimed.
///
/// A `BOTH` segment becomes `REMOTE` in the catalog (the bucket copy is the
/// durable archive) and the local file is unlinked. A `LOCAL`-only segment has no
/// durable copy, so eviction is destructive — the catalog row is dropped.
/// Production eviction routes through `select_eviction_candidate` (BOTH only);
/// the destructive branch serves direct callers and compaction's stale-reference
/// repair.
///
/// Takes the query-visibility WRITE lock (via `blocking_write`) before the unlink
/// so an in-flight query that snapshotted this path drains first. Same lock order
/// as `commit_swap` (query_visibility -> view lock).
pub fn evict_segment(
    table: &str,
    catalog: &Catalog,
    segments: &SegmentView,
    query_visibility: &Arc<RwLock<()>>,
    path: &str,
) -> i64 {
    let _write_guard = query_visibility.blocking_write();
    let removed = segments.remove(path);
    if removed.as_ref().map(|segment| segment.location) == Some(SegmentLocation::Both) {
        let _ = catalog.set_location(table, path, SegmentLocation::Remote);
    } else {
        let _ = catalog.remove_segment(table, path);
    }
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            tracing::warn!(namespace = %table, path = %path, %error, "failed to delete evicted segment");
        }
    }
    // Derived indexes are local-only, so they are unlinked with the local
    // Parquet on eviction.
    remove_index_artifacts(path);
    removed.map(|segment| segment.size_bytes).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::store::catalog::Catalog;
    use crate::store::segment::discover_segments;
    use crate::store::table::maintenance::{self, TableWork};
    use crate::store::table::test_tables::*;
    use crate::store::types::basename;

    /// Local L1 segment files under `dir`.
    fn local_l1(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        discover_segments(dir)
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("seg_L1_")
            })
            .collect()
    }

    async fn maintain(table: &Arc<crate::store::table::TableRuntime>, force_compact_l0: bool) {
        maintenance::run(table, TableWork::Cycle { force_compact_l0 })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn maintain_uploads_compacted_segment_and_flips_both() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table_remote(
            "iris.worker",
            worker_schema(),
            Some(table_dir),
            Arc::clone(&catalog),
            remote.to_str().unwrap(),
            StoragePolicy::default(),
        );
        write_one(&table).await;
        // L0 promoted to L1, then sync uploads it -> BOTH; remote file present.
        maintain(&table, true).await;
        let files = remote_files(&remote, "iris.worker");
        assert_eq!(files.len(), 1, "one compacted L1 segment uploaded");
        let segments = catalog.list_segments("iris.worker").unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].level, 1);
        assert_eq!(segments[0].location, SegmentLocation::Both);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn eviction_drops_oldest_both_preserving_remote_archive() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // cap = 1 segment: after two compaction+upload cycles the oldest is
        // evicted (BOTH -> REMOTE + local unlink), remote archive survives.
        let table = open_table_remote(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            Arc::clone(&catalog),
            remote.to_str().unwrap(),
            StoragePolicy {
                max_segments: Some(1),
                ..Default::default()
            },
        );

        write_one(&table).await;
        maintain(&table, true).await; // L1 #1, uploaded, BOTH
        let first_l1 = local_l1(&table_dir);
        assert_eq!(first_l1.len(), 1);

        write_one(&table).await;
        maintain(&table, true).await; // L1 #2; cap=1 evicts oldest

        // Local L1 files: exactly one remains, and it is NOT the first one.
        assert_eq!(local_l1(&table_dir).len(), 1, "evicted oldest local L1");
        assert!(!first_l1[0].exists(), "oldest local file unlinked");

        // Remote keeps BOTH segments (durable archive preserved).
        assert_eq!(remote_files(&remote, "iris.worker").len(), 2);

        // Catalog: the evicted segment is REMOTE; stats exclude it.
        let segments = catalog.list_segments("iris.worker").unwrap();
        let remote_rows = segments
            .iter()
            .filter(|segment| segment.location == SegmentLocation::Remote)
            .count();
        assert_eq!(remote_rows, 1);
        let stats = table.stats();
        assert_eq!(stats.segment_count, 1, "REMOTE excluded from stats");
        assert_eq!(stats.row_count, 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn eviction_skips_local_only_when_no_remote() {
        let dir = tempdir();
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        // cap = 1, NO remote: nothing is BOTH, so nothing is evictable — two L1
        // segments must survive (eviction must never destroy LOCAL-only data).
        let table = open_table_with_policy(
            "iris.worker",
            worker_schema(),
            Some(table_dir.clone()),
            catalog,
            StoragePolicy {
                max_segments: Some(1),
                ..Default::default()
            },
        );
        write_one(&table).await;
        maintain(&table, true).await;
        write_one(&table).await;
        maintain(&table, true).await;
        assert_eq!(
            local_l1(&table_dir).len(),
            2,
            "LOCAL-only segments are never evicted"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn age_eviction_drops_backdated_both_segment() {
        let dir = tempdir();
        let remote = dir.join("remote");
        let table_dir = dir.join("iris.worker");
        let catalog = Arc::new(Catalog::open(Some(&dir)).unwrap());
        let table = open_table_remote(
            "iris.worker",
            worker_schema(),
            Some(table_dir),
            Arc::clone(&catalog),
            remote.to_str().unwrap(),
            StoragePolicy {
                max_age_seconds: Some(60),
                ..Default::default()
            },
        );
        write_one(&table).await;
        maintain(&table, true).await; // L1, BOTH
        let segments = catalog.list_segments("iris.worker").unwrap();
        assert_eq!(segments.len(), 1);
        let base = basename(&segments[0].path);

        // Within window: a fresh maintain keeps it.
        maintain(&table, false).await;
        assert_eq!(table.stats().segment_count, 1);

        // Backdate past the cutoff (now - 60s); maintain age-evicts it.
        table.backdate_segment(&base, 1).unwrap();
        maintain(&table, false).await;
        assert_eq!(table.stats().segment_count, 0, "aged-out segment dropped");
        // Remote archive preserved.
        assert_eq!(remote_files(&remote, "iris.worker").len(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
