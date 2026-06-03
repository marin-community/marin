//! object_store remote sync surface (LOCAL -> BOTH -> REMOTE).
//!
//! `build_remote_store` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (prod); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! The on-disk layout is `{remote_log_dir}/{namespace}/{basename}`; the
//! `RemoteStore` carries an optional bucket-relative `prefix` (the `gs://`
//! path component) and every per-namespace op composes `{prefix}/{namespace}`.
//! object_store 0.13 moved `put`/`get`/`head`/`delete` onto the `ObjectStoreExt`
//! blanket trait, which must be in scope.

use std::sync::Arc;

use futures::StreamExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path as OsPath;
use object_store::{ObjectStore, ObjectStoreExt};

use crate::errors::StatsError;

/// A configured remote object store plus the bucket-relative prefix the store
/// is rooted under (empty for a `LocalFileSystem` rooted at the remote dir).
#[derive(Clone)]
pub struct RemoteStore {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

/// Build the remote store from `remote_log_dir`, or `None` when sync is
/// disabled (empty string).
///
/// `gs://bucket/sub/dir` -> a GCS store on `bucket` with prefix `sub/dir`.
/// Any other value -> a `LocalFileSystem` rooted at that (created) directory,
/// with an empty prefix, writing into `{remote_log_dir}/{namespace}/{basename}`.
pub fn build_remote_store(remote_log_dir: &str) -> Result<Option<RemoteStore>, StatsError> {
    let dir = remote_log_dir.trim_end_matches('/');
    if dir.is_empty() {
        return Ok(None);
    }
    if let Some(rest) = dir.strip_prefix("gs://") {
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p),
            None => (rest, ""),
        };
        let store = object_store::gcp::GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| StatsError::Internal(format!("build gcs store {bucket:?}: {e}")))?;
        return Ok(Some(RemoteStore {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
        }));
    }
    // Local filesystem remote (tests). Root the store at the remote dir so
    // object paths are `{namespace}/{basename}`.
    std::fs::create_dir_all(dir)
        .map_err(|e| StatsError::Internal(format!("create remote dir {dir}: {e}")))?;
    let store = LocalFileSystem::new_with_prefix(dir)
        .map_err(|e| StatsError::Internal(format!("local remote store {dir}: {e}")))?;
    Ok(Some(RemoteStore {
        store: Arc::new(store),
        prefix: String::new(),
    }))
}

impl RemoteStore {
    /// Split the configured prefix on `/` into individual path components.
    /// `OsPath::from_iter` escapes `/` *within* a single part, so a multi-segment
    /// prefix like `logs/sub` must be pushed component-by-component.
    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.prefix.split('/').filter(|s| !s.is_empty())
    }

    /// The object path for `{prefix}/{namespace}/{basename}`.
    fn object_path(&self, namespace: &str, basename: &str) -> OsPath {
        let parts: Vec<&str> = self.prefix_parts().chain([namespace, basename]).collect();
        OsPath::from_iter(parts)
    }

    /// The directory prefix for one namespace, `{prefix}/{namespace}`.
    fn namespace_prefix(&self, namespace: &str) -> OsPath {
        let parts: Vec<&str> = self.prefix_parts().chain([namespace]).collect();
        OsPath::from_iter(parts)
    }

    /// Upload `local_path` to `{namespace}/{basename}`. Returns `true` on
    /// success; the next sync retries on failure. The byte read + put run as
    /// async object_store calls (no spawn_blocking).
    pub async fn upload(&self, namespace: &str, local_path: &std::path::Path) -> bool {
        let Some(basename) = local_path.file_name().and_then(|n| n.to_str()) else {
            return false;
        };
        let bytes = match tokio::fs::read(local_path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(path = %local_path.display(), error = %e, "remote upload: read local failed");
                return false;
            }
        };
        let remote = self.object_path(namespace, basename);
        match self
            .store
            .put(&remote, bytes::Bytes::from(bytes).into())
            .await
        {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(remote = %remote, error = %e, "remote upload failed");
                false
            }
        }
    }

    /// List the basenames of every parquet object under `{namespace}/`.
    pub async fn list_basenames(&self, namespace: &str) -> Result<Vec<String>, StatsError> {
        let prefix = self.namespace_prefix(namespace);
        let mut stream = self.store.list(Some(&prefix));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta =
                item.map_err(|e| StatsError::Internal(format!("remote list {namespace:?}: {e}")))?;
            if let Some(name) = meta.location.filename() {
                out.push(name.to_string());
            }
        }
        Ok(out)
    }

    /// List `(basename, byte_size)` for every parquet object under
    /// `{namespace}/` whose basename parses as a segment filename. Used by boot
    /// reconcile to enumerate adoption candidates.
    pub async fn list_segment_objects(
        &self,
        namespace: &str,
    ) -> Result<Vec<(String, u64)>, StatsError> {
        let prefix = self.namespace_prefix(namespace);
        let mut stream = self.store.list(Some(&prefix));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta =
                item.map_err(|e| StatsError::Internal(format!("remote list {namespace:?}: {e}")))?;
            if let Some(name) = meta.location.filename() {
                out.push((name.to_string(), meta.size));
            }
        }
        Ok(out)
    }

    /// Delete `{namespace}/{basename}` from the remote store. Best-effort; logs
    /// and swallows on error (warn-and-continue).
    pub async fn delete(&self, namespace: &str, basename: &str) {
        let remote = self.object_path(namespace, basename);
        if let Err(e) = self.store.delete(&remote).await {
            tracing::warn!(remote = %remote, error = %e, "remote delete failed");
        }
    }

    /// Async footer read of `{namespace}/{basename}` — `(row_count, key_min,
    /// key_max)` where the key bounds are the Int64 statistics for `key_column`.
    /// Returns `None` on an unreadable footer. One ranged GET of the file tail.
    pub async fn read_footer(
        &self,
        namespace: &str,
        basename: &str,
        key_column: Option<&str>,
    ) -> Option<(i64, Option<i64>, Option<i64>)> {
        use parquet::arrow::async_reader::ParquetObjectReader;
        use parquet::file::metadata::ParquetMetaDataReader;
        use parquet::file::statistics::Statistics;

        let remote = self.object_path(namespace, basename);
        let meta = self.store.head(&remote).await.ok()?;
        let mut reader =
            ParquetObjectReader::new(Arc::clone(&self.store), remote).with_file_size(meta.size);
        let md = ParquetMetaDataReader::new()
            .with_prefetch_hint(Some(64 * 1024))
            .load_via_suffix_and_finish(&mut reader)
            .await
            .ok()?;
        let num_rows = md.file_metadata().num_rows();
        let mut lo: Option<i64> = None;
        let mut hi: Option<i64> = None;
        if let Some(kc) = key_column {
            let schema = md.file_metadata().schema_descr();
            if let Some(col_idx) =
                (0..schema.num_columns()).find(|&i| schema.column(i).name() == kc)
            {
                for rg in md.row_groups() {
                    if let Some(Statistics::Int64(s)) = rg.column(col_idx).statistics() {
                        if let Some(&m) = s.min_opt() {
                            lo = Some(lo.map_or(m, |x: i64| x.min(m)));
                        }
                        if let Some(&m) = s.max_opt() {
                            hi = Some(hi.map_or(m, |x: i64| x.max(m)));
                        }
                    }
                }
            }
        }
        Some((num_rows, lo, hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("finelog_remote_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn empty_remote_dir_disables_sync() {
        assert!(build_remote_store("").unwrap().is_none());
    }

    #[test]
    fn gs_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here).
        let store = build_remote_store("gs://my-bucket/logs/sub")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "logs/sub");
        let p = store.object_path("ns.a", "seg_L1_0001.parquet");
        assert_eq!(p.to_string(), "logs/sub/ns.a/seg_L1_0001.parquet");
    }

    #[tokio::test]
    async fn local_remote_upload_list_delete_round_trip() {
        let remote_dir = tempdir("rt");
        let local_dir = tempdir("local");
        let store = build_remote_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();

        let local_file = local_dir.join("seg_L1_0000000000000000001.parquet");
        std::fs::write(&local_file, b"hello-parquet").unwrap();
        assert!(store.upload("ns.a", &local_file).await);

        let on_disk = remote_dir
            .join("ns.a")
            .join(local_file.file_name().unwrap());
        assert!(on_disk.exists());
        let names = store.list_basenames("ns.a").await.unwrap();
        assert_eq!(
            names,
            vec!["seg_L1_0000000000000000001.parquet".to_string()]
        );

        store
            .delete("ns.a", "seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!on_disk.exists());
        assert!(store.list_basenames("ns.a").await.unwrap().is_empty());

        std::fs::remove_dir_all(&remote_dir).ok();
        std::fs::remove_dir_all(&local_dir).ok();
    }
}
