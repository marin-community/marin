//! object_store remote sync surface (LOCAL -> BOTH -> REMOTE).
//!
//! `build_remote_store` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (GCP prod);
//! `s3://bucket/prefix` -> `AmazonS3Builder` for any S3-compatible store
//! (Cloudflare R2 / CoreWeave Object Storage on CoreWeave clusters); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! The on-disk layout is `{remote_log_dir}/{namespace}/{relative segment key}`; the
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

const GCS_SCHEME: &str = "gs://";
const S3_SCHEME: &str = "s3://";

/// Whether `remote_log_dir` names an object store rather than a local directory.
pub fn is_object_store(remote_log_dir: &str) -> bool {
    let dir = remote_log_dir.trim();
    dir.starts_with(GCS_SCHEME) || dir.starts_with(S3_SCHEME)
}

/// Build the remote store from `remote_log_dir`, or `None` when sync is
/// disabled (empty string).
///
/// `gs://bucket/sub/dir` -> a GCS store on `bucket` with prefix `sub/dir`.
/// `s3://bucket/sub/dir` -> an S3-compatible store on `bucket` with prefix
/// `sub/dir`. Everything else about the connection comes from the standard
/// `AWS_*` env the deploy environment injects (on CoreWeave: the `iris-task-env`
/// Secret's R2 creds). `AmazonS3Builder::from_env` reads credentials, region,
/// the custom `AWS_ENDPOINT_URL`, and `AWS_VIRTUAL_HOSTED_STYLE_REQUEST` — so
/// iris owns the addressing-style decision (path-style for R2, virtual-hosted
/// for CoreWeave Object Storage) and this server stays endpoint-agnostic.
/// Any other value -> a `LocalFileSystem` rooted at that (created) directory,
/// with an empty prefix, writing into `{remote_log_dir}/{namespace}/{relative segment key}`.
pub fn build_remote_store(remote_log_dir: &str) -> Result<Option<RemoteStore>, StatsError> {
    let dir = remote_log_dir.trim_end_matches('/');
    if dir.is_empty() {
        return Ok(None);
    }
    if let Some(rest) = dir.strip_prefix(GCS_SCHEME) {
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
    if let Some(rest) = dir.strip_prefix(S3_SCHEME) {
        let (bucket, prefix) = match rest.split_once('/') {
            Some((b, p)) => (b, p),
            None => (rest, ""),
        };
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(bucket)
            .build()
            .map_err(|e| StatsError::Internal(format!("build s3 store {bucket:?}: {e}")))?;
        return Ok(Some(RemoteStore {
            store: Arc::new(store),
            prefix: prefix.trim_matches('/').to_string(),
        }));
    }
    // Local filesystem remote (tests). Root the store at the remote dir so
    // object paths are `{namespace}/{relative segment key}`.
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

    /// The object path for `{prefix}/{namespace}/{relative segment key}`.
    fn object_path(&self, namespace: &str, relative_key: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain([namespace])
            .chain(relative_key.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    /// The directory prefix for one namespace, `{prefix}/{namespace}`.
    fn namespace_prefix(&self, namespace: &str) -> OsPath {
        let parts: Vec<&str> = self.prefix_parts().chain([namespace]).collect();
        OsPath::from_iter(parts)
    }

    /// Upload `local_path` to `{namespace}/{relative_key}`. Returns `true` on
    /// success; the next sync retries on failure. The byte read + put run as
    /// async object_store calls (no spawn_blocking).
    pub async fn upload(
        &self,
        namespace: &str,
        relative_key: &str,
        local_path: &std::path::Path,
    ) -> bool {
        let bytes = match tokio::fs::read(local_path).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(path = %local_path.display(), error = %e, "remote upload: read local failed");
                return false;
            }
        };
        let remote = self.object_path(namespace, relative_key);
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

    /// Copy one object within a namespace without downloading it through the
    /// Finelog process. Layout migration uses this to publish a new key before
    /// atomically changing the catalog pointer to it.
    pub async fn copy(&self, namespace: &str, from_key: &str, to_key: &str) -> bool {
        let from = self.object_path(namespace, from_key);
        let to = self.object_path(namespace, to_key);
        match self.store.copy(&from, &to).await {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(from = %from, to = %to, %error, "remote copy failed");
                false
            }
        }
    }

    /// List the relative keys of every object under `{namespace}/`.
    pub async fn list_keys(&self, namespace: &str) -> Result<Vec<String>, StatsError> {
        let prefix = self.namespace_prefix(namespace);
        let mut stream = self.store.list(Some(&prefix));
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            let meta =
                item.map_err(|e| StatsError::Internal(format!("remote list {namespace:?}: {e}")))?;
            if let Some(parts) = meta.location.prefix_match(&prefix) {
                out.push(
                    parts
                        .map(|part| part.as_ref().to_string())
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            };
        }
        Ok(out)
    }

    /// List `(relative_key, byte_size)` for every parquet object under
    /// `{namespace}/` whose filename parses as a segment filename. Used by boot
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
            let Some(filename) = meta.location.filename() else {
                continue;
            };
            if crate::store::types::parse_seg_filename(filename).is_none() {
                continue;
            }
            if let Some(parts) = meta.location.prefix_match(&prefix) {
                let key = parts
                    .map(|part| part.as_ref().to_string())
                    .collect::<Vec<_>>()
                    .join("/");
                out.push((key, meta.size));
            };
        }
        Ok(out)
    }

    /// Delete `{namespace}/{relative_key}` from the remote store. Best-effort; logs
    /// and swallows on error (warn-and-continue).
    pub async fn delete(&self, namespace: &str, relative_key: &str) {
        let remote = self.object_path(namespace, relative_key);
        if let Err(e) = self.store.delete(&remote).await {
            tracing::warn!(remote = %remote, error = %e, "remote delete failed");
        }
    }

    /// Async footer read of `{namespace}/{relative_key}`, including actual seq
    /// bounds and hidden partition metadata.
    /// Returns `None` on an unreadable footer.
    ///
    /// `file_size` is the object size already known from `list_segment_objects`,
    /// passed in so this is a single ranged GET of the file tail with NO preceding
    /// `head` round-trip — halving the cross-region RPCs per segment on reconcile.
    pub async fn read_footer(
        &self,
        namespace: &str,
        relative_key: &str,
        file_size: u64,
        key_column: Option<&str>,
    ) -> Option<crate::store::segment::SegmentMetadata> {
        use parquet::arrow::async_reader::ParquetObjectReader;
        use parquet::file::metadata::ParquetMetaDataReader;

        let filename = std::path::Path::new(relative_key).file_name()?.to_str()?;
        let (level, filename_min_seq) = crate::store::types::parse_seg_filename(filename)?;

        let remote = self.object_path(namespace, relative_key);
        let mut reader =
            ParquetObjectReader::new(Arc::clone(&self.store), remote).with_file_size(file_size);
        let md = ParquetMetaDataReader::new()
            .with_prefetch_hint(Some(64 * 1024))
            .load_via_suffix_and_finish(&mut reader)
            .await
            .ok()?;
        crate::store::segment::segment_metadata_from_parquet(
            &md,
            level,
            filename_min_seq,
            key_column,
        )
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

    #[test]
    fn s3_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here). No
        // AWS_ENDPOINT_URL is set, so the builder keeps its default endpoint.
        let store = build_remote_store("s3://my-bucket/finelog/cw-us-east-02a")
            .unwrap()
            .unwrap();
        assert_eq!(store.prefix, "finelog/cw-us-east-02a");
        let p = store.object_path("iris.worker", "seg_L1_0001.parquet");
        assert_eq!(
            p.to_string(),
            "finelog/cw-us-east-02a/iris.worker/seg_L1_0001.parquet"
        );
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
        assert!(
            store
                .upload(
                    "ns.a",
                    "run_id/07/seg_L1_0000000000000000001.parquet",
                    &local_file
                )
                .await
        );

        let on_disk = remote_dir
            .join("ns.a")
            .join("run_id/07")
            .join(local_file.file_name().unwrap());
        assert!(on_disk.exists());

        assert!(
            store
                .copy(
                    "ns.a",
                    "run_id/07/seg_L1_0000000000000000001.parquet",
                    "run_id/08/seg_L1_0000000000000000001.parquet",
                )
                .await
        );
        let copied = remote_dir
            .join("ns.a")
            .join("run_id/08")
            .join(local_file.file_name().unwrap());
        assert_eq!(std::fs::read(&copied).unwrap(), b"hello-parquet");

        let mut names = store.list_keys("ns.a").await.unwrap();
        names.sort();
        assert_eq!(
            names,
            vec![
                "run_id/07/seg_L1_0000000000000000001.parquet".to_string(),
                "run_id/08/seg_L1_0000000000000000001.parquet".to_string(),
            ]
        );

        store
            .delete("ns.a", "run_id/07/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!on_disk.exists());
        store
            .delete("ns.a", "run_id/08/seg_L1_0000000000000000001.parquet")
            .await;
        assert!(!copied.exists());
        assert!(store.list_keys("ns.a").await.unwrap().is_empty());

        std::fs::remove_dir_all(&remote_dir).ok();
        std::fs::remove_dir_all(&local_dir).ok();
    }
}
