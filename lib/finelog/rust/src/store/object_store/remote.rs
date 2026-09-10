//! Canonical-layout implementation of the Finelog object-store contract.
//!
//! `build_remote_object_store` dispatches on the configured `remote_log_dir`:
//! `gs://bucket/prefix` -> `GoogleCloudStorageBuilder` (GCP prod);
//! `s3://bucket/prefix` -> `AmazonS3Builder` for any S3-compatible store
//! (Cloudflare R2 / CoreWeave Object Storage on CoreWeave clusters); any other
//! non-empty value -> `LocalFileSystem` rooted at that directory (tests pass a
//! plain tmp path). An empty `remote_log_dir` disables sync (returns `None`).
//!
//! Canonical objects use root-relative [`ObjectId`] values under
//! `_finelog/tables`. Historical layout translation lives in `legacy`; both
//! implementations satisfy [`ObjectStore`].
//! object_store 0.13 moved `put`/`get`/`head`/`delete` onto the `ObjectStoreExt`
//! blanket trait, which must be in scope.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use futures::{stream, StreamExt};
use object_store::path::Path as OsPath;
use object_store::{ObjectStoreExt, PutMode, PutOptions};

use crate::errors::StatsError;
use crate::store::object_store::{
    DeleteManyOutcome, ObjectId, ObjectMetadata, ObjectPrefix, ObjectStore, ObjectVersion,
    StoredObject, FINELOG_ROOT_COMPONENT, TABLES_COMPONENT,
};

use super::provider::Provider;

/// A configured remote object store plus the bucket-relative prefix the store
/// is rooted under (empty for a `LocalFileSystem` rooted at the remote dir).
#[derive(Clone)]
pub struct RemoteObjectStore {
    provider: Provider,
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
pub fn build_remote_object_store(
    remote_log_dir: &str,
) -> Result<Option<RemoteObjectStore>, StatsError> {
    Ok(Provider::from_remote_log_dir(remote_log_dir)?
        .map(|provider| RemoteObjectStore { provider }))
}

impl RemoteObjectStore {
    pub(super) fn provider(&self) -> Provider {
        self.provider.clone()
    }

    /// The backend store and base URL a query engine registers to scan this
    /// provider's objects directly. `None` for a local-directory provider,
    /// whose objects scan through the engine's default file store.
    pub fn scan_registration(&self) -> Option<(String, Arc<dyn object_store::ObjectStore>)> {
        self.provider
            .base_url()
            .map(|base| (base.to_string(), Arc::clone(self.provider.backend())))
    }

    /// Split the configured prefix on `/` into individual path components.
    /// `OsPath::from_iter` escapes `/` *within* a single part, so a multi-segment
    /// prefix like `logs/sub` must be pushed component-by-component.
    fn prefix_parts(&self) -> impl Iterator<Item = &str> {
        self.provider.prefix_parts()
    }

    /// The physical backend path for a root-relative key (an [`ObjectId`] or
    /// [`ObjectPrefix`] rendering), under the configured prefix.
    fn canonical(&self, key: &str) -> OsPath {
        let parts: Vec<&str> = self
            .prefix_parts()
            .chain(key.split('/').filter(|part| !part.is_empty()))
            .collect();
        OsPath::from_iter(parts)
    }

    async fn list_tables_under_root(&self) -> Result<Vec<String>, StatsError> {
        let root = OsPath::from_iter(
            self.prefix_parts()
                .chain([FINELOG_ROOT_COMPONENT, TABLES_COMPONENT]),
        );
        let result = self
            .provider
            .backend()
            .list_with_delimiter(Some(&root))
            .await
            .map_err(|error| StatsError::Internal(format!("list object tables {root}: {error}")))?;
        let mut tables = result
            .common_prefixes
            .into_iter()
            .filter_map(|prefix| {
                prefix
                    .prefix_match(&root)
                    .and_then(|mut parts| parts.next())
                    .map(|namespace| namespace.as_ref().to_string())
            })
            .collect::<Vec<_>>();
        tables.sort();
        Ok(tables)
    }
}

#[async_trait]
impl ObjectStore for RemoteObjectStore {
    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError> {
        self.provider
            .get_path(self.provider.object_path(id), "object")
            .await
    }

    async fn exists(&self, id: &ObjectId) -> Result<bool, StatsError> {
        let path = self.provider.object_path(id);
        match self.provider.backend().head(&path).await {
            Ok(_) => Ok(true),
            Err(object_store::Error::NotFound { .. }) => Ok(false),
            Err(error) => Err(StatsError::Internal(format!(
                "inspect object {path}: {error}"
            ))),
        }
    }

    /// Create an immutable object, accepting an identical retry.
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        let path = self.provider.object_path(id);
        let byte_size = bytes.len() as u64;
        let result = self
            .provider
            .backend()
            .put_opts(
                &path,
                bytes.clone().into(),
                PutOptions {
                    mode: PutMode::Create,
                    ..Default::default()
                },
            )
            .await;
        match result {
            Ok(result) => Ok(ObjectVersion {
                e_tag: result.e_tag,
                provider_version: result.version,
                byte_size,
                local_value: None,
            }),
            Err(object_store::Error::AlreadyExists { .. }) => {
                let existing = self.read(id).await?.ok_or_else(|| {
                    StatsError::Internal(format!("object {path} disappeared after create conflict"))
                })?;
                if existing.bytes == bytes {
                    Ok(existing.version)
                } else {
                    Err(StatsError::SchemaConflict(format!(
                        "immutable object {path} already exists with different contents"
                    )))
                }
            }
            Err(error) => Err(StatsError::Internal(format!(
                "create object {path}: {error}"
            ))),
        }
    }

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        let local_path = self
            .provider
            .local_root()
            .map(|root| local_object_path(root, id));
        self.provider
            .compare_and_swap_path(
                self.provider.object_path(id),
                local_path,
                expected,
                bytes,
                "object",
            )
            .await
    }

    fn remote_scan_url(&self, id: &ObjectId) -> Option<String> {
        Some(self.provider.scan_url(id))
    }

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError> {
        let path = self.canonical(prefix.as_str());
        let mut stream = self.provider.backend().list(Some(&path));
        let mut objects = Vec::new();
        while let Some(result) = stream.next().await {
            let meta = result
                .map_err(|error| StatsError::Internal(format!("list objects {path}: {error}")))?;
            let root = OsPath::from_iter(self.prefix_parts());
            let Some(parts) = meta.location.prefix_match(&root) else {
                continue;
            };
            let id = parts
                .map(|part| part.as_ref().to_string())
                .collect::<Vec<_>>()
                .join("/");
            objects.push(ObjectMetadata {
                id: ObjectId::parse(&id)?,
                modified_at_ms: meta.last_modified.timestamp_millis(),
            });
        }
        Ok(objects)
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError> {
        let path = self.provider.object_path(id);
        match self.provider.backend().delete(&path).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(error) => Err(StatsError::Internal(format!(
                "delete object {path}: {error}"
            ))),
        }
    }

    async fn delete_many(&self, ids: Vec<ObjectId>) -> DeleteManyOutcome {
        let by_path = ids
            .into_iter()
            .map(|id| (self.provider.object_path(&id).to_string(), id))
            .collect::<std::collections::HashMap<_, _>>();
        let paths = by_path
            .keys()
            .cloned()
            .map(OsPath::from)
            .map(Ok)
            .collect::<Vec<Result<OsPath, object_store::Error>>>();
        let mut results = self
            .provider
            .backend()
            .delete_stream(stream::iter(paths).boxed());
        let mut deleted = Vec::with_capacity(by_path.len());
        let mut first_error = None;
        while let Some(result) = results.next().await {
            match result {
                Ok(path) => {
                    if let Some(id) = by_path.get(path.as_ref()) {
                        deleted.push(id.clone());
                    }
                }
                Err(object_store::Error::NotFound { path, .. }) => {
                    let logical_path = self
                        .provider
                        .local_root()
                        .and_then(|root| Path::new(&path).strip_prefix(root).ok())
                        .map(|relative| relative.to_string_lossy().into_owned())
                        .unwrap_or(path);
                    if let Some(id) = by_path.get(&logical_path) {
                        deleted.push(id.clone());
                    }
                }
                Err(error) => {
                    first_error.get_or_insert_with(|| {
                        StatsError::Internal(format!("delete object batch: {error}"))
                    });
                }
            }
        }
        DeleteManyOutcome {
            deleted,
            error: first_error,
        }
    }

    async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        self.list_tables_under_root().await
    }
}

fn local_object_path(root: &Path, id: &ObjectId) -> PathBuf {
    let mut path = root.to_path_buf();
    for part in id.as_str().split('/').filter(|part| !part.is_empty()) {
        path.push(part);
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_dir;

    #[test]
    fn empty_remote_dir_disables_sync() {
        assert!(build_remote_object_store("").unwrap().is_none());
    }

    #[test]
    fn gs_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here).
        let store = build_remote_object_store("gs://my-bucket/logs/sub")
            .unwrap()
            .unwrap();
        let id = ObjectId::table("ns.a", "seg_L1_0001.parquet").unwrap();
        assert_eq!(
            store.canonical(id.as_str()).to_string(),
            "logs/sub/_finelog/tables/ns.a/seg_L1_0001.parquet"
        );
    }

    #[test]
    fn s3_url_parses_bucket_and_prefix() {
        // from_env() builds without credentials; the parse + prefix split is the
        // logic under test (no network — we never call put/list here). No
        // AWS_ENDPOINT_URL is set, so the builder keeps its default endpoint.
        let store = build_remote_object_store("s3://my-bucket/finelog/cw-us-east-02a")
            .unwrap()
            .unwrap();
        let id = ObjectId::table("iris.worker", "seg_L1_0001.parquet").unwrap();
        let p = store.canonical(id.as_str());
        assert_eq!(
            p.to_string(),
            "finelog/cw-us-east-02a/_finelog/tables/iris.worker/seg_L1_0001.parquet"
        );
    }

    #[tokio::test]
    async fn canonical_objects_round_trip_by_typed_id() {
        let remote_dir = unique_dir("remote_objects");
        let store = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let id = ObjectId::table("iris.worker", "objects/v1/l1/hash/segment.parquet").unwrap();

        let version = store
            .write(&id, bytes::Bytes::from_static(b"parquet"))
            .await
            .unwrap();
        assert_eq!(version.byte_size, 7);
        assert_eq!(
            store.read(&id).await.unwrap().unwrap().bytes,
            b"parquet"[..]
        );
        assert!(store.exists(&id).await.unwrap());
        let listed = store
            .list(&ObjectPrefix::table("iris.worker", "objects/v1").unwrap())
            .await
            .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, id);

        store.delete(&id).await.unwrap();
        assert!(store.read(&id).await.unwrap().is_none());
        assert!(!store.exists(&id).await.unwrap());
        std::fs::remove_dir_all(&remote_dir).ok();
    }

    #[tokio::test]
    async fn batch_delete_confirms_existing_and_missing_objects() {
        let remote_dir = unique_dir("remote_batch_delete");
        let store = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let existing = [
            ObjectId::table("iris.task", "catalogs/1.json").unwrap(),
            ObjectId::table("iris.task", "catalogs/2.json").unwrap(),
        ];
        for id in &existing {
            store
                .write(id, bytes::Bytes::from_static(b"catalog"))
                .await
                .unwrap();
        }
        let missing = ObjectId::table("iris.task", "catalogs/missing.json").unwrap();
        let outcome = store
            .delete_many(
                existing
                    .iter()
                    .cloned()
                    .chain(std::iter::once(missing.clone()))
                    .collect(),
            )
            .await;

        assert!(outcome.error.is_none());
        assert_eq!(outcome.deleted.len(), 3);
        for id in existing.iter().chain(std::iter::once(&missing)) {
            assert!(!store.exists(id).await.unwrap());
        }
        std::fs::remove_dir_all(&remote_dir).ok();
    }

    #[tokio::test]
    async fn table_listing_returns_only_immediate_table_prefixes() {
        let remote_dir = unique_dir("remote_table_listing");
        let store = build_remote_object_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        for id in [
            ObjectId::table("iris.task", "objects/v1/a.parquet").unwrap(),
            ObjectId::table("iris.task", "states/1.json").unwrap(),
            ObjectId::table("log", "HEAD.json").unwrap(),
        ] {
            store
                .write(&id, bytes::Bytes::from_static(b"data"))
                .await
                .unwrap();
        }

        assert_eq!(
            store.list_tables().await.unwrap(),
            vec!["iris.task".to_string(), "log".to_string()]
        );
        std::fs::remove_dir_all(&remote_dir).ok();
    }
}
