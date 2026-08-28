//! Immutable object-native catalogs selected by one conditional HEAD update.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use buffa::MessageField;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    CatalogHead, CatalogSegment, MigrationPhase, NamespaceCatalog, ObjectRef, TableVersionSegments,
};
use crate::store::catalog::Catalog;
use crate::store::remote::{RemoteStore, RemoteVersion};
use crate::store::types::{basename, segment_relative_key, SegmentLocation};

pub const NATIVE_FORMAT_VERSION: u64 = 1;
const HEAD_KEY: &str = "HEAD.json";

#[derive(Debug, Clone)]
pub struct CatalogSnapshot {
    pub head: CatalogHead,
    pub catalog: NamespaceCatalog,
    head_version: RemoteVersion,
}

#[derive(Clone)]
pub struct NativeCatalog {
    remote: RemoteStore,
}

impl NativeCatalog {
    pub fn new(remote: RemoteStore) -> Self {
        Self { remote }
    }

    pub async fn load(&self, namespace: &str) -> Result<Option<CatalogSnapshot>, StatsError> {
        let Some(head_object) = self.remote.get_native(namespace, HEAD_KEY).await? else {
            return Ok(None);
        };
        let head: CatalogHead = serde_json::from_slice(&head_object.bytes).map_err(|error| {
            StatsError::Internal(format!("decode native HEAD for {namespace:?}: {error}"))
        })?;
        validate_head(namespace, &head)?;
        let catalog_ref = head.catalog.as_option().ok_or_else(|| {
            StatsError::Internal(format!(
                "native HEAD for {namespace:?} has no catalog reference"
            ))
        })?;
        let catalog_key = catalog_ref.uri.as_deref().ok_or_else(|| {
            StatsError::Internal(format!(
                "native HEAD for {namespace:?} has an empty catalog URI"
            ))
        })?;
        let catalog_object = self
            .remote
            .get_native(namespace, catalog_key)
            .await?
            .ok_or_else(|| {
                StatsError::Internal(format!(
                    "native HEAD for {namespace:?} references missing catalog {catalog_key:?}"
                ))
            })?;
        if catalog_ref.sha256.as_deref() != Some(catalog_object.version.content_sha256.as_slice()) {
            return Err(StatsError::Internal(format!(
                "native catalog {catalog_key:?} for {namespace:?} failed SHA-256 validation"
            )));
        }
        let catalog: NamespaceCatalog =
            serde_json::from_slice(&catalog_object.bytes).map_err(|error| {
                StatsError::Internal(format!(
                    "decode native catalog {catalog_key:?} for {namespace:?}: {error}"
                ))
            })?;
        validate_catalog(namespace, &head, &catalog)?;
        Ok(Some(CatalogSnapshot {
            head,
            catalog,
            head_version: head_object.version,
        }))
    }

    pub async fn publish(
        &self,
        namespace: &str,
        writer_epoch: u64,
        catalog: NamespaceCatalog,
        expected: Option<&CatalogSnapshot>,
    ) -> Result<CatalogSnapshot, StatsError> {
        let generation = catalog.catalog_generation.unwrap_or(0);
        let previous_generation =
            expected.map(|snapshot| snapshot.head.catalog_generation.unwrap_or(0));
        if generation == 0 || previous_generation.is_some_and(|previous| generation <= previous) {
            return Err(StatsError::SchemaConflict(format!(
                "native catalog generation {generation} does not advance {previous_generation:?} for {namespace:?}"
            )));
        }
        if catalog.format_version.unwrap_or(0) != NATIVE_FORMAT_VERSION
            || catalog.namespace.as_deref() != Some(namespace)
        {
            return Err(StatsError::SchemaValidation(format!(
                "native catalog identity does not match namespace {namespace:?}"
            )));
        }

        let catalog_bytes = serde_json::to_vec(&catalog).map_err(|error| {
            StatsError::Internal(format!("encode native catalog for {namespace:?}: {error}"))
        })?;
        let catalog_sha256: [u8; 32] = Sha256::digest(&catalog_bytes).into();
        let catalog_key = format!(
            "catalogs/{generation:020}-{}.json",
            short_hex(&catalog_sha256)
        );
        let catalog_version = self
            .remote
            .put_native_immutable(namespace, &catalog_key, Bytes::from(catalog_bytes.clone()))
            .await?;
        let head = CatalogHead {
            format_version: Some(NATIVE_FORMAT_VERSION),
            namespace: Some(namespace.to_string()),
            writer_epoch: Some(writer_epoch),
            catalog_generation: Some(generation),
            active_table_spec_version: catalog.active_table_spec_version,
            catalog: buffa::MessageField::some(ObjectRef {
                uri: Some(catalog_key),
                provider_version: catalog_version.provider_version.clone(),
                etag: catalog_version.e_tag.clone(),
                byte_size: Some(catalog_bytes.len() as u64),
                sha256: Some(catalog_sha256.to_vec()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let head_bytes = serde_json::to_vec(&head).map_err(|error| {
            StatsError::Internal(format!("encode native HEAD for {namespace:?}: {error}"))
        })?;
        let head_version = self
            .remote
            .compare_and_swap_native(
                namespace,
                HEAD_KEY,
                expected.map(|snapshot| &snapshot.head_version),
                Bytes::from(head_bytes),
            )
            .await?;
        Ok(CatalogSnapshot {
            head,
            catalog,
            head_version,
        })
    }

    #[cfg(test)]
    async fn catalog_keys(&self, namespace: &str) -> Result<Vec<String>, StatsError> {
        self.remote.list_native_keys(namespace, "catalogs").await
    }

    /// Remove superseded catalog documents after the maximum query lifetime.
    pub async fn gc_obsolete_catalogs(
        &self,
        namespace: &str,
        now_ms: i64,
        max_query_time_ms: u64,
    ) -> Result<usize, StatsError> {
        let Some(snapshot) = self.load(namespace).await? else {
            return Ok(0);
        };
        let current = snapshot
            .head
            .catalog
            .as_option()
            .and_then(|reference| reference.uri.as_deref())
            .ok_or_else(|| {
                StatsError::Internal(format!("native HEAD for {namespace:?} has no catalog URI"))
            })?;
        let cutoff = now_ms.saturating_sub(i64::try_from(max_query_time_ms).unwrap_or(i64::MAX));
        let current_generation = snapshot.head.catalog_generation.unwrap_or(0);
        let catalog_objects = self
            .remote
            .list_native_objects(namespace, "catalogs")
            .await?;
        let mut removed = 0;
        for (key, meta) in &catalog_objects {
            if key == current {
                continue;
            }
            let Some(generation) = catalog_generation_from_key(key) else {
                tracing::warn!(namespace, key, "retaining unrecognized native catalog key");
                continue;
            };
            let obsolete_at_ms = if generation < current_generation {
                catalog_objects
                    .iter()
                    .filter_map(|(candidate_key, candidate_meta)| {
                        let candidate_generation = catalog_generation_from_key(candidate_key)?;
                        (candidate_generation > generation
                            && candidate_generation <= current_generation)
                            .then_some(candidate_meta.last_modified.timestamp_millis())
                    })
                    .min()
            } else {
                // Same-generation and future-generation objects never won HEAD.
                Some(meta.last_modified.timestamp_millis())
            };
            if obsolete_at_ms.is_none_or(|obsolete_at_ms| obsolete_at_ms > cutoff) {
                continue;
            }
            self.remote.delete_native(namespace, key).await?;
            removed += 1;
        }
        let mut referenced = referenced_object_keys(&snapshot.catalog);
        for key in self.remote.list_native_keys(namespace, "catalogs").await? {
            if key == current {
                continue;
            }
            let Some(object) = self.remote.get_native(namespace, &key).await? else {
                continue;
            };
            let catalog: NamespaceCatalog =
                serde_json::from_slice(&object.bytes).map_err(|error| {
                    StatsError::Internal(format!(
                        "decode retained native catalog {key:?} for {namespace:?}: {error}"
                    ))
                })?;
            referenced.extend(referenced_object_keys(&catalog));
        }
        for (key, meta) in self
            .remote
            .list_native_objects(namespace, "objects")
            .await?
        {
            if referenced.contains(&key) || meta.last_modified.timestamp_millis() > cutoff {
                continue;
            }
            self.remote.delete_native(namespace, &key).await?;
            removed += 1;
        }
        Ok(removed)
    }

    /// Publish the recoverable SQLite view unless HEAD already selects it.
    pub async fn publish_local(
        &self,
        catalog: &Catalog,
        namespace: &str,
        namespace_dir: &Path,
        writer_epoch: u64,
    ) -> Result<CatalogSnapshot, StatsError> {
        let contents = build_namespace_catalog(catalog, namespace, namespace_dir)?;
        let remote = self.load(namespace).await?;
        if let Some(remote) = &remote {
            let remote_generation = remote.head.catalog_generation.unwrap_or(0);
            let local_generation = contents.catalog_generation.unwrap_or(0);
            if remote_generation == local_generation {
                if remote.catalog != contents {
                    return Err(StatsError::SchemaConflict(format!(
                        "local catalog generation {local_generation} for {namespace:?} differs from the published generation"
                    )));
                }
                return Ok(remote.clone());
            }
            if remote_generation >= local_generation {
                return Err(StatsError::SchemaConflict(format!(
                    "local catalog generation {local_generation} for {namespace:?} does not advance remote generation {remote_generation}"
                )));
            }
        }
        self.publish(namespace, writer_epoch, contents, remote.as_ref())
            .await
    }
}

fn referenced_object_keys(catalog: &NamespaceCatalog) -> std::collections::HashSet<String> {
    catalog
        .version_segments
        .iter()
        .flat_map(|version| {
            version
                .live_segments
                .iter()
                .chain(version.retired_segments.iter())
        })
        .filter_map(|segment| {
            segment
                .source
                .as_option()
                .and_then(|source| source.uri.clone())
        })
        .collect()
}

pub fn build_namespace_catalog(
    catalog: &Catalog,
    namespace: &str,
    namespace_dir: &Path,
) -> Result<NamespaceCatalog, StatsError> {
    let status = catalog.table_spec_status(namespace)?;
    if status.catalog_generation == 0 {
        return Err(StatsError::SchemaValidation(format!(
            "namespace {namespace:?} has no versioned table specification"
        )));
    }
    let native_segments: HashMap<_, _> = catalog
        .native_segments(namespace)?
        .into_iter()
        .map(|record| (record.path.clone(), record))
        .collect();
    let mut by_version: BTreeMap<u64, Vec<CatalogSegment>> = BTreeMap::new();
    for row in catalog.list_segments(namespace)? {
        let (version, source) = match native_segments.get(&row.path) {
            Some(record) => (record.table_spec_version, record.source.clone()),
            None if row.location != SegmentLocation::Local => {
                let Some(relative_key) = segment_relative_key(namespace_dir, &row.path) else {
                    continue;
                };
                (
                    0,
                    ObjectRef {
                        uri: Some(relative_key),
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
                .and_then(|partition| serde_json::to_string(partition).ok()),
            schema_revision: Some(version),
            migration_source_id: native_segments
                .get(&row.path)
                .and_then(|record| record.migration_source_id.clone()),
            migration_source_rows: native_segments
                .get(&row.path)
                .and_then(|record| record.migration_source_rows),
            migration_backfill: native_segments
                .get(&row.path)
                .map(|record| record.migration_backfill),
            ..Default::default()
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
                && native_segments
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
    let version_segments: Vec<TableVersionSegments> = by_version
        .into_iter()
        .map(|(version, live_segments)| TableVersionSegments {
            table_spec_version: Some(version),
            live_segments,
            ..Default::default()
        })
        .collect();
    let desired_version = status.desired_version();
    let stats = catalog.aggregate_namespace_stats(namespace)?;
    let max_query_time_ms = status
        .active
        .as_ref()
        .or(status.desired.as_ref())
        .and_then(|spec| spec.operating_policy.as_option())
        .and_then(|policy| policy.max_query_time_ms)
        .unwrap_or(crate::store::table_spec::DEFAULT_MAX_QUERY_TIME_MS);
    Ok(NamespaceCatalog {
        format_version: Some(NATIVE_FORMAT_VERSION),
        namespace: Some(namespace.to_string()),
        catalog_generation: Some(status.catalog_generation),
        active_table_spec_version: Some(status.active_version()),
        desired_table_spec_version: Some(desired_version),
        retained_table_specs: catalog.retained_table_specs(namespace)?,
        persisted_high_water: Some(stats.max_seq),
        version_segments,
        migration: status.migration.into(),
        max_query_time_ms: Some(max_query_time_ms),
        forward_cursors: catalog.forward_cursors(namespace)?,
        ..Default::default()
    })
}

fn validate_head(namespace: &str, head: &CatalogHead) -> Result<(), StatsError> {
    if head.format_version.unwrap_or(0) != NATIVE_FORMAT_VERSION
        || head.namespace.as_deref() != Some(namespace)
        || head.catalog_generation.unwrap_or(0) == 0
    {
        return Err(StatsError::Internal(format!(
            "invalid native HEAD for namespace {namespace:?}"
        )));
    }
    Ok(())
}

fn validate_catalog(
    namespace: &str,
    head: &CatalogHead,
    catalog: &NamespaceCatalog,
) -> Result<(), StatsError> {
    if catalog.format_version.unwrap_or(0) != NATIVE_FORMAT_VERSION
        || catalog.namespace.as_deref() != Some(namespace)
        || catalog.catalog_generation != head.catalog_generation
        || catalog.active_table_spec_version != head.active_table_spec_version
    {
        return Err(StatsError::Internal(format!(
            "native catalog does not match HEAD for namespace {namespace:?}"
        )));
    }
    Ok(())
}

fn short_hex(bytes: &[u8; 32]) -> String {
    bytes[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn catalog_generation_from_key(key: &str) -> Option<u64> {
    key.strip_prefix("catalogs/")?
        .split_once('-')?
        .0
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::store::remote::build_remote_store;

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("finelog_native_catalog_{tag}_{nanos}"));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn catalog(namespace: &str, generation: u64, active_version: u64) -> NamespaceCatalog {
        NamespaceCatalog {
            format_version: Some(NATIVE_FORMAT_VERSION),
            namespace: Some(namespace.to_string()),
            catalog_generation: Some(generation),
            active_table_spec_version: Some(active_version),
            max_query_time_ms: Some(600_000),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn local_head_cas_publishes_one_complete_generation() {
        let remote_dir = tempdir("cas");
        let remote = build_remote_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let native = NativeCatalog::new(remote);

        let first = native
            .publish("iris.worker", 11, catalog("iris.worker", 1, 1), None)
            .await
            .unwrap();
        let loaded = native.load("iris.worker").await.unwrap().unwrap();
        assert_eq!(loaded.head.writer_epoch, Some(11));
        assert_eq!(loaded.catalog.catalog_generation, Some(1));

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let second = native
            .publish(
                "iris.worker",
                11,
                catalog("iris.worker", 2, 2),
                Some(&first),
            )
            .await
            .unwrap();
        let stale = native
            .publish(
                "iris.worker",
                12,
                catalog("iris.worker", 2, 3),
                Some(&first),
            )
            .await
            .unwrap_err();
        assert!(matches!(stale, StatsError::SchemaConflict(_)));

        let loaded = native.load("iris.worker").await.unwrap().unwrap();
        assert_eq!(loaded.catalog.active_table_spec_version, Some(2));
        assert_eq!(loaded.head.writer_epoch, Some(11));
        assert_eq!(second.catalog, loaded.catalog);

        // A losing writer may leave one immutable, unreachable catalog. HEAD is
        // still the sole visibility boundary, and later GC can remove the orphan.
        let keys = native.catalog_keys("iris.worker").await.unwrap();
        assert_eq!(keys.len(), 3);
        let current_key = second
            .head
            .catalog
            .as_option()
            .unwrap()
            .uri
            .as_deref()
            .unwrap();
        let current_modified_ms = native
            .remote
            .list_native_objects("iris.worker", "catalogs")
            .await
            .unwrap()
            .into_iter()
            .find(|(key, _)| key == current_key)
            .unwrap()
            .1
            .last_modified
            .timestamp_millis();
        assert_eq!(
            native
                .gc_obsolete_catalogs("iris.worker", current_modified_ms + 5, 10)
                .await
                .unwrap(),
            0
        );
        let future = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            + 601_000;
        assert_eq!(
            native
                .gc_obsolete_catalogs("iris.worker", future, 600_000)
                .await
                .unwrap(),
            2
        );
        assert_eq!(native.catalog_keys("iris.worker").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn immutable_catalog_retries_require_identical_bytes() {
        let remote_dir = tempdir("immutable");
        let remote = build_remote_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let native = NativeCatalog::new(remote.clone());
        let published = native
            .publish("iris.worker", 1, catalog("iris.worker", 1, 1), None)
            .await
            .unwrap();
        let key = published
            .head
            .catalog
            .as_option()
            .unwrap()
            .uri
            .as_deref()
            .unwrap();
        let existing = remote
            .get_native("iris.worker", key)
            .await
            .unwrap()
            .unwrap();
        remote
            .put_native_immutable("iris.worker", key, existing.bytes)
            .await
            .unwrap();
        let error = remote
            .put_native_immutable("iris.worker", key, Bytes::from_static(b"different"))
            .await
            .unwrap_err();
        assert!(matches!(error, StatsError::SchemaConflict(_)));
    }

    #[tokio::test]
    async fn garbage_collection_removes_unreferenced_objects_after_query_grace() {
        let remote_dir = tempdir("orphan_gc");
        let remote = build_remote_store(remote_dir.to_str().unwrap())
            .unwrap()
            .unwrap();
        let native = NativeCatalog::new(remote.clone());
        native
            .publish("iris.worker", 1, catalog("iris.worker", 1, 1), None)
            .await
            .unwrap();
        remote
            .put_native_immutable(
                "iris.worker",
                "objects/v1/l0/orphan/source.parquet",
                Bytes::from_static(b"orphan"),
            )
            .await
            .unwrap();

        let future = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            + 601_000;
        assert_eq!(
            native
                .gc_obsolete_catalogs("iris.worker", future, 600_000)
                .await
                .unwrap(),
            1
        );
        assert!(remote
            .get_native("iris.worker", "objects/v1/l0/orphan/source.parquet")
            .await
            .unwrap()
            .is_none());
    }
}
