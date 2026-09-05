//! Provider-independent object persistence contract.
//!
//! Callers use validated logical IDs and object versions. Concrete provider
//! paths, local cache membership, and legacy layout translation are private to
//! the implementations of [`ObjectStore`].

mod cached;
mod legacy;
mod local_file;
mod provider;
mod remote;

pub use cached::CachedObjectStore;
pub use legacy::LegacyObjectStore;
pub use provider::is_remote_object_store;
pub use remote::{build_remote_object_store, RemoteObjectStore};

use std::path::PathBuf;

use crate::errors::StatsError;
use crate::proto::finelog::stats::ObjectRef as ProtoObjectRef;
use async_trait::async_trait;

pub const FINELOG_ROOT_COMPONENT: &str = "_finelog";
pub const TABLES_COMPONENT: &str = "tables";

/// Key prefix under a table for the data objects its states reference.
pub const OBJECTS_PREFIX: &str = "objects";
/// Key prefix under a table for segment index bundles.
pub(crate) const INDICES_PREFIX: &str = "indices";
/// Key prefix under a table for covering-projection artifacts.
pub(crate) const PROJECTIONS_PREFIX: &str = "projections";
/// The two components above joined: the prefix every table-scoped key carries.
const TABLE_ROOT_PREFIX: &str = "_finelog/tables/";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    pub bytes: bytes::Bytes,
    pub version: ObjectVersion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectVersion {
    pub e_tag: Option<String>,
    pub provider_version: Option<String>,
    pub byte_size: u64,
    /// Exact bytes for a mutable pointer read from a local provider.
    ///
    /// Remote providers compare their opaque version or ETag. The local
    /// provider has neither, so its locked compare-and-swap compares this
    /// value directly. It shares the read buffer and lives only as long as the
    /// returned object version; durable references never retain it.
    pub(crate) local_value: Option<bytes::Bytes>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectReference {
    pub id: ObjectId,
    pub version: ObjectVersion,
}

impl TryFrom<&ProtoObjectRef> for ObjectReference {
    type Error = StatsError;

    fn try_from(reference: &ProtoObjectRef) -> Result<Self, Self::Error> {
        let id = ObjectId::parse(reference.object_id.as_deref().ok_or_else(|| {
            StatsError::Internal("object reference has no object ID".to_string())
        })?)?;
        Ok(Self {
            id,
            version: ObjectVersion {
                e_tag: reference.etag.clone(),
                provider_version: reference.provider_version.clone(),
                byte_size: reference.byte_size.ok_or_else(|| {
                    StatsError::Internal("object reference has no byte size".to_string())
                })?,
                local_value: None,
            },
        })
    }
}

/// Complete, validated key relative to the configured object-store root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(String);

impl ObjectId {
    pub fn parse(value: &str) -> Result<Self, StatsError> {
        validate_relative_prefix(value)?;
        let rest = value.strip_prefix(TABLE_ROOT_PREFIX).ok_or_else(|| {
            StatsError::Internal(format!(
                "object ID {value:?} is outside {TABLE_ROOT_PREFIX}"
            ))
        })?;
        if !rest.contains('/') {
            return Err(StatsError::Internal(format!(
                "object ID {value:?} has no table-relative key"
            )));
        }
        Ok(Self(value.to_string()))
    }

    pub fn table(table: &str, relative_key: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let relative_key = validate_relative_key(relative_key)?;
        Ok(Self(format!("{TABLE_ROOT_PREFIX}{table}/{relative_key}")))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The `(table, table-relative key)` pair every constructor guarantees.
    fn split(&self) -> (&str, &str) {
        self.0[TABLE_ROOT_PREFIX.len()..]
            .split_once('/')
            .expect("object IDs carry a table and key by construction")
    }

    pub fn table_relative<'a>(&'a self, table: &str) -> Option<&'a str> {
        let (name, key) = self.split();
        (name == table).then_some(key)
    }

    pub fn table_name(&self) -> &str {
        self.split().0
    }

    pub fn relative_key(&self) -> &str {
        self.split().1
    }
}

/// Validated key prefix used only for listings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectPrefix(String);

impl ObjectPrefix {
    pub fn table(table: &str, relative_prefix: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let suffix = validate_relative_prefix(relative_prefix)?;
        let root = format!("{TABLE_ROOT_PREFIX}{table}");
        Ok(Self(if suffix.is_empty() {
            root
        } else {
            format!("{root}/{suffix}")
        }))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The `(table, relative prefix)` pair the constructor guarantees; the
    /// prefix is empty for a whole-table listing.
    fn split(&self) -> (&str, &str) {
        let rest = &self.0[TABLE_ROOT_PREFIX.len()..];
        rest.split_once('/').unwrap_or((rest, ""))
    }

    pub fn table_name(&self) -> &str {
        self.split().0
    }

    pub fn relative_prefix(&self) -> &str {
        self.split().1
    }
}

#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub id: ObjectId,
    pub modified_at_ms: i64,
}

/// Opaque persistence boundary for immutable data and mutable catalog pointers.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError>;

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError>;

    /// Make `bytes` locally durable under `id` for a later
    /// [`ObjectStore::upload_staged`]. A store without local staging uploads
    /// immediately instead, so callers get remote durability either way.
    async fn stage(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError> {
        self.write(id, bytes).await
    }

    /// Make a staged object remotely durable. A store without local staging
    /// already uploaded it at [`ObjectStore::stage`].
    async fn upload_staged(&self, reference: &ObjectReference) -> Result<(), StatsError> {
        let _ = reference;
        Ok(())
    }

    /// Return a local file usable by DataFusion.
    async fn local_path(&self, reference: &ObjectReference) -> Result<PathBuf, StatsError> {
        Err(StatsError::Internal(format!(
            "object store has no local file for {:?}",
            reference.id.as_str()
        )))
    }

    /// The filename [`ObjectStore::local_path`] resolves this object to.
    ///
    /// Derived from the object's identity alone: recovery names objects it has
    /// not localized without reading or downloading them.
    fn planned_local_path(&self, id: &ObjectId) -> Result<PathBuf, StatsError> {
        Err(StatsError::Internal(format!(
            "object store has no local file for {:?}",
            id.as_str()
        )))
    }

    /// The URL a registered query-engine scan reads this object from directly,
    /// when the backing provider supports remote scans.
    fn remote_scan_url(&self, _id: &ObjectId) -> Option<String> {
        None
    }

    /// The local cache file for `reference` when one is already
    /// present, without fetching anything.
    async fn cached_path(
        &self,
        _reference: &ObjectReference,
    ) -> Result<Option<PathBuf>, StatsError> {
        Ok(None)
    }

    /// Begin materializing `reference` into the local cache without waiting
    /// for it. A store without a cache does nothing.
    fn warm(&self, _reference: &ObjectReference) {}

    /// Conditionally replace a mutable pointer object. Stores whose objects
    /// are all immutable reject this.
    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        _expected: Option<&ObjectVersion>,
        _bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError> {
        Err(StatsError::Internal(format!(
            "object store cannot conditionally write {:?}",
            id.as_str()
        )))
    }

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError>;

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError>;

    /// Enumerate the tables under `_finelog/tables`. Required so a store that
    /// cannot enumerate reports so instead of silently discovering nothing.
    async fn list_tables(&self) -> Result<Vec<String>, StatsError>;

    /// Reclaim implementation-owned state. Implementations may retain
    /// everything; `CachedObjectStore` evicts least-recently-used cache files
    /// when a capacity is configured.
    async fn gc(&self) -> Result<(), StatsError> {
        Ok(())
    }
}

fn validate_component(value: &str, description: &str) -> Result<(), StatsError> {
    if value.is_empty()
        || matches!(value, "." | "..")
        || value.contains('/')
        || value.contains('\\')
    {
        return Err(StatsError::InvalidNamespace(format!(
            "invalid object {description} {value:?}"
        )));
    }
    Ok(())
}

fn validate_relative_key(value: &str) -> Result<&str, StatsError> {
    if value.is_empty() {
        return Err(StatsError::Internal(
            "object key must not be empty".to_string(),
        ));
    }
    validate_relative_prefix(value)
}

fn validate_relative_prefix(value: &str) -> Result<&str, StatsError> {
    if value != value.trim_matches('/')
        || (!value.is_empty() && value.split('/').any(str::is_empty))
    {
        return Err(StatsError::Internal(format!(
            "object key {value:?} is not canonical"
        )));
    }
    for component in value.split('/').filter(|component| !component.is_empty()) {
        validate_component(component, "key component")?;
    }
    Ok(value)
}
