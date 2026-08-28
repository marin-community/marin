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
pub use remote::{build_remote_object_store, is_object_store, RemoteObjectStore};

use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;

use crate::errors::StatsError;
use crate::proto::finelog::stats::ObjectRef as ProtoObjectRef;

pub const FINELOG_ROOT_COMPONENT: &str = "_finelog";
pub const TABLES_COMPONENT: &str = "tables";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredObject {
    pub bytes: bytes::Bytes,
    pub version: ObjectVersion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectVersion {
    pub e_tag: Option<String>,
    pub provider_version: Option<String>,
    pub content_sha256: [u8; 32],
    pub byte_size: u64,
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
        let content_sha256: [u8; 32] = reference
            .sha256
            .as_deref()
            .ok_or_else(|| StatsError::Internal("object reference has no SHA-256".to_string()))?
            .try_into()
            .map_err(|_| {
                StatsError::Internal("object reference SHA-256 is not 32 bytes".to_string())
            })?;
        Ok(Self {
            id,
            version: ObjectVersion {
                e_tag: reference.etag.clone(),
                provider_version: reference.provider_version.clone(),
                content_sha256,
                byte_size: reference.byte_size.ok_or_else(|| {
                    StatsError::Internal("object reference has no byte size".to_string())
                })?,
            },
        })
    }
}

/// Complete, validated key relative to the configured object-store root.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObjectId(String);

impl ObjectId {
    pub fn parse(value: &str) -> Result<Self, StatsError> {
        let value = validate_relative_key(value)?;
        let mut components = value.split('/');
        if components.next() != Some(FINELOG_ROOT_COMPONENT)
            || components.next() != Some(TABLES_COMPONENT)
        {
            return Err(StatsError::Internal(format!(
                "object ID {value:?} is outside {FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}"
            )));
        }
        let table = components
            .next()
            .ok_or_else(|| StatsError::Internal(format!("object ID {value:?} has no table")))?;
        validate_component(table, "table")?;
        if components.next().is_none() {
            return Err(StatsError::Internal(format!(
                "object ID {value:?} has no table-relative key"
            )));
        }
        Ok(Self(value.to_string()))
    }

    pub fn table(table: &str, relative_key: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let relative_key = validate_relative_key(relative_key)?;
        Ok(Self(format!(
            "{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}/{relative_key}"
        )))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn table_relative<'a>(&'a self, table: &str) -> Option<&'a str> {
        self.0.strip_prefix(&format!(
            "{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}/"
        ))
    }

    pub fn table_name(&self) -> &str {
        self.0
            .split('/')
            .nth(2)
            .expect("validated object IDs contain a table")
    }

    pub fn relative_key(&self) -> &str {
        self.0
            .splitn(4, '/')
            .nth(3)
            .expect("validated object IDs contain a table-relative key")
    }
}

/// Validated key prefix used only for listings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectPrefix(String);

impl ObjectPrefix {
    pub fn table(table: &str, relative_prefix: &str) -> Result<Self, StatsError> {
        validate_component(table, "table")?;
        let suffix = validate_relative_prefix(relative_prefix)?;
        let root = format!("{FINELOG_ROOT_COMPONENT}/{TABLES_COMPONENT}/{table}");
        Ok(Self(if suffix.is_empty() {
            root
        } else {
            format!("{root}/{suffix}")
        }))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn table_name(&self) -> &str {
        self.0
            .split('/')
            .nth(2)
            .expect("validated object prefixes contain a table")
    }

    pub fn relative_prefix(&self) -> &str {
        self.0.splitn(4, '/').nth(3).unwrap_or("")
    }
}

#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub id: ObjectId,
    pub modified_at_ms: i64,
}

pub type ObjectByteStream = BoxStream<'static, Result<bytes::Bytes, StatsError>>;

/// Opaque persistence boundary for immutable data and mutable catalog pointers.
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn write(&self, id: &ObjectId, bytes: bytes::Bytes) -> Result<ObjectVersion, StatsError>;

    async fn write_stream(
        &self,
        id: &ObjectId,
        mut stream: ObjectByteStream,
    ) -> Result<ObjectVersion, StatsError> {
        let mut bytes = bytes::BytesMut::new();
        while let Some(chunk) = stream.next().await {
            bytes.extend_from_slice(&chunk?);
        }
        self.write(id, bytes.freeze()).await
    }

    async fn read(&self, id: &ObjectId) -> Result<Option<StoredObject>, StatsError>;

    /// Return a verified local file usable by DataFusion.
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

    async fn compare_and_swap(
        &self,
        id: &ObjectId,
        expected: Option<&ObjectVersion>,
        bytes: bytes::Bytes,
    ) -> Result<ObjectVersion, StatsError>;

    async fn delete(&self, id: &ObjectId) -> Result<(), StatsError>;

    async fn list(&self, prefix: &ObjectPrefix) -> Result<Vec<ObjectMetadata>, StatsError>;

    async fn list_tables(&self) -> Result<Vec<String>, StatsError> {
        Ok(Vec::new())
    }

    /// Reclaim implementation-owned state. Implementations may choose to retain
    /// everything; `CachedObjectStore` does so until an eviction policy exists.
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
    validate_relative_prefix(value)?;
    Ok(value.trim_matches('/'))
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
