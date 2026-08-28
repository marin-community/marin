//! Local query cache for canonical object-store keys.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::store::object_store::{atomic_write_file, ObjectId};

/// Map a canonical object ID to the same relative path below the local root.
pub(crate) fn object_cache_path(
    namespace_dir: &Path,
    table: &str,
    relative_key: &str,
) -> Result<PathBuf, StatsError> {
    let root = namespace_dir.parent().ok_or_else(|| {
        StatsError::Internal(format!(
            "namespace directory {} has no cache root",
            namespace_dir.display()
        ))
    })?;
    relative_cache_path(root, ObjectId::table(table, relative_key)?.as_str())
}

pub(crate) fn table_cache_root(root: &Path, table: &str) -> Result<PathBuf, StatsError> {
    let probe = relative_cache_path(root, ObjectId::table(table, "_")?.as_str())?;
    Ok(probe
        .parent()
        .expect("canonical table object has a parent")
        .to_path_buf())
}

/// Map a legacy archive key below the table's historical local directory.
pub(crate) fn legacy_cache_path(
    namespace_dir: &Path,
    relative_key: &str,
) -> Result<PathBuf, StatsError> {
    relative_cache_path(namespace_dir, relative_key)
}

fn relative_cache_path(root: &Path, relative_key: &str) -> Result<PathBuf, StatsError> {
    let mut path = root.to_path_buf();
    for component in relative_key
        .split('/')
        .filter(|component| !component.is_empty())
    {
        if matches!(component, "." | "..") || component.contains('\\') {
            return Err(StatsError::Internal(format!(
                "object key {relative_key:?} is not a safe relative path"
            )));
        }
        path.push(component);
    }
    Ok(path)
}

/// Write through to the local mirror, accepting an identical retry.
pub(crate) fn write_cache_file(path: &Path, bytes: &[u8]) -> Result<(), StatsError> {
    if path.exists() {
        let existing = std::fs::read(path).map_err(|error| {
            StatsError::Internal(format!("read object cache {}: {error}", path.display()))
        })?;
        if Sha256::digest(&existing).as_slice() == Sha256::digest(bytes).as_slice() {
            return Ok(());
        }
    }
    atomic_write_file(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_cache_mirrors_canonical_layout() {
        let namespace_dir = Path::new("/var/lib/finelog/iris.worker");
        let path = object_cache_path(
            namespace_dir,
            "iris.worker",
            "objects/v2/l1/abc/segment.parquet",
        )
        .unwrap();
        assert_eq!(
            path,
            Path::new(
                "/var/lib/finelog/_finelog/tables/iris.worker/objects/v2/l1/abc/segment.parquet"
            )
        );
    }
}
