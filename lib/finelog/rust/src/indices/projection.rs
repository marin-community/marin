//! Covering-projection descriptors referenced from a segment's FIDX bundle.
//!
//! A covering projection is a narrow Parquet file holding the rows a predicate
//! selects. The bundle stores only the descriptor; the Parquet file itself is a
//! separate artifact object the segment references.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::indices::exact::{self, ProjectionDescriptor};
use crate::indices::format;
use crate::store::schema::CoveringProjection;

/// Row identity postings use inside a covering projection.
pub const SOURCE_ROW_OFFSET_IDENTITY: &str = "source_segment_row_offset";

/// Header-resident descriptor for an external covering-projection Parquet file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionReference {
    pub spec_id: String,
    pub descriptor: ProjectionDescriptor,
    pub file_segment_id: String,
    pub file_bytes: u64,
    /// Postings use zero-based row offsets in the bound source segment.
    pub row_identity: String,
}

pub fn projection_section_id(name: &str) -> String {
    std::format!("projection:{name}")
}

pub fn parse_projection_reference(bytes: &[u8]) -> Option<ProjectionReference> {
    serde_json::from_slice(bytes).ok()
}

/// Content identity of one projection specification, so a policy change is
/// detectable without reopening the projection file.
pub fn projection_spec_id(projection: &CoveringProjection) -> String {
    let digest = format::fingerprint(
        &serde_json::to_vec(projection).expect("projection serialization never fails"),
    );
    crate::hex::encode(&digest)
}

/// Local file a built projection was staged to.
pub fn projection_path(parquet_path: &Path, reference: &ProjectionReference) -> PathBuf {
    exact::named_projection_path(parquet_path, &reference.descriptor.name)
}

fn covering_projection_paths_with_suffix(
    parquet_path: &Path,
    suffix: &str,
) -> std::io::Result<Vec<PathBuf>> {
    let (Some(directory), Some(file_name)) = (parquet_path.parent(), parquet_path.file_name())
    else {
        return Ok(Vec::new());
    };
    let prefix = std::format!(
        "{}{}",
        file_name.to_string_lossy(),
        exact::NAMED_PROJECTION_MARKER
    );
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with(&prefix) && name.ends_with(suffix) {
            paths.push(path);
        }
    }
    Ok(paths)
}

/// Projection files staged beside `parquet_path` by a completed build.
pub fn covering_projection_paths(parquet_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    covering_projection_paths_with_suffix(parquet_path, ".parquet")
}

/// Projection files left behind by an interrupted build.
pub fn covering_projection_staging_paths(parquet_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    covering_projection_paths_with_suffix(parquet_path, ".parquet.tmp")
}
