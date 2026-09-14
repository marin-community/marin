//! The concrete index subsystem: build artifacts for a segment, open the ones a
//! segment advertises.
//!
//! [`IndexRegistry`] is the single owner of the derived-artifact formats. It
//! builds over the closed [`IndexSpec`](crate::indices::IndexSpec) family and
//! opens artifacts by exact reference. It never decides when to build, which
//! segments to index, or when to delete an artifact; those are table-state
//! decisions the controller commits.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::record_batch::RecordBatch;

use crate::indices::cache::IndexCache;
use crate::indices::exact::ExactSection;
use crate::indices::format::{BundleHeader, SectionKind};
use crate::indices::group_extrema::{GroupExtremaConfig, GroupExtremaSection};
use crate::indices::trigram::ColumnIndex;
use crate::indices::{write_segment_index, SegmentIndexConfig, TrigramCoverage};
use crate::store::segment::segment_id_and_row_group_rows;
use crate::store::table_state::LocalArtifacts;

/// The artifacts each snapshotted segment advertises, keyed by the local path
/// of its source Parquet. A query resolves this once from the table state it
/// pinned and never re-derives an artifact filename while scanning.
pub type SegmentArtifacts = BTreeMap<String, LocalArtifacts>;

/// One segment's artifact build: the batches it holds and the policy to apply.
pub struct IndexBuildRequest<'a> {
    /// The local Parquet the artifacts bind to. Its footer supplies the segment
    /// UUID and row count recorded in the FIDX container.
    pub source: &'a Path,
    pub batches: &'a [RecordBatch],
    pub config: &'a SegmentIndexConfig,
}

/// The local files one build produced. Empty when the policy asks for no index
/// or the segment cannot satisfy any configured method.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BuiltArtifacts {
    pub bundle: Option<PathBuf>,
    pub projections: Vec<PathBuf>,
}

impl BuiltArtifacts {
    pub fn is_empty(&self) -> bool {
        self.bundle.is_none() && self.projections.is_empty()
    }
}

/// A validated bundle bound to the source segment that advertised it.
pub struct OpenedIndexes {
    bundle_path: PathBuf,
    pub header: Arc<BundleHeader>,
    pub row_group_rows: Arc<[usize]>,
}

/// A bundle validated from catalog-resident source identity and row count,
/// without opening the source Parquet footer.
pub struct OpenedIndexSummary {
    bundle_path: PathBuf,
    header: Arc<BundleHeader>,
}

impl OpenedIndexes {
    pub fn bundle_path(&self) -> &Path {
        &self.bundle_path
    }
}

pub struct IndexRegistry {
    cache: Arc<IndexCache>,
}

impl std::fmt::Debug for IndexRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("IndexRegistry")
            .field("cache", &self.cache)
            .finish()
    }
}

impl IndexRegistry {
    pub fn new(cache: Arc<IndexCache>) -> Self {
        Self { cache }
    }

    pub fn cache(&self) -> &Arc<IndexCache> {
        &self.cache
    }

    /// Build every artifact `request.config` asks for and return the local files
    /// written. A method the segment cannot satisfy is omitted rather than
    /// failing the build.
    pub fn build(&self, request: IndexBuildRequest<'_>) -> std::io::Result<BuiltArtifacts> {
        write_segment_index(request.source, request.batches, request.config)
    }

    /// Open the bundle `artifacts` advertises for `source`.
    ///
    /// Returns `None` — and the caller scans the source Parquet — when the
    /// segment advertises no bundle, the bundle file is missing or corrupt, or
    /// it is bound to a different physical segment than the one on disk.
    pub fn open(&self, source: &Path, artifacts: &LocalArtifacts) -> Option<OpenedIndexes> {
        let bundle = artifacts.bundle.as_ref()?;
        let (source_id, row_group_rows) = segment_id_and_row_group_rows(source)?;
        let source_rows = row_group_rows.iter().sum::<usize>() as u64;
        let header = self.cache.get_header(bundle, source_id, source_rows)?;
        Some(OpenedIndexes {
            bundle_path: bundle.clone(),
            header,
            row_group_rows,
        })
    }

    /// Open the bundle the segment at `parquet_path` advertises in `artifacts`.
    pub fn open_segment(
        &self,
        parquet_path: &Path,
        artifacts: &SegmentArtifacts,
    ) -> Option<OpenedIndexes> {
        let references = artifacts.get(parquet_path.to_str()?)?;
        self.open(parquet_path, references)
    }

    /// Open an advertised bundle using the immutable source binding carried by
    /// the table state. This is safe for segment-level pruning before the
    /// Parquet scan is planned: a missing or malformed binding declines the
    /// optimization.
    pub fn open_summary(&self, artifacts: &LocalArtifacts) -> Option<OpenedIndexSummary> {
        let bundle_path = artifacts.bundle.clone()?;
        let source_id = uuid::Uuid::parse_str(artifacts.binding.segment_uuid.as_deref()?).ok()?;
        let row_count = u64::try_from(artifacts.binding.row_count).ok()?;
        let header = self.cache.get_header(&bundle_path, source_id, row_count)?;
        Some(OpenedIndexSummary {
            bundle_path,
            header,
        })
    }

    pub fn summary_trigram(
        &self,
        opened: &OpenedIndexSummary,
        column: &str,
    ) -> Option<(TrigramCoverage, Arc<ColumnIndex>)> {
        self.cache
            .get_trigram(&opened.bundle_path, &opened.header, column)
    }

    /// The local covering-projection file `name` resolves to for the segment at
    /// `parquet_path`.
    pub fn projection_file(
        &self,
        parquet_path: &Path,
        artifacts: &SegmentArtifacts,
        name: &str,
    ) -> Option<PathBuf> {
        Some(
            artifacts
                .get(parquet_path.to_str()?)?
                .projections
                .get(name)?
                .clone(),
        )
    }

    pub fn trigram(
        &self,
        opened: &OpenedIndexes,
        column: &str,
    ) -> Option<(TrigramCoverage, Arc<ColumnIndex>)> {
        self.cache
            .get_trigram(&opened.bundle_path, &opened.header, column)
    }

    pub fn exact(&self, opened: &OpenedIndexes, kind: SectionKind) -> Option<Arc<ExactSection>> {
        self.cache
            .get_exact(&opened.bundle_path, &opened.header, kind)
    }

    pub fn group_extrema(
        &self,
        opened: &OpenedIndexes,
        config: &GroupExtremaConfig,
    ) -> Option<Arc<GroupExtremaSection>> {
        self.cache
            .get_group_extrema(&opened.bundle_path, &opened.header, config)
    }

    /// Drop every parsed section cached for a bundle file that was replaced.
    pub fn invalidate(&self, bundle_path: &Path) {
        self.cache.invalidate(bundle_path);
    }
}

/// A registry over a small cache, for tests that build artifacts beside a
/// segment and read them back.
#[cfg(test)]
pub fn test_index_registry() -> Arc<IndexRegistry> {
    Arc::new(IndexRegistry::new(Arc::new(IndexCache::new(16))))
}

/// The artifacts the segments at `paths` carry as local sidecars.
#[cfg(test)]
pub fn sidecar_artifacts<S: AsRef<str>>(paths: &[S]) -> SegmentArtifacts {
    paths
        .iter()
        .map(|path| {
            (
                path.as_ref().to_string(),
                crate::indices::local_sidecar_artifacts(Path::new(path.as_ref())),
            )
        })
        .collect()
}
