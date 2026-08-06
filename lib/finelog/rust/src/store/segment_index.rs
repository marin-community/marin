//! Build and read the typed index bundle for one immutable Parquet segment.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::store::exact::{
    self, ExactColumn, ExactIndexConfig, ExactSection, ProjectionDescriptor,
};
use crate::store::group_extrema::{self, GroupExtremaConfig, GroupExtremaSection};
use crate::store::index_bundle::{self, Exactness, SectionInput, SectionKind, SegmentBinding};
use crate::store::schema::CoveringProjection;
use crate::store::segment::{segment_id, segment_id_and_row_group_rows};
use crate::store::trigram::{self, TrigramIndex, SIDECAR_SPAN_ROWS};

const TRIGRAM_METHOD_VERSION: u8 = 1;
const EXACT_POSTINGS_METHOD_VERSION: u8 = 1;
const VALUE_COUNTS_METHOD_VERSION: u8 = 1;
const PROJECTION_METHOD_VERSION: u8 = 1;
const GROUP_EXTREMA_METHOD_VERSION: u8 = 1;
const EXACT_POSTINGS_SECTION_ID: &str = "exact-postings";
const VALUE_COUNTS_SECTION_ID: &str = "value-counts";
pub const SOURCE_ROW_OFFSET_IDENTITY: &str = "source_segment_row_offset";

/// Complete secondary-index policy for a namespace segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SegmentIndexConfig {
    pub indexes: Vec<IndexSpec>,
    pub key_column: Option<String>,
}

/// Closed planner-facing family of index methods Finelog knows how to build.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum IndexSpec {
    TrigramBloom { column: String },
    ExactPostings { column: String, values: Vec<String> },
    ValueCounts { column: String },
    AdaptiveValueCounts { column: String },
    AdaptiveGroupExtrema { config: GroupExtremaConfig },
    CoveringProjection { projection: CoveringProjection },
}

impl SegmentIndexConfig {
    pub fn from_policies(
        trigram_columns: impl IntoIterator<Item = impl Into<String>>,
        exact: &[ExactIndexConfig],
        projections: &[CoveringProjection],
        key_column: Option<String>,
    ) -> Self {
        let mut indexes = Vec::new();
        indexes.extend(
            trigram_columns
                .into_iter()
                .map(|column| IndexSpec::TrigramBloom {
                    column: column.into(),
                }),
        );
        for config in exact {
            if !config.exact_values.is_empty() {
                indexes.push(IndexSpec::ExactPostings {
                    column: config.column.clone(),
                    values: config.exact_values.clone(),
                });
            }
            if config.value_counts {
                indexes.push(IndexSpec::ValueCounts {
                    column: config.column.clone(),
                });
            }
        }
        indexes.extend(
            projections
                .iter()
                .cloned()
                .map(|projection| IndexSpec::CoveringProjection { projection }),
        );
        Self {
            indexes,
            key_column,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    pub fn with_adaptive_value_counts(
        mut self,
        columns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let required: BTreeSet<String> = self
            .indexes
            .iter()
            .filter_map(|index| match index {
                IndexSpec::ValueCounts { column } => Some(column.clone()),
                _ => None,
            })
            .collect();
        self.indexes.extend(
            columns
                .into_iter()
                .map(Into::into)
                .filter(|column| !required.contains(column))
                .map(|column| IndexSpec::AdaptiveValueCounts { column }),
        );
        self
    }

    pub fn with_adaptive_group_extrema(
        mut self,
        configs: impl IntoIterator<Item = GroupExtremaConfig>,
    ) -> Self {
        self.indexes.extend(
            configs
                .into_iter()
                .map(|config| IndexSpec::AdaptiveGroupExtrema { config }),
        );
        self
    }

    /// Columns a one-pass backfill must read from the source segment.
    pub fn input_columns(&self) -> Vec<&str> {
        let mut columns: BTreeSet<&str> = BTreeSet::new();
        for index in &self.indexes {
            match index {
                IndexSpec::TrigramBloom { column }
                | IndexSpec::ExactPostings { column, .. }
                | IndexSpec::ValueCounts { column }
                | IndexSpec::AdaptiveValueCounts { column } => {
                    columns.insert(column.as_str());
                }
                IndexSpec::AdaptiveGroupExtrema { config } => {
                    columns.insert(config.filter_column.as_str());
                    columns.insert(config.json_column.as_str());
                    columns.insert(config.extrema_column.as_str());
                }
                IndexSpec::CoveringProjection { projection } => {
                    columns.extend(projection.columns.iter().map(String::as_str));
                }
            }
        }
        if let Some(key) = self.key_column.as_deref() {
            columns.insert(key);
        }
        columns.into_iter().collect()
    }

    pub fn policy_fingerprint(&self) -> [u8; 32] {
        index_bundle::fingerprint(
            &serde_json::to_vec(self).expect("segment index policy serialization never fails"),
        )
    }

    fn exact_configs(&self) -> Vec<ExactIndexConfig> {
        let mut configs = BTreeMap::new();
        for index in &self.indexes {
            let (column, values, value_counts): (&str, &[String], bool) = match index {
                IndexSpec::ExactPostings { column, values } => {
                    (column.as_str(), values.as_slice(), false)
                }
                IndexSpec::ValueCounts { column } | IndexSpec::AdaptiveValueCounts { column } => {
                    (column.as_str(), &[], true)
                }
                IndexSpec::CoveringProjection { projection } => (
                    projection.predicate_column.as_str(),
                    projection.predicate_values.as_slice(),
                    false,
                ),
                IndexSpec::TrigramBloom { .. } => continue,
                IndexSpec::AdaptiveGroupExtrema { .. } => continue,
            };
            let config = configs
                .entry(column.to_string())
                .or_insert_with(|| ExactIndexConfig {
                    column: column.to_string(),
                    exact_values: Vec::new(),
                    value_counts: false,
                });
            config.exact_values.extend(values.iter().cloned());
            config.value_counts |= value_counts;
        }
        for config in configs.values_mut() {
            config.exact_values.sort();
            config.exact_values.dedup();
        }
        configs.into_values().collect()
    }

    fn projections(&self) -> impl Iterator<Item = &CoveringProjection> {
        self.indexes.iter().filter_map(|index| match index {
            IndexSpec::CoveringProjection { projection } => Some(projection),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentIndexWrite {
    Written,
    NotApplicable,
}

/// Header-resident coverage for one trigram section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrigramCoverage {
    pub column: String,
    pub span_count: u32,
    pub span_rows: u32,
    pub key_column: String,
    pub key_min: Option<Vec<u8>>,
    pub key_max: Option<Vec<u8>>,
}

impl TrigramCoverage {
    pub fn key_band_overlaps(&self, lo: Option<&[u8]>, hi: Option<&[u8]>) -> bool {
        let (Some(key_min), Some(key_max)) = (self.key_min.as_deref(), self.key_max.as_deref())
        else {
            return true;
        };
        !hi.is_some_and(|value| value < key_min) && !lo.is_some_and(|value| value > key_max)
    }
}

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

pub fn trigram_section_id(column: &str) -> String {
    format!("trigram:{column}")
}

pub fn projection_section_id(name: &str) -> String {
    format!("projection:{name}")
}

pub fn parse_trigram_coverage(bytes: &[u8]) -> Option<TrigramCoverage> {
    serde_json::from_slice(bytes).ok()
}

pub fn parse_projection_reference(bytes: &[u8]) -> Option<ProjectionReference> {
    serde_json::from_slice(bytes).ok()
}

fn projection_spec_id(projection: &CoveringProjection) -> String {
    let digest = index_bundle::fingerprint(
        &serde_json::to_vec(projection).expect("projection serialization never fails"),
    );
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to a string never fails");
    }
    out
}

pub fn read_exact_section(
    bundle_path: &Path,
    header: &index_bundle::BundleHeader,
    kind: SectionKind,
) -> Option<ExactSection> {
    let section = header
        .sections
        .iter()
        .find(|section| section.kind == kind)?;
    exact::parse(&index_bundle::read_section(
        bundle_path,
        header,
        &section.id,
    )?)
}

fn section_covers_column(
    header: &index_bundle::BundleHeader,
    section_id: &str,
    column: &str,
) -> bool {
    header
        .section(section_id)
        .and_then(|section| std::str::from_utf8(&section.coverage).ok())
        .is_some_and(|columns| columns.split('\0').any(|covered| covered == column))
}

/// Whether a segment lacks a complete bundle for the current policy.
pub fn needs_rebuild(parquet_path: &Path, expected_rows: i64, config: &SegmentIndexConfig) -> bool {
    if config.is_empty() {
        return false;
    }
    let Some((source_id, row_groups)) = segment_id_and_row_group_rows(parquet_path) else {
        return true;
    };
    let source_rows = row_groups.iter().sum::<usize>() as u64;
    let path = index_bundle::bundle_path(parquet_path);
    let Some(header) = index_bundle::read_header(&path) else {
        return true;
    };
    if i64::try_from(source_rows).ok() != Some(expected_rows)
        || !header.matches(source_id, source_rows)
        || header.binding.policy_fingerprint != config.policy_fingerprint()
    {
        return true;
    }
    for index in &config.indexes {
        let covered = match index {
            IndexSpec::TrigramBloom { column } => {
                header.section(&trigram_section_id(column)).is_some()
            }
            IndexSpec::ExactPostings { column, .. } => {
                section_covers_column(&header, EXACT_POSTINGS_SECTION_ID, column)
            }
            IndexSpec::ValueCounts { column } => {
                section_covers_column(&header, VALUE_COUNTS_SECTION_ID, column)
            }
            IndexSpec::AdaptiveValueCounts { .. } | IndexSpec::CoveringProjection { .. } => true,
            IndexSpec::AdaptiveGroupExtrema { .. } => true,
        };
        if !covered {
            return true;
        }
    }
    for projection in config.projections() {
        let Some(section) = header.section(&projection_section_id(&projection.name)) else {
            return true;
        };
        let Some(reference) = parse_projection_reference(&section.coverage) else {
            return true;
        };
        if reference.spec_id != projection_spec_id(projection) {
            return true;
        }
        let path = projection_path(parquet_path, &reference);
        if std::fs::metadata(&path).ok().map(|metadata| metadata.len())
            != Some(reference.file_bytes)
            || segment_id(&path).map(|id| id.to_string()) != Some(reference.file_segment_id)
        {
            return true;
        }
    }
    false
}

/// Build every configured index from `batches` and publish the bundle last.
pub fn write_segment_index(
    parquet_path: &Path,
    batches: &[RecordBatch],
    config: &SegmentIndexConfig,
) -> std::io::Result<SegmentIndexWrite> {
    if config.is_empty() {
        return Ok(SegmentIndexWrite::NotApplicable);
    }
    let started = Instant::now();
    let Some(first) = batches.first() else {
        return Ok(SegmentIndexWrite::NotApplicable);
    };
    let total_rows = batches.iter().try_fold(0_u64, |total, batch| {
        total.checked_add(batch.num_rows() as u64)
    });
    let Some(total_rows) = total_rows else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "segment row count overflow",
        ));
    };
    let (source_segment_id, _) = segment_id_and_row_group_rows(parquet_path).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot read source segment footer for index build: {}",
                parquet_path.display()
            ),
        )
    })?;

    let mut sections = build_trigram_sections(batches, config);
    append_group_extrema_sections(&mut sections, batches, config);
    let mut referenced_projection_paths = BTreeSet::new();
    let exact_configs = config.exact_configs();
    let exact = exact::build_sidecar(batches, &exact_configs);
    if let Some(sidecar) = exact.as_ref() {
        append_exact_sections(&mut sections, sidecar);
        for projection in config.projections() {
            let descriptor = match exact::write_covering_projection(
                parquet_path,
                batches,
                sidecar,
                projection,
            ) {
                Ok(descriptor) => descriptor,
                Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => {
                    tracing::debug!(projection = %projection.name, %error, "segment cannot satisfy covering projection");
                    continue;
                }
                Err(error) => return Err(error),
            };
            let projection_path = exact::named_projection_path(parquet_path, &projection.name);
            referenced_projection_paths.insert(projection_path.clone());
            let Some(projection_segment_id) = segment_id(&projection_path) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("projection {:?} has no segment identity", projection.name),
                ));
            };
            let reference = ProjectionReference {
                spec_id: projection_spec_id(projection),
                descriptor,
                file_segment_id: projection_segment_id.to_string(),
                file_bytes: std::fs::metadata(&projection_path)?.len(),
                row_identity: SOURCE_ROW_OFFSET_IDENTITY.to_string(),
            };
            sections.push(SectionInput {
                id: projection_section_id(&projection.name),
                kind: SectionKind::CoveringProjection,
                method_version: PROJECTION_METHOD_VERSION,
                exactness: Exactness::Covering,
                coverage: serde_json::to_vec(&reference)
                    .expect("projection descriptor serialization never fails"),
                payload: Vec::new(),
            });
        }
    }
    if sections.is_empty() {
        remove_if_exists(&index_bundle::bundle_path(parquet_path))?;
        remove_if_exists(&index_bundle::staging_path(parquet_path))?;
        remove_unreferenced_projections(parquet_path, &referenced_projection_paths);
        remove_legacy_artifacts(parquet_path);
        tracing::info!(
            path = %parquet_path.display(),
            rows = total_rows,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "segment index produced no applicable sections"
        );
        return Ok(SegmentIndexWrite::NotApplicable);
    }
    sections.sort_by(|left, right| left.id.cmp(&right.id));
    let schema_fingerprint = index_bundle::fingerprint(format!("{:?}", first.schema()).as_bytes());
    let binding = SegmentBinding {
        segment_id: source_segment_id,
        row_count: total_rows,
        schema_fingerprint,
        policy_fingerprint: config.policy_fingerprint(),
    };
    index_bundle::write_bundle(parquet_path, &binding, &sections)?;
    let bundle_bytes = std::fs::metadata(index_bundle::bundle_path(parquet_path))?.len();
    let external_bytes = referenced_projection_paths.iter().try_fold(
        0_u64,
        |total, path| -> std::io::Result<u64> {
            Ok(total.saturating_add(std::fs::metadata(path)?.len()))
        },
    )?;
    remove_unreferenced_projections(parquet_path, &referenced_projection_paths);
    remove_legacy_artifacts(parquet_path);
    tracing::info!(
        path = %parquet_path.display(),
        rows = total_rows,
        sections = sections.len(),
        bundle_bytes,
        external_bytes,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "segment index built"
    );
    Ok(SegmentIndexWrite::Written)
}

fn build_trigram_sections(
    batches: &[RecordBatch],
    config: &SegmentIndexConfig,
) -> Vec<SectionInput> {
    let indexes: Vec<TrigramIndex> = config
        .indexes
        .iter()
        .filter_map(|index| match index {
            IndexSpec::TrigramBloom { column } => TrigramIndex::build(batches, column),
            _ => None,
        })
        .filter(|index| !index.is_empty())
        .collect();
    let (key_min, key_max) = config
        .key_column
        .as_deref()
        .map(|column| trigram::string_key_bounds(batches, column))
        .unwrap_or((None, None));
    indexes
        .into_iter()
        .map(|index| {
            let coverage = TrigramCoverage {
                column: index.column().to_string(),
                span_count: index.len() as u32,
                span_rows: SIDECAR_SPAN_ROWS as u32,
                key_column: config.key_column.clone().unwrap_or_default(),
                key_min: key_min.clone(),
                key_max: key_max.clone(),
            };
            SectionInput {
                id: trigram_section_id(index.column()),
                kind: SectionKind::TrigramBloom,
                method_version: TRIGRAM_METHOD_VERSION,
                exactness: Exactness::Lossy,
                coverage: serde_json::to_vec(&coverage)
                    .expect("trigram coverage serialization never fails"),
                payload: trigram::serialize_index(&index),
            }
        })
        .collect()
}

fn group_extrema_section_id(config: &GroupExtremaConfig) -> String {
    format!(
        "group-extrema:{}:{}:{}:{}",
        config.filter_column, config.json_column, config.json_key, config.extrema_column
    )
}

fn append_group_extrema_sections(
    sections: &mut Vec<SectionInput>,
    batches: &[RecordBatch],
    config: &SegmentIndexConfig,
) {
    for method in &config.indexes {
        let IndexSpec::AdaptiveGroupExtrema { config } = method else {
            continue;
        };
        let Some(section) = group_extrema::build(batches, config) else {
            continue;
        };
        let Some(payload) = group_extrema::serialize(&section) else {
            continue;
        };
        sections.push(SectionInput {
            id: group_extrema_section_id(config),
            kind: SectionKind::GroupExtrema,
            method_version: GROUP_EXTREMA_METHOD_VERSION,
            exactness: Exactness::ExactAggregate,
            coverage: serde_json::to_vec(config)
                .expect("group-extrema coverage serialization never fails"),
            payload,
        });
    }
}

pub fn parse_group_extrema_config(bytes: &[u8]) -> Option<GroupExtremaConfig> {
    serde_json::from_slice(bytes).ok()
}

pub fn read_group_extrema_section(
    bundle_path: &Path,
    header: &index_bundle::BundleHeader,
    config: &GroupExtremaConfig,
) -> Option<GroupExtremaSection> {
    let id = group_extrema_section_id(config);
    let section = header.section(&id)?;
    if section.kind != SectionKind::GroupExtrema
        || parse_group_extrema_config(&section.coverage).as_ref() != Some(config)
    {
        return None;
    }
    group_extrema::parse(&index_bundle::read_section(bundle_path, header, &id)?)
}

fn append_exact_sections(sections: &mut Vec<SectionInput>, sidecar: &ExactSection) {
    let postings = ExactSection {
        total_rows: sidecar.total_rows,
        columns: sidecar
            .columns
            .iter()
            .filter(|(_, column)| !column.rows.is_empty())
            .map(|(name, column)| {
                (
                    name.clone(),
                    ExactColumn {
                        counts: None,
                        rows: column.rows.clone(),
                    },
                )
            })
            .collect(),
    };
    if !postings.columns.is_empty() {
        sections.push(SectionInput {
            id: EXACT_POSTINGS_SECTION_ID.to_string(),
            kind: SectionKind::ExactPostings,
            method_version: EXACT_POSTINGS_METHOD_VERSION,
            exactness: Exactness::ExactRows,
            coverage: postings
                .columns
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join("\0")
                .into_bytes(),
            payload: exact::serialize(&postings),
        });
    }

    let counts = ExactSection {
        total_rows: sidecar.total_rows,
        columns: sidecar
            .columns
            .iter()
            .filter_map(|(name, column)| {
                column.counts.as_ref().map(|counts| {
                    (
                        name.clone(),
                        ExactColumn {
                            counts: Some(counts.clone()),
                            rows: BTreeMap::new(),
                        },
                    )
                })
            })
            .collect(),
    };
    if !counts.columns.is_empty() {
        sections.push(SectionInput {
            id: VALUE_COUNTS_SECTION_ID.to_string(),
            kind: SectionKind::ValueCounts,
            method_version: VALUE_COUNTS_METHOD_VERSION,
            exactness: Exactness::ExactAggregate,
            coverage: counts
                .columns
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join("\0")
                .into_bytes(),
            payload: exact::serialize(&counts),
        });
    }
}

pub fn legacy_artifact_paths(parquet_path: &Path) -> [PathBuf; 3] {
    [
        trigram::sidecar_path(parquet_path),
        exact::sidecar_path(parquet_path),
        exact::projection_path(parquet_path),
    ]
}

fn covering_projection_paths_with_suffix(
    parquet_path: &Path,
    suffix: &str,
) -> std::io::Result<Vec<PathBuf>> {
    let (Some(directory), Some(file_name)) = (parquet_path.parent(), parquet_path.file_name())
    else {
        return Ok(Vec::new());
    };
    let prefix = format!(
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

pub fn covering_projection_paths(parquet_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    covering_projection_paths_with_suffix(parquet_path, ".parquet")
}

pub fn covering_projection_staging_paths(parquet_path: &Path) -> std::io::Result<Vec<PathBuf>> {
    covering_projection_paths_with_suffix(parquet_path, ".parquet.tmp")
}

pub fn remove_if_exists(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn remove_legacy_artifacts(parquet_path: &Path) {
    for path in legacy_artifact_paths(parquet_path) {
        if let Err(error) = remove_if_exists(&path) {
            tracing::warn!(path = %path.display(), %error, "failed to remove legacy index artifact");
        }
    }
}

fn remove_unreferenced_projections(parquet_path: &Path, referenced: &BTreeSet<PathBuf>) {
    let paths = match covering_projection_paths(parquet_path) {
        Ok(paths) => paths,
        Err(error) => {
            tracing::warn!(path = %parquet_path.display(), %error, "failed to enumerate covering projections");
            return;
        }
    };
    for path in paths {
        if referenced.contains(&path) {
            continue;
        }
        match remove_if_exists(&path) {
            Ok(()) => {
                tracing::debug!(path = %path.display(), "removed unreferenced covering projection")
            }
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "failed to remove unreferenced covering projection")
            }
        }
    }
}

pub fn projection_path(parquet_path: &Path, reference: &ProjectionReference) -> PathBuf {
    exact::named_projection_path(parquet_path, &reference.descriptor.name)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    use super::*;
    use crate::store::segment::write_segment_to_dir;

    struct Fixture {
        parquet: PathBuf,
        batch: RecordBatch,
        config: SegmentIndexConfig,
    }

    fn fixture() -> Fixture {
        let directory = crate::test_support::unique_dir("segment_index_bundle");
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("timestamp_ms", DataType::Int64, false),
            Field::new("service", DataType::Utf8, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Float64, true),
            Field::new("blob", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0, 1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
                Arc::new(StringArray::from(vec![
                    "levanter", "other", "levanter", "levanter",
                ])),
                Arc::new(StringArray::from(vec!["phase", "noise", "step", "loss"])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(StringArray::from(vec![
                    "wide-a", "wide-b", "wide-c", "wide-d",
                ])),
            ],
        )
        .unwrap();
        let (parquet, _) = write_segment_to_dir(&directory, 1, 0, &batch).unwrap();
        let config = SegmentIndexConfig {
            indexes: vec![
                IndexSpec::TrigramBloom {
                    column: "name".to_string(),
                },
                IndexSpec::ExactPostings {
                    column: "name".to_string(),
                    values: vec!["phase".to_string(), "step".to_string()],
                },
                IndexSpec::ValueCounts {
                    column: "service".to_string(),
                },
                IndexSpec::CoveringProjection {
                    projection: CoveringProjection::new(
                        "training-status",
                        "name",
                        ["phase", "step"],
                        ["seq", "timestamp_ms", "service", "name", "value"],
                    ),
                },
            ],
            key_column: Some("service".to_string()),
        };
        Fixture {
            parquet,
            batch,
            config,
        }
    }

    #[test]
    fn one_bundle_contains_independent_methods_and_narrow_projection() {
        let Fixture {
            parquet,
            batch,
            config,
        } = fixture();
        assert_eq!(
            write_segment_index(&parquet, &[batch], &config).unwrap(),
            SegmentIndexWrite::Written
        );
        let bundle = index_bundle::bundle_path(&parquet);
        let header = index_bundle::read_header(&bundle).unwrap();
        assert_eq!(
            header.binding.policy_fingerprint,
            config.policy_fingerprint()
        );
        assert!(header.section("trigram:name").is_some());
        assert!(header.section("exact-postings").is_some());
        assert!(header.section("value-counts").is_some());
        let projection = header.section("projection:training-status").unwrap();
        let reference = parse_projection_reference(&projection.coverage).unwrap();
        assert_eq!(reference.spec_id.len(), 64);
        assert_eq!(reference.descriptor.row_count, 2);

        let projection_path = projection_path(&parquet, &reference);
        let reader =
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(projection_path).unwrap())
                .unwrap();
        assert_eq!(
            reader
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            vec!["seq", "timestamp_ms", "service", "name", "value"]
        );
        assert_eq!(reader.metadata().file_metadata().num_rows(), 2);
        std::fs::remove_dir_all(parquet.parent().unwrap()).ok();
    }

    #[test]
    fn policy_change_requires_rebuild() {
        let Fixture {
            parquet,
            batch,
            config,
        } = fixture();
        write_segment_index(&parquet, &[batch], &config).unwrap();
        assert!(!needs_rebuild(&parquet, 4, &config));

        let mut changed = config.clone();
        changed.indexes.push(IndexSpec::TrigramBloom {
            column: "service".to_string(),
        });
        assert!(needs_rebuild(&parquet, 4, &changed));
        std::fs::remove_dir_all(parquet.parent().unwrap()).ok();
    }

    #[test]
    fn rebuilding_removes_projection_no_longer_in_policy() {
        let Fixture {
            parquet,
            batch,
            config,
        } = fixture();
        write_segment_index(&parquet, std::slice::from_ref(&batch), &config).unwrap();
        let projection = exact::named_projection_path(&parquet, "training-status");
        assert!(projection.exists());

        let mut changed = config;
        changed
            .indexes
            .retain(|index| !matches!(index, IndexSpec::CoveringProjection { .. }));
        write_segment_index(&parquet, &[batch], &changed).unwrap();

        assert!(!projection.exists());
        let header = index_bundle::read_header(&index_bundle::bundle_path(&parquet)).unwrap();
        assert!(header.section("projection:training-status").is_none());
        std::fs::remove_dir_all(parquet.parent().unwrap()).ok();
    }
}
