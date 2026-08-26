//! In-place migration from the production telemetry root into semantic namespaces.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{new_null_array, Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use parquet::arrow::ProjectionMask;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::ingestion_policy::IngestionBatchSource;
use crate::partition_policy::{select_rows, SegmentPartition};
use crate::policies::{
    eager_storage_namespaces_for, physical_partition_policy_for, schema_for_namespace,
    storage_policy_for, PolicyRegistry,
};
use crate::server::telemetry::TELEMETRY_MAX_ROW_GROUP_ROWS;
use crate::store::catalog::{Catalog, CATALOG_DB_FILENAME};
use crate::store::schema::{schema_to_arrow, stored_form, IMPLICIT_SEQ_COLUMN};
use crate::store::segment::{
    discover_segments, read_segment_footer, segment_writer_properties_with_partition,
};
use crate::store::store::acquire_exclusive_store_lock;
use crate::store::types::{seg_filename, SegmentLocation, SegmentRow};
use crate::telemetry_policy::TELEMETRY_NAMESPACE;

const MANIFEST_FILENAME: &str = ".finelog-telemetry-v1-migration.json";
const MANIFEST_VERSION: u32 = 7;
const DUAL_WRITE_FENCE_FILENAME: &str = "dual-write-fence.json";
const DUAL_WRITE_FENCE_VERSION: u32 = 1;
const POLICY_REVISION: &str = "typed-levanter-run-partition-v2-step-index";
const MIGRATION_SOURCE_NAMESPACES: [&str; 1] = [TELEMETRY_NAMESPACE];
const OUTPUT_LEVEL: i32 = 0;
const MIGRATION_DIRECTORY: &str = ".finelog-telemetry-v1-migration";
const SOURCE_SNAPSHOT_DIRECTORY: &str = "source";
const STAGED_DIRECTORY: &str = "staged";
const ROLLBACK_DIRECTORY: &str = "rollback";
const CATALOG_BUILD_DIRECTORY: &str = "catalog-build";
const MIGRATED_SEQ_START: i64 = -4_000_000_000_000_000_000;
const MIGRATED_SEQ_ROWS_PER_OUTPUT: i64 = 10_000_000_000;
const MIGRATED_SEQ_OUTPUTS_PER_SOURCE: i64 = 100_000;
const MIGRATED_SEQ_ROWS_PER_SOURCE: i64 =
    MIGRATED_SEQ_ROWS_PER_OUTPUT * MIGRATED_SEQ_OUTPUTS_PER_SOURCE;
const SNAPSHOT_ATTEMPTS: usize = 8;
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(10);

fn migration_source_namespaces() -> impl Iterator<Item = &'static str> {
    MIGRATION_SOURCE_NAMESPACES.iter().copied()
}

fn migration_schema_for(namespace: &str) -> Result<crate::store::schema::Schema, StatsError> {
    schema_for_namespace(namespace).ok_or_else(|| {
        validation_error(format!(
            "migration policy produced namespace {namespace:?} without a registered schema"
        ))
    })
}

#[derive(Debug, Clone)]
pub struct InPlaceConfig {
    pub store_dir: PathBuf,
    pub batch_rows: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MigrationPhase {
    Staged,
    Published,
    Retired,
}

#[derive(Debug, Clone)]
pub struct PrepareConfig {
    pub source_dir: PathBuf,
    pub output_dir: PathBuf,
    pub final_log_dir: PathBuf,
    pub batch_rows: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MigrationManifest {
    pub version: u32,
    pub policy_revision: String,
    pub source_dir: String,
    pub source_catalog_sha256: String,
    pub final_log_dir: String,
    pub complete: bool,
    pub phase: MigrationPhase,
    #[serde(default)]
    pub verified_at_ms: Option<i64>,
    #[serde(default)]
    pub legacy_max_seq: Option<i64>,
    pub input_rows: i64,
    pub output_rows: i64,
    pub suppressed_rows: i64,
    pub source_segments: Vec<SourceSegment>,
    #[serde(default)]
    pub published_files: Vec<String>,
    #[serde(default)]
    pub retired_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DualWriteFence {
    pub version: u32,
    pub policy_revision: String,
    pub legacy_max_seq: i64,
    pub armed_at_ms: i64,
}

/// Persist the root sequence boundary before a dual-write server accepts traffic.
pub fn ensure_dual_write_fence(
    store_dir: &Path,
    legacy_max_seq: i64,
) -> Result<DualWriteFence, StatsError> {
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    std::fs::create_dir_all(&migration_dir)
        .map_err(internal_error("create telemetry migration directory"))?;
    let path = migration_dir.join(DUAL_WRITE_FENCE_FILENAME);
    if path.exists() {
        return read_dual_write_fence(&path);
    }
    let fence = DualWriteFence {
        version: DUAL_WRITE_FENCE_VERSION,
        policy_revision: POLICY_REVISION.to_string(),
        legacy_max_seq,
        armed_at_ms: now_ms()?,
    };
    write_json_atomically(&path, &fence, "dual-write fence")?;
    Ok(fence)
}

fn read_dual_write_fence(path: &Path) -> Result<DualWriteFence, StatsError> {
    let raw = std::fs::read(path).map_err(internal_error("read dual-write fence"))?;
    let fence: DualWriteFence = serde_json::from_slice(&raw).map_err(|error| {
        validation_error(format!(
            "decode dual-write fence {}: {error}",
            path.display()
        ))
    })?;
    if fence.version != DUAL_WRITE_FENCE_VERSION || fence.policy_revision != POLICY_REVISION {
        return Err(validation_error(
            "dual-write fence policy version does not match this binary",
        ));
    }
    Ok(fence)
}

fn dual_write_fence(store_dir: &Path) -> Result<Option<DualWriteFence>, StatsError> {
    let path = store_dir
        .join(MIGRATION_DIRECTORY)
        .join(DUAL_WRITE_FENCE_FILENAME);
    path.exists()
        .then(|| read_dual_write_fence(&path))
        .transpose()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSegment {
    pub namespace: String,
    pub relative_path: String,
    pub byte_size: u64,
    pub file_sha256: Option<String>,
    pub rows: i64,
    #[serde(default)]
    pub complete: bool,
    #[serde(default)]
    pub suppressed_rows: i64,
    pub outputs: Vec<PlannedOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedOutput {
    pub namespace: String,
    #[serde(default)]
    pub partition: Option<SegmentPartition>,
    pub relative_path: String,
    pub min_seq: i64,
    pub max_seq: i64,
    pub rows: i64,
    pub min_timestamp_ms: i64,
    pub max_timestamp_ms: i64,
    pub identity_sha256: String,
    pub file_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MigrationDestination {
    namespace: String,
    partition: Option<SegmentPartition>,
}

struct DestinationWriter {
    destination: MigrationDestination,
    min_seq: i64,
    next_seq: i64,
    rows: i64,
    min_timestamp_ms: Option<i64>,
    max_timestamp_ms: Option<i64>,
    identity: Sha256,
    temporary_path: PathBuf,
    final_path: PathBuf,
    target_schema: SchemaRef,
    writer: ArrowWriter<File>,
}

pub fn prepare_store(config: &PrepareConfig) -> Result<MigrationManifest, StatsError> {
    validate_config(config)?;
    let manifest_path = config.output_dir.join(MANIFEST_FILENAME);
    if config.output_dir.exists()
        && !manifest_path.exists()
        && std::fs::read_dir(&config.output_dir)
            .map_err(internal_error("list existing output store"))?
            .next()
            .is_some()
    {
        return Err(validation_error(
            "output_dir is non-empty and has no migration manifest",
        ));
    }
    std::fs::create_dir_all(&config.output_dir).map_err(internal_error("create output store"))?;

    let mut manifest = if manifest_path.exists() {
        let manifest = read_manifest(&manifest_path)?;
        validate_manifest_config(&manifest, config)?;
        manifest
    } else {
        let manifest = plan_migration(config)?;
        write_manifest(&manifest_path, &manifest)?;
        manifest
    };

    tracing::info!(
        source_segments = manifest.source_segments.len(),
        input_rows = manifest.input_rows,
        legacy_max_seq = ?manifest.legacy_max_seq,
        completed_segments = manifest
            .source_segments
            .iter()
            .filter(|source| source.complete)
            .count(),
        "telemetry migration plan ready"
    );

    verify_source_segments(&config.source_dir, &manifest)?;
    if manifest.complete {
        return Ok(manifest);
    }
    write_planned_outputs(config, &mut manifest)?;
    if manifest
        .source_segments
        .iter()
        .any(|source| !source.complete)
    {
        return Err(validation_error(
            "migration finished writing with incomplete source segments",
        ));
    }
    if manifest.input_rows != manifest.output_rows + manifest.suppressed_rows {
        return Err(validation_error(format!(
            "migration row mismatch: {} source rows, {} output rows, {} intentionally suppressed rows",
            manifest.input_rows, manifest.output_rows, manifest.suppressed_rows
        )));
    }
    manifest.complete = true;
    manifest.phase = MigrationPhase::Staged;
    write_manifest(&manifest_path, &manifest)?;
    Ok(manifest)
}

pub fn verify_store(
    source_dir: &Path,
    output_dir: &Path,
    batch_rows: usize,
) -> Result<MigrationManifest, StatsError> {
    if batch_rows == 0 {
        return Err(validation_error("batch_rows must be positive"));
    }
    let manifest = read_manifest(&output_dir.join(MANIFEST_FILENAME))?;
    if !manifest.complete {
        return Err(validation_error(
            "migration manifest is incomplete; rerun prepare",
        ));
    }
    let total_outputs = manifest
        .source_segments
        .iter()
        .map(|source| source.outputs.len())
        .sum::<usize>();
    tracing::info!(
        source_segments = manifest.source_segments.len(),
        output_files = total_outputs,
        output_rows = manifest.output_rows,
        "telemetry migration verification started"
    );
    verify_source_segments(source_dir, &manifest)?;
    let output_verification_started = Instant::now();
    let mut last_progress = Instant::now();
    let mut verified_outputs = 0_usize;
    let mut verified_rows = 0_i64;
    for source in &manifest.source_segments {
        for output in &source.outputs {
            let path = output_dir.join(&output.relative_path);
            verify_output_file(&path, output, batch_rows)?;
            verified_outputs += 1;
            verified_rows += output.rows;
            if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
                tracing::info!(
                    verified_outputs,
                    total_outputs,
                    verified_rows,
                    output_rows = manifest.output_rows,
                    elapsed_seconds = output_verification_started.elapsed().as_secs(),
                    "telemetry migration output verification progress"
                );
                last_progress = Instant::now();
            }
        }
    }
    tracing::info!(
        verified_outputs,
        total_outputs,
        verified_rows,
        output_rows = manifest.output_rows,
        elapsed_seconds = output_verification_started.elapsed().as_secs(),
        "telemetry migration output verification complete"
    );
    if manifest.input_rows != manifest.output_rows + manifest.suppressed_rows {
        return Err(validation_error(format!(
            "migration row mismatch: {} source rows, {} output rows, {} intentionally suppressed rows",
            manifest.input_rows, manifest.output_rows, manifest.suppressed_rows
        )));
    }
    Ok(manifest)
}

/// Stage the complete root hot set while Finelog remains available.
///
/// Deploy the row-aware ingestion policy first so no new rows enter the root
/// namespace after preparation begins. Repeated calls resume the same stage.
pub fn prepare_in_place(config: &InPlaceConfig) -> Result<MigrationManifest, StatsError> {
    if config.batch_rows == 0 {
        return Err(validation_error("batch_rows must be positive"));
    }
    let store_dir = std::fs::canonicalize(&config.store_dir)
        .map_err(internal_error("resolve in-place telemetry store"))?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let source_dir = migration_dir.join(SOURCE_SNAPSHOT_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    if !source_dir.exists() {
        snapshot_migration_sources(&store_dir, &source_dir)?;
    }
    prepare_store(&PrepareConfig {
        source_dir,
        output_dir: staged_dir,
        final_log_dir: store_dir,
        batch_rows: config.batch_rows,
    })
}

/// Verify the current in-place phase without modifying the store.
pub fn verify_in_place(config: &InPlaceConfig) -> Result<MigrationManifest, StatsError> {
    let store_dir = std::fs::canonicalize(&config.store_dir)
        .map_err(internal_error("resolve in-place telemetry store"))?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let source_dir = migration_dir.join(SOURCE_SNAPSHOT_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    let mut manifest = verify_store(&source_dir, &staged_dir, config.batch_rows)?;
    if manifest.phase != MigrationPhase::Staged {
        verify_published_layout(&store_dir, &manifest)?;
    }
    if manifest.phase == MigrationPhase::Retired {
        verify_root_namespace_retired(&store_dir)?;
    }
    if manifest.phase == MigrationPhase::Staged {
        manifest.verified_at_ms = Some(now_ms()?);
        write_manifest(&staged_dir.join(MANIFEST_FILENAME), &manifest)?;
    }
    Ok(manifest)
}

/// Make the staged semantic rows queryable during a stopped-server cutover.
pub fn publish_in_place(store_dir: &Path) -> Result<MigrationManifest, StatsError> {
    let store_dir = std::fs::canonicalize(store_dir)
        .map_err(internal_error("resolve in-place telemetry store"))?;
    let _store_lock = acquire_exclusive_store_lock(&store_dir)?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    let manifest_path = staged_dir.join(MANIFEST_FILENAME);
    let mut manifest = read_manifest(&manifest_path)?;
    if manifest.version != MANIFEST_VERSION || manifest.policy_revision != POLICY_REVISION {
        return Err(validation_error(
            "migration manifest policy version does not match this binary",
        ));
    }
    if !manifest.complete {
        return Err(validation_error(
            "migration manifest is incomplete; rerun prepare",
        ));
    }
    if manifest.verified_at_ms.is_none() {
        return Err(validation_error(
            "staged telemetry migration has not been verified; run verify-telemetry-v1 before publish",
        ));
    }
    if manifest.phase != MigrationPhase::Staged {
        return Err(validation_error(
            "telemetry migration has already been published",
        ));
    }

    let mut published_files = Vec::new();
    for output in manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
    {
        let source = staged_dir.join(&output.relative_path);
        let destination = store_dir.join(&output.relative_path);
        link_verified_file(&source, &destination, output.file_sha256.as_deref())?;
        published_files.push(output.relative_path.clone());
    }
    let publish_backup = catalog_backup_path(&migration_dir, "pre-publish");
    if publish_backup.exists() {
        verify_published_files(&store_dir, &manifest)?;
    } else {
        replace_catalog_for_publish(&store_dir, &migration_dir, &manifest)?;
    }
    manifest.published_files = published_files;
    verify_published_files(&store_dir, &manifest)?;
    manifest.phase = MigrationPhase::Published;
    write_manifest(&manifest_path, &manifest)?;
    Ok(manifest)
}

/// Remove the root namespace after every query has switched to semantic names.
///
/// Stop Finelog before this command and restart it after the command returns.
pub fn retire_in_place(config: &InPlaceConfig) -> Result<MigrationManifest, StatsError> {
    let store_dir = std::fs::canonicalize(&config.store_dir)
        .map_err(internal_error("resolve in-place telemetry store"))?;
    let _store_lock = acquire_exclusive_store_lock(&store_dir)?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    let manifest_path = staged_dir.join(MANIFEST_FILENAME);
    let mut manifest = read_manifest(&manifest_path)?;
    if manifest.phase != MigrationPhase::Published {
        return Err(validation_error(
            "telemetry migration must be published before root retirement",
        ));
    }
    verify_published_layout(&store_dir, &manifest)?;

    let rollback_sources = migration_dir.join(ROLLBACK_DIRECTORY).join("root-files");
    for namespace in migration_source_namespaces() {
        let namespace_dir = store_dir.join(namespace);
        for source in discover_segments(&namespace_dir) {
            let filename = source.file_name().ok_or_else(|| {
                validation_error(format!(
                    "root telemetry segment has no filename: {}",
                    source.display()
                ))
            })?;
            let destination = rollback_sources.join(namespace).join(filename);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(internal_error("create root rollback directory"))?;
            }
            std::fs::rename(&source, &destination)
                .map_err(internal_error("move root segment into rollback"))?;
        }
    }
    let retired_files = migration_source_namespaces()
        .flat_map(|namespace| {
            discover_segments(&rollback_sources.join(namespace))
                .into_iter()
                .map(move |path| {
                    Path::new(namespace)
                        .join(path.file_name().expect("discovered segment has a filename"))
                        .to_string_lossy()
                        .into_owned()
                })
        })
        .collect();
    let retire_backup = catalog_backup_path(&migration_dir, "pre-retire");
    if retire_backup.exists() {
        verify_root_namespace_retired(&store_dir)?;
    } else {
        replace_catalog_for_retirement(&store_dir, &migration_dir)?;
    }
    manifest.retired_files = retired_files;
    manifest.phase = MigrationPhase::Retired;
    write_manifest(&manifest_path, &manifest)?;
    verify_root_namespace_retired(&store_dir)?;
    Ok(manifest)
}

fn snapshot_migration_sources(store_dir: &Path, source_dir: &Path) -> Result<(), StatsError> {
    let migration_dir = source_dir
        .parent()
        .ok_or_else(|| validation_error("migration source snapshot has no parent"))?;
    std::fs::create_dir_all(migration_dir).map_err(internal_error("create migration directory"))?;
    let temporary = migration_dir.join(format!("{SOURCE_SNAPSHOT_DIRECTORY}.tmp"));
    for attempt in 0..SNAPSHOT_ATTEMPTS {
        if temporary.exists() {
            std::fs::remove_dir_all(&temporary)
                .map_err(internal_error("remove interrupted source snapshot"))?;
        }
        std::fs::create_dir_all(&temporary).map_err(internal_error("create source snapshot"))?;
        copy_catalog_consistently(
            &store_dir.join(CATALOG_DB_FILENAME),
            &temporary.join(CATALOG_DB_FILENAME),
        )?;
        if link_catalog_snapshot_segments(store_dir, &temporary)? {
            std::fs::rename(&temporary, source_dir)
                .map_err(internal_error("publish source snapshot"))?;
            return Ok(());
        }
        if attempt + 1 == SNAPSHOT_ATTEMPTS {
            return Err(validation_error(
                "telemetry segments kept changing while the migration snapshot was taken",
            ));
        }
    }
    unreachable!()
}

fn link_catalog_snapshot_segments(
    store_dir: &Path,
    snapshot_dir: &Path,
) -> Result<bool, StatsError> {
    let catalog = Catalog::open(Some(snapshot_dir))?;
    for namespace in migration_source_namespaces() {
        let expected_directory = store_dir.join(namespace);
        for segment in catalog.list_segments(namespace)? {
            if segment.location == SegmentLocation::Remote {
                continue;
            }
            let source = Path::new(&segment.path);
            if !source.starts_with(&expected_directory) {
                return Err(validation_error(format!(
                    "catalog segment escaped namespace directory: {}",
                    source.display()
                )));
            }
            let filename = source.file_name().ok_or_else(|| {
                validation_error(format!(
                    "source segment has no filename: {}",
                    source.display()
                ))
            })?;
            let destination = snapshot_dir.join(namespace).join(filename);
            if let Some(parent) = destination.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(internal_error("create snapshot namespace directory"))?;
            }
            match std::fs::hard_link(source, destination) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(error) => {
                    return Err(StatsError::Internal(format!(
                        "hard-link source telemetry segment: {error}"
                    )))
                }
            }
        }
    }
    Ok(true)
}

fn copy_catalog_consistently(source: &Path, destination: &Path) -> Result<(), StatsError> {
    if destination.exists() {
        std::fs::remove_file(destination).map_err(internal_error("remove old catalog snapshot"))?;
    }
    let connection =
        rusqlite::Connection::open_with_flags(source, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(internal_error("open catalog for snapshot"))?;
    connection
        .execute(
            "VACUUM main INTO ?1",
            [destination.to_string_lossy().as_ref()],
        )
        .map_err(internal_error("snapshot catalog"))?;
    Ok(())
}

fn link_verified_file(
    source: &Path,
    destination: &Path,
    expected_sha256: Option<&str>,
) -> Result<(), StatsError> {
    let expected_sha256 = expected_sha256
        .ok_or_else(|| validation_error("staged telemetry output has no checksum"))?;
    if destination.exists() {
        if file_sha256(destination)? == expected_sha256 {
            return Ok(());
        }
        return Err(validation_error(format!(
            "publish destination already exists with different content: {}",
            destination.display()
        )));
    }
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(internal_error("create output namespace"))?;
    }
    std::fs::hard_link(source, destination).map_err(internal_error("publish migrated segment"))?;
    Ok(())
}

fn replace_catalog_for_publish(
    store_dir: &Path,
    migration_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let build_dir = reset_catalog_build_directory(migration_dir)?;
    copy_catalog_consistently(
        &store_dir.join(CATALOG_DB_FILENAME),
        &build_dir.join(CATALOG_DB_FILENAME),
    )?;
    let catalog = Catalog::open(Some(&build_dir))?;
    for namespace in eager_storage_namespaces_for(TELEMETRY_NAMESPACE) {
        catalog.upsert(namespace, &stored_form(migration_schema_for(namespace)?))?;
        catalog.upsert_policy(namespace, &storage_policy_for(namespace)?)?;
    }
    let output_namespaces: BTreeSet<_> = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| output.namespace.as_str())
        .collect();
    for namespace in output_namespaces {
        catalog.upsert(namespace, &stored_form(migration_schema_for(namespace)?))?;
        catalog.upsert_policy(namespace, &storage_policy_for(namespace)?)?;
    }
    let created_at_ms = now_ms()?;
    let rows = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| {
            let path = store_dir.join(&output.relative_path);
            Ok(SegmentRow {
                namespace: output.namespace.clone(),
                path: path.to_string_lossy().into_owned(),
                level: OUTPUT_LEVEL,
                min_seq: output.min_seq,
                max_seq: output.max_seq,
                row_count: output.rows,
                byte_size: std::fs::metadata(&path)
                    .map_err(internal_error("stat published telemetry segment"))?
                    .len() as i64,
                created_at_ms,
                min_key_value: Some(output.min_timestamp_ms.to_string()),
                max_key_value: Some(output.max_timestamp_ms.to_string()),
                partition: output.partition.clone(),
                location: SegmentLocation::Local,
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    catalog.upsert_segments(&rows)?;
    drop(catalog);
    replace_catalog_file(store_dir, migration_dir, &build_dir, "pre-publish")
}

fn replace_catalog_for_retirement(
    store_dir: &Path,
    migration_dir: &Path,
) -> Result<(), StatsError> {
    let build_dir = reset_catalog_build_directory(migration_dir)?;
    copy_catalog_consistently(
        &store_dir.join(CATALOG_DB_FILENAME),
        &build_dir.join(CATALOG_DB_FILENAME),
    )?;
    let catalog = Catalog::open(Some(&build_dir))?;
    for namespace in migration_source_namespaces() {
        catalog.delete(namespace)?;
    }
    drop(catalog);
    replace_catalog_file(store_dir, migration_dir, &build_dir, "pre-retire")
}

fn reset_catalog_build_directory(migration_dir: &Path) -> Result<PathBuf, StatsError> {
    let build_dir = migration_dir.join(CATALOG_BUILD_DIRECTORY);
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir)
            .map_err(internal_error("remove interrupted catalog build"))?;
    }
    std::fs::create_dir_all(&build_dir)
        .map_err(internal_error("create catalog build directory"))?;
    Ok(build_dir)
}

fn replace_catalog_file(
    store_dir: &Path,
    migration_dir: &Path,
    build_dir: &Path,
    backup_name: &str,
) -> Result<(), StatsError> {
    let rollback_dir = migration_dir.join(ROLLBACK_DIRECTORY);
    std::fs::create_dir_all(&rollback_dir)
        .map_err(internal_error("create catalog rollback directory"))?;
    let current = store_dir.join(CATALOG_DB_FILENAME);
    let candidate = build_dir.join(CATALOG_DB_FILENAME);
    let backup = catalog_backup_path(migration_dir, backup_name);
    if backup.exists() {
        return Err(validation_error(format!(
            "catalog rollback already exists: {}",
            backup.display()
        )));
    }
    std::fs::rename(&current, &backup).map_err(internal_error("save catalog rollback"))?;
    if let Err(publish_error) = std::fs::rename(&candidate, &current) {
        if let Err(rollback_error) = std::fs::rename(&backup, &current) {
            return Err(StatsError::Internal(format!(
                "publish telemetry catalog: {publish_error}; restore original catalog: {rollback_error}"
            )));
        }
        return Err(StatsError::Internal(format!(
            "publish telemetry catalog: {publish_error}"
        )));
    }
    let journal = store_dir.join(format!("{CATALOG_DB_FILENAME}-journal"));
    if journal.exists() {
        let backup_journal = rollback_dir.join(format!("{backup_name}.sqlite-journal"));
        std::fs::rename(&journal, backup_journal)
            .map_err(internal_error("save catalog rollback journal"))?;
    }
    Ok(())
}

fn catalog_backup_path(migration_dir: &Path, backup_name: &str) -> PathBuf {
    migration_dir
        .join(ROLLBACK_DIRECTORY)
        .join(format!("{backup_name}.sqlite"))
}

fn verify_published_files(
    store_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        store_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(internal_error("open published catalog"))?;
    for output in manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
    {
        let path = store_dir.join(&output.relative_path);
        if !path.is_file() {
            return Err(validation_error(format!(
                "published telemetry file is missing: {}",
                path.display()
            )));
        }
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM segments WHERE namespace = ?1 AND path = ?2 AND row_count = ?3",
                rusqlite::params![output.namespace, path.to_string_lossy(), output.rows],
                |row| row.get(0),
            )
            .map_err(internal_error("verify published telemetry catalog row"))?;
        if count != 1 {
            return Err(validation_error(format!(
                "published telemetry catalog row is missing for {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn verify_published_layout(
    store_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let expected_files = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| output.relative_path.as_str())
        .collect::<BTreeSet<_>>();
    let published_files = manifest
        .published_files
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if manifest.published_files.len() != expected_files.len() || published_files != expected_files {
        return Err(validation_error(
            "published telemetry file set differs from the migration plan",
        ));
    }

    let connection = rusqlite::Connection::open_with_flags(
        store_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(internal_error("open published catalog"))?;
    let output_namespaces = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| output.namespace.as_str())
        .collect::<BTreeSet<_>>();
    for namespace in output_namespaces {
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM namespaces WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(internal_error("verify published telemetry namespace"))?;
        if count != 1 {
            return Err(validation_error(format!(
                "published telemetry namespace is missing: {namespace:?}"
            )));
        }
    }
    Ok(())
}

fn verify_root_namespace_retired(store_dir: &Path) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        store_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(internal_error("open retired telemetry catalog"))?;
    for namespace in migration_source_namespaces() {
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM namespaces WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(internal_error("verify retired telemetry namespace"))?;
        if count != 0 || !discover_segments(&store_dir.join(namespace)).is_empty() {
            return Err(validation_error(format!(
                "root telemetry namespace {namespace:?} is still visible"
            )));
        }
    }
    Ok(())
}

fn validate_config(config: &PrepareConfig) -> Result<(), StatsError> {
    if config.batch_rows == 0 {
        return Err(validation_error("batch_rows must be positive"));
    }
    if !config.final_log_dir.is_absolute() {
        return Err(validation_error("final_log_dir must be absolute"));
    }
    let source = std::fs::canonicalize(&config.source_dir)
        .map_err(internal_error("resolve source store"))?;
    let output = absolute_path(&config.output_dir)?;
    if source == output || output.starts_with(&source) {
        return Err(validation_error(
            "staged output must be outside the immutable source snapshot",
        ));
    }
    if !config.source_dir.join(CATALOG_DB_FILENAME).is_file() {
        return Err(validation_error(format!(
            "source store has no {} catalog",
            CATALOG_DB_FILENAME
        )));
    }
    assert_catalog_is_quiescent(&config.source_dir)?;
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf, StatsError> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(internal_error("resolve relative path"))
}

fn assert_catalog_is_quiescent(source_dir: &Path) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        source_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(internal_error("open source catalog read-only"))?;
    let check: String = connection
        .query_row("PRAGMA quick_check", [], |row| row.get(0))
        .map_err(internal_error("check source catalog"))?;
    if check != "ok" {
        return Err(validation_error(format!(
            "source catalog quick_check failed: {check}"
        )));
    }
    let forwarding_rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM forward_state", [], |row| row.get(0))
        .map_err(internal_error("inspect forwarding state"))?;
    if forwarding_rows != 0 {
        return Err(validation_error(
            "telemetry migration is hub-only; source catalog has forwarding state",
        ));
    }
    Ok(())
}

fn plan_migration(config: &PrepareConfig) -> Result<MigrationManifest, StatsError> {
    let mut source_segments = Vec::new();
    let mut input_rows = 0_i64;
    let legacy_max_seq = dual_write_fence(&config.final_log_dir)?.map(|fence| fence.legacy_max_seq);

    for namespace in migration_source_namespaces() {
        let namespace_dir = config.source_dir.join(namespace);
        if !namespace_dir.is_dir() {
            continue;
        }
        for path in discover_segments(&namespace_dir) {
            let metadata =
                std::fs::metadata(&path).map_err(internal_error("stat source segment"))?;
            let footer = read_segment_footer(&path, Some("timestamp_ms")).ok_or_else(|| {
                validation_error(format!("could not read source segment {}", path.display()))
            })?;
            let rows = match legacy_max_seq {
                Some(fence) if footer.min_seq > fence => 0,
                Some(fence) if footer.max_seq > fence => {
                    count_rows_at_or_before_seq(&path, config.batch_rows, fence)?
                }
                _ => footer.row_count,
            };
            input_rows += rows;
            let relative_path = path
                .strip_prefix(&config.source_dir)
                .map_err(|_| validation_error("source segment escaped source_dir"))?
                .to_string_lossy()
                .into_owned();
            source_segments.push(SourceSegment {
                namespace: namespace.to_string(),
                relative_path,
                byte_size: metadata.len(),
                file_sha256: None,
                rows,
                complete: false,
                suppressed_rows: 0,
                outputs: Vec::new(),
            });
        }
    }
    Ok(MigrationManifest {
        version: MANIFEST_VERSION,
        policy_revision: POLICY_REVISION.to_string(),
        source_dir: std::fs::canonicalize(&config.source_dir)
            .map_err(internal_error("resolve source store"))?
            .to_string_lossy()
            .into_owned(),
        source_catalog_sha256: file_sha256(&config.source_dir.join(CATALOG_DB_FILENAME))?,
        final_log_dir: config.final_log_dir.to_string_lossy().into_owned(),
        complete: false,
        phase: MigrationPhase::Staged,
        verified_at_ms: None,
        legacy_max_seq,
        input_rows,
        output_rows: 0,
        suppressed_rows: 0,
        source_segments,
        published_files: Vec::new(),
        retired_files: Vec::new(),
    })
}

fn write_planned_outputs(
    config: &PrepareConfig,
    manifest: &mut MigrationManifest,
) -> Result<(), StatsError> {
    let policies = PolicyRegistry::default();
    index_migration_state(config, manifest, &policies)?;
    let manifest_path = config.output_dir.join(MANIFEST_FILENAME);
    let conversion_started = Instant::now();
    let mut last_conversion_progress = Instant::now();
    let total_sources = manifest.source_segments.len();
    for source_index in 0..manifest.source_segments.len() {
        let source_path = config
            .source_dir
            .join(&manifest.source_segments[source_index].relative_path);
        write_source_outputs(
            &policies,
            source_index,
            total_sources,
            &source_path,
            config,
            manifest.legacy_max_seq,
            &mut manifest.source_segments[source_index],
        )?;
        manifest.output_rows = manifest
            .source_segments
            .iter()
            .map(|source| source.outputs.iter().map(|output| output.rows).sum::<i64>())
            .sum();
        manifest.suppressed_rows = manifest
            .source_segments
            .iter()
            .map(|source| source.suppressed_rows)
            .sum();
        write_manifest(&manifest_path, manifest)?;
        if last_conversion_progress.elapsed() >= PROGRESS_LOG_INTERVAL
            || source_index + 1 == total_sources
        {
            tracing::info!(
                completed_segments = manifest
                    .source_segments
                    .iter()
                    .filter(|source| source.complete)
                    .count(),
                total_segments = total_sources,
                output_rows = manifest.output_rows,
                suppressed_rows = manifest.suppressed_rows,
                elapsed_seconds = conversion_started.elapsed().as_secs(),
                "telemetry migration conversion progress"
            );
            last_conversion_progress = Instant::now();
        }
    }
    Ok(())
}

fn index_migration_state(
    config: &PrepareConfig,
    manifest: &MigrationManifest,
    policies: &PolicyRegistry,
) -> Result<(), StatsError> {
    const INDEX_COLUMNS: [&str; 10] = [
        "service",
        "name",
        "value",
        "timestamp_ms",
        "seq",
        "record_index",
        "process_index",
        "execution_uid",
        "resource_attributes_json",
        "attributes_json",
    ];
    let index_started = Instant::now();
    let mut last_progress = Instant::now();
    let mut indexed_rows = 0_i64;
    let total_sources = manifest.source_segments.len();
    for (source_index, source) in manifest.source_segments.iter().enumerate() {
        let source_path = config.source_dir.join(&source.relative_path);
        for batch in parquet_reader_projected(&source_path, config.batch_rows, &INDEX_COLUMNS)? {
            let batch = batch.map_err(internal_error("read migration index batch"))?;
            let batch = filter_legacy_rows(&batch, manifest.legacy_max_seq)?;
            if batch.num_rows() == 0 {
                continue;
            }
            indexed_rows += batch.num_rows() as i64;
            policies.index_migration_batch(
                IngestionBatchSource::Stored(source.namespace.as_str()),
                &batch,
            )?;
            if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
                tracing::info!(
                    indexed_rows,
                    input_rows = manifest.input_rows,
                    source_segment = source_index + 1,
                    total_segments = total_sources,
                    elapsed_seconds = index_started.elapsed().as_secs(),
                    "telemetry migration indexing progress"
                );
                last_progress = Instant::now();
            }
        }
    }
    policies.finish_migration_index();
    tracing::info!(
        indexed_rows,
        input_rows = manifest.input_rows,
        total_segments = total_sources,
        elapsed_seconds = index_started.elapsed().as_secs(),
        "telemetry migration indexing complete"
    );
    Ok(())
}

fn write_source_outputs(
    policies: &PolicyRegistry,
    source_index: usize,
    total_sources: usize,
    source_path: &Path,
    config: &PrepareConfig,
    legacy_max_seq: Option<i64>,
    source: &mut SourceSegment,
) -> Result<(), StatsError> {
    if verify_completed_source(config, source)? {
        return Ok(());
    }
    let source_offset = migrated_source_offset(source_index)?;
    let mut writers = BTreeMap::new();
    let source_started = Instant::now();
    let mut last_progress = Instant::now();
    let mut source_rows_processed = 0_i64;
    for batch in parquet_reader(source_path, config.batch_rows)? {
        let batch = batch.map_err(internal_error("read source batch"))?;
        let batch = filter_legacy_rows(&batch, legacy_max_seq)?;
        if batch.num_rows() == 0 {
            continue;
        }
        source_rows_processed += batch.num_rows() as i64;
        for partition in policies.route_ingestion_batch(
            IngestionBatchSource::Stored(source.namespace.as_str()),
            &batch,
        )? {
            for (destination, batch) in physical_migration_batches(
                partition.destination.logical_namespace,
                partition.batch,
            )? {
                if !writers.contains_key(&destination) {
                    let writer = create_destination_writer(
                        &destination,
                        writers.len(),
                        source_offset,
                        config,
                    )?;
                    writers.insert(destination.clone(), writer);
                }
                write_destination_batch(
                    writers
                        .get_mut(&destination)
                        .expect("destination writer was just created"),
                    &batch,
                )?;
            }
        }
        if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
            tracing::info!(
                source_segment = source_index + 1,
                total_segments = total_sources,
                source_rows_processed,
                source_rows = source.rows,
                elapsed_seconds = source_started.elapsed().as_secs(),
                "telemetry migration source progress"
            );
            last_progress = Instant::now();
        }
    }
    let outputs = finish_destination_writers(writers, &config.output_dir)?;
    record_source_outputs(source, source_path, outputs)
}

fn verify_completed_source(
    config: &PrepareConfig,
    source: &SourceSegment,
) -> Result<bool, StatsError> {
    if source.complete {
        for output in &source.outputs {
            verify_output_file(
                &config.output_dir.join(&output.relative_path),
                output,
                config.batch_rows,
            )?;
        }
        return Ok(true);
    }
    if !source.outputs.is_empty() {
        return Err(validation_error(
            "incomplete metadata-only migration source has planned outputs",
        ));
    }
    Ok(false)
}

fn migrated_source_offset(source_index: usize) -> Result<i64, StatsError> {
    let source_index = i64::try_from(source_index)
        .map_err(|_| validation_error("migration has too many source segments"))?;
    source_index
        .checked_mul(MIGRATED_SEQ_ROWS_PER_SOURCE)
        .and_then(|offset| MIGRATED_SEQ_START.checked_add(offset))
        .filter(|min_seq| *min_seq < 0)
        .ok_or_else(|| validation_error("migration has too many source segments"))
}

fn create_destination_writer(
    destination: &MigrationDestination,
    output_index: usize,
    source_offset: i64,
    config: &PrepareConfig,
) -> Result<DestinationWriter, StatsError> {
    let output_index = i64::try_from(output_index)
        .map_err(|_| validation_error("source produced too many outputs"))?;
    if output_index >= MIGRATED_SEQ_OUTPUTS_PER_SOURCE {
        return Err(validation_error("source produced too many outputs"));
    }
    let min_seq = output_index
        .checked_mul(MIGRATED_SEQ_ROWS_PER_OUTPUT)
        .and_then(|offset| source_offset.checked_add(offset))
        .ok_or_else(|| validation_error("migrated sequence range overflowed"))?;
    let relative_path = Path::new(&destination.namespace).join(seg_filename(OUTPUT_LEVEL, min_seq));
    let final_path = config.output_dir.join(relative_path);
    let parent = final_path
        .parent()
        .ok_or_else(|| validation_error("output segment has no parent"))?;
    std::fs::create_dir_all(parent).map_err(internal_error("create output namespace"))?;
    let temporary_path = temporary_path(&final_path);
    for interrupted_path in [&temporary_path, &final_path] {
        if interrupted_path.exists() {
            std::fs::remove_file(interrupted_path)
                .map_err(internal_error("remove interrupted output segment"))?;
        }
    }
    let file = File::create(&temporary_path).map_err(internal_error("create output segment"))?;
    let target_schema =
        schema_to_arrow(&stored_form(migration_schema_for(&destination.namespace)?));
    let options =
        ArrowWriterOptions::new().with_properties(segment_writer_properties_with_partition(
            usize::try_from(TELEMETRY_MAX_ROW_GROUP_ROWS)
                .expect("telemetry row-group limit fits usize"),
            destination.partition.as_ref(),
        )?);
    let writer = ArrowWriter::try_new_with_options(file, Arc::clone(&target_schema), options)
        .map_err(internal_error("create output parquet writer"))?;
    Ok(DestinationWriter {
        destination: destination.clone(),
        min_seq,
        next_seq: min_seq,
        rows: 0,
        min_timestamp_ms: None,
        max_timestamp_ms: None,
        identity: Sha256::new(),
        temporary_path,
        final_path,
        target_schema,
        writer,
    })
}

fn write_destination_batch(
    writer: &mut DestinationWriter,
    batch: &RecordBatch,
) -> Result<(), StatsError> {
    let batch_rows = i64::try_from(batch.num_rows())
        .map_err(|_| validation_error("migration batch has too many rows"))?;
    if writer.rows + batch_rows > MIGRATED_SEQ_ROWS_PER_OUTPUT {
        return Err(validation_error(
            "migration output exceeded its reserved sequence range",
        ));
    }
    let migrated = align_migrated_batch(batch, &writer.target_schema, writer.next_seq)?;
    let timestamps = int64_column(&migrated, "timestamp_ms")?;
    if let Some(batch_min) = timestamps.iter().flatten().min() {
        writer.min_timestamp_ms = Some(
            writer
                .min_timestamp_ms
                .map_or(batch_min, |current| current.min(batch_min)),
        );
    }
    if let Some(batch_max) = timestamps.iter().flatten().max() {
        writer.max_timestamp_ms = Some(
            writer
                .max_timestamp_ms
                .map_or(batch_max, |current| current.max(batch_max)),
        );
    }
    update_batch_identity(&mut writer.identity, &migrated)?;
    writer
        .writer
        .write(&migrated)
        .map_err(internal_error("write migrated telemetry"))?;
    writer.rows += migrated.num_rows() as i64;
    writer.next_seq += migrated.num_rows() as i64;
    Ok(())
}

fn finish_destination_writers(
    writers: BTreeMap<MigrationDestination, DestinationWriter>,
    output_dir: &Path,
) -> Result<Vec<PlannedOutput>, StatsError> {
    let mut outputs = Vec::with_capacity(writers.len());
    for (_destination, writer) in writers {
        let file = writer
            .writer
            .into_inner()
            .map_err(internal_error("close output segment"))?;
        file.sync_all()
            .map_err(internal_error("fsync output segment"))?;
        std::fs::rename(&writer.temporary_path, &writer.final_path)
            .map_err(internal_error("publish output segment"))?;
        let relative_path = writer
            .final_path
            .strip_prefix(output_dir)
            .map_err(|_| validation_error("output segment escaped output_dir"))?
            .to_string_lossy()
            .into_owned();
        outputs.push(PlannedOutput {
            namespace: writer.destination.namespace,
            partition: writer.destination.partition,
            relative_path,
            min_seq: writer.min_seq,
            max_seq: writer.next_seq - 1,
            rows: writer.rows,
            min_timestamp_ms: writer
                .min_timestamp_ms
                .ok_or_else(|| validation_error("migrated output has no timestamp"))?,
            max_timestamp_ms: writer
                .max_timestamp_ms
                .ok_or_else(|| validation_error("migrated output has no timestamp"))?,
            identity_sha256: digest_hex(writer.identity),
            file_sha256: Some(file_sha256(&writer.final_path)?),
        });
    }
    outputs.sort_by(|left, right| {
        (&left.namespace, &left.partition).cmp(&(&right.namespace, &right.partition))
    });
    Ok(outputs)
}

fn record_source_outputs(
    source: &mut SourceSegment,
    source_path: &Path,
    outputs: Vec<PlannedOutput>,
) -> Result<(), StatsError> {
    let output_rows = outputs.iter().map(|output| output.rows).sum::<i64>();
    if output_rows > source.rows {
        return Err(validation_error(format!(
            "source {} produced more rows than it contained",
            source.relative_path
        )));
    }
    source.file_sha256 = Some(file_sha256(source_path)?);
    source.suppressed_rows = source.rows - output_rows;
    source.outputs = outputs;
    source.complete = true;
    Ok(())
}

fn physical_migration_batches(
    namespace: String,
    batch: RecordBatch,
) -> Result<Vec<(MigrationDestination, RecordBatch)>, StatsError> {
    let Some(policy) = physical_partition_policy_for(&namespace) else {
        return Ok(vec![(
            MigrationDestination {
                namespace,
                partition: None,
            },
            batch,
        )]);
    };
    Ok(policy
        .partition_batches(&[batch])?
        .into_iter()
        .flat_map(|output| {
            let destination = MigrationDestination {
                namespace: namespace.clone(),
                partition: Some(output.partition),
            };
            output
                .batches
                .into_iter()
                .map(move |batch| (destination.clone(), batch))
        })
        .collect())
}

fn align_migrated_batch(
    source: &RecordBatch,
    target_schema: &SchemaRef,
    min_seq: i64,
) -> Result<RecordBatch, StatsError> {
    // Catalog schema evolution is additive, so retired nullable columns remain
    // in old Parquets after the current telemetry schema stops writing them.
    // Project those columns away; an unknown required field still signals that
    // the current layout cannot represent the source record.
    for field in source.schema().fields() {
        if field.name() != IMPLICIT_SEQ_COLUMN
            && target_schema.field_with_name(field.name()).is_err()
            && !field.is_nullable()
        {
            return Err(validation_error(format!(
                "source telemetry has unknown required column {:?}",
                field.name()
            )));
        }
    }
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        if field.name() == IMPLICIT_SEQ_COLUMN {
            columns.push(Arc::new(Int64Array::from_iter_values(
                min_seq..min_seq + source.num_rows() as i64,
            )));
            continue;
        }
        match source.schema().index_of(field.name()) {
            Ok(index) => {
                let array = source.column(index);
                columns.push(if array.data_type() == field.data_type() {
                    Arc::clone(array)
                } else {
                    cast(array, field.data_type())
                        .map_err(internal_error("align telemetry column"))?
                });
            }
            Err(_) if field.is_nullable() => {
                columns.push(new_null_array(field.data_type(), source.num_rows()));
            }
            Err(_) => {
                return Err(validation_error(format!(
                    "source telemetry is missing required column {:?}",
                    field.name()
                )));
            }
        }
    }
    RecordBatch::try_new(Arc::clone(target_schema), columns)
        .map_err(internal_error("build migrated telemetry batch"))
}

fn verify_source_segments(
    source_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let verification_started = Instant::now();
    let mut last_progress = Instant::now();
    let mut verified_segments = 0_usize;
    let mut verified_bytes = 0_u64;
    let total_segments = manifest.source_segments.len();
    let total_bytes = manifest
        .source_segments
        .iter()
        .map(|segment| segment.byte_size)
        .sum::<u64>();
    let canonical_source = std::fs::canonicalize(source_dir)
        .map_err(internal_error("resolve source store for verification"))?;
    if canonical_source.to_string_lossy() != manifest.source_dir {
        return Err(validation_error(
            "source_dir differs from the migration manifest",
        ));
    }
    if file_sha256(&source_dir.join(CATALOG_DB_FILENAME))? != manifest.source_catalog_sha256 {
        return Err(validation_error(
            "source catalog changed after migration planning",
        ));
    }
    let expected: BTreeSet<&str> = manifest
        .source_segments
        .iter()
        .map(|segment| segment.relative_path.as_str())
        .collect();
    let mut actual = BTreeSet::new();
    for namespace in migration_source_namespaces() {
        let dir = source_dir.join(namespace);
        if !dir.is_dir() {
            continue;
        }
        for path in discover_segments(&dir) {
            let relative = path
                .strip_prefix(source_dir)
                .map_err(|_| validation_error("source segment escaped source_dir"))?
                .to_string_lossy()
                .into_owned();
            actual.insert(relative);
        }
    }
    if actual.iter().map(String::as_str).collect::<BTreeSet<_>>() != expected {
        return Err(validation_error(
            "source telemetry segment set changed after migration planning",
        ));
    }
    for segment in &manifest.source_segments {
        let path = source_dir.join(&segment.relative_path);
        let metadata = std::fs::metadata(&path).map_err(internal_error("stat source segment"))?;
        if metadata.len() != segment.byte_size {
            return Err(validation_error(format!(
                "source segment changed after planning: {}",
                path.display()
            )));
        }
        match &segment.file_sha256 {
            Some(expected) if &file_sha256(&path)? != expected => {
                return Err(validation_error(format!(
                    "source segment changed after planning: {}",
                    path.display()
                )));
            }
            Some(_) => {}
            None if segment.complete || manifest.complete => {
                return Err(validation_error(format!(
                    "completed source segment has no checksum: {}",
                    path.display()
                )));
            }
            None => {}
        }
        verified_segments += 1;
        verified_bytes += segment.byte_size;
        if last_progress.elapsed() >= PROGRESS_LOG_INTERVAL {
            tracing::info!(
                verified_segments,
                total_segments,
                verified_bytes,
                total_bytes,
                elapsed_seconds = verification_started.elapsed().as_secs(),
                "telemetry migration source verification progress"
            );
            last_progress = Instant::now();
        }
    }
    tracing::info!(
        verified_segments,
        total_segments,
        verified_bytes,
        total_bytes,
        elapsed_seconds = verification_started.elapsed().as_secs(),
        "telemetry migration source verification complete"
    );
    Ok(())
}

fn verify_output_file(
    path: &Path,
    output: &PlannedOutput,
    batch_rows: usize,
) -> Result<(), StatsError> {
    let footer = read_segment_footer(path, Some("timestamp_ms")).ok_or_else(|| {
        validation_error(format!("could not read output segment {}", path.display()))
    })?;
    if footer.row_count != output.rows
        || footer.min_seq != output.min_seq
        || footer.max_seq != output.max_seq
        || footer.partition != output.partition
    {
        return Err(validation_error(format!(
            "output segment metadata differs from plan: {}",
            path.display()
        )));
    }
    let mut identity = Sha256::new();
    for batch in parquet_reader(path, batch_rows)? {
        update_batch_identity(
            &mut identity,
            &batch.map_err(internal_error("read output batch"))?,
        )?;
    }
    if digest_hex(identity) != output.identity_sha256 {
        return Err(validation_error(format!(
            "output segment identity differs from plan: {}",
            path.display()
        )));
    }
    if let Some(expected) = &output.file_sha256 {
        if &file_sha256(path)? != expected {
            return Err(validation_error(format!(
                "output segment bytes changed: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn update_batch_identity(identity: &mut Sha256, batch: &RecordBatch) -> Result<(), StatsError> {
    let clusters = optional_string_column(batch, "cluster")?;
    let batch_ids = string_column(batch, "batch_id")?;
    let record_indices = int64_column(batch, "record_index")?;
    for row in 0..batch.num_rows() {
        update_identity(
            identity,
            clusters
                .as_ref()
                .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
            batch_ids.value(row),
            record_indices.value(row),
        );
    }
    Ok(())
}

fn update_identity(
    identity: &mut Sha256,
    cluster: Option<&str>,
    batch_id: &str,
    record_index: i64,
) {
    update_optional_string(identity, cluster);
    update_string(identity, batch_id);
    identity.update(record_index.to_le_bytes());
}

fn update_optional_string(identity: &mut Sha256, value: Option<&str>) {
    match value {
        Some(value) => {
            identity.update([1]);
            update_string(identity, value);
        }
        None => identity.update([0]),
    }
}

fn update_string(identity: &mut Sha256, value: &str) {
    identity.update((value.len() as u64).to_le_bytes());
    identity.update(value.as_bytes());
}

fn string_column(batch: &RecordBatch, name: &str) -> Result<StringArray, StatsError> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| validation_error(format!("telemetry batch is missing {name:?}")))?;
    let values = cast(batch.column(index), &DataType::Utf8)
        .map_err(internal_error("cast telemetry string column"))?;
    values
        .as_any()
        .downcast_ref::<StringArray>()
        .cloned()
        .ok_or_else(|| validation_error(format!("telemetry column {name:?} is not a string")))
}

fn optional_string_column(
    batch: &RecordBatch,
    name: &str,
) -> Result<Option<StringArray>, StatsError> {
    match batch.schema().index_of(name) {
        Ok(_) => string_column(batch, name).map(Some),
        Err(_) => Ok(None),
    }
}

fn int64_column(batch: &RecordBatch, name: &str) -> Result<Int64Array, StatsError> {
    let index = batch
        .schema()
        .index_of(name)
        .map_err(|_| validation_error(format!("telemetry batch is missing {name:?}")))?;
    let values = cast(batch.column(index), &DataType::Int64)
        .map_err(internal_error("cast telemetry int64 column"))?;
    values
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| validation_error(format!("telemetry column {name:?} is not int64")))
}

fn filter_legacy_rows(
    batch: &RecordBatch,
    legacy_max_seq: Option<i64>,
) -> Result<RecordBatch, StatsError> {
    let Some(legacy_max_seq) = legacy_max_seq else {
        return Ok(batch.clone());
    };
    let seq = int64_column(batch, IMPLICIT_SEQ_COLUMN)?;
    let mut row_indices = Vec::with_capacity(batch.num_rows());
    for row_index in 0..batch.num_rows() {
        if seq.is_null(row_index) {
            return Err(validation_error("telemetry seq values must be non-null"));
        }
        if seq.value(row_index) <= legacy_max_seq {
            row_indices.push(row_index as u32);
        }
    }
    select_rows(batch, row_indices)
}

fn count_rows_at_or_before_seq(
    path: &Path,
    batch_rows: usize,
    legacy_max_seq: i64,
) -> Result<i64, StatsError> {
    let mut rows = 0_i64;
    for batch in parquet_reader_projected(path, batch_rows, &[IMPLICIT_SEQ_COLUMN])? {
        rows += filter_legacy_rows(
            &batch.map_err(internal_error("read migration fence batch"))?,
            Some(legacy_max_seq),
        )?
        .num_rows() as i64;
    }
    Ok(rows)
}

fn parquet_reader(
    path: &Path,
    batch_rows: usize,
) -> Result<impl Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>, StatsError> {
    let file = File::open(path).map_err(internal_error("open parquet segment"))?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(internal_error("open parquet reader"))?
        .with_batch_size(batch_rows)
        .build()
        .map_err(internal_error("build parquet reader"))
}

fn parquet_reader_projected(
    path: &Path,
    batch_rows: usize,
    columns: &[&str],
) -> Result<impl Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>, StatsError> {
    let file = File::open(path).map_err(internal_error("open projected parquet segment"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(internal_error("open projected parquet reader"))?;
    let projection = {
        let parquet_schema = builder.parquet_schema();
        let indices = (0..parquet_schema.num_columns())
            .filter(|index| columns.contains(&parquet_schema.column(*index).name()))
            .collect::<Vec<_>>();
        ProjectionMask::leaves(parquet_schema, indices)
    };
    builder
        .with_projection(projection)
        .with_batch_size(batch_rows)
        .build()
        .map_err(internal_error("build projected parquet reader"))
}

fn file_sha256(path: &Path) -> Result<String, StatsError> {
    let mut file = File::open(path).map_err(internal_error("open file for checksum"))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(internal_error("read file for checksum"))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest_hex(digest))
}

fn digest_hex(digest: Sha256) -> String {
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_manifest_config(
    manifest: &MigrationManifest,
    config: &PrepareConfig,
) -> Result<(), StatsError> {
    if manifest.version != MANIFEST_VERSION || manifest.policy_revision != POLICY_REVISION {
        return Err(validation_error(
            "migration manifest policy version does not match this binary",
        ));
    }
    if manifest.final_log_dir != config.final_log_dir.to_string_lossy() {
        return Err(validation_error(
            "final_log_dir differs from the existing migration manifest",
        ));
    }
    let current_fence = dual_write_fence(&config.final_log_dir)?.map(|fence| fence.legacy_max_seq);
    if manifest.legacy_max_seq != current_fence {
        return Err(validation_error(
            "dual-write fence differs from the existing migration manifest",
        ));
    }
    Ok(())
}

fn read_manifest(path: &Path) -> Result<MigrationManifest, StatsError> {
    let raw = std::fs::read(path).map_err(internal_error("read migration manifest"))?;
    serde_json::from_slice(&raw).map_err(|error| {
        validation_error(format!(
            "decode migration manifest {}: {error}",
            path.display()
        ))
    })
}

fn write_manifest(path: &Path, manifest: &MigrationManifest) -> Result<(), StatsError> {
    write_json_atomically(path, manifest, "migration manifest")
}

fn write_json_atomically<T: Serialize>(
    path: &Path,
    value: &T,
    description: &str,
) -> Result<(), StatsError> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| validation_error(format!("encode {description}: {error}")))?;
    let temporary = temporary_path(path);
    let mut file = File::create(&temporary)
        .map_err(|error| StatsError::Internal(format!("create {description}: {error}")))?;
    file.write_all(&bytes)
        .map_err(|error| StatsError::Internal(format!("write {description}: {error}")))?;
    file.write_all(b"\n")
        .map_err(|error| StatsError::Internal(format!("finish {description}: {error}")))?;
    file.sync_all()
        .map_err(|error| StatsError::Internal(format!("fsync {description}: {error}")))?;
    std::fs::rename(&temporary, path)
        .map_err(|error| StatsError::Internal(format!("publish {description}: {error}")))?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.migration.tmp", path.display()))
}

fn now_ms() -> Result<i64, StatsError> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| StatsError::Internal(format!("read system clock: {error}")))?
        .as_millis();
    i64::try_from(millis)
        .map_err(|error| StatsError::Internal(format!("system clock is out of range: {error}")))
}

fn validation_error(message: impl Into<String>) -> StatsError {
    StatsError::SchemaValidation(message.into())
}

fn internal_error<E: Display>(context: &'static str) -> impl FnOnce(E) -> StatsError {
    move |error| StatsError::Internal(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::time::Duration;

    use arrow::array::{Float64Array, Int32Array, UInt32Array};
    use arrow::datatypes::{Field, Schema};
    use uuid::Uuid;

    use super::*;
    use crate::server::telemetry::telemetry_schema;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::store::{ServeMode, Store};
    use crate::telemetry_policy::{
        IRIS_RPC_NAMESPACE, LEVANTER_NAMESPACE, NODE_AGENT_NAMESPACE, VLLM_NAMESPACE,
        ZEPHYR_NAMESPACE,
    };

    struct TestDirs {
        root: PathBuf,
        store: PathBuf,
    }

    struct PreparedMigration {
        dirs: TestDirs,
        manifest: MigrationManifest,
        source_sha: String,
    }

    impl TestDirs {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "finelog_telemetry_migration_{name}_{}",
                Uuid::new_v4()
            ));
            let store = root.join("store");
            fs::create_dir_all(&store).unwrap();
            Self { root, store }
        }
    }

    impl Drop for TestDirs {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn telemetry_batch(services: &[&str], names: &[&str], min_seq: i64) -> RecordBatch {
        assert_eq!(services.len(), names.len());
        let rows = services.len();
        let schema = schema_to_arrow(&stored_form(telemetry_schema()));
        let columns = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match field.name().as_str() {
                    "seq" => Arc::new(Int64Array::from_iter_values(min_seq..min_seq + rows as i64)),
                    "schema_version" => Arc::new(Int32Array::from(vec![1; rows])),
                    "timestamp_ms" => Arc::new(Int64Array::from_iter_values(
                        (0..rows).map(|row| 1_800_000_000_000_i64 + row as i64),
                    )),
                    "batch_id" => Arc::new(StringArray::from_iter_values(
                        (0..rows).map(|row| format!("batch-{min_seq}-{row}")),
                    )),
                    "record_index" => Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                    "service" => Arc::new(StringArray::from(services.to_vec())),
                    "kind" => Arc::new(StringArray::from(vec!["gauge"; rows])),
                    "name" => Arc::new(StringArray::from(names.to_vec())),
                    "value" => Arc::new(Float64Array::from_iter_values(
                        (0..rows).map(|row| row as f64),
                    )),
                    "resource_attributes_json" | "attributes_json" => {
                        Arc::new(StringArray::from(vec!["{}"; rows]))
                    }
                    "cluster" => Arc::new(StringArray::from(vec![Some("marin"); rows])),
                    _ => new_null_array(field.data_type(), rows),
                }
            })
            .collect();
        RecordBatch::try_new(schema, columns).unwrap()
    }

    fn legacy_levanter_batch(names: &[&str], values: &[f64], min_seq: i64) -> RecordBatch {
        assert_eq!(names.len(), values.len());
        let rows = names.len();
        let schema = schema_to_arrow(&stored_form(telemetry_schema()));
        let columns = schema
            .fields()
            .iter()
            .map(|field| -> ArrayRef {
                match field.name().as_str() {
                    "seq" => Arc::new(Int64Array::from_iter_values(min_seq..min_seq + rows as i64)),
                    "schema_version" => Arc::new(Int32Array::from(vec![1; rows])),
                    "timestamp_ms" => Arc::new(Int64Array::from_iter_values(
                        (0..rows).map(|row| 1_800_000_000_000_i64 + row as i64),
                    )),
                    "batch_id" => Arc::new(StringArray::from(vec!["legacy-batch"; rows])),
                    "record_index" => Arc::new(Int64Array::from_iter_values(0..rows as i64)),
                    "service" => Arc::new(StringArray::from(vec!["levanter"; rows])),
                    "process_index" => Arc::new(StringArray::from(vec![Some("0"); rows])),
                    "kind" => Arc::new(StringArray::from(vec!["gauge"; rows])),
                    "name" => Arc::new(StringArray::from(names.to_vec())),
                    "value" => Arc::new(Float64Array::from(values.to_vec())),
                    "resource_attributes_json" => Arc::new(StringArray::from(vec![
                        "{\"execution_uid\":\"attempt-1\",\"root_run_uid\":\"run/+long\",\"job_id\":\"/job\",\"node_name\":\"node-a\",\"process_index\":\"0\"}";
                        rows
                    ])),
                    "attributes_json" => Arc::new(StringArray::from(vec![
                        "{\"source_kind\":\"gauge\"}";
                        rows
                    ])),
                    "cluster" => Arc::new(StringArray::from(vec![Some("marin"); rows])),
                    _ => new_null_array(field.data_type(), rows),
                }
            })
            .collect();
        RecordBatch::try_new(schema, columns).unwrap()
    }

    fn with_legacy_alert_tag(batch: RecordBatch, nullable: bool) -> RecordBatch {
        let mut fields = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields.push(Field::new("alert_tag", DataType::Utf8, nullable));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(StringArray::from(vec![
            Some("hero");
            batch.num_rows()
        ])));
        RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
    }

    fn add_segment(
        catalog: &Catalog,
        source: &Path,
        namespace: &str,
        level: i32,
        min_seq: i64,
        batch: &RecordBatch,
    ) -> PathBuf {
        let directory = source.join(namespace);
        fs::create_dir_all(&directory).unwrap();
        let (path, byte_size) = write_segment_to_dir(&directory, level, min_seq, batch).unwrap();
        let footer = read_segment_footer(&path, Some("timestamp_ms")).unwrap();
        catalog
            .upsert(namespace, &stored_form(telemetry_schema()))
            .unwrap();
        catalog
            .upsert_segment(&SegmentRow {
                namespace: namespace.to_string(),
                path: path.to_string_lossy().into_owned(),
                level,
                min_seq: footer.min_seq,
                max_seq: footer.max_seq,
                row_count: footer.row_count,
                byte_size,
                created_at_ms: now_ms().unwrap(),
                min_key_value: footer.min_key_value.map(|value| value.to_string()),
                max_key_value: footer.max_key_value.map(|value| value.to_string()),
                partition: footer.partition,
                location: SegmentLocation::Both,
            })
            .unwrap();
        path
    }

    fn add_orphan_segment(
        source: &Path,
        namespace: &str,
        level: i32,
        min_seq: i64,
        batch: &RecordBatch,
    ) -> PathBuf {
        let directory = source.join(namespace);
        fs::create_dir_all(&directory).unwrap();
        write_segment_to_dir(&directory, level, min_seq, batch)
            .unwrap()
            .0
    }

    fn prepared_migration() -> PreparedMigration {
        let dirs = TestDirs::new("prepare");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        let root = add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &telemetry_batch(
                &[
                    "levanter",
                    "levanter",
                    "iris-node-agent",
                    "iris-controller",
                    "vllm",
                    "zephyr",
                    "unowned-service",
                ],
                &[
                    "train_loss",
                    "grad_histogram",
                    "node_cpu_utilization_percent",
                    "rpc_requests_total",
                    "vllm_request_latency",
                    "progress_time_seconds",
                    "custom_metric",
                ],
                1,
            ),
        );
        add_orphan_segment(
            &dirs.store,
            TELEMETRY_NAMESPACE,
            0,
            100,
            &telemetry_batch(&["levanter"], &["orphan_root_row"], 100),
        );
        add_segment(
            &catalog,
            &dirs.store,
            LEVANTER_NAMESPACE,
            1,
            1,
            &telemetry_batch(&["levanter"], &["existing_detail"], 1),
        );
        add_orphan_segment(
            &dirs.store,
            LEVANTER_NAMESPACE,
            0,
            100,
            &telemetry_batch(&["levanter"], &["orphan_detail"], 100),
        );
        drop(catalog);
        let source_sha = file_sha256(&root).unwrap();
        let manifest = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        PreparedMigration {
            dirs,
            manifest,
            source_sha,
        }
    }

    #[test]
    fn prepare_in_place_routes_every_root_row_and_preserves_the_live_store() {
        let PreparedMigration {
            dirs,
            manifest,
            source_sha,
        } = prepared_migration();

        assert_eq!(manifest.input_rows, 7);
        assert_eq!(manifest.output_rows, 7);
        assert!(manifest.complete);
        assert_eq!(manifest.phase, MigrationPhase::Staged);
        assert!(manifest.verified_at_ms.is_none());
        assert_eq!(
            file_sha256(
                &dirs
                    .store
                    .join("telemetry_v1/seg_L1_0000000000000000001.parquet")
            )
            .unwrap(),
            source_sha
        );
        assert!(manifest
            .source_segments
            .iter()
            .flat_map(|source| source.outputs.iter())
            .all(|output| output.min_seq < 0 && output.max_seq < 0));
        let levanter_outputs = manifest
            .source_segments
            .iter()
            .flat_map(|source| source.outputs.iter())
            .filter(|output| output.namespace == LEVANTER_NAMESPACE)
            .collect::<Vec<_>>();
        assert_eq!(levanter_outputs.len(), 1);
        assert!(levanter_outputs[0].partition.is_none());
        let staged_dir = dirs.store.join(MIGRATION_DIRECTORY).join(STAGED_DIRECTORY);
        assert!(manifest
            .source_segments
            .iter()
            .flat_map(|source| source.outputs.iter())
            .all(|output| staged_dir.join(&output.relative_path).is_file()));
    }

    #[test]
    fn migration_stops_at_the_persistent_dual_write_fence() {
        let dirs = TestDirs::new("dual-write-fence");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &telemetry_batch(
                &[
                    "iris-node-agent",
                    "iris-node-agent",
                    "iris-node-agent",
                    "iris-node-agent",
                ],
                &["cpu_1", "cpu_2", "cpu_3", "cpu_4"],
                1,
            ),
        );
        drop(catalog);

        let first = ensure_dual_write_fence(&dirs.store, 2).unwrap();
        let repeated = ensure_dual_write_fence(&dirs.store, 99).unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first.legacy_max_seq, 2);

        let manifest = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        assert_eq!(manifest.legacy_max_seq, Some(2));
        assert_eq!(manifest.input_rows, 2);
        assert_eq!(manifest.output_rows, 2);
    }

    #[test]
    fn prepare_in_place_projects_legacy_optional_columns_into_current_schema() {
        let dirs = TestDirs::new("legacy_optional_column");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &with_legacy_alert_tag(telemetry_batch(&["levanter"], &["train_loss"], 1), true),
        );
        drop(catalog);

        let manifest = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();

        assert_eq!(manifest.input_rows, 1);
        assert_eq!(manifest.output_rows, 1);
        let staged_path = dirs
            .store
            .join(MIGRATION_DIRECTORY)
            .join(STAGED_DIRECTORY)
            .join(&manifest.source_segments[0].outputs[0].relative_path);
        let batch = parquet_reader(&staged_path, 2)
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        assert!(batch.column_by_name("alert_tag").is_none());
        assert_eq!(
            string_column(&batch, "name").unwrap().value(0),
            "train_loss"
        );
    }

    #[test]
    fn prepare_in_place_indexes_steps_independently_of_physical_row_order() {
        let dirs = TestDirs::new("legacy_levanter_metrics");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        let chronological =
            legacy_levanter_batch(&["step", "global_step", "train_loss"], &[7.0, 8.0, 0.25], 1);
        let physical_order = UInt32Array::from(vec![2, 0, 1]);
        let reordered = RecordBatch::try_new(
            chronological.schema(),
            chronological
                .columns()
                .iter()
                .map(|column| arrow::compute::take(column, &physical_order, None).unwrap())
                .collect(),
        )
        .unwrap();
        add_segment(&catalog, &dirs.store, TELEMETRY_NAMESPACE, 1, 1, &reordered);
        drop(catalog);

        let manifest = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();

        assert_eq!(manifest.input_rows, 3);
        assert_eq!(manifest.output_rows, 1);
        assert_eq!(manifest.suppressed_rows, 2);
        let output = &manifest.source_segments[0].outputs[0];
        assert_eq!(
            output.namespace,
            crate::levanter_metrics_policy::LEVANTER_METRICS_NAMESPACE
        );
        assert_eq!(output.rows, 1);
        assert_eq!(
            output
                .partition
                .as_ref()
                .and_then(|partition| partition.value("run_id")),
            Some("run/+long")
        );

        let staged_path = dirs
            .store
            .join(MIGRATION_DIRECTORY)
            .join(STAGED_DIRECTORY)
            .join(&output.relative_path);
        let batch = ParquetRecordBatchReaderBuilder::try_new(File::open(staged_path).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .next()
            .unwrap()
            .unwrap();
        let names = batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let steps = batch
            .column_by_name("step")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(names.value(0), "train_loss");
        assert_eq!(steps.value(0), 8);
    }

    #[test]
    fn prepare_in_place_rejects_unknown_required_columns() {
        let dirs = TestDirs::new("unknown_required_column");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &with_legacy_alert_tag(telemetry_batch(&["levanter"], &["train_loss"], 1), false),
        );
        drop(catalog);

        assert!(matches!(
            prepare_in_place(&InPlaceConfig {
                store_dir: dirs.store.clone(),
                batch_rows: 2,
            }),
            Err(StatsError::SchemaValidation(_))
        ));
    }

    #[test]
    fn migration_plan_reads_metadata_without_evaluating_rows() {
        let dirs = TestDirs::new("metadata_only_plan");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &with_legacy_alert_tag(telemetry_batch(&["levanter"], &["train_loss"], 1), false),
        );
        drop(catalog);
        let output_dir = dirs.root.join("staged");
        let config = PrepareConfig {
            source_dir: dirs.store.clone(),
            output_dir,
            final_log_dir: dirs.store.clone(),
            batch_rows: 2,
        };

        let manifest = plan_migration(&config).unwrap();

        assert_eq!(manifest.input_rows, 1);
        assert_eq!(manifest.output_rows, 0);
        assert!(!manifest.complete);
        assert_eq!(manifest.source_segments.len(), 1);
        assert!(!manifest.source_segments[0].complete);
        assert!(manifest.source_segments[0].file_sha256.is_none());
        assert!(manifest.source_segments[0].outputs.is_empty());
    }

    #[test]
    fn prepare_in_place_resume_reuses_verified_outputs_without_duplicates() {
        let PreparedMigration {
            dirs,
            manifest: first,
            ..
        } = prepared_migration();
        let second = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 3,
        })
        .unwrap();

        assert_eq!(second, first);
    }

    #[tokio::test]
    async fn publish_then_retire_switches_visibility_without_rewriting_again() {
        let PreparedMigration { dirs, .. } = prepared_migration();
        verify_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        let published = publish_in_place(&dirs.store).unwrap();
        assert_eq!(published.phase, MigrationPhase::Published);
        assert!(published.verified_at_ms.is_some());
        assert_eq!(
            published.published_files.len(),
            published
                .source_segments
                .iter()
                .map(|source| source.outputs.len())
                .sum::<usize>()
        );
        let rewritten = &published.source_segments[0].outputs[0];
        let published_path = dirs.store.join(&rewritten.relative_path);
        let compacted_path = published_path
            .parent()
            .unwrap()
            .join(seg_filename(1, rewritten.min_seq));
        fs::rename(&published_path, &compacted_path).unwrap();
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        catalog
            .remove_segment(&rewritten.namespace, &published_path.to_string_lossy())
            .unwrap();
        catalog
            .upsert_segment(&SegmentRow {
                namespace: rewritten.namespace.clone(),
                path: compacted_path.to_string_lossy().into_owned(),
                level: 1,
                min_seq: rewritten.min_seq,
                max_seq: rewritten.max_seq,
                row_count: rewritten.rows,
                byte_size: fs::metadata(&compacted_path).unwrap().len() as i64,
                created_at_ms: now_ms().unwrap(),
                min_key_value: Some(rewritten.min_timestamp_ms.to_string()),
                max_key_value: Some(rewritten.max_timestamp_ms.to_string()),
                partition: None,
                location: SegmentLocation::Local,
            })
            .unwrap();
        drop(catalog);
        verify_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        let repeated_prepare = prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        assert_eq!(repeated_prepare.phase, MigrationPhase::Published);

        let store = Store::new(
            Some(dirs.store.clone()),
            String::new(),
            1,
            ServeMode::Shadow,
        )
        .expect("prepared store should boot");
        let provider_names = store
            .query_providers()
            .expect("prepared store should build query providers")
            .into_iter()
            .map(|provider| provider.name)
            .collect::<BTreeSet<_>>();

        assert!(provider_names.contains(TELEMETRY_NAMESPACE));
        assert!(provider_names.contains(LEVANTER_NAMESPACE));
        assert!(provider_names.contains(NODE_AGENT_NAMESPACE));
        assert!(provider_names.contains(IRIS_RPC_NAMESPACE));
        assert!(provider_names.contains(VLLM_NAMESPACE));
        assert!(provider_names.contains(ZEPHYR_NAMESPACE));
        assert!(provider_names.contains("telemetry_v1.unowned_service"));

        store.shutdown(Duration::from_secs(1)).await;
        drop(store);

        let retired = retire_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        assert_eq!(retired.phase, MigrationPhase::Retired);
        assert!(!retired.retired_files.is_empty());
        verify_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();

        let store = Store::new(
            Some(dirs.store.clone()),
            String::new(),
            1,
            ServeMode::Shadow,
        )
        .expect("retired store should boot");
        let provider_names = store
            .query_providers()
            .unwrap()
            .into_iter()
            .map(|provider| provider.name)
            .collect::<BTreeSet<_>>();
        assert!(!provider_names.contains(TELEMETRY_NAMESPACE));
        assert!(provider_names.contains(LEVANTER_NAMESPACE));
        store.shutdown(Duration::from_secs(1)).await;
    }

    #[tokio::test]
    async fn publish_refuses_a_store_held_by_a_running_server() {
        let PreparedMigration { dirs, .. } = prepared_migration();
        verify_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        let store = Store::new(
            Some(dirs.store.clone()),
            String::new(),
            1,
            ServeMode::Shadow,
        )
        .unwrap();

        assert!(publish_in_place(&dirs.store).is_err());
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
    }

    #[test]
    fn publish_requires_a_completed_pre_cutover_verification() {
        let PreparedMigration { dirs, .. } = prepared_migration();

        assert!(publish_in_place(&dirs.store).is_err());
        let manifest = read_manifest(
            &dirs
                .store
                .join(MIGRATION_DIRECTORY)
                .join(STAGED_DIRECTORY)
                .join(MANIFEST_FILENAME),
        )
        .unwrap();
        assert_eq!(manifest.phase, MigrationPhase::Staged);
        assert!(manifest.published_files.is_empty());
    }

    #[test]
    fn verify_in_place_rejects_a_changed_source_snapshot() {
        let PreparedMigration { dirs, .. } = prepared_migration();
        let root = dirs
            .store
            .join(MIGRATION_DIRECTORY)
            .join(SOURCE_SNAPSHOT_DIRECTORY)
            .join("telemetry_v1/seg_L1_0000000000000000001.parquet");
        let mut file = OpenOptions::new().append(true).open(root).unwrap();
        file.write_all(b"changed").unwrap();

        assert!(verify_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .is_err());
    }

    #[test]
    fn prepare_in_place_rejects_a_forwarding_sender() {
        let dirs = TestDirs::new("forwarding");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        catalog
            .set_forward_cursor("hub", TELEMETRY_NAMESPACE, 12)
            .unwrap();
        drop(catalog);

        assert!(prepare_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .is_err());
    }
}
