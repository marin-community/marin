//! In-place migration from the production telemetry root into semantic storage shards.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{new_null_array, Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::ingestion_policy::IngestionBatchSource;
use crate::policies::{eager_storage_namespaces_for, route_ingestion_batch, storage_policy_for};
use crate::server::telemetry::{telemetry_schema, TELEMETRY_MAX_ROW_GROUP_ROWS};
use crate::store::catalog::{Catalog, CATALOG_DB_FILENAME};
use crate::store::schema::{schema_to_arrow, stored_form, IMPLICIT_SEQ_COLUMN};
use crate::store::segment::{
    discover_segments, read_segment_footer, segment_writer_properties_with_max_rows,
};
use crate::store::store::acquire_exclusive_store_lock;
use crate::store::types::{seg_filename, SegmentLocation, SegmentRow};
use crate::telemetry_policy::TELEMETRY_NAMESPACE;

const MANIFEST_FILENAME: &str = ".finelog-telemetry-v1-migration.json";
const MANIFEST_VERSION: u32 = 1;
const POLICY_REVISION: &str = "semantic-storage-v4";
const MIGRATION_SOURCE_NAMESPACES: [&str; 1] = [TELEMETRY_NAMESPACE];
const OUTPUT_LEVEL: i32 = 0;
const MIGRATION_DIRECTORY: &str = ".finelog-telemetry-v1-migration";
const SOURCE_SNAPSHOT_DIRECTORY: &str = "source";
const STAGED_DIRECTORY: &str = "staged";
const ROLLBACK_DIRECTORY: &str = "rollback";
const CATALOG_BUILD_DIRECTORY: &str = "catalog-build";
const MIGRATED_SEQ_START: i64 = -4_000_000_000_000_000_000;
const SNAPSHOT_ATTEMPTS: usize = 8;

fn migration_source_namespaces() -> impl Iterator<Item = &'static str> {
    MIGRATION_SOURCE_NAMESPACES.iter().copied()
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
    pub input_rows: i64,
    pub output_rows: i64,
    pub residual_rows: i64,
    pub source_segments: Vec<SourceSegment>,
    #[serde(default)]
    pub published_files: Vec<String>,
    #[serde(default)]
    pub retired_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSegment {
    pub namespace: String,
    pub relative_path: String,
    pub byte_size: u64,
    pub file_sha256: String,
    pub rows: i64,
    pub outputs: Vec<PlannedOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlannedOutput {
    pub namespace: String,
    pub relative_path: String,
    pub min_seq: i64,
    pub max_seq: i64,
    pub rows: i64,
    pub min_timestamp_ms: i64,
    pub max_timestamp_ms: i64,
    pub identity_sha256: String,
    pub file_sha256: Option<String>,
}

#[derive(Default)]
struct DestinationStats {
    rows: i64,
    min_timestamp_ms: Option<i64>,
    max_timestamp_ms: Option<i64>,
    identity: Sha256,
}

struct DestinationWriter {
    output_index: usize,
    next_seq: i64,
    rows: i64,
    identity: Sha256,
    temporary_path: PathBuf,
    final_path: PathBuf,
    writer: ArrowWriter<File>,
}

pub fn prepare_store(config: &PrepareConfig) -> Result<MigrationManifest, StatsError> {
    validate_config(config)?;
    let manifest_path = config.output_dir.join(MANIFEST_FILENAME);
    if config.output_dir.exists()
        && !manifest_path.exists()
        && std::fs::read_dir(&config.output_dir)
            .map_err(io_error("list existing output store"))?
            .next()
            .is_some()
    {
        return Err(validation_error(
            "output_dir is non-empty and has no migration manifest",
        ));
    }
    std::fs::create_dir_all(&config.output_dir).map_err(io_error("create output store"))?;

    let mut manifest = if manifest_path.exists() {
        let manifest = read_manifest(&manifest_path)?;
        validate_manifest_config(&manifest, config)?;
        manifest
    } else {
        let manifest = plan_migration(config)?;
        write_manifest(&manifest_path, &manifest)?;
        manifest
    };

    verify_source_segments(&config.source_dir, &manifest)?;
    if manifest.complete {
        return verify_store(&config.source_dir, &config.output_dir, config.batch_rows);
    }
    write_planned_outputs(config, &mut manifest)?;
    manifest.complete = true;
    manifest.phase = MigrationPhase::Staged;
    write_manifest(&manifest_path, &manifest)?;
    verify_store(&config.source_dir, &config.output_dir, config.batch_rows)
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
    verify_source_segments(source_dir, &manifest)?;
    for source in &manifest.source_segments {
        for output in &source.outputs {
            let path = output_dir.join(&output.relative_path);
            verify_output_file(&path, output, batch_rows)?;
        }
    }
    if manifest.input_rows != manifest.output_rows {
        return Err(validation_error(format!(
            "migration row mismatch: {} source rows, {} output rows",
            manifest.input_rows, manifest.output_rows
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
        .map_err(io_error("resolve in-place telemetry store"))?;
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
        .map_err(io_error("resolve in-place telemetry store"))?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let source_dir = migration_dir.join(SOURCE_SNAPSHOT_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    let manifest = verify_store(&source_dir, &staged_dir, config.batch_rows)?;
    if manifest.phase != MigrationPhase::Staged {
        verify_published_catalog(&store_dir, &manifest)?;
    }
    if manifest.phase == MigrationPhase::Retired {
        verify_root_namespace_retired(&store_dir)?;
    }
    Ok(manifest)
}

/// Make the staged semantic rows queryable during a stopped-server cutover.
pub fn publish_in_place(config: &InPlaceConfig) -> Result<MigrationManifest, StatsError> {
    let store_dir = std::fs::canonicalize(&config.store_dir)
        .map_err(io_error("resolve in-place telemetry store"))?;
    let _store_lock = acquire_exclusive_store_lock(&store_dir)?;
    let migration_dir = store_dir.join(MIGRATION_DIRECTORY);
    let source_dir = migration_dir.join(SOURCE_SNAPSHOT_DIRECTORY);
    let staged_dir = migration_dir.join(STAGED_DIRECTORY);
    let manifest_path = staged_dir.join(MANIFEST_FILENAME);
    let mut manifest = verify_store(&source_dir, &staged_dir, config.batch_rows)?;
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
        verify_published_catalog(&store_dir, &manifest)?;
    } else {
        replace_catalog_for_publish(&store_dir, &migration_dir, &manifest)?;
    }
    manifest.published_files = published_files;
    manifest.phase = MigrationPhase::Published;
    write_manifest(&manifest_path, &manifest)?;
    verify_published_catalog(&store_dir, &manifest)?;
    Ok(manifest)
}

/// Remove the root namespace after every query has switched to semantic names.
///
/// Stop Finelog before this command and restart it after the command returns.
pub fn retire_in_place(config: &InPlaceConfig) -> Result<MigrationManifest, StatsError> {
    let store_dir = std::fs::canonicalize(&config.store_dir)
        .map_err(io_error("resolve in-place telemetry store"))?;
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
    verify_published_catalog(&store_dir, &manifest)?;

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
                    .map_err(io_error("create root rollback directory"))?;
            }
            std::fs::rename(&source, &destination)
                .map_err(io_error("move root segment into rollback"))?;
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
    std::fs::create_dir_all(migration_dir).map_err(io_error("create migration directory"))?;
    let temporary = migration_dir.join(format!("{SOURCE_SNAPSHOT_DIRECTORY}.tmp"));
    for attempt in 0..SNAPSHOT_ATTEMPTS {
        if temporary.exists() {
            std::fs::remove_dir_all(&temporary)
                .map_err(io_error("remove interrupted source snapshot"))?;
        }
        std::fs::create_dir_all(&temporary).map_err(io_error("create source snapshot"))?;
        copy_catalog_consistently(
            &store_dir.join(CATALOG_DB_FILENAME),
            &temporary.join(CATALOG_DB_FILENAME),
        )?;
        if link_catalog_snapshot_segments(store_dir, &temporary)? {
            std::fs::rename(&temporary, source_dir).map_err(io_error("publish source snapshot"))?;
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
                    .map_err(io_error("create snapshot namespace directory"))?;
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
        std::fs::remove_file(destination).map_err(io_error("remove old catalog snapshot"))?;
    }
    let connection =
        rusqlite::Connection::open_with_flags(source, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(sqlite_error("open catalog for snapshot"))?;
    connection
        .execute(
            "VACUUM main INTO ?1",
            [destination.to_string_lossy().as_ref()],
        )
        .map_err(sqlite_error("snapshot catalog"))?;
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
        std::fs::create_dir_all(parent).map_err(io_error("create physical namespace"))?;
    }
    std::fs::hard_link(source, destination).map_err(io_error("publish migrated segment"))?;
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
    let stored_schema = stored_form(telemetry_schema());
    for namespace in eager_storage_namespaces_for(TELEMETRY_NAMESPACE) {
        catalog.upsert(namespace, &stored_schema)?;
        catalog.upsert_policy(namespace, &storage_policy_for(namespace)?)?;
    }
    let output_namespaces: BTreeSet<_> = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| output.namespace.as_str())
        .collect();
    for namespace in output_namespaces {
        catalog.upsert(namespace, &stored_schema)?;
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
                    .map_err(io_error("stat published telemetry segment"))?
                    .len() as i64,
                created_at_ms,
                min_key_value: Some(output.min_timestamp_ms.to_string()),
                max_key_value: Some(output.max_timestamp_ms.to_string()),
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
            .map_err(io_error("remove interrupted catalog build"))?;
    }
    std::fs::create_dir_all(&build_dir).map_err(io_error("create catalog build directory"))?;
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
        .map_err(io_error("create catalog rollback directory"))?;
    let current = store_dir.join(CATALOG_DB_FILENAME);
    let candidate = build_dir.join(CATALOG_DB_FILENAME);
    let backup = catalog_backup_path(migration_dir, backup_name);
    if backup.exists() {
        return Err(validation_error(format!(
            "catalog rollback already exists: {}",
            backup.display()
        )));
    }
    std::fs::rename(&current, &backup).map_err(io_error("save catalog rollback"))?;
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
            .map_err(io_error("save catalog rollback journal"))?;
    }
    Ok(())
}

fn catalog_backup_path(migration_dir: &Path, backup_name: &str) -> PathBuf {
    migration_dir
        .join(ROLLBACK_DIRECTORY)
        .join(format!("{backup_name}.sqlite"))
}

fn verify_published_catalog(
    store_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        store_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(sqlite_error("open published catalog"))?;
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
            .map_err(sqlite_error("verify published telemetry catalog row"))?;
        if count != 1 {
            return Err(validation_error(format!(
                "published telemetry catalog row is missing for {}",
                path.display()
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
    .map_err(sqlite_error("open retired telemetry catalog"))?;
    for namespace in migration_source_namespaces() {
        let count: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM namespaces WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_error("verify retired telemetry namespace"))?;
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
    let source =
        std::fs::canonicalize(&config.source_dir).map_err(io_error("resolve source store"))?;
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
        .map_err(io_error("resolve relative path"))
}

fn assert_catalog_is_quiescent(source_dir: &Path) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        source_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(sqlite_error("open source catalog read-only"))?;
    let check: String = connection
        .query_row("PRAGMA quick_check", [], |row| row.get(0))
        .map_err(sqlite_error("check source catalog"))?;
    if check != "ok" {
        return Err(validation_error(format!(
            "source catalog quick_check failed: {check}"
        )));
    }
    let forwarding_rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM forward_state", [], |row| row.get(0))
        .map_err(sqlite_error("inspect forwarding state"))?;
    if forwarding_rows != 0 {
        return Err(validation_error(
            "telemetry migration is hub-only; source catalog has forwarding state",
        ));
    }
    Ok(())
}

fn plan_migration(config: &PrepareConfig) -> Result<MigrationManifest, StatsError> {
    let mut next_seq = BTreeMap::new();
    let mut source_segments = Vec::new();
    let mut input_rows = 0_i64;
    let mut output_rows = 0_i64;
    let residual_rows = 0_i64;

    for namespace in migration_source_namespaces() {
        let namespace_dir = config.source_dir.join(namespace);
        if !namespace_dir.is_dir() {
            continue;
        }
        for path in discover_segments(&namespace_dir) {
            let metadata = std::fs::metadata(&path).map_err(io_error("stat source segment"))?;
            let stats = scan_source_segment(&path, namespace, config.batch_rows)?;
            let rows = stats.values().map(|item| item.rows).sum::<i64>();
            let footer = read_segment_footer(&path, Some("timestamp_ms")).ok_or_else(|| {
                validation_error(format!("could not read source segment {}", path.display()))
            })?;
            if rows != footer.row_count {
                return Err(validation_error(format!(
                    "classified {rows}/{} rows in {}",
                    footer.row_count,
                    path.display()
                )));
            }
            let mut outputs = Vec::new();
            for (destination, item) in stats {
                if item.rows == 0 {
                    continue;
                }
                let min_seq = *next_seq
                    .entry(destination.clone())
                    .or_insert(MIGRATED_SEQ_START);
                let max_seq = min_seq + item.rows - 1;
                next_seq.insert(destination.clone(), max_seq + 1);
                let relative_path = Path::new(&destination)
                    .join(seg_filename(OUTPUT_LEVEL, min_seq))
                    .to_string_lossy()
                    .into_owned();
                let min_timestamp_ms = item.min_timestamp_ms.ok_or_else(|| {
                    validation_error(format!(
                        "destination {destination:?} has rows but no minimum timestamp"
                    ))
                })?;
                let max_timestamp_ms = item.max_timestamp_ms.ok_or_else(|| {
                    validation_error(format!(
                        "destination {destination:?} has rows but no maximum timestamp"
                    ))
                })?;
                debug_assert_ne!(destination, TELEMETRY_NAMESPACE);
                outputs.push(PlannedOutput {
                    namespace: destination,
                    relative_path,
                    min_seq,
                    max_seq,
                    rows: item.rows,
                    min_timestamp_ms,
                    max_timestamp_ms,
                    identity_sha256: digest_hex(item.identity),
                    file_sha256: None,
                });
            }
            outputs.sort_by(|left, right| left.namespace.cmp(&right.namespace));
            input_rows += rows;
            output_rows += outputs.iter().map(|output| output.rows).sum::<i64>();
            let relative_path = path
                .strip_prefix(&config.source_dir)
                .map_err(|_| validation_error("source segment escaped source_dir"))?
                .to_string_lossy()
                .into_owned();
            source_segments.push(SourceSegment {
                namespace: namespace.to_string(),
                relative_path,
                byte_size: metadata.len(),
                file_sha256: file_sha256(&path)?,
                rows,
                outputs,
            });
        }
    }
    Ok(MigrationManifest {
        version: MANIFEST_VERSION,
        policy_revision: POLICY_REVISION.to_string(),
        source_dir: std::fs::canonicalize(&config.source_dir)
            .map_err(io_error("resolve source store"))?
            .to_string_lossy()
            .into_owned(),
        source_catalog_sha256: file_sha256(&config.source_dir.join(CATALOG_DB_FILENAME))?,
        final_log_dir: config.final_log_dir.to_string_lossy().into_owned(),
        complete: false,
        phase: MigrationPhase::Staged,
        input_rows,
        output_rows,
        residual_rows,
        source_segments,
        published_files: Vec::new(),
        retired_files: Vec::new(),
    })
}

fn scan_source_segment(
    path: &Path,
    source_namespace: &str,
    batch_rows: usize,
) -> Result<BTreeMap<String, DestinationStats>, StatsError> {
    let reader = parquet_reader(path, batch_rows)?;
    let mut stats: BTreeMap<String, DestinationStats> = BTreeMap::new();
    for batch in reader {
        let batch = batch.map_err(arrow_error("read source batch"))?;
        for partition in
            route_ingestion_batch(IngestionBatchSource::Stored(source_namespace), &batch)?
        {
            let destination = partition.destination.physical_namespace;
            let batch_ids = string_column(&partition.batch, "batch_id")?;
            let record_indices = int64_column(&partition.batch, "record_index")?;
            let timestamps = int64_column(&partition.batch, "timestamp_ms")?;
            let clusters = optional_string_column(&partition.batch, "cluster")?;
            let item = stats.entry(destination).or_default();
            for row in 0..partition.batch.num_rows() {
                let timestamp = timestamps.value(row);
                item.rows += 1;
                item.min_timestamp_ms = Some(
                    item.min_timestamp_ms
                        .map_or(timestamp, |value| value.min(timestamp)),
                );
                item.max_timestamp_ms = Some(
                    item.max_timestamp_ms
                        .map_or(timestamp, |value| value.max(timestamp)),
                );
                update_identity(
                    &mut item.identity,
                    clusters
                        .as_ref()
                        .and_then(|values| (!values.is_null(row)).then(|| values.value(row))),
                    batch_ids.value(row),
                    record_indices.value(row),
                );
            }
        }
    }
    Ok(stats)
}

fn write_planned_outputs(
    config: &PrepareConfig,
    manifest: &mut MigrationManifest,
) -> Result<(), StatsError> {
    let target_schema = schema_to_arrow(&stored_form(telemetry_schema()));
    let manifest_path = config.output_dir.join(MANIFEST_FILENAME);
    for source_index in 0..manifest.source_segments.len() {
        let source_path = config
            .source_dir
            .join(&manifest.source_segments[source_index].relative_path);
        write_source_outputs(
            &source_path,
            config,
            &target_schema,
            &mut manifest.source_segments[source_index],
        )?;
        write_manifest(&manifest_path, manifest)?;
    }
    Ok(())
}

fn write_source_outputs(
    source_path: &Path,
    config: &PrepareConfig,
    target_schema: &SchemaRef,
    source: &mut SourceSegment,
) -> Result<(), StatsError> {
    for output in &mut source.outputs {
        let path = config.output_dir.join(&output.relative_path);
        if path.exists() {
            verify_output_file(&path, output, config.batch_rows)?;
            output.file_sha256 = Some(file_sha256(&path)?);
        }
    }
    let missing: BTreeSet<usize> = source
        .outputs
        .iter()
        .enumerate()
        .filter_map(|(index, output)| output.file_sha256.is_none().then_some(index))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    let mut writers = BTreeMap::new();
    for index in missing {
        let output = &source.outputs[index];
        let final_path = config.output_dir.join(&output.relative_path);
        let parent = final_path
            .parent()
            .ok_or_else(|| validation_error("output segment has no parent"))?;
        std::fs::create_dir_all(parent).map_err(io_error("create output namespace"))?;
        let temporary_path = temporary_path(&final_path);
        if temporary_path.exists() {
            std::fs::remove_file(&temporary_path)
                .map_err(io_error("remove interrupted output segment"))?;
        }
        let file = File::create(&temporary_path).map_err(io_error("create output segment"))?;
        let options =
            ArrowWriterOptions::new().with_properties(segment_writer_properties_with_max_rows(
                usize::try_from(TELEMETRY_MAX_ROW_GROUP_ROWS)
                    .expect("telemetry row-group limit fits usize"),
            )?);
        let writer = ArrowWriter::try_new_with_options(file, Arc::clone(target_schema), options)
            .map_err(parquet_error("create output parquet writer"))?;
        writers.insert(
            output.namespace.clone(),
            DestinationWriter {
                output_index: index,
                next_seq: output.min_seq,
                rows: 0,
                identity: Sha256::new(),
                temporary_path,
                final_path,
                writer,
            },
        );
    }

    let reader = parquet_reader(source_path, config.batch_rows)?;
    for batch in reader {
        let batch = batch.map_err(arrow_error("read source batch"))?;
        for partition in route_ingestion_batch(
            IngestionBatchSource::Stored(source.namespace.as_str()),
            &batch,
        )? {
            let namespace = partition.destination.physical_namespace;
            let Some(writer) = writers.get_mut(&namespace) else {
                continue;
            };
            let migrated = align_migrated_batch(&partition.batch, target_schema, writer.next_seq)?;
            update_batch_identity(&mut writer.identity, &migrated)?;
            writer
                .writer
                .write(&migrated)
                .map_err(parquet_error("write migrated telemetry"))?;
            writer.rows += migrated.num_rows() as i64;
            writer.next_seq += migrated.num_rows() as i64;
        }
    }

    for (_namespace, writer) in writers {
        let output = &mut source.outputs[writer.output_index];
        if writer.rows != output.rows || digest_hex(writer.identity) != output.identity_sha256 {
            return Err(validation_error(format!(
                "output {} did not match its plan",
                output.relative_path
            )));
        }
        let file = writer
            .writer
            .into_inner()
            .map_err(parquet_error("close output segment"))?;
        file.sync_all().map_err(io_error("fsync output segment"))?;
        std::fs::rename(&writer.temporary_path, &writer.final_path)
            .map_err(io_error("publish output segment"))?;
        output.file_sha256 = Some(file_sha256(&writer.final_path)?);
    }
    Ok(())
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
                    cast(array, field.data_type()).map_err(arrow_error("align telemetry column"))?
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
        .map_err(arrow_error("build migrated telemetry batch"))
}

fn verify_source_segments(
    source_dir: &Path,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let canonical_source = std::fs::canonicalize(source_dir)
        .map_err(io_error("resolve source store for verification"))?;
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
        let metadata = std::fs::metadata(&path).map_err(io_error("stat source segment"))?;
        if metadata.len() != segment.byte_size || file_sha256(&path)? != segment.file_sha256 {
            return Err(validation_error(format!(
                "source segment changed after planning: {}",
                path.display()
            )));
        }
    }
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
            &batch.map_err(arrow_error("read output batch"))?,
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
        .map_err(arrow_error("cast telemetry string column"))?;
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
        .map_err(arrow_error("cast telemetry int64 column"))?;
    values
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| validation_error(format!("telemetry column {name:?} is not int64")))
}

fn parquet_reader(
    path: &Path,
    batch_rows: usize,
) -> Result<impl Iterator<Item = Result<RecordBatch, arrow::error::ArrowError>>, StatsError> {
    let file = File::open(path).map_err(io_error("open parquet segment"))?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(parquet_error("open parquet reader"))?
        .with_batch_size(batch_rows)
        .build()
        .map_err(parquet_error("build parquet reader"))
}

fn file_sha256(path: &Path) -> Result<String, StatsError> {
    let mut file = File::open(path).map_err(io_error("open file for checksum"))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(io_error("read file for checksum"))?;
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
    Ok(())
}

fn read_manifest(path: &Path) -> Result<MigrationManifest, StatsError> {
    let raw = std::fs::read(path).map_err(io_error("read migration manifest"))?;
    serde_json::from_slice(&raw).map_err(|error| {
        validation_error(format!(
            "decode migration manifest {}: {error}",
            path.display()
        ))
    })
}

fn write_manifest(path: &Path, manifest: &MigrationManifest) -> Result<(), StatsError> {
    let bytes = serde_json::to_vec_pretty(manifest)
        .map_err(|error| validation_error(format!("encode migration manifest: {error}")))?;
    let temporary = temporary_path(path);
    let mut file = File::create(&temporary).map_err(io_error("create migration manifest"))?;
    file.write_all(&bytes)
        .map_err(io_error("write migration manifest"))?;
    file.write_all(b"\n")
        .map_err(io_error("finish migration manifest"))?;
    file.sync_all()
        .map_err(io_error("fsync migration manifest"))?;
    std::fs::rename(&temporary, path).map_err(io_error("publish migration manifest"))?;
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

fn io_error(context: &'static str) -> impl FnOnce(std::io::Error) -> StatsError {
    move |error| StatsError::Internal(format!("{context}: {error}"))
}

fn sqlite_error(context: &'static str) -> impl FnOnce(rusqlite::Error) -> StatsError {
    move |error| StatsError::Internal(format!("{context}: {error}"))
}

fn parquet_error(
    context: &'static str,
) -> impl FnOnce(parquet::errors::ParquetError) -> StatsError {
    move |error| StatsError::Internal(format!("{context}: {error}"))
}

fn arrow_error(context: &'static str) -> impl FnOnce(arrow::error::ArrowError) -> StatsError {
    move |error| StatsError::Internal(format!("{context}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::fs::{self, OpenOptions};
    use std::time::Duration;

    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::store::{ServeMode, Store};
    use crate::telemetry_policy::{
        IRIS_RPC_NAMESPACE, LEVANTER_DETAIL_STORAGE_NAMESPACE, LEVANTER_NAMESPACE,
        LEVANTER_STATUS_STORAGE_NAMESPACE, NODE_AGENT_NAMESPACE, VLLM_NAMESPACE, ZEPHYR_NAMESPACE,
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
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let root =
                std::env::temp_dir().join(format!("finelog_telemetry_migration_{name}_{nonce}"));
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

    fn with_legacy_alert_tag(batch: RecordBatch, nullable: bool) -> RecordBatch {
        let mut fields = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields.push(Field::new("alert_tag", DataType::Utf8, nullable));
        let mut columns = batch.columns().to_vec();
        columns.push(Arc::new(StringArray::from(vec![Some("hero"); batch.num_rows()])));
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
            "telemetry_storage_v1.levanter.detail",
            1,
            1,
            &telemetry_batch(&["levanter"], &["existing_detail"], 1),
        );
        add_orphan_segment(
            &dirs.store,
            "telemetry_storage_v1.levanter.detail",
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
        assert_eq!(manifest.residual_rows, 0);
        assert!(manifest.complete);
        assert_eq!(manifest.phase, MigrationPhase::Staged);
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
        let staged_dir = dirs.store.join(MIGRATION_DIRECTORY).join(STAGED_DIRECTORY);
        assert!(manifest
            .source_segments
            .iter()
            .flat_map(|source| source.outputs.iter())
            .all(|output| staged_dir.join(&output.relative_path).is_file()));
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
            &with_legacy_alert_tag(
                telemetry_batch(&["levanter"], &["train_loss"], 1),
                true,
            ),
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
    fn prepare_in_place_rejects_unknown_required_columns() {
        let dirs = TestDirs::new("unknown_required_column");
        let catalog = Catalog::open(Some(&dirs.store)).unwrap();
        add_segment(
            &catalog,
            &dirs.store,
            TELEMETRY_NAMESPACE,
            1,
            1,
            &with_legacy_alert_tag(
                telemetry_batch(&["levanter"], &["train_loss"], 1),
                false,
            ),
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
        let published = publish_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .unwrap();
        assert_eq!(published.phase, MigrationPhase::Published);
        assert_eq!(
            published.published_files.len(),
            published
                .source_segments
                .iter()
                .map(|source| source.outputs.len())
                .sum::<usize>()
        );
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
        assert!(!provider_names.contains(LEVANTER_STATUS_STORAGE_NAMESPACE));
        assert!(!provider_names.contains(LEVANTER_DETAIL_STORAGE_NAMESPACE));

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
        let store = Store::new(
            Some(dirs.store.clone()),
            String::new(),
            1,
            ServeMode::Shadow,
        )
        .unwrap();

        assert!(publish_in_place(&InPlaceConfig {
            store_dir: dirs.store.clone(),
            batch_rows: 2,
        })
        .is_err());
        store.shutdown(Duration::from_secs(1)).await;
        drop(store);
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
