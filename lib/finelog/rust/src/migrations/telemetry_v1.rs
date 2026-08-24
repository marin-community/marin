//! Offline migration from legacy telemetry namespaces into semantic storage shards.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{new_null_array, Array, ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::compute::{cast, filter_record_batch};
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::{ArrowWriter, ArrowWriterOptions};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::server::telemetry::telemetry_schema;
use crate::store::catalog::{Catalog, CATALOG_DB_FILENAME};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_arrow, stored_form, IMPLICIT_SEQ_COLUMN};
use crate::store::segment::{
    discover_segments, read_segment_footer, segment_writer_properties_with_max_rows,
};
use crate::store::types::{seg_filename, SegmentLocation, SegmentRow};
use crate::telemetry_policy::{
    ingest_storage_namespace, legacy_storage_namespace, migration_source_logical_namespace,
    migration_source_namespaces, storage_max_bytes, TELEMETRY_NAMESPACE, TELEMETRY_STORAGE_SHARDS,
};

const MANIFEST_FILENAME: &str = ".finelog-telemetry-v1-migration.json";
const MANIFEST_VERSION: u32 = 1;
const POLICY_REVISION: &str = "semantic-storage-v1";
const OUTPUT_LEVEL: i32 = 0;

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
    pub input_rows: i64,
    pub output_rows: i64,
    pub residual_rows: i64,
    pub source_segments: Vec<SourceSegment>,
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

    clone_unchanged_store(config)?;
    verify_source_segments(&config.source_dir, &manifest)?;
    write_planned_outputs(config, &mut manifest)?;
    finalize_catalog(config, &manifest)?;
    manifest.complete = true;
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
    let catalog_path = output_dir.join(CATALOG_DB_FILENAME);
    if !catalog_path.is_file() {
        return Err(validation_error(format!(
            "prepared store has no catalog at {}",
            catalog_path.display()
        )));
    }
    verify_catalog(output_dir, &manifest)?;
    Ok(manifest)
}

fn verify_catalog(output_dir: &Path, manifest: &MigrationManifest) -> Result<(), StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        output_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(sqlite_error("open replacement catalog read-only"))?;
    let check: String = connection
        .query_row("PRAGMA quick_check", [], |row| row.get(0))
        .map_err(sqlite_error("check replacement catalog"))?;
    if check != "ok" {
        return Err(validation_error(format!(
            "replacement catalog quick_check failed: {check}"
        )));
    }

    let final_log_dir = Path::new(&manifest.final_log_dir);
    for source in &manifest.source_segments {
        for output in &source.outputs {
            let catalog_path = final_log_dir.join(&output.relative_path);
            let row: Option<(i64, i64, i64, String)> = connection
                .query_row(
                    "SELECT min_seq, max_seq, row_count, location FROM segments \
                     WHERE namespace = ?1 AND path = ?2",
                    rusqlite::params![output.namespace, catalog_path.to_string_lossy()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()
                .map_err(sqlite_error("read migrated catalog row"))?;
            if row
                != Some((
                    output.min_seq,
                    output.max_seq,
                    output.rows,
                    SegmentLocation::Local.as_str().to_string(),
                ))
            {
                return Err(validation_error(format!(
                    "replacement catalog does not expose migrated output {}",
                    output.relative_path
                )));
            }
        }

        let source_filename = Path::new(&source.relative_path)
            .file_name()
            .ok_or_else(|| validation_error("source segment has no filename"))?;
        let legacy_path = final_log_dir
            .join(&source.namespace)
            .join(source_filename)
            .to_string_lossy()
            .into_owned();
        let location: Option<String> = connection
            .query_row(
                "SELECT location FROM segments WHERE namespace = ?1 AND path = ?2",
                rusqlite::params![source.namespace, legacy_path],
                |row| row.get(0),
            )
            .optional()
            .map_err(sqlite_error("read legacy catalog row"))?;
        if location.is_some_and(|value| value != SegmentLocation::Remote.as_str()) {
            return Err(validation_error(format!(
                "legacy source remains locally queryable: {}",
                source.relative_path
            )));
        }
    }

    let mut statement = connection
        .prepare("SELECT namespace, path FROM segments WHERE location != 'REMOTE'")
        .map_err(sqlite_error("prepare local catalog verification"))?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(sqlite_error("read local catalog paths"))?;
    for row in rows {
        let (namespace, path) = row.map_err(sqlite_error("decode local catalog path"))?;
        let filename = Path::new(&path)
            .file_name()
            .ok_or_else(|| validation_error(format!("catalog segment has no filename: {path}")))?;
        let expected_path = final_log_dir.join(&namespace).join(filename);
        if Path::new(&path) != expected_path {
            return Err(validation_error(format!(
                "catalog segment path is outside final_log_dir: {path}"
            )));
        }
        let prepared_path = output_dir.join(namespace).join(filename);
        if !prepared_path.is_file() {
            return Err(validation_error(format!(
                "catalogued local segment is absent from replacement store: {}",
                prepared_path.display()
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
            "output_dir must be outside source_dir so the source remains a rollback",
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
            "telemetry replacement-store migration is hub-only; source catalog has forwarding state",
        ));
    }
    Ok(())
}

fn clone_unchanged_store(config: &PrepareConfig) -> Result<(), StatsError> {
    let migration_sources: BTreeSet<&str> = migration_source_namespaces().collect();
    for entry in std::fs::read_dir(&config.source_dir).map_err(io_error("list source store"))? {
        let entry = entry.map_err(io_error("read source store entry"))?;
        let name = entry.file_name();
        let name_text = name.to_string_lossy();
        if name_text.starts_with(CATALOG_DB_FILENAME)
            || name_text == MANIFEST_FILENAME
            || migration_sources.contains(name_text.as_ref())
        {
            continue;
        }
        clone_path(&entry.path(), &config.output_dir.join(name))?;
    }
    Ok(())
}

fn clone_path(source: &Path, destination: &Path) -> Result<(), StatsError> {
    let metadata = std::fs::symlink_metadata(source).map_err(io_error("stat source entry"))?;
    if metadata.is_dir() {
        std::fs::create_dir_all(destination).map_err(io_error("create output directory"))?;
        for entry in std::fs::read_dir(source).map_err(io_error("list source directory"))? {
            let entry = entry.map_err(io_error("read source directory entry"))?;
            clone_path(&entry.path(), &destination.join(entry.file_name()))?;
        }
        return Ok(());
    }
    if !metadata.is_file() {
        return Err(validation_error(format!(
            "source store contains unsupported entry {}",
            source.display()
        )));
    }
    if destination.exists() {
        let existing = std::fs::metadata(destination).map_err(io_error("stat cloned file"))?;
        if existing.len() != metadata.len() {
            return Err(validation_error(format!(
                "existing clone {} has {} bytes; expected {}",
                destination.display(),
                existing.len(),
                metadata.len()
            )));
        }
        return Ok(());
    }
    let temporary = temporary_path(destination);
    if temporary.exists() {
        std::fs::remove_file(&temporary).map_err(io_error("remove interrupted clone"))?;
    }
    if std::fs::hard_link(source, &temporary).is_err() {
        std::fs::copy(source, &temporary).map_err(io_error("copy source file"))?;
    }
    std::fs::rename(&temporary, destination).map_err(io_error("publish cloned file"))?;
    Ok(())
}

fn plan_migration(config: &PrepareConfig) -> Result<MigrationManifest, StatsError> {
    let mut next_seq = destination_next_sequences(&config.source_dir)?;
    let mut source_segments = Vec::new();
    let mut input_rows = 0_i64;
    let mut output_rows = 0_i64;
    let mut residual_rows = 0_i64;

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
                let min_seq = *next_seq.entry(destination.clone()).or_insert(1);
                let max_seq = min_seq + item.rows - 1;
                next_seq.insert(destination.clone(), max_seq + 1);
                let relative_path = Path::new(&destination)
                    .join(seg_filename(OUTPUT_LEVEL, min_seq))
                    .to_string_lossy()
                    .into_owned();
                if destination == TELEMETRY_NAMESPACE {
                    residual_rows += item.rows;
                }
                outputs.push(PlannedOutput {
                    namespace: destination,
                    relative_path,
                    min_seq,
                    max_seq,
                    rows: item.rows,
                    min_timestamp_ms: item.min_timestamp_ms.unwrap_or_default(),
                    max_timestamp_ms: item.max_timestamp_ms.unwrap_or_default(),
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
        input_rows,
        output_rows,
        residual_rows,
        source_segments,
    })
}

fn destination_next_sequences(source_dir: &Path) -> Result<BTreeMap<String, i64>, StatsError> {
    let connection = rusqlite::Connection::open_with_flags(
        source_dir.join(CATALOG_DB_FILENAME),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(sqlite_error("open source catalog for sequence planning"))?;
    let mut next = BTreeMap::new();
    for namespace in std::iter::once(TELEMETRY_NAMESPACE).chain(
        TELEMETRY_STORAGE_SHARDS
            .iter()
            .map(|shard| shard.storage_namespace),
    ) {
        let catalog_max_seq: Option<i64> = connection
            .query_row(
                "SELECT MAX(max_seq) FROM segments WHERE namespace = ?1",
                [namespace],
                |row| row.get(0),
            )
            .map_err(sqlite_error("read destination sequence high-water"))?;
        let mut max_seq = catalog_max_seq.unwrap_or(0);
        let namespace_dir = source_dir.join(namespace);
        if namespace_dir.is_dir() {
            for path in discover_segments(&namespace_dir) {
                let footer = read_segment_footer(&path, Some("timestamp_ms")).ok_or_else(|| {
                    validation_error(format!(
                        "could not read destination segment {}",
                        path.display()
                    ))
                })?;
                max_seq = max_seq.max(footer.max_seq);
            }
        }
        next.insert(namespace.to_string(), max_seq + 1);
    }
    Ok(next)
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
        let services = string_column(&batch, "service")?;
        let names = string_column(&batch, "name")?;
        let batch_ids = string_column(&batch, "batch_id")?;
        let record_indices = int64_column(&batch, "record_index")?;
        let timestamps = int64_column(&batch, "timestamp_ms")?;
        let clusters = optional_string_column(&batch, "cluster")?;
        for row in 0..batch.num_rows() {
            let service = services.value(row);
            let name = names.value(row);
            let destination = row_destination(source_namespace, service, name)?;
            let item = stats.entry(destination).or_default();
            let timestamp = timestamps.value(row);
            item.rows += 1;
            item.min_timestamp_ms = Some(
                item.min_timestamp_ms
                    .map_or(timestamp, |v| v.min(timestamp)),
            );
            item.max_timestamp_ms = Some(
                item.max_timestamp_ms
                    .map_or(timestamp, |v| v.max(timestamp)),
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
    Ok(stats)
}

fn row_destination(
    source_namespace: &str,
    service: &str,
    name: &str,
) -> Result<String, StatsError> {
    if source_namespace == TELEMETRY_NAMESPACE {
        return Ok(legacy_storage_namespace(service, name)
            .unwrap_or(TELEMETRY_NAMESPACE)
            .to_string());
    }
    let logical = migration_source_logical_namespace(source_namespace).ok_or_else(|| {
        validation_error(format!(
            "unsupported migration source namespace {source_namespace:?}"
        ))
    })?;
    ingest_storage_namespace(logical, name).ok_or_else(|| {
        validation_error(format!(
            "could not route legacy namespace {source_namespace:?}"
        ))
    })
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
        let options = ArrowWriterOptions::new()
            .with_properties(segment_writer_properties_with_max_rows(128 * 1024)?);
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
        let services = string_column(&batch, "service")?;
        let names = string_column(&batch, "name")?;
        let mut masks: BTreeMap<String, Vec<bool>> = writers
            .keys()
            .map(|namespace| (namespace.clone(), Vec::with_capacity(batch.num_rows())))
            .collect();
        for row in 0..batch.num_rows() {
            let destination = row_destination(
                source.namespace.as_str(),
                services.value(row),
                names.value(row),
            )?;
            for (namespace, mask) in &mut masks {
                mask.push(namespace == &destination);
            }
        }
        for (namespace, mask) in masks {
            let writer = writers
                .get_mut(&namespace)
                .expect("mask is built from writer namespaces");
            let mask = arrow::array::BooleanArray::from(mask);
            let filtered = filter_record_batch(&batch, &mask)
                .map_err(arrow_error("filter source telemetry"))?;
            if filtered.num_rows() == 0 {
                continue;
            }
            let migrated = align_migrated_batch(&filtered, target_schema, writer.next_seq)?;
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
    for field in source.schema().fields() {
        if field.name() != IMPLICIT_SEQ_COLUMN
            && target_schema.field_with_name(field.name()).is_err()
        {
            return Err(validation_error(format!(
                "source telemetry has unknown column {:?}",
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

fn finalize_catalog(
    config: &PrepareConfig,
    manifest: &MigrationManifest,
) -> Result<(), StatsError> {
    let build_dir = config.output_dir.join(".finelog-catalog-build");
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir)
            .map_err(io_error("remove interrupted catalog build"))?;
    }
    std::fs::create_dir_all(&build_dir).map_err(io_error("create catalog build directory"))?;
    std::fs::copy(
        config.source_dir.join(CATALOG_DB_FILENAME),
        build_dir.join(CATALOG_DB_FILENAME),
    )
    .map_err(io_error("copy source catalog"))?;

    let catalog = Catalog::open(Some(&build_dir))?;
    remap_catalog_paths(&catalog, &config.final_log_dir)?;
    for namespace in migration_source_namespaces() {
        for row in catalog.list_segments(namespace)? {
            match row.location {
                SegmentLocation::Local => {
                    catalog.remove_segments(namespace, &[row.path])?;
                }
                SegmentLocation::Both => {
                    catalog.set_location(namespace, &row.path, SegmentLocation::Remote)?;
                }
                SegmentLocation::Remote => {}
            }
        }
    }
    let stored_schema = stored_form(telemetry_schema());
    catalog.upsert(TELEMETRY_NAMESPACE, &stored_schema)?;
    catalog.upsert_policy(
        TELEMETRY_NAMESPACE,
        &StoragePolicy {
            max_bytes: storage_max_bytes(TELEMETRY_NAMESPACE),
            ..StoragePolicy::default()
        },
    )?;
    for shard in TELEMETRY_STORAGE_SHARDS {
        catalog.upsert(shard.storage_namespace, &stored_schema)?;
        catalog.upsert_policy(
            shard.storage_namespace,
            &StoragePolicy {
                max_bytes: Some(shard.max_bytes),
                ..StoragePolicy::default()
            },
        )?;
    }
    adopt_unchanged_physical_segments(config, &catalog)?;

    let created_at_ms = now_ms();
    let rows = manifest
        .source_segments
        .iter()
        .flat_map(|source| source.outputs.iter())
        .map(|output| {
            let relative = Path::new(&output.relative_path);
            let actual = config.output_dir.join(relative);
            let byte_size = std::fs::metadata(&actual)
                .map_err(io_error("stat migrated segment"))?
                .len() as i64;
            Ok(SegmentRow {
                namespace: output.namespace.clone(),
                path: config
                    .final_log_dir
                    .join(relative)
                    .to_string_lossy()
                    .into_owned(),
                level: OUTPUT_LEVEL,
                min_seq: output.min_seq,
                max_seq: output.max_seq,
                row_count: output.rows,
                byte_size,
                created_at_ms,
                min_key_value: Some(output.min_timestamp_ms.to_string()),
                max_key_value: Some(output.max_timestamp_ms.to_string()),
                location: SegmentLocation::Local,
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    catalog.upsert_segments(&rows)?;
    drop(catalog);

    let built_catalog = build_dir.join(CATALOG_DB_FILENAME);
    let final_catalog = config.output_dir.join(CATALOG_DB_FILENAME);
    let temporary_catalog = temporary_path(&final_catalog);
    if temporary_catalog.exists() {
        std::fs::remove_file(&temporary_catalog)
            .map_err(io_error("remove interrupted catalog publication"))?;
    }
    std::fs::rename(&built_catalog, &temporary_catalog)
        .map_err(io_error("stage replacement catalog"))?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&temporary_catalog)
        .map_err(io_error("open replacement catalog"))?;
    file.sync_all()
        .map_err(io_error("fsync replacement catalog"))?;
    std::fs::rename(&temporary_catalog, &final_catalog)
        .map_err(io_error("publish replacement catalog"))?;
    std::fs::remove_dir_all(&build_dir).map_err(io_error("remove catalog build directory"))?;
    Ok(())
}

fn adopt_unchanged_physical_segments(
    config: &PrepareConfig,
    catalog: &Catalog,
) -> Result<(), StatsError> {
    for shard in TELEMETRY_STORAGE_SHARDS {
        let namespace = shard.storage_namespace;
        let known_paths = catalog
            .list_segments(namespace)?
            .into_iter()
            .map(|row| row.path)
            .collect::<BTreeSet<_>>();
        let namespace_dir = config.output_dir.join(namespace);
        if !namespace_dir.is_dir() {
            continue;
        }
        for path in discover_segments(&namespace_dir) {
            let filename = path.file_name().ok_or_else(|| {
                validation_error(format!(
                    "physical segment has no filename: {}",
                    path.display()
                ))
            })?;
            let final_path = config
                .final_log_dir
                .join(namespace)
                .join(filename)
                .to_string_lossy()
                .into_owned();
            if known_paths.contains(&final_path) {
                continue;
            }
            let footer = read_segment_footer(&path, Some("timestamp_ms")).ok_or_else(|| {
                validation_error(format!(
                    "could not read physical segment {}",
                    path.display()
                ))
            })?;
            let metadata = std::fs::metadata(&path).map_err(io_error("stat physical segment"))?;
            let created_at_ms = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis() as i64)
                .unwrap_or_else(now_ms);
            catalog.upsert_segment(&SegmentRow {
                namespace: namespace.to_string(),
                path: final_path,
                level: footer.level,
                min_seq: footer.min_seq,
                max_seq: footer.max_seq,
                row_count: footer.row_count,
                byte_size: metadata.len() as i64,
                created_at_ms,
                min_key_value: footer.min_key_value.map(|value| value.to_string()),
                max_key_value: footer.max_key_value.map(|value| value.to_string()),
                location: SegmentLocation::Local,
            })?;
        }
    }
    Ok(())
}

fn remap_catalog_paths(catalog: &Catalog, final_log_dir: &Path) -> Result<(), StatsError> {
    for (namespace, _schema) in catalog.list_all()? {
        let existing = catalog.list_segments(&namespace)?;
        let mut removed = Vec::new();
        let mut added = Vec::new();
        for mut row in existing {
            let path = Path::new(&row.path);
            let filename = path.file_name().ok_or_else(|| {
                validation_error(format!(
                    "catalog segment has no filename: {}",
                    path.display()
                ))
            })?;
            removed.push(row.path.clone());
            row.path = final_log_dir
                .join(&namespace)
                .join(filename)
                .to_string_lossy()
                .into_owned();
            added.push(row);
        }
        catalog.replace_segments(&namespace, &removed, &added)?;
    }
    Ok(())
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

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
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
    use std::fs;
    use std::time::Duration;

    use arrow::array::{Float64Array, Int32Array};

    use super::*;
    use crate::store::segment::write_segment_to_dir;
    use crate::store::store::{ServeMode, Store};
    use crate::telemetry_policy::{
        IRIS_RPC_NAMESPACE, LEVANTER_DETAIL_STORAGE_NAMESPACE, LEVANTER_NAMESPACE,
        LEVANTER_STATUS_STORAGE_NAMESPACE, NODE_AGENT_NAMESPACE, VLLM_NAMESPACE,
    };

    struct TestDirs {
        root: PathBuf,
        source: PathBuf,
        output: PathBuf,
    }

    struct PreparedStore {
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
            let source = root.join("source");
            let output = root.join("output");
            fs::create_dir_all(&source).unwrap();
            Self {
                root,
                source,
                output,
            }
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
                created_at_ms: now_ms(),
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

    fn prepared_store() -> PreparedStore {
        let dirs = TestDirs::new("prepare");
        let catalog = Catalog::open(Some(&dirs.source)).unwrap();
        let root = add_segment(
            &catalog,
            &dirs.source,
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
                    "unowned-service",
                ],
                &[
                    "train_loss",
                    "grad_histogram",
                    "node_cpu_utilization_percent",
                    "rpc_requests_total",
                    "vllm_request_latency",
                    "custom_metric",
                ],
                1,
            ),
        );
        add_segment(
            &catalog,
            &dirs.source,
            "telemetry_v1.levanter.extra",
            1,
            1,
            &telemetry_batch(&["old-levanter-client"], &["step"], 1),
        );
        add_segment(
            &catalog,
            &dirs.source,
            "telemetry_storage_v1.levanter.detail",
            1,
            1,
            &telemetry_batch(&["levanter"], &["existing_detail"], 1),
        );
        add_orphan_segment(
            &dirs.source,
            "telemetry_storage_v1.levanter.detail",
            0,
            100,
            &telemetry_batch(&["levanter"], &["orphan_detail"], 100),
        );
        drop(catalog);
        fs::create_dir_all(dirs.source.join("iris.task")).unwrap();
        fs::write(dirs.source.join("iris.task/unchanged.marker"), b"preserved").unwrap();
        let source_sha = file_sha256(&root).unwrap();
        let manifest = prepare_store(&PrepareConfig {
            source_dir: dirs.source.clone(),
            output_dir: dirs.output.clone(),
            final_log_dir: dirs.output.clone(),
            batch_rows: 2,
        })
        .unwrap();
        PreparedStore {
            dirs,
            manifest,
            source_sha,
        }
    }

    #[test]
    fn prepare_store_routes_every_legacy_row_and_preserves_the_source() {
        let PreparedStore {
            dirs,
            manifest,
            source_sha,
        } = prepared_store();

        assert_eq!(manifest.input_rows, 7);
        assert_eq!(manifest.output_rows, 7);
        assert_eq!(manifest.residual_rows, 1);
        assert!(manifest.complete);
        assert_eq!(
            fs::read(dirs.output.join("iris.task/unchanged.marker")).unwrap(),
            b"preserved"
        );
        assert_eq!(
            file_sha256(
                &dirs
                    .source
                    .join("telemetry_v1/seg_L1_0000000000000000001.parquet")
            )
            .unwrap(),
            source_sha
        );

        let catalog = Catalog::open(Some(&dirs.output)).unwrap();
        let local_rows = |namespace: &str| {
            catalog
                .list_segments(namespace)
                .unwrap()
                .iter()
                .filter(|segment| segment.location != SegmentLocation::Remote)
                .map(|segment| segment.row_count)
                .sum::<i64>()
        };
        assert_eq!(local_rows("telemetry_v1"), 1);
        assert_eq!(local_rows("telemetry_storage_v1.levanter.status"), 2);
        assert_eq!(local_rows("telemetry_storage_v1.levanter.detail"), 3);
        assert_eq!(local_rows("telemetry_storage_v1.node_agent"), 1);
        assert_eq!(local_rows("telemetry_storage_v1.iris_rpc"), 1);
        assert_eq!(local_rows("telemetry_storage_v1.vllm"), 1);
        assert!(catalog
            .list_segments("telemetry_v1.levanter.extra")
            .unwrap()
            .iter()
            .all(|segment| segment.location == SegmentLocation::Remote));
        let detail = catalog
            .list_segments("telemetry_storage_v1.levanter.detail")
            .unwrap();
        assert_eq!(
            detail
                .iter()
                .map(|segment| (segment.min_seq, segment.max_seq, segment.location))
                .collect::<Vec<_>>(),
            vec![
                (1, 1, SegmentLocation::Both),
                (100, 100, SegmentLocation::Local),
                (101, 101, SegmentLocation::Local)
            ]
        );
        assert!(detail
            .iter()
            .all(|segment| Path::new(&segment.path).starts_with(&dirs.output)));
    }

    #[test]
    fn prepare_store_resume_reuses_verified_outputs_without_duplicates() {
        let PreparedStore {
            dirs,
            manifest: first,
            ..
        } = prepared_store();
        let second = prepare_store(&PrepareConfig {
            source_dir: dirs.source.clone(),
            output_dir: dirs.output.clone(),
            final_log_dir: dirs.output.clone(),
            batch_rows: 3,
        })
        .unwrap();

        assert_eq!(second, first);
        let catalog = Catalog::open(Some(&dirs.output)).unwrap();
        assert_eq!(
            catalog
                .list_segments("telemetry_storage_v1.levanter.status")
                .unwrap()
                .iter()
                .filter(|segment| segment.location != SegmentLocation::Remote)
                .map(|segment| segment.row_count)
                .sum::<i64>(),
            2
        );
    }

    #[tokio::test]
    async fn prepared_store_boots_with_semantic_query_aliases() {
        let PreparedStore { dirs, .. } = prepared_store();
        let store = Store::new(
            Some(dirs.output.clone()),
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
        assert!(!provider_names.contains(LEVANTER_STATUS_STORAGE_NAMESPACE));
        assert!(!provider_names.contains(LEVANTER_DETAIL_STORAGE_NAMESPACE));

        store.shutdown(Duration::from_secs(1)).await;
    }

    #[test]
    fn verify_store_rejects_a_changed_source_snapshot() {
        let PreparedStore { dirs, .. } = prepared_store();
        let root = dirs
            .source
            .join("telemetry_v1/seg_L1_0000000000000000001.parquet");
        let mut file = OpenOptions::new().append(true).open(root).unwrap();
        file.write_all(b"changed").unwrap();

        assert!(verify_store(&dirs.source, &dirs.output, 2).is_err());
    }

    #[test]
    fn verify_store_rejects_a_missing_migrated_catalog_row() {
        let PreparedStore { dirs, manifest, .. } = prepared_store();
        let output = &manifest.source_segments[0].outputs[0];
        let catalog_path = Path::new(&manifest.final_log_dir).join(&output.relative_path);
        let connection = rusqlite::Connection::open(dirs.output.join(CATALOG_DB_FILENAME)).unwrap();
        connection
            .execute(
                "DELETE FROM segments WHERE namespace = ?1 AND path = ?2",
                rusqlite::params![output.namespace, catalog_path.to_string_lossy()],
            )
            .unwrap();

        assert!(verify_store(&dirs.source, &dirs.output, 2).is_err());
    }

    #[test]
    fn prepare_store_rejects_a_forwarding_sender() {
        let dirs = TestDirs::new("forwarding");
        let catalog = Catalog::open(Some(&dirs.source)).unwrap();
        catalog
            .set_forward_cursor("hub", TELEMETRY_NAMESPACE, 12)
            .unwrap();
        drop(catalog);

        assert!(prepare_store(&PrepareConfig {
            source_dir: dirs.source.clone(),
            output_dir: dirs.output.clone(),
            final_log_dir: dirs.output.clone(),
            batch_rows: 2,
        })
        .is_err());
    }
}
