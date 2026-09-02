//! The flush pipeline: seal a buffer, encode Parquet, make it durable, commit.
//!
//! Two destinations share one shape. A legacy table writes a local L0 file and
//! upserts its catalog row. An object-backed table sorts and partitions the
//! sealed batch as its source layout declares, stages one immutable object per
//! partition on local disk, and commits their descriptors as a new table
//! revision.
//!
//! Both acknowledge on local durability: the sealed rows are on disk and their
//! catalog rows are committed before the durability high-water mark advances.
//! An object-backed table then owes the revision to publication, which uploads
//! the staged objects and swaps HEAD — attempted immediately after the ack and
//! retried by maintenance, so a remote outage delays HEAD, never the ack. On
//! failure before the local commit the sealed rows go back to the buffer.
//!
//! Callers serialize flushes; this module takes no locks of its own beyond the
//! short buffer and view locks.

use std::collections::BTreeMap;
use std::path::Path;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{cast, lexsort_to_indices, take, SortColumn, SortOptions};
use arrow::datatypes::DataType;
use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::errors::StatsError;
use crate::partition_policy::{select_rows, SegmentPartition};
use crate::proto::finelog::stats::{partition_field, SourceLayout};
use crate::store::catalog::Catalog;
use crate::store::segment::{
    write_segment_to_dir_with_max_row_group_rows, write_segment_with_max_row_group_rows,
};
use crate::store::table::controller::TableController;
use crate::store::table::ingest::IngestBuffer;
use crate::store::table::segment_format::SegmentFormat;
use crate::store::table::segment_view::SegmentView;
use crate::store::table_spec::TablePolicy;
use crate::store::table_state::{ArtifactReferences, LocalArtifacts, SegmentDescriptor};
use crate::store::types::{segment_to_row, LocalSegment, SegmentLocation};

/// Hive's sentinel for a null partition value.
const NULL_PARTITION_VALUE: &str = "__HIVE_DEFAULT_PARTITION__";

/// Everything one flush reads and commits through.
pub struct FlushTarget<'a> {
    pub table: &'a str,
    pub format: &'a SegmentFormat,
    pub buffer: &'a IngestBuffer,
    pub segments: &'a SegmentView,
    pub catalog: &'a Catalog,
    pub controller: &'a TableController,
}

/// Drain the buffer to one local L0 segment under `table_dir`.
///
/// Returns `Ok(())` when there was nothing to flush. On write failure the
/// in-flight buffer is restored and the durability mark is not advanced.
pub fn flush_local(target: FlushTarget<'_>, table_dir: &Path) -> Result<(), StatsError> {
    let Some(sealed) = target.buffer.seal() else {
        return Ok(());
    };
    let (path, size) = match write_segment_to_dir_with_max_row_group_rows(
        table_dir,
        0,
        sealed.min_seq,
        &sealed.batch,
        target.format.max_row_group_rows(),
    ) {
        Ok(written) => written,
        Err(error) => return Err(restore(&target, error)),
    };
    // L0 files are small and short-lived. Derived indexes are built after
    // compaction promotes them to L1+, keeping flush acknowledgement fast while
    // query plans merge indexed counts with uncovered L0 data.
    let (min_key, max_key) = target.format.key_bounds(&sealed.batch);
    let segment = LocalSegment {
        path: path.to_string_lossy().into_owned(),
        size_bytes: size,
        level: 0,
        min_seq: sealed.min_seq,
        max_seq: sealed.max_seq,
        row_count: sealed.batch.num_rows() as i64,
        created_at_ms: crate::store::table::now_ms(),
        min_key_value: min_key,
        max_key_value: max_key,
        partition: None,
        location: SegmentLocation::Local,
        artifacts: LocalArtifacts::default(),
    };
    // Persist the catalog row BEFORE committing the in-RAM flush: the file is
    // already renamed into place, so on an upsert error the sealed rows are
    // still intact and are returned to the buffer for retry (rather than being
    // silently cleared with the catalog row missing).
    if let Err(error) = target
        .catalog
        .upsert_segment(&segment_to_row(target.table, &segment))
    {
        return Err(restore(&target, error));
    }
    target.segments.extend(vec![segment]);
    target.buffer.commit_sealed();
    // Durability-before-ack: the file is renamed and the catalog row is
    // committed before the new high-water seq is published.
    target.buffer.publish_persisted(sealed.max_seq);
    Ok(())
}

/// Drain the buffer to locally staged immutable objects, commit their
/// descriptors, and acknowledge; then attempt the owed publication.
pub async fn flush_to_objects(
    target: FlushTarget<'_>,
    policy: &TablePolicy,
) -> Result<(), StatsError> {
    let Some(sealed) = target.buffer.seal() else {
        return Ok(());
    };
    let max_seq = sealed.max_seq;
    if let Err(error) = write_sealed_objects(&target, sealed.batch.clone(), policy).await {
        target.buffer.restore_sealed();
        tracing::warn!(namespace = %target.table, %error, "object-backed flush failed; restored RAM buffer");
        return Err(error);
    }
    // Local durability is the ack: staged objects and catalog rows are on
    // disk. HEAD is owed, and the caller publishes it outside the flush gate.
    target.buffer.publish_persisted(max_seq);
    Ok(())
}

fn restore(target: &FlushTarget<'_>, error: StatsError) -> StatsError {
    target.buffer.restore_sealed();
    tracing::warn!(namespace = %target.table, %error, "flush failed; restored RAM buffer");
    error
}

async fn write_sealed_objects(
    target: &FlushTarget<'_>,
    batch: RecordBatch,
    policy: &TablePolicy,
) -> Result<(), StatsError> {
    let source_layout = policy.source_layout.clone();
    let max_row_group_rows = source_layout
        .as_ref()
        .and_then(|layout| layout.max_row_group_rows)
        .map(|rows| rows as usize)
        .unwrap_or(target.format.max_row_group_rows());
    let encoded = tokio::task::spawn_blocking(move || {
        let sorted = sorted_object_batch(&batch, source_layout.as_ref())?;
        partition_object_batch(&sorted, source_layout.as_ref())?
            .into_iter()
            .map(|(partition, batch)| {
                let (min_seq, max_seq) = batch_seq_bounds(&batch)?;
                let parquet = write_segment_with_max_row_group_rows(&batch, max_row_group_rows)?;
                Ok((partition, batch, parquet, min_seq, max_seq))
            })
            .collect::<Result<Vec<_>, StatsError>>()
    })
    .await
    .map_err(|error| StatsError::Internal(format!("object-backed parquet task panicked: {error}")))
    .and_then(|encoded| encoded)?;

    let mut segments = Vec::with_capacity(encoded.len());
    let mut descriptors = Vec::with_capacity(encoded.len());
    for (partition, batch, parquet, min_seq, max_seq) in encoded {
        let stored = target
            .controller
            .stage_parquet(Bytes::from(parquet))
            .await?;
        let (min_key, max_key) = target.format.key_bounds(&batch);
        let segment = LocalSegment {
            path: stored.path.to_string_lossy().into_owned(),
            size_bytes: stored.byte_size,
            level: 0,
            min_seq,
            max_seq,
            row_count: batch.num_rows() as i64,
            created_at_ms: crate::store::table::now_ms(),
            min_key_value: min_key,
            max_key_value: max_key,
            partition,
            location: SegmentLocation::Both,
            artifacts: LocalArtifacts::default(),
        };
        descriptors.push(SegmentDescriptor {
            row: segment_to_row(target.table, &segment),
            source: stored.source,
            // L0 is unindexed: a flush advertises no derived artifacts.
            artifacts: ArtifactReferences::default(),
        });
        segments.push(segment);
    }

    let table_spec_version = policy.table_spec_version;
    // The local commit owns these objects; the revision is owed to publication
    // from the moment it is durable here.
    target
        .controller
        .commit_owing_publication(|| {
            let revision =
                target
                    .catalog
                    .commit_object_segments(&descriptors, table_spec_version, false)?;
            Ok((revision, ()))
        })
        .map_err(StatsError::from)?;
    target.segments.extend(segments);
    target.buffer.commit_sealed();
    Ok(())
}

/// Sort a sealed batch into the order the source layout declares, always ending
/// with `seq` so equal keys stay in ingest order.
fn sorted_object_batch(
    batch: &RecordBatch,
    source_layout: Option<&SourceLayout>,
) -> Result<RecordBatch, StatsError> {
    let Some(layout) = source_layout else {
        return Ok(batch.clone());
    };
    let mut names = layout.sort_columns.clone();
    if !names.iter().any(|name| name == "seq") {
        names.push("seq".to_string());
    }
    if names.is_empty() {
        return Ok(batch.clone());
    }
    let columns = names
        .iter()
        .map(|name| {
            let column = batch.column_by_name(name).ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "object source layout sort column {name:?} is missing"
                ))
            })?;
            Ok(SortColumn {
                values: column.clone(),
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: false,
                }),
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    let indices = lexsort_to_indices(&columns, None)
        .map_err(|error| StatsError::Internal(format!("sort object-backed batch: {error}")))?;
    let arrays = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| StatsError::Internal(format!("apply object-backed sort: {error}")))?;
    RecordBatch::try_new(batch.schema(), arrays)
        .map_err(|error| StatsError::Internal(format!("build sorted object batch: {error}")))
}

/// Split a batch into one output per physical partition the source layout
/// declares, or a single unpartitioned output when it declares none.
pub fn partition_object_batch(
    batch: &RecordBatch,
    source_layout: Option<&SourceLayout>,
) -> Result<Vec<(Option<SegmentPartition>, RecordBatch)>, StatsError> {
    let Some(partition) = source_layout.and_then(|layout| layout.partition.as_option()) else {
        return Ok(vec![(None, batch.clone())]);
    };
    if partition.fields.is_empty() {
        return Ok(vec![(None, batch.clone())]);
    }
    let spec_id = u32::try_from(partition.spec_id.unwrap_or(0)).map_err(|_| {
        StatsError::SchemaValidation("partition spec_id exceeds the supported range".to_string())
    })?;
    let rendered_columns = partition
        .fields
        .iter()
        .map(|field| {
            let source = field.source_column.as_deref().unwrap_or("");
            let column = batch.column_by_name(source).ok_or_else(|| {
                StatsError::SchemaValidation(format!(
                    "partition source column {source:?} is missing"
                ))
            })?;
            cast(column, &DataType::Utf8).map_err(|error| {
                StatsError::SchemaValidation(format!(
                    "partition source column {source:?} cannot be rendered: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, StatsError>>()?;
    let mut indices: BTreeMap<SegmentPartition, Vec<u32>> = BTreeMap::new();
    for row in 0..batch.num_rows() {
        let mut values = BTreeMap::new();
        for (field, rendered) in partition.fields.iter().zip(&rendered_columns) {
            let values_array = rendered
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Arrow UTF-8 cast returns StringArray");
            let value = match field.transform.as_ref() {
                Some(partition_field::Transform::Identity(_)) if values_array.is_null(row) => {
                    NULL_PARTITION_VALUE.to_string()
                }
                Some(partition_field::Transform::Identity(_)) => {
                    values_array.value(row).to_string()
                }
                Some(partition_field::Transform::Bucket(_)) if values_array.is_null(row) => {
                    NULL_PARTITION_VALUE.to_string()
                }
                Some(partition_field::Transform::Bucket(bucket)) => {
                    let buckets = bucket.buckets.unwrap_or(0);
                    if buckets == 0 {
                        return Err(StatsError::SchemaValidation(format!(
                            "partition field {:?} bucket count must be positive",
                            field.name.as_deref().unwrap_or("")
                        )));
                    }
                    let digest = Sha256::digest(values_array.value(row).as_bytes());
                    let hash = u32::from_be_bytes(
                        digest[..4]
                            .try_into()
                            .expect("SHA-256 prefix is four bytes"),
                    );
                    (hash % buckets).to_string()
                }
                None => {
                    return Err(StatsError::SchemaValidation(format!(
                        "partition field {:?} has no transform",
                        field.name.as_deref().unwrap_or("")
                    )))
                }
            };
            values.insert(field.name.as_deref().unwrap_or("").to_string(), value);
        }
        indices
            .entry(SegmentPartition { spec_id, values })
            .or_default()
            .push(row as u32);
    }
    indices
        .into_iter()
        .map(|(partition, indices)| {
            select_rows(batch, indices).map(|batch| (Some(partition), batch))
        })
        .collect()
}

fn batch_seq_bounds(batch: &RecordBatch) -> Result<(i64, i64), StatsError> {
    let seq = batch
        .column_by_name("seq")
        .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| StatsError::Internal("object-backed batch has no Int64 seq".to_string()))?;
    (0..seq.len())
        .filter(|index| !seq.is_null(*index))
        .map(|index| seq.value(index))
        .fold(None, |bounds: Option<(i64, i64)>, value| {
            let (min, max) = bounds.unwrap_or((value, value));
            Some((min.min(value), max.max(value)))
        })
        .ok_or_else(|| {
            StatsError::Internal("object-backed batch has no sequence values".to_string())
        })
}
