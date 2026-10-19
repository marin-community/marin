//! Bounded reads of the durable forwarding spool, independent of SQL queries.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{AsArray, BooleanArray, RecordBatch};
use arrow::compute::{concat_batches, filter_record_batch, sort_to_indices, take};
use arrow::datatypes::{Int64Type, SchemaRef};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::file::statistics::Statistics;

use crate::errors::StatsError;
use crate::store::compaction::merge::project_to_schema;
use crate::store::schema::{IMPLICIT_CLUSTER_COLUMN, IMPLICIT_SEQ_COLUMN};
use crate::store::string_column::StringColumn;
use crate::store::Store;

const DECODE_BATCH_ROWS: usize = 8_192;
const SLOW_READ: Duration = Duration::from_secs(5);

struct ForwardSource {
    path: PathBuf,
    bounds: Option<(i64, i64)>,
}

/// Limits for one source prefix. A single oversized row may exceed the byte
/// target; decoding, ordering, and cold object downloads need extra working space.
#[derive(Clone, Copy)]
pub(crate) struct ReadBudget {
    pub rows: usize,
    pub bytes: usize,
    pub segments: usize,
    pub timeout: Duration,
}

/// A complete examined source prefix, including positions excluded by origin.
pub(crate) struct ForwardRead {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
    pub resume_at: Option<i64>,
    pub scanned_through: i64,
}

/// Read ordered, locally produced rows after `cursor` under the host's durability
/// boundary. Missing objects fail the read; they never count as examined rows.
pub(crate) async fn read_forwarding(
    store: Arc<Store>,
    namespace: &str,
    cursor: i64,
    persisted: i64,
    budget: ReadBudget,
) -> Result<ForwardRead, StatsError> {
    assert!(budget.rows > 0 && budget.bytes > 0 && budget.segments > 0);
    let timeout = store
        .object_query_bound()
        .map_or(budget.timeout, |bound| bound.min(budget.timeout));
    let deadline = Instant::now() + timeout;
    tokio::time::timeout(
        timeout,
        read_source(store, namespace, cursor, persisted, budget, deadline),
    )
    .await
    .map_err(|_| StatsError::DeadlineExceeded(format!("forward read of {namespace:?} timed out")))?
}

async fn read_source(
    store: Arc<Store>,
    namespace: &str,
    cursor: i64,
    persisted: i64,
    budget: ReadBudget,
    deadline: Instant,
) -> Result<ForwardRead, StatsError> {
    let started = Instant::now();
    // Legacy compaction and eviction can unlink paths. Object-backed reads also
    // retain their existing snapshot deadline while this guard is held.
    let visibility = Arc::clone(store.query_visibility()).read_owned().await;
    let snapshot_started = Instant::now();
    let snapshot_store = Arc::clone(&store);
    let name = namespace.to_owned();
    let snapshot = tokio::task::spawn_blocking(move || snapshot_store.query_snapshot(&name))
        .await
        .map_err(|error| StatsError::Internal(format!("forward snapshot task: {error}")))??;
    let snapshot_ms = snapshot_started.elapsed().as_millis() as u64;
    let resume_at = resume_after_eviction(cursor, snapshot.min_seq);
    let read_from = resume_at.unwrap_or(cursor);
    let scanned_through = bounded_forward_read_through(
        &snapshot.paths,
        &snapshot.seq_bounds,
        read_from,
        persisted,
        budget.segments,
    )
    .min(read_from.saturating_add(budget.rows.min(i64::MAX as usize) as i64));
    let live_segments = snapshot.paths.len();
    let paths: Vec<_> = snapshot
        .paths
        .into_iter()
        .filter(|path| {
            snapshot
                .seq_bounds
                .get(path)
                .is_none_or(|&(minimum, maximum)| maximum > read_from && minimum <= scanned_through)
        })
        .collect();
    let selected_segments = paths.len();
    let localize_started = Instant::now();
    let mut sources = Vec::with_capacity(paths.len());
    // Cache misses materialize whole objects. Resolve one at a time to avoid
    // holding several object download buffers for one namespace read.
    for path in paths {
        let bounds = snapshot.seq_bounds.get(&path).copied();
        let path = if let Some((objects, segments)) = &snapshot.sources {
            let source = segments.get(&path).ok_or_else(|| {
                StatsError::Internal(format!("forward source reference missing for {path:?}"))
            })?;
            objects.local_path(&source.source).await?
        } else {
            PathBuf::from(path)
        };
        sources.push(ForwardSource { path, bounds });
    }
    let localize_ms = localize_started.elapsed().as_millis() as u64;
    let schema = snapshot.schema;
    let decode_schema = Arc::clone(&schema);
    let scan_started = Instant::now();
    let (batches, scanned_through) = tokio::task::spawn_blocking(move || {
        // A timed-out async caller cannot cancel a running blocking decoder.
        // Keep its paths visible until that decoder actually releases them.
        let _visibility = visibility;
        decode_prefix(
            sources,
            decode_schema,
            read_from,
            scanned_through,
            budget,
            deadline,
        )
    })
    .await
    .map_err(|error| StatsError::Internal(format!("forward decode task: {error}")))??;
    let scan_ms = scan_started.elapsed().as_millis() as u64;
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let retained_bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
    if started.elapsed() >= SLOW_READ {
        tracing::info!(
            namespace,
            cursor,
            persisted,
            scanned_through,
            live_segments,
            selected_segments,
            rows,
            retained_bytes,
            snapshot_ms,
            localize_ms,
            scan_ms,
            total_ms = started.elapsed().as_millis() as u64,
            "finelog forwarder: slow source read completed"
        );
    }
    Ok(ForwardRead {
        schema,
        batches,
        resume_at,
        scanned_through,
    })
}

fn source_reader(
    path: &Path,
    cursor: i64,
    through: i64,
) -> Result<ParquetRecordBatchReader, StatsError> {
    let file = File::open(path).map_err(|error| {
        StatsError::Internal(format!(
            "open forwarding source {}: {error}",
            path.display()
        ))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| StatsError::Internal(format!("open forwarding parquet: {error}")))?;
    // Compacted legacy files can be much larger than a source window. Keep
    // unknown statistics, and prune only on the top-level sequence column.
    let seq_column = builder
        .parquet_schema()
        .columns()
        .iter()
        .position(|column| column.path().parts() == [IMPLICIT_SEQ_COLUMN]);
    let row_groups = builder
        .metadata()
        .row_groups()
        .iter()
        .enumerate()
        .filter_map(|(index, group)| {
            let bounds = seq_column.and_then(|column| {
                let Statistics::Int64(stats) = group.column(column).statistics()? else {
                    return None;
                };
                Some((*stats.min_opt()?, *stats.max_opt()?))
            });
            bounds
                .is_none_or(|(minimum, maximum)| maximum > cursor && minimum <= through)
                .then_some(index)
        })
        .collect();
    builder
        .with_row_groups(row_groups)
        .with_batch_size(DECODE_BATCH_ROWS)
        .build()
        .map_err(|error| StatsError::Internal(format!("open forwarding parquet: {error}")))
}

fn decode_prefix(
    paths: Vec<ForwardSource>,
    schema: SchemaRef,
    cursor: i64,
    mut through: i64,
    budget: ReadBudget,
    deadline: Instant,
) -> Result<(Vec<RecordBatch>, i64), StatsError> {
    let mut batches = Vec::new();
    let mut retained_bytes = 0;
    for source in paths {
        if source
            .bounds
            .is_some_and(|(minimum, maximum)| maximum <= cursor || minimum > through)
        {
            continue;
        }
        if Instant::now() >= deadline {
            return Err(StatsError::DeadlineExceeded(
                "forward decode timed out".into(),
            ));
        }
        for batch in source_reader(&source.path, cursor, through)? {
            if Instant::now() >= deadline {
                return Err(StatsError::DeadlineExceeded(
                    "forward decode timed out".into(),
                ));
            }
            let batch = batch.map_err(|error| {
                StatsError::Internal(format!("decode forwarding parquet: {error}"))
            })?;
            let batch = project_to_schema(&batch, &schema).map_err(|error| {
                StatsError::Internal(format!("align forwarding schema: {error}"))
            })?;
            let batch = filter_source(&batch, cursor, through)?;
            if batch.num_rows() == 0 {
                continue;
            }
            retained_bytes += batch.get_array_memory_size();
            batches.push(batch);
            // Narrow the source prefix, not just the returned row count. Later
            // files can hold lower sequences, so all overlapping files still
            // have to be examined before this boundary is safe to settle.
            while retained_bytes > budget.bytes && through > cursor.saturating_add(1) {
                through = cursor + (through - cursor) / 2;
                let narrowed = batches
                    .into_iter()
                    .map(|batch| filter_source(&batch, cursor, through))
                    .collect::<Result<Vec<_>, _>>()?;
                batches = narrowed
                    .into_iter()
                    .filter(|batch| batch.num_rows() > 0)
                    .collect();
                retained_bytes = batches.iter().map(RecordBatch::get_array_memory_size).sum();
            }
        }
    }
    Ok((order_batches(schema, batches)?, through))
}

fn order_batches(
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<Vec<RecordBatch>, StatsError> {
    if batches.is_empty() || sequence_ordered(&batches) {
        return Ok(batches);
    }
    let batch = concat_batches(&schema, &batches)
        .map_err(|error| StatsError::Internal(format!("combine forwarding rows: {error}")))?;
    drop(batches);
    let seq = batch
        .column_by_name(IMPLICIT_SEQ_COLUMN)
        .expect("stored seq");
    let order = sort_to_indices(seq.as_ref(), None, None)
        .map_err(|error| StatsError::Internal(format!("order forwarding sequences: {error}")))?;
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &order, None))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| StatsError::Internal(format!("gather forwarding rows: {error}")))?;
    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|error| StatsError::Internal(format!("build forwarding rows: {error}")))?;
    Ok(vec![batch])
}

fn filter_source(
    batch: &RecordBatch,
    cursor: i64,
    through: i64,
) -> Result<RecordBatch, StatsError> {
    let seq = batch
        .column_by_name(IMPLICIT_SEQ_COLUMN)
        .expect("stored seq");
    let seq = seq.as_primitive::<Int64Type>();
    let origin = batch
        .column_by_name(IMPLICIT_CLUSTER_COLUMN)
        .and_then(|column| StringColumn::new(column.as_ref()));
    let keep = BooleanArray::from_iter((0..batch.num_rows()).map(|row| {
        let sequence = seq.value(row);
        Some(
            sequence > cursor
                && sequence <= through
                && origin
                    .as_ref()
                    .is_none_or(|values| values.value(row).is_none_or(str::is_empty)),
        )
    }));
    filter_record_batch(batch, &keep)
        .map_err(|error| StatsError::Internal(format!("filter forwarding rows: {error}")))
}

fn sequence_ordered(batches: &[RecordBatch]) -> bool {
    let mut previous = i64::MIN;
    for batch in batches {
        let seq = batch
            .column_by_name(IMPLICIT_SEQ_COLUMN)
            .expect("stored seq");
        for value in seq.as_primitive::<Int64Type>().values() {
            if *value < previous {
                return false;
            }
            previous = *value;
        }
    }
    true
}

/// Bound intentional segment fanout without excluding overlapping ranges.
pub(crate) fn bounded_forward_read_through(
    paths: &[String],
    seq_bounds: &BTreeMap<String, (i64, i64)>,
    read_from: i64,
    persisted: i64,
    max_segments: usize,
) -> i64 {
    assert!(max_segments > 0);
    paths
        .iter()
        .filter_map(|path| seq_bounds.get(path))
        .filter(|(minimum, maximum)| *maximum > read_from && *minimum <= persisted)
        .take(max_segments)
        .map(|(_, maximum)| *maximum)
        .max()
        .unwrap_or(read_from)
        .min(persisted)
}

/// Return `Some(minimum - 1)` when retention removed the next source position.
/// Return `None` when that position is retained or the source has no segments.
pub(crate) fn resume_after_eviction(cursor: i64, minimum: Option<i64>) -> Option<i64> {
    minimum
        .filter(|minimum| cursor.saturating_add(1) < *minimum)
        .map(|minimum| minimum - 1)
}

#[cfg(test)]
#[path = "forwarding_tests.rs"]
mod tests;
