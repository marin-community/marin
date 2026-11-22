//! Per-table segment-count history.
//!
//! A hub that enables it writes one row per table to
//! [`TABLE_SEGMENTS_NAMESPACE`] every [`SEGMENT_HISTORY_INTERVAL`], so the
//! growth of a compaction backlog can be graphed after the fact. The rows carry
//! the same statistics `ListNamespaces` reports at that moment.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use tokio::task::JoinHandle;

use crate::errors::StatsError;
use crate::proto::finelog::stats::ColumnType;
use crate::server::diagnostics::spawn_periodic;
use crate::store::ipc::encode_ipc;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_arrow, Column, Schema};
use crate::store::types::NamespaceStats;
use crate::store::Store;

pub const TABLE_SEGMENTS_NAMESPACE: &str = "finelog.table_segments";
pub const SEGMENT_HISTORY_INTERVAL: Duration = Duration::from_secs(60);

/// One column per compaction level. L3 is the terminal level of the default
/// compaction policy, so a segment above it has no column to be counted in.
const LEVEL_COLUMNS: [&str; 4] = [
    "l0_segment_count",
    "l1_segment_count",
    "l2_segment_count",
    "l3_segment_count",
];

/// Bounds the hub's local copy. The remote archive keeps every row, as it does
/// for every other table.
const LOCAL_CACHE_MAX_AGE_SECONDS: i64 = 30 * 24 * 60 * 60;

fn table_segments_schema() -> Schema {
    let mut columns = vec![
        Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
        Column::new("namespace", ColumnType::COLUMN_TYPE_STRING, false),
        Column::new("segment_count", ColumnType::COLUMN_TYPE_INT64, false),
    ];
    columns.extend(
        LEVEL_COLUMNS
            .iter()
            .map(|name| Column::new(*name, ColumnType::COLUMN_TYPE_INT64, false)),
    );
    columns.push(Column::new(
        "row_count",
        ColumnType::COLUMN_TYPE_INT64,
        false,
    ));
    columns.push(Column::new(
        "byte_size",
        ColumnType::COLUMN_TYPE_INT64,
        false,
    ));
    Schema::new(columns, "timestamp_ms")
}

fn table_segments_batch(
    schema: &Schema,
    timestamp_ms: i64,
    tables: &[(String, NamespaceStats)],
) -> Result<RecordBatch, StatsError> {
    let mut levels = vec![Vec::with_capacity(tables.len()); LEVEL_COLUMNS.len()];
    for (namespace, stats) in tables {
        if stats.segment_count_by_level.len() > LEVEL_COLUMNS.len() {
            return Err(StatsError::Internal(format!(
                "{namespace} has segments above L{}, which {TABLE_SEGMENTS_NAMESPACE} has no column for",
                LEVEL_COLUMNS.len() - 1
            )));
        }
        for (level, counts) in levels.iter_mut().enumerate() {
            counts.push(i64::from(
                stats
                    .segment_count_by_level
                    .get(level)
                    .copied()
                    .unwrap_or(0),
            ));
        }
    }
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![timestamp_ms; tables.len()])),
        Arc::new(StringArray::from_iter_values(
            tables.iter().map(|(namespace, _)| namespace.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(
            tables
                .iter()
                .map(|(_, stats)| i64::from(stats.segment_count)),
        )),
    ];
    columns.extend(
        levels
            .into_iter()
            .map(|counts| Arc::new(Int64Array::from(counts)) as ArrayRef),
    );
    columns.push(Arc::new(Int64Array::from_iter_values(
        tables.iter().map(|(_, stats)| stats.row_count),
    )));
    columns.push(Arc::new(Int64Array::from_iter_values(
        tables.iter().map(|(_, stats)| stats.byte_size),
    )));
    RecordBatch::try_new(schema_to_arrow(schema), columns).map_err(|error| {
        StatsError::Internal(format!("building {TABLE_SEGMENTS_NAMESPACE} rows: {error}"))
    })
}

/// Write one row per live table with its current segment statistics.
pub fn record_table_segments(store: &Store, timestamp_ms: i64) -> Result<(), StatsError> {
    let schema = table_segments_schema();
    store.register_managed_table(
        TABLE_SEGMENTS_NAMESPACE,
        schema.clone(),
        StoragePolicy {
            max_age_seconds: Some(LOCAL_CACHE_MAX_AGE_SECONDS),
            ..StoragePolicy::default()
        },
    )?;
    let tables: Vec<(String, NamespaceStats)> = store
        .list_namespaces_with_stats()?
        .into_iter()
        .map(|(namespace, _, stats, _)| (namespace, stats))
        .collect();
    let batch = table_segments_batch(&schema, timestamp_ms, &tables)?;
    let ipc = encode_ipc(&batch.schema(), &[batch]).map_err(|error| {
        StatsError::Internal(format!("encoding {TABLE_SEGMENTS_NAMESPACE} rows: {error}"))
    })?;
    store.write_ingestion_rows(TABLE_SEGMENTS_NAMESPACE, &ipc, None)?;
    Ok(())
}

async fn record_now(store: Arc<Store>) {
    let recorded = tokio::task::spawn_blocking(move || {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                StatsError::Internal(format!("system clock is before the unix epoch: {error}"))
            })?
            .as_millis() as i64;
        record_table_segments(&store, timestamp_ms)
    })
    .await;
    // One failed write must not end the history for the rest of the process.
    match recorded {
        Ok(Ok(())) => {}
        Ok(Err(error)) => tracing::warn!(%error, "table segment history write failed"),
        Err(error) => tracing::warn!(%error, "table segment history task failed"),
    }
}

/// Spawn the history task. It stops the same way the diagnostics task does.
pub fn spawn_table_segment_history(
    store: Arc<Store>,
    stop: Arc<AtomicBool>,
    shutdown: Arc<tokio::sync::Notify>,
) -> JoinHandle<()> {
    spawn_periodic(SEGMENT_HISTORY_INTERVAL, stop, shutdown, move || {
        record_now(Arc::clone(&store))
    })
}

#[cfg(test)]
mod tests {
    use arrow::array::AsArray;
    use arrow::datatypes::{DataType, Int64Type};

    use super::*;
    use crate::query::{make_ctx, query_namespaces, run_query_over};
    use crate::store::compaction::config::CompactionConfig;
    use crate::store::ServeMode;
    use crate::test_support::unique_dir;

    fn worker_schema() -> Schema {
        Schema::new(
            vec![Column::new(
                "worker_id",
                ColumnType::COLUMN_TYPE_STRING,
                false,
            )],
            "worker_id",
        )
    }

    fn write_worker(store: &Store, worker: &str) {
        let schema = schema_to_arrow(&worker_schema());
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![worker]))],
        )
        .unwrap();
        let ipc = encode_ipc(&schema, &[batch]).unwrap();
        store.write_rows("iris.worker", &ipc, None).unwrap();
    }

    /// `(namespace, segment_count, l0, l1)` for every recorded row, by namespace.
    async fn recorded_rows(store: &Store) -> Vec<(String, i64, i64, i64)> {
        let sql = format!(
            "SELECT namespace, segment_count, l0_segment_count, l1_segment_count \
             FROM \"{TABLE_SEGMENTS_NAMESPACE}\" ORDER BY namespace"
        );
        let _guard = store.query_visibility().read().await;
        let providers = store
            .query_providers(&query_namespaces(&make_ctx(), &sql).unwrap())
            .unwrap();
        let batches = run_query_over(&make_ctx(), providers, &sql)
            .await
            .unwrap()
            .batches;
        let mut rows = Vec::new();
        for batch in batches {
            let namespaces = arrow::compute::cast(batch.column(0), &DataType::Utf8).unwrap();
            let namespaces = namespaces.as_string::<i32>();
            let int = |column: usize| batch.column(column).as_primitive::<Int64Type>().clone();
            let (segments, l0, l1) = (int(1), int(2), int(3));
            for row in 0..batch.num_rows() {
                rows.push((
                    namespaces.value(row).to_string(),
                    segments.value(row),
                    l0.value(row),
                    l1.value(row),
                ));
            }
        }
        rows
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_tick_records_each_table_with_its_segments_split_by_level() {
        let store = Store::new(
            Some(unique_dir("segment-history")),
            String::new(),
            crate::indices::cache::DEFAULT_INDEX_CACHE_MB,
            ServeMode::Live,
        )
        .unwrap();
        store
            .register_table("iris.worker", worker_schema(), StoragePolicy::default())
            .unwrap();
        write_worker(&store, "w-a");
        // A forced maintenance pass flushes the row and promotes its segment to L1.
        store.maintain_namespace("iris.worker", true).await.unwrap();
        write_worker(&store, "w-b");
        store
            .maintain_namespace("iris.worker", false)
            .await
            .unwrap();

        record_table_segments(&store, 1_000).unwrap();
        store
            .maintain_namespace(TABLE_SEGMENTS_NAMESPACE, false)
            .await
            .unwrap();

        let rows = recorded_rows(&store).await;
        assert!(
            rows.contains(&("iris.worker".to_string(), 2, 1, 1)),
            "{rows:?}"
        );
        assert!(rows.contains(&("log".to_string(), 0, 0, 0)), "{rows:?}");
    }

    #[test]
    fn a_segment_above_the_last_level_column_is_an_error() {
        let stats = NamespaceStats {
            segment_count: 1,
            segment_count_by_level: vec![0, 0, 0, 0, 1],
            ..NamespaceStats::default()
        };

        let result = table_segments_batch(
            &table_segments_schema(),
            1_000,
            &[("iris.worker".to_string(), stats)],
        );

        assert!(matches!(result, Err(StatsError::Internal(_))));
    }

    #[test]
    fn the_level_columns_cover_the_default_compaction_levels() {
        let terminal_level = CompactionConfig::default().terminal_level();

        assert_eq!(LEVEL_COLUMNS.len(), terminal_level as usize + 1);
    }
}
