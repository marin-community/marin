//! Table fixtures shared by the tests of the modules that operate on a
//! [`TableRuntime`]: the runtime itself, both compaction drivers, the index
//! backfill, and the legacy archive and layout paths.
//!
//! Everything here builds a real runtime through [`TableRuntime::open`], so a
//! test drives production entry points rather than a hand-assembled state.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field};
use tokio::sync::{Notify, RwLock};

use crate::levanter_metrics_policy::levanter_metrics_schema;
use crate::proto::finelog::stats::ColumnType;
use crate::store::catalog::object_state_store::ObjectTableStateStore;
use crate::store::catalog::state_store::TableStateStore;
use crate::store::catalog::Catalog;
use crate::store::object_store::{
    build_remote_object_store, CachedObjectStore, LegacyObjectStore, ObjectStore,
};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_arrow, with_implicit_seq, AlignedBatch, Column, Schema};
use crate::store::table::controller::TableController;
use crate::store::table::runtime::TableRuntime;
use crate::store::table::ObjectPersistence;
use crate::store::table_state::WriterFence;

pub fn worker_schema() -> Schema {
    with_implicit_seq(Schema::new(
        vec![
            Column::new("worker_id", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
        ],
        "timestamp_ms",
    ))
}

pub fn aligned(n: i64) -> AlignedBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("w{i}")).collect();
    let ts: Vec<i64> = (0..n).map(|i| 1000 + i).collect();
    AlignedBatch {
        arrays: vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(Int64Array::from(ts)),
        ],
        fields: vec![
            Field::new("worker_id", DataType::Utf8, false),
            Field::new("timestamp_ms", DataType::Int64, false),
        ],
        num_rows: n as usize,
        byte_size: 16 * n,
    }
}

pub fn metrics_aligned(run_ids: &[&str]) -> AlignedBatch {
    let schema = schema_to_arrow(&levanter_metrics_schema());
    let rows = run_ids.len();
    let column_names = ["timestamp_ms", "run_id", "step", "name", "kind", "value"];
    AlignedBatch {
        arrays: vec![
            Arc::new(Int64Array::from_iter_values(
                (0..rows).map(|row| 1_700_000_000_000 + row as i64),
            )),
            Arc::new(StringArray::from(run_ids.to_vec())),
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(StringArray::from(vec!["training_loss"; rows])),
            Arc::new(StringArray::from(vec!["scalar"; rows])),
            Arc::new(Float64Array::from_iter_values(
                (0..rows).map(|row| row as f64 / 10.0),
            )),
        ],
        fields: column_names
            .iter()
            .map(|name| schema.field_with_name(name).unwrap().clone())
            .collect(),
        num_rows: rows,
        byte_size: 128 * rows as i64,
    }
}

pub fn tempdir() -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("finelog_table_test_{nanos}"));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Open a table with default wiring (a fresh shared query-visibility lock, no
/// remote, empty policy).
pub fn open_table(
    name: &str,
    schema: Schema,
    data_dir: Option<PathBuf>,
    catalog: Arc<Catalog>,
) -> Arc<TableRuntime> {
    open_table_with_policy(name, schema, data_dir, catalog, StoragePolicy::default())
}

pub fn open_table_with_policy(
    name: &str,
    schema: Schema,
    data_dir: Option<PathBuf>,
    catalog: Arc<Catalog>,
    policy: StoragePolicy,
) -> Arc<TableRuntime> {
    TableRuntime::open(
        name,
        schema,
        data_dir,
        Arc::clone(&catalog),
        Arc::new(RwLock::new(())),
        crate::indices::test_index_registry(),
        crate::maintenance::MaintenanceLimits::new(),
        Arc::new(Notify::new()),
        TableController::start(
            name.to_string(),
            catalog,
            None,
            crate::store::table_state::WriterFence::UNCLAIMED,
        ),
        policy,
    )
    .unwrap()
}

/// Open a table with a configured remote dir + per-table retention policy.
pub fn open_table_remote(
    name: &str,
    schema: Schema,
    data_dir: Option<PathBuf>,
    catalog: Arc<Catalog>,
    remote_log_dir: &str,
    policy: StoragePolicy,
) -> Arc<TableRuntime> {
    let provider = build_remote_object_store(remote_log_dir).unwrap().unwrap();
    let cache_root = data_dir
        .as_ref()
        .and_then(|path| path.parent())
        .unwrap()
        .to_path_buf();
    let object_store =
        Arc::new(CachedObjectStore::new(Arc::new(provider.clone()), cache_root.clone()).unwrap())
            as Arc<dyn ObjectStore>;
    let legacy_object_store = Arc::new(LegacyObjectStore::new(&provider));
    let state_store =
        Arc::new(ObjectTableStateStore::new(object_store.clone())) as Arc<dyn TableStateStore>;
    let controller = TableController::start(
        name.to_string(),
        Arc::clone(&catalog),
        Some(ObjectPersistence {
            table_dir: data_dir.clone().unwrap(),
            store: object_store,
            legacy_store: legacy_object_store,
            state_store,
        }),
        WriterFence::new(1),
    );
    TableRuntime::open(
        name,
        schema,
        data_dir,
        catalog,
        Arc::new(RwLock::new(())),
        crate::indices::test_index_registry(),
        crate::maintenance::MaintenanceLimits::new(),
        Arc::new(Notify::new()),
        controller,
        policy,
    )
    .unwrap()
}

/// Append one batch and force it durable on a sealed L0 segment.
pub async fn write_one(table: &Arc<TableRuntime>) {
    let last = table.append_aligned_batch(&aligned(1));
    table.flush().await.unwrap();
    table
        .await_persisted(last, Duration::from_secs(10))
        .await
        .unwrap();
}

/// The `.parquet` object names the legacy archive holds for `table`.
pub fn remote_files(remote: &std::path::Path, table: &str) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(remote.join(table))
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.file_name().into_string().ok())
                .filter(|name| name.ends_with(".parquet"))
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}
