// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Int64Array, StringArray};
use arrow::record_batch::RecordBatch;

use finelog::proto::finelog::stats::ColumnType;
use finelog::store::policy::StoragePolicy;
use finelog::store::schema::{schema_to_arrow, Column};
use finelog::test_support::unique_dir;

use crate::support::{
    assert_metadata_only_bootstrap, register_v1, run_sql, seq_column, worker_schema, write_row,
    Cluster, PERSIST_BUDGET, TABLE,
};

/// A producer's additive schema evolution reaches a migrated table through the
/// legacy RegisterTable path — the route every forwarder takes. The evolution
/// must land in the durable spec: a spec-silent column serves fine until a
/// restart, then vanishes because recovery rebuilds the schema from a spec
/// that never learned it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_additive_evolution_on_a_migrated_table_survives_recovery() {
    let cluster = Cluster::new("journey_schema_evolution");
    let (store, _faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;

    // The forwarder-shaped evolution: a legacy registration adding one
    // nullable column folds into spec version 2, active immediately.
    let mut evolved = worker_schema();
    evolved
        .columns
        .push(Column::new("gpu_id", ColumnType::COLUMN_TYPE_STRING, true));
    tokio::task::block_in_place(|| {
        store.register_table(TABLE, evolved.clone(), StoragePolicy::default())
    })
    .unwrap();
    assert_eq!(store.spec_lifecycle(TABLE).unwrap().active_version(), 2);

    // A row that uses the column, flushed and published under version 2.
    let batch_schema = schema_to_arrow(&evolved);
    let batch = RecordBatch::try_new(
        batch_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["w-2"])),
            Arc::new(Int64Array::from(vec![20])),
            Arc::new(Int64Array::from(vec![20])),
            Arc::new(StringArray::from(vec![Some("gpu-7")])),
        ],
    )
    .unwrap();
    let ipc = finelog::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
    let (_, seq) = store.write_rows(TABLE, &ipc, None).unwrap();
    store.maintain_namespace(TABLE, false).await.unwrap();
    store
        .await_persisted(TABLE, seq, PERSIST_BUDGET)
        .await
        .unwrap();
    store.publish_object_catalog(TABLE).await.unwrap();
    store.shutdown(Duration::from_secs(1)).await;
    drop(store);

    // A cold boot from the remote root alone keeps the column and its value.
    let cold_dir = unique_dir("journey_schema_evolution_cold");
    let (cold, faults) = cluster.open_from(&cold_dir);
    cold.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&faults);
    let batches = run_sql(
        &cold,
        &format!("SELECT gpu_id FROM \"{TABLE}\" WHERE seq = {seq}"),
    )
    .await;
    let gpu = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(0);
    assert_eq!(gpu, "gpu-7", "the evolved column must survive recovery");
    let seqs = seq_column(&run_sql(&cold, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(seqs, vec![1, 2]);

    cold.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&cold_dir).ok();
    cluster.cleanup();
}
