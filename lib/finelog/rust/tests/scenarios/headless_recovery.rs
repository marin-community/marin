// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use finelog::store::object_store::ObjectId;
use finelog::test_support::unique_dir;

use crate::support::{
    assert_metadata_only_bootstrap, register_v1, run_sql, seq_column, write_row,
    write_row_despite_deferral, Cluster, TABLE,
};

/// A remote root that lost its HEAD — a bucket lifecycle rule or human error;
/// the software never deletes one — while its catalogs survived. The server
/// flags the table degraded but keeps serving it and accumulating writes
/// locally, refuses to start a second history over the surviving catalogs,
/// and drains everything once an operator restores HEAD and restarts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_head_degrades_the_table_without_starting_a_new_history() {
    let cluster = Cluster::new("journey_headless");
    let (store, _faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;
    store.publish_object_catalog(TABLE).await.unwrap();
    store.shutdown(Duration::from_secs(1)).await;
    drop(store);

    // HEAD vanishes out from under the root; the catalogs stay.
    let head_id = ObjectId::table(TABLE, "HEAD.json").unwrap();
    let saved_head = cluster.objects().read(&head_id).await.unwrap().unwrap();
    cluster.objects().delete(&head_id).await.unwrap();

    // A warm restart over the same data directory flags the confusion.
    let (degraded, _faults) = cluster.open();
    degraded.recover_tables().await.unwrap();
    let reason = degraded
        .table_degraded_reason(TABLE)
        .expect("a headless table must be flagged degraded");
    assert!(reason.contains("no HEAD"), "{reason}");

    // Writes still accumulate and acknowledge on local durability, and reads
    // serve them; the publish attempt inside maintenance stays refused.
    write_row_despite_deferral(&degraded, "w-2", 20).await;
    let deferred = degraded.maintain_namespace(TABLE, false).await;
    assert!(deferred.is_err(), "maintenance must report the deferral");
    let seqs = seq_column(&run_sql(&degraded, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(seqs, vec![1, 2]);
    assert!(
        cluster.objects().read(&head_id).await.unwrap().is_none(),
        "a headless table must not start a second history"
    );
    degraded.shutdown(Duration::from_secs(1)).await;
    drop(degraded);

    // Operator repair: restore HEAD, restart. The local revisions committed
    // while headless roll forward and publish.
    cluster
        .objects()
        .write(&head_id, saved_head.bytes)
        .await
        .unwrap();
    let (repaired, _faults) = cluster.open();
    repaired.recover_tables().await.unwrap();
    assert_eq!(repaired.table_degraded_reason(TABLE), None);
    repaired.maintain_namespace(TABLE, false).await.unwrap();
    repaired.shutdown(Duration::from_secs(1)).await;
    drop(repaired);

    // A cold boot from the repaired root alone holds every row, the ones
    // written while headless included.
    let cold_dir = unique_dir("journey_headless_cold");
    let (cold, cold_faults) = cluster.open_from(&cold_dir);
    cold.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&cold_faults);
    let seqs = seq_column(&run_sql(&cold, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(seqs, vec![1, 2]);

    cold.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&cold_dir).ok();
    cluster.cleanup();
}
