// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use finelog::test_support::{FaultAction, FaultGate, ObjectFault};

use crate::support::{
    data_object_upload, encode_worker_row, register_v1, retargeted_spec, write_row, Cluster,
    Invariants, TABLE,
};

/// A write acks while a migration backfill batch is mid-rewrite. The backfill
/// holds the table's flush gate only while assembling its batch from a catalog
/// snapshot; the rewrite itself runs outside the gate, so a concurrent
/// WriteRows flush commits and acks instead of stalling for the whole batch.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_acks_while_a_backfill_batch_is_mid_rewrite() {
    let cluster = Cluster::new("scenario_write_during_backfill");
    let mut invariants = Invariants::new(&cluster.remote_dir);

    let (store, faults) = cluster.open();
    register_v1(&store).await;
    for (worker, mem_bytes) in [("w-1", 10), ("w-2", 20), ("w-3", 30)] {
        write_row(&store, worker, mem_bytes).await;
    }

    // Park the backfill's first output upload, freezing the batch mid-rewrite.
    let mid_rewrite = FaultGate::new();
    let (op, pattern) = data_object_upload();
    faults.arm(ObjectFault::new(
        op,
        pattern,
        FaultAction::Park(Arc::clone(&mid_rewrite)),
    ));
    store
        .register_versioned_table(TABLE, retargeted_spec(2))
        .unwrap();
    store.publish_object_catalog(TABLE).await.unwrap();
    let backfilling = {
        let tables = store.tables().clone();
        tokio::spawn(async move { tables.maintain(TABLE, false).await })
    };
    mid_rewrite.entered().await;

    // The ack path a WriteRows request takes: append, flush, await durability.
    let runtime = store.tables().get(TABLE).unwrap();
    let ipc = encode_worker_row("w-4", 40);
    let (_, seq) = store.write_rows(TABLE, &ipc, None).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        runtime.flush().await.unwrap();
        store
            .await_persisted(TABLE, seq, Duration::from_secs(5))
            .await
            .unwrap();
    })
    .await
    .expect("a write must ack while the backfill batch is mid-rewrite");

    mid_rewrite.release();
    backfilling.await.unwrap().unwrap();
    for _ in 0..4 {
        if store.spec_lifecycle(TABLE).unwrap().active_version() == 2 {
            break;
        }
        store.maintain_namespace(TABLE, false).await.unwrap();
    }
    assert_eq!(store.spec_lifecycle(TABLE).unwrap().active_version(), 2);
    let after = invariants.check(&store).await;
    assert_eq!(
        after.seqs,
        vec![1, 2, 3, 4],
        "the concurrent write lands exactly once"
    );

    drop(store);
    cluster.cleanup();
}
