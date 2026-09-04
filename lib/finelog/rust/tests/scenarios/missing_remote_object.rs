// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use finelog::store::object_store::ObjectId;
use finelog::test_support::unique_dir;

use crate::support::{
    assert_metadata_only_bootstrap, live_objects, register_v1, try_run_sql, write_row, Cluster,
    TABLE,
};

/// A remote root that lost a referenced data object — the state the
/// first-publication guard exists to prevent — still recovers metadata-only,
/// fails queries with the missing object named, and survives repeated
/// maintenance instead of wedging the table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_missing_remote_object_fails_reads_cleanly_without_wedging_recovery() {
    let cluster = Cluster::new("journey_missing_object");
    let (store, _faults) = cluster.open();
    register_v1(&store).await;
    for (worker, mem_bytes) in [("w-1", 10), ("w-2", 20), ("w-3", 30)] {
        write_row(&store, worker, mem_bytes).await;
    }
    store.shutdown(Duration::from_secs(1)).await;
    drop(store);

    // Delete one live data object out from under the published state.
    let state = cluster.states().load(TABLE).await.unwrap().unwrap();
    let victim = live_objects(&state.catalog)
        .into_iter()
        .next()
        .expect("a flushed table references at least one object");
    cluster
        .objects()
        .delete(&ObjectId::parse(&victim).unwrap())
        .await
        .unwrap();

    // A cold server still bootstraps: recovery is metadata-only by design.
    let cold_dir = unique_dir("journey_missing_object_cold");
    let (cold, faults) = cluster.open_from(&cold_dir);
    cold.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&faults);

    // Reads fail with the lost object named rather than returning partial rows.
    let error = try_run_sql(&cold, &format!("SELECT seq FROM \"{TABLE}\""))
        .await
        .expect_err("a scan over a missing object must fail, not drop rows");
    let basename = victim.rsplit('/').next().unwrap();
    assert!(
        error.contains(basename),
        "the error must name the missing object; got: {error}"
    );

    // Maintenance keeps returning — the table degrades, the process does not.
    for _ in 0..3 {
        let _ = cold.maintain_namespace(TABLE, false).await;
    }
    assert_eq!(cold.spec_lifecycle(TABLE).unwrap().active_version(), 1);

    cold.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&cold_dir).ok();
    cluster.cleanup();
}
