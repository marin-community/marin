// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use finelog::errors::StatsError;
use finelog::test_support::{unique_dir, FaultAction, ObjectFault, ObjectOp, ObjectPattern};

use crate::support::{
    assert_metadata_only_bootstrap, register_v1, run_sql, seq_column, write_row,
    write_row_despite_deferral, Cluster, Invariants, TABLE,
};

fn outage() -> StatsError {
    StatsError::Internal("remote object store outage".to_string())
}

/// A sustained remote outage: every upload and HEAD swap fails for rounds on
/// end while writes keep acknowledging on local durability and reads keep
/// serving. When the remote returns, one maintenance round drains the owed
/// revisions — every row exactly once — and a cold boot from the drained root
/// alone serves them all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sustained_remote_outage_buffers_locally_and_drains_exactly_once() {
    let cluster = Cluster::new("journey_remote_outage");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;
    let healthy = invariants.check(&store).await;
    assert_eq!(healthy.seqs, vec![1]);

    // The remote goes away: every upload and every HEAD swap fails.
    faults.arm(
        ObjectFault::new(
            ObjectOp::Write,
            ObjectPattern::Contains("/".to_string()),
            FaultAction::Fail(outage()),
        )
        .forever(),
    );
    faults.arm(
        ObjectFault::new(
            ObjectOp::CompareAndSwap,
            ObjectPattern::Contains("/".to_string()),
            FaultAction::Fail(outage()),
        )
        .forever(),
    );

    // Five write-and-flush rounds through the outage: each acknowledges on
    // local durability and each maintenance round reports the deferral.
    for row in 2..=6 {
        write_row_despite_deferral(&store, &format!("w-{row}"), row * 10).await;
    }
    let mid_outage = seq_column(&run_sql(&store, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(
        mid_outage,
        (1..=6).collect::<Vec<_>>(),
        "reads serve local bytes mid-outage"
    );

    // The remote returns; one maintenance round uploads the staged objects
    // and publishes the owed revision. Invariants re-verify remotely:
    // referenced objects exist and no acknowledged seq is lost or duplicated.
    faults.clear_faults();
    store.maintain_namespace(TABLE, false).await.unwrap();
    let drained = invariants.check(&store).await;
    assert_eq!(drained.seqs, (1..=6).collect::<Vec<_>>());
    assert!(
        drained.state.catalog_generation > healthy.state.catalog_generation,
        "the drained root must carry the outage-era revisions"
    );

    // A cold boot from the drained root alone holds every acknowledged row.
    store.shutdown(Duration::from_secs(1)).await;
    drop(store);
    let cold_dir = unique_dir("journey_remote_outage_cold");
    let (cold, cold_faults) = cluster.open_from(&cold_dir);
    cold.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&cold_faults);
    let seqs = seq_column(&run_sql(&cold, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(seqs, (1..=6).collect::<Vec<_>>());

    cold.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&cold_dir).ok();
    cluster.cleanup();
}
