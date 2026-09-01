// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use finelog::proto::finelog::stats::{MigrationPhase, SourceLayout};
use finelog::test_support::{unique_dir, FaultAction, FaultGate, ObjectFault};

use crate::support::{
    assert_metadata_only_bootstrap, content_digest, data_object_upload, drive_to_phase,
    object_backed_spec, register_legacy, run_sql, seq_column, write_row, Cluster, TABLE,
};

/// The full rollout rehearsal in a box: a legacy version-0 table migrates to
/// object-backed storage while it keeps serving queries and taking writes, its
/// content digest never changes, and after retirement a cold server recovers
/// everything from the remote root alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_legacy_import_preserves_content_through_retirement_and_cold_boot() {
    let cluster = Cluster::new("journey_rehearsal");
    let (store, faults) = cluster.open();
    register_legacy(&store);
    for row in 0..6 {
        write_row(&store, &format!("w-{row}"), row * 10).await;
    }
    let baseline_seq = 6;
    let pre = content_digest(&store, baseline_seq).await;

    // Registering the object spec starts the version-0 import. Park the writer
    // mid-backfill and query through it: the legacy table must keep serving.
    store
        .register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
        .unwrap();
    store.publish_object_catalog(TABLE).await.unwrap();
    let mid_backfill = FaultGate::new();
    let (op, pattern) = data_object_upload();
    faults.arm(ObjectFault::new(
        op,
        pattern,
        FaultAction::Park(Arc::clone(&mid_backfill)),
    ));
    let backfilling = {
        let tables = store.tables().clone();
        tokio::spawn(async move { tables.maintain(TABLE, false).await })
    };
    mid_backfill.entered().await;
    assert_eq!(content_digest(&store, baseline_seq).await, pre, "mid-backfill");
    mid_backfill.release();
    backfilling.await.unwrap().unwrap();

    // Activation, a write during the observation window, retirement.
    drive_to_phase(&store, MigrationPhase::MIGRATION_PHASE_OBSERVING, 8).await;
    let continuity_seq = write_row(&store, "w-continuity", 999).await;
    assert_eq!(content_digest(&store, baseline_seq).await, pre, "observing");
    store.expire_migration_observation(TABLE).unwrap();
    drive_to_phase(&store, MigrationPhase::MIGRATION_PHASE_RETIRED, 8).await;
    assert_eq!(content_digest(&store, baseline_seq).await, pre, "retired");

    // Cold boot: a fresh data directory recovering from the remote root alone
    // holds the same content, the continuity row included.
    store.shutdown(Duration::from_secs(1)).await;
    drop(store);
    let cold_dir = unique_dir("journey_rehearsal_cold");
    let (cold, cold_faults) = cluster.open_from(&cold_dir);
    cold.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&cold_faults);
    assert_eq!(content_digest(&cold, baseline_seq).await, pre, "cold boot");
    let seqs = seq_column(&run_sql(&cold, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(seqs, (1..=continuity_seq).collect::<Vec<_>>());

    cold.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&cold_dir).ok();
    cluster.cleanup();
}
