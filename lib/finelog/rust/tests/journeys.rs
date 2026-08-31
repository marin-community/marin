// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Small-scale reproductions of the failure points the rollout rehearsals hit,
//! each driven through a real [`finelog::store::Store`] via the shared
//! `support` fixture. Where a rehearsal validated a 5 GB production copy with
//! external digest scripts, these journeys validate the same invariants —
//! content equality across a migration, exactly-once sequences, clean failure
//! on missing durable objects — in seconds.

use std::sync::Arc;
use std::time::Duration;

use finelog::proto::finelog::stats::{MigrationPhase, SourceLayout};
use finelog::store::object_store::ObjectId;
use finelog::test_support::{unique_dir, FaultAction, FaultGate, ObjectFault};

mod support;
use support::{
    assert_metadata_only_bootstrap, content_digest, data_object_upload, drive_to_phase,
    live_objects, object_backed_spec, register_legacy, register_v1, run_sql, seq_column,
    try_run_sql, write_row, Cluster, Invariants, TABLE,
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

/// A producer's organic registration and an operator's rollout script race:
/// identical specs are idempotent whoever wins, a divergent spec at the same
/// version is refused deterministically, and the migration the winner started
/// still runs to a healthy activated table.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_registrations_are_idempotent_and_divergent_specs_are_refused() {
    let cluster = Cluster::new("journey_registration_race");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, _faults) = cluster.open();
    register_legacy(&store);
    write_row(&store, "w-1", 10).await;

    // Producer and operator submit the same version-1 spec concurrently.
    let outcomes = std::thread::scope(|scope| {
        [scope.spawn(|| {
            store.register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
        }), scope.spawn(|| {
            store.register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
        })]
        .map(|handle| handle.join().unwrap())
    });
    for outcome in outcomes {
        let registration = outcome.expect("an identical concurrent registration is idempotent");
        assert_eq!(registration.spec_lifecycle.desired_version(), 1);
    }

    // The same version with different contents is a deterministic refusal, not
    // a second migration.
    let divergent = match store.register_versioned_table(
        TABLE,
        object_backed_spec(
            1,
            SourceLayout {
                target_object_bytes: Some(8 * 1024 * 1024),
                ..Default::default()
            },
        ),
    ) {
        Err(error) => error,
        Ok(_) => panic!("a divergent spec at the same version must be refused"),
    };
    assert!(
        divergent.to_string().contains("already registered"),
        "unexpected refusal: {divergent}"
    );

    // The race left one healthy migration behind.
    store.publish_object_catalog(TABLE).await.unwrap();
    drive_to_phase(&store, MigrationPhase::MIGRATION_PHASE_OBSERVING, 8).await;
    assert_eq!(store.spec_lifecycle(TABLE).unwrap().active_version(), 1);
    let observed = invariants.check(&store).await;
    assert_eq!(observed.seqs, vec![1]);

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

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
