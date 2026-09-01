// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use finelog::proto::finelog::stats::{MigrationPhase, SourceLayout};

use crate::support::{
    drive_to_phase, object_backed_spec, register_legacy, write_row, Cluster, Invariants, TABLE,
};

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
        [
            scope.spawn(|| {
                store
                    .register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
            }),
            scope.spawn(|| {
                store
                    .register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
            }),
        ]
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
