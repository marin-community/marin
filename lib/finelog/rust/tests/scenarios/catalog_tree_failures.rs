// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Failure journeys for the parent-linked catalog commit protocol.

use std::time::Duration;

use finelog::errors::StatsError;
use finelog::test_support::{
    lost_head_response, FaultAction, ObjectFault, ObjectOp, ObjectPattern,
};

use crate::support::{
    assert_metadata_only_bootstrap, encode_worker_row, object_backed_spec, register_v1, write_row,
    Cluster, Invariants, TABLE,
};
use finelog::proto::finelog::stats::SourceLayout;

fn rejected_head_swap() -> StatsError {
    StatsError::Internal("injected failure before HEAD CAS".to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_first_head_cas_retries_past_its_unselected_checkpoint() {
    let cluster = Cluster::new("journey_catalog_first_head_orphan");
    let (store, faults) = cluster.open();
    store
        .register_versioned_table(TABLE, object_backed_spec(1, SourceLayout::default()))
        .unwrap();
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));

    assert!(store.publish_object_catalog(TABLE).await.is_err());
    assert!(cluster.states().load(TABLE).await.unwrap().is_none());
    store.publish_object_catalog(TABLE).await.unwrap();
    assert_eq!(
        cluster
            .states()
            .load(TABLE)
            .await
            .unwrap()
            .unwrap()
            .revision()
            .get(),
        1
    );

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn uploaded_node_that_loses_head_cas_is_an_orphan_and_retry_converges() {
    let cluster = Cluster::new("journey_catalog_node_orphan");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;
    let before = invariants.check(&store).await;

    faults.clear_calls();
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));
    let ipc = encode_worker_row("w-2", 20);
    let (_, seq) = store.write_rows(TABLE, &ipc, None).unwrap();
    assert!(store.maintain_namespace(TABLE, false).await.is_err());

    let still_selected = cluster.states().load(TABLE).await.unwrap().unwrap();
    assert_eq!(
        still_selected.revision().get(),
        before.state.catalog_generation.unwrap()
    );
    assert!(
        faults
            .keys_for(ObjectOp::Write)
            .iter()
            .any(|key| key.contains("/catalogs/")),
        "the immutable node must upload before HEAD is attempted"
    );

    store.maintain_namespace(TABLE, false).await.unwrap();
    store
        .await_persisted(TABLE, seq, Duration::from_secs(30))
        .await
        .unwrap();
    let after = invariants.check(&store).await;
    assert_eq!(after.seqs, vec![1, 2]);
    assert!(after.state.catalog_generation > before.state.catalog_generation);

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lost_head_response_is_settled_only_by_the_exact_selected_tip() {
    let cluster = Cluster::new("journey_catalog_lost_head_response");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;

    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::LoseResponse {
            error: lost_head_response(),
            gate: None,
        },
    ));
    let seq = write_row(&store, "w-2", 20).await;
    let selected = invariants.check(&store).await;
    assert_eq!(selected.seqs, vec![1, seq]);

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn crash_with_a_local_tail_replays_it_as_a_child_of_remote_head() {
    let cluster = Cluster::new("journey_catalog_local_tail");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    write_row(&store, "w-1", 10).await;
    let before = invariants.check(&store).await;

    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));
    let ipc = encode_worker_row("w-2", 20);
    let (_, second_seq) = store.write_rows(TABLE, &ipc, None).unwrap();
    assert!(store.maintain_namespace(TABLE, false).await.is_err());
    drop(store);

    let (restarted, restart_faults) = cluster.open();
    restarted.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&restart_faults);
    restarted.maintain_namespace(TABLE, false).await.unwrap();
    restarted
        .await_persisted(TABLE, second_seq, Duration::from_secs(30))
        .await
        .unwrap();
    let after = invariants.check(&restarted).await;
    assert_eq!(after.seqs, vec![1, 2]);
    assert!(after.state.catalog_generation > before.state.catalog_generation);

    restarted.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}
