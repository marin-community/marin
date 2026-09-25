// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Failure journeys for forwarding settlement after hub durability.

use std::time::Duration;

use finelog::errors::StatsError;
use finelog::test_support::{
    lost_head_response, FaultAction, ObjectFault, ObjectOp, ObjectPattern,
};

use crate::support::{register_v1, write_row, Cluster, Invariants, TABLE};

fn rejected_head_swap() -> StatsError {
    StatsError::Internal("injected failure before settlement HEAD CAS".to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn settlement_advances_the_cursor_and_releases_only_covered_segments() {
    let cluster = Cluster::new("journey_relay_settlement");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, _) = cluster.open();
    register_v1(&store).await;
    let first = write_row(&store, "w-1", 10).await;
    let second = write_row(&store, "w-2", 20).await;

    let settlement = store.settle_forwarding("hub", TABLE, first).await.unwrap();
    assert_eq!(settlement.cursor, first);
    assert_eq!(settlement.removed_rows, 1);
    assert_eq!(settlement.removed_paths.len(), 1);
    assert_eq!(store.forward_cursor("hub", TABLE).unwrap(), Some(first));
    let live = store.list_segments(TABLE).unwrap();
    assert_eq!(live.len(), 1);
    assert_eq!((live[0].min_seq, live[0].max_seq), (second, second));

    let selected = invariants.check(&store).await;
    assert_eq!(selected.seqs, vec![second]);
    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn settlement_keeps_segments_until_every_target_covers_them() {
    let cluster = Cluster::new("journey_relay_multi_target_settlement");
    let (store, _) = cluster.open();
    register_v1(&store).await;
    let settled = write_row(&store, "w-1", 10).await;
    store
        .set_forward_cursor("backup-hub", TABLE, 0)
        .await
        .unwrap();

    let first = store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .unwrap();
    assert!(first.removed_paths.is_empty());
    assert_eq!(store.list_segments(TABLE).unwrap().len(), 1);

    let second = store
        .settle_forwarding("backup-hub", TABLE, settled)
        .await
        .unwrap();
    assert_eq!(second.removed_paths.len(), 1);
    assert!(store.list_segments(TABLE).unwrap().is_empty());

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_settlement_publication_replays_the_local_tail_after_restart() {
    let cluster = Cluster::new("journey_relay_settlement_restart");
    let mut invariants = Invariants::new(&cluster.remote_dir);
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    let settled = write_row(&store, "w-1", 10).await;
    let before = invariants.check(&store).await;
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));

    assert!(store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .is_err());
    assert_eq!(store.forward_cursor("hub", TABLE).unwrap(), Some(settled));
    assert!(store.list_segments(TABLE).unwrap().is_empty());
    let still_selected = cluster.states().load(TABLE).await.unwrap().unwrap();
    assert_eq!(
        still_selected.revision().get(),
        before.state.catalog_generation.unwrap()
    );
    drop(store);

    let (restarted, _) = cluster.open();
    restarted.recover_tables().await.unwrap();
    restarted.maintain_namespace(TABLE, false).await.unwrap();
    assert_eq!(
        restarted.forward_cursor("hub", TABLE).unwrap(),
        Some(settled)
    );
    assert!(restarted.list_segments(TABLE).unwrap().is_empty());
    let recovered = cluster.states().load(TABLE).await.unwrap().unwrap();
    assert_eq!(recovered.catalog.forward_cursors.len(), 1);
    assert_eq!(recovered.catalog.forward_cursors[0].cursor, Some(settled));
    assert!(recovered
        .catalog
        .version_segments
        .iter()
        .all(|version| version.live_segments.is_empty()));
    assert!(recovered.catalog.direct_query_segments.is_empty());

    restarted.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn segment_free_relay_replays_a_metadata_tail_after_restart() {
    let cluster = Cluster::new("journey_relay_empty_metadata_restart");
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    let settled = write_row(&store, "w-1", 10).await;
    store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .unwrap();
    assert!(store.list_segments(TABLE).unwrap().is_empty());

    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));
    assert!(store
        .set_forward_cursor("backup-hub", TABLE, settled)
        .await
        .is_err());
    drop(store);

    let (restarted, _) = cluster.open();
    restarted.recover_tables().await.unwrap();
    assert_eq!(restarted.namespace_persisted_seq(TABLE).unwrap(), settled);
    assert_eq!(
        restarted.forward_cursor("hub", TABLE).unwrap(),
        Some(settled)
    );
    assert_eq!(
        restarted.forward_cursor("backup-hub", TABLE).unwrap(),
        Some(settled)
    );
    restarted.maintain_namespace(TABLE, false).await.unwrap();
    let selected = cluster.states().load(TABLE).await.unwrap().unwrap();
    assert_eq!(selected.catalog.persisted_high_water, Some(settled));
    assert!(selected
        .catalog
        .version_segments
        .iter()
        .all(|version| version.live_segments.is_empty()));

    restarted.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_process_retry_reconciles_accounting_after_publication_failure() {
    let cluster = Cluster::new("journey_relay_settlement_retry");
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    let settled = write_row(&store, "w-1", 10).await;
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::Fail(rejected_head_swap()),
    ));

    assert!(store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .is_err());
    let before_retry = store
        .list_namespaces_with_stats()
        .unwrap()
        .into_iter()
        .find(|(name, _, _, _)| name == TABLE)
        .unwrap()
        .2;
    assert_eq!(before_retry.segment_count, 1);

    let retried = store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .unwrap();
    assert!(retried.removed_paths.is_empty());
    let after_retry = store
        .list_namespaces_with_stats()
        .unwrap()
        .into_iter()
        .find(|(name, _, _, _)| name == TABLE)
        .unwrap()
        .2;
    assert_eq!(after_retry.segment_count, 0);

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lost_successful_settlement_response_is_resolved_by_the_exact_tip() {
    let cluster = Cluster::new("journey_relay_settlement_lost_response");
    let (store, faults) = cluster.open();
    register_v1(&store).await;
    let settled = write_row(&store, "w-1", 10).await;
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::LoseResponse {
            error: lost_head_response(),
            gate: None,
        },
    ));

    let result = store
        .settle_forwarding("hub", TABLE, settled)
        .await
        .unwrap();
    assert_eq!(result.cursor, settled);
    assert_eq!(store.forward_cursor("hub", TABLE).unwrap(), Some(settled));
    assert!(store.list_segments(TABLE).unwrap().is_empty());

    store.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}
