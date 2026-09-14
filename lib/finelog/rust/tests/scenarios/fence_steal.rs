// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use finelog::test_support::{
    lost_head_response, unique_dir, FaultAction, FaultGate, ObjectFault, ObjectOp, ObjectPattern,
};

use crate::support::{encode_worker_row, register_v1, write_row, Cluster, Invariants, TABLE};

/// A replacement writer that claims the fence while the original writer's HEAD
/// swap has applied but not been reported leaves exactly one writer standing.
/// The original resolves to fenced; whichever revision HEAD holds, the
/// replacement reads it, keeps writing from it, and never loses or duplicates
/// an acknowledged row.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_fence_steal_during_an_ambiguous_flush_commit_leaves_one_writer() {
    let cluster = Cluster::new("scenario_fence_steal");
    let mut invariants = Invariants::new(&cluster.remote_dir);

    let (original, faults) = cluster.open();
    register_v1(&original).await;
    write_row(&original, "w-1", 10).await;
    let settled = invariants.check(&original).await;
    assert_eq!(settled.seqs, vec![1]);
    let settled_revision = settled.state.catalog_generation.unwrap();

    // The next flush's HEAD swap applies, then parks, then reports itself as
    // ambiguous — the lost-response case.
    let ambiguous = FaultGate::new();
    faults.arm(ObjectFault::new(
        ObjectOp::CompareAndSwap,
        ObjectPattern::EndsWith("HEAD.json".to_string()),
        FaultAction::LoseResponse {
            error: lost_head_response(),
            gate: Some(Arc::clone(&ambiguous)),
        },
    ));
    let ipc = encode_worker_row("w-2", 20);
    let (_, ambiguous_seq) = original.write_rows(TABLE, &ipc, None).unwrap();
    let flushing = {
        let tables = original.tables().clone();
        tokio::spawn(async move { tables.maintain(TABLE, false).await })
    };
    ambiguous.entered().await;

    // A replacement process claims the table while the original is still inside
    // its unresolved commit.
    let replacement_dir = unique_dir("scenario_fence_steal_replacement");
    let (replacement, _replacement_faults) = cluster.open_from(&replacement_dir);
    replacement.recover_tables().await.unwrap();

    ambiguous.release();
    let _ = flushing.await.unwrap();

    // Exactly one writer may still commit. The original observes the steal and
    // stops accepting writes.
    let rejected = original.write_rows(TABLE, &ipc, None).unwrap_err();
    assert!(
        matches!(rejected, finelog::errors::StatsError::SchemaConflict(_)),
        "the fenced writer must refuse writes, got {rejected:?}"
    );
    assert!(
        original.publish_object_catalog(TABLE).await.is_err(),
        "the fenced writer must not publish again"
    );

    // The design allows either outcome for the ambiguous commit. Both keep the
    // acknowledged prefix and never move the revision backwards.
    let observed = invariants.check(&replacement).await;
    let revision = observed.state.catalog_generation.unwrap();
    assert!(
        revision >= settled_revision,
        "the replacement must not publish behind the settled revision"
    );
    let ambiguous_row_is_durable = observed.seqs.contains(&ambiguous_seq);
    assert!(
        observed.seqs.starts_with(&[1]),
        "the settled prefix must survive the steal, got {:?}",
        observed.seqs
    );
    assert_eq!(
        observed.seqs.len(),
        if ambiguous_row_is_durable { 2 } else { 1 },
        "the ambiguous commit is either durable or absent, never partial"
    );
    // With this interleaving the swap applied before the claim, so the design's
    // durable branch is the one that must be taken: the replacement inherits the
    // revision the fenced writer never learned it had published.
    assert!(
        ambiguous_row_is_durable,
        "a HEAD swap that applied before the claim must be visible to the replacement"
    );
    assert!(revision > settled_revision);

    // The replacement owns the table and keeps writing from what it loaded.
    let next = write_row(&replacement, "w-3", 30).await;
    let after = invariants.check(&replacement).await;
    assert_eq!(*after.seqs.last().unwrap(), next);
    assert_eq!(
        after.seqs.len(),
        observed.seqs.len() + 1,
        "the replacement's write is the only new row"
    );

    replacement.shutdown(Duration::from_secs(1)).await;
    original.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&replacement_dir).ok();
    cluster.cleanup();
}
