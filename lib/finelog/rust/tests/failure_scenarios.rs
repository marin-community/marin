// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Composed failure journeys for an object-backed table, driven through the
//! crate's public surface plus the `test-util` seams (`test_support`, the
//! interposed-store constructor).
//!
//! Each test drives a complete [`Store`] — real table manager, controllers,
//! `ObjectTableStateStore`, compaction, and migration — over a local object
//! directory, then interrupts it the way a process failure or a replacement
//! writer would. A crash is dropping the store without shutting it down; a
//! restart is a new store over the same object directory; a replacement writer
//! is a second store over that directory from its own data directory.
//!
//! Faults come from [`FaultInjectingObjectStore`], armed as an explicit queue,
//! so every interleaving here reproduces exactly.
//!
//! After each step the scenario re-checks [`Invariants`]: the durable revision
//! never decreases, HEAD names a complete state whose objects all exist, no
//! acknowledged sequence number is lost or duplicated, and a tombstone never
//! turns back into a live table.

use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Int64Array, StringArray};
use arrow::record_batch::RecordBatch;

use finelog::store::object_store::{ObjectId, OBJECTS_PREFIX};
use finelog::store::schema::schema_to_arrow;
use finelog::store::table_state::{CommitError, TableRevision};
use finelog::test_support::{
    lost_head_response, unique_dir, FaultAction, FaultGate, ObjectFault, ObjectOp, ObjectPattern,
};

mod support;
use support::{
    assert_metadata_only_bootstrap, cached_data_objects, data_object_upload, live_levels,
    live_objects, localized_data_objects, referenced_objects, register_v1, retargeted_spec,
    run_sql, seq_column, wait_for_cache_fill, worker_schema, write_row, Cluster, Invariants, TABLE,
};

/// A crash during a migration backfill leaves the table recoverable: the
/// restart loads metadata only, the migration resumes from its checkpoints and
/// activates, and the compaction lease the crashed process held can no longer
/// commit — its uploaded output stays unreferenced while the table keeps
/// serving every acknowledged row exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_crash_during_migration_backfill_resumes_and_rejects_the_stale_lease() {
    let cluster = Cluster::new("scenario_migration_crash");
    let mut invariants = Invariants::new(&cluster.remote_dir);

    let (store, faults) = cluster.open();
    register_v1(&store).await;
    for (worker, mem_bytes) in [("w-1", 10), ("w-2", 20), ("w-3", 30)] {
        write_row(&store, worker, mem_bytes).await;
    }
    let before = invariants.check(&store).await;
    assert_eq!(before.seqs, vec![1, 2, 3]);
    assert_eq!(live_objects(&before.state).len(), 3);

    // A compaction lease over the live inputs, plus the output object such a
    // compaction would have uploaded before its commit.
    let inputs = store.query_snapshot(TABLE).unwrap().paths;
    assert_eq!(inputs.len(), 3);
    let lease = store.tables().begin_compaction(TABLE).unwrap();
    let staging = unique_dir("scenario_migration_crash_staging");
    let staged = staging.join("compaction-output.parquet");
    std::fs::write(&staged, b"compaction output that never commits").unwrap();
    let orphan = store
        .tables()
        .controller(TABLE)
        .write_staged_object(OBJECTS_PREFIX, "parquet", &staged)
        .await
        .unwrap();
    let orphan_id = orphan.source.object_id.clone().unwrap();

    // A compatible layout change starts an automatic migration.
    store
        .register_versioned_table(TABLE, retargeted_spec(2))
        .unwrap();
    store.publish_object_catalog(TABLE).await.unwrap();

    // Park the writer while it uploads the second migration output, so the
    // crash lands after one source is checkpointed and before activation.
    let mid_backfill = FaultGate::new();
    let (op, pattern) = data_object_upload();
    faults
        .arm(ObjectFault::new(op, pattern, FaultAction::Park(Arc::clone(&mid_backfill))).after(1));
    let backfilling = {
        let tables = store.tables().clone();
        tokio::spawn(async move { tables.maintain(TABLE, false).await })
    };
    mid_backfill.entered().await;

    // Crash: the process dies mid-upload, so nothing else it owned runs again.
    backfilling.abort();
    let _ = backfilling.await;
    drop(store);

    let crashed = cluster.states().load(TABLE).await.unwrap().unwrap();
    assert_eq!(
        crashed.catalog.active_table_spec_version,
        Some(1),
        "the crash must land before activation"
    );
    let checkpointed = crashed
        .catalog
        .version_segments
        .iter()
        .find(|version| version.table_spec_version == Some(2))
        .expect("at least one migration segment committed before the crash");
    assert_eq!(checkpointed.live_segments.len(), 1);

    // Restart over the same directories.
    let (restarted, restart_faults) = cluster.open();
    restarted.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&restart_faults);
    let recovered = invariants.check(&restarted).await;
    assert_eq!(recovered.seqs, vec![1, 2, 3]);

    // The migration resumes from its checkpoint and activates.
    for _ in 0..4 {
        restarted.maintain_namespace(TABLE, false).await.unwrap();
        invariants.check(&restarted).await;
        if restarted.spec_lifecycle(TABLE).unwrap().active_version() == 2 {
            break;
        }
    }
    let status = restarted.spec_lifecycle(TABLE).unwrap();
    assert_eq!(status.active_version(), 2, "the migration must activate");

    // The lease belongs to the fence the crashed process held, so its commit is
    // refused before it can touch durable state.
    let rejected = restarted
        .tables()
        .controller(TABLE)
        .commit_maintenance(
            &lease,
            || -> Result<(TableRevision, ()), finelog::errors::StatsError> {
                unreachable!("a lease from a dead writer must not run its mutation")
            },
        )
        .await
        .map(|committed| committed.token.revision());
    assert!(
        matches!(rejected, Err(CommitError::Fenced(_))),
        "a stale lease must be fenced, got {rejected:?}"
    );

    let after = invariants.check(&restarted).await;
    assert_eq!(after.seqs, vec![1, 2, 3], "no row was lost or duplicated");
    assert!(
        !referenced_objects(&after.state).contains(&orphan_id),
        "the abandoned compaction output must stay unreferenced"
    );
    assert!(
        cluster
            .objects()
            .read(&ObjectId::parse(&orphan_id).unwrap())
            .await
            .unwrap()
            .is_some(),
        "an unreferenced output is abandoned, not deleted"
    );
    // The table is healthy: it still takes writes under the restarted fence.
    let next = write_row(&restarted, "w-4", 40).await;
    assert_eq!(next, 4);
    let healthy = invariants.check(&restarted).await;
    assert_eq!(healthy.seqs, vec![1, 2, 3, 4]);

    restarted.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(&staging).ok();
    cluster.cleanup();
}

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
    let batch_schema = schema_to_arrow(&worker_schema());
    let batch = RecordBatch::try_new(
        batch_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["w-2"])),
            Arc::new(Int64Array::from(vec![20])),
            Arc::new(Int64Array::from(vec![20])),
        ],
    )
    .unwrap();
    let ipc = finelog::store::ipc::encode_ipc(&batch_schema, &[batch]).unwrap();
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

/// A store that restarts with no local cache and no local catalog bootstraps
/// from the object directory alone. Recovery downloads no data, the first reads
/// scan the object directory directly while the cache fills in the background
/// with only the live objects, and the table compacts cleanly afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cold_restart_scans_remotely_and_warms_only_the_live_objects() {
    let cluster = Cluster::new("scenario_cold_restart");
    let mut invariants = Invariants::new(&cluster.remote_dir);

    let (store, _faults) = cluster.open();
    register_v1(&store).await;
    for (worker, mem_bytes) in [("w-1", 10), ("w-2", 20), ("w-3", 30)] {
        write_row(&store, worker, mem_bytes).await;
    }
    let flushed = invariants.check(&store).await;
    let inputs = live_objects(&flushed.state);
    assert_eq!(inputs.len(), 3);
    // One compaction, so the object directory holds an L1 with index artifacts
    // beside the three L0 objects it replaced.
    store.maintain_namespace(TABLE, true).await.unwrap();
    let last_seq = write_row(&store, "w-4", 40).await;
    let before = invariants.check(&store).await;
    assert_eq!(before.seqs, vec![1, 2, 3, 4]);
    let live = live_objects(&before.state);
    assert_eq!(live.len(), 2, "one compacted L1 and one later L0");
    assert!(
        live.is_disjoint(&inputs),
        "compaction replaced every input it merged"
    );

    // Crash, then lose the local cache and the local catalog entirely.
    drop(store);
    std::fs::remove_dir_all(&cluster.data_dir).unwrap();

    let (restarted, faults) = cluster.open();
    restarted.recover_tables().await.unwrap();
    assert_metadata_only_bootstrap(&faults);

    // A full scan, before any maintenance cycle. The cold cache never blocks
    // the read: the scan runs against the object directory itself and the
    // cache fills behind it.
    faults.clear_calls();
    assert!(cached_data_objects(&cluster.data_dir).is_empty());
    let scanned = seq_column(&run_sql(&restarted, &format!("SELECT seq FROM \"{TABLE}\"")).await);
    assert_eq!(scanned, vec![1, 2, 3, 4]);
    assert!(
        localized_data_objects(&faults).is_empty(),
        "a cold scan must not block on object downloads"
    );
    let warmed = wait_for_cache_fill(&cluster.data_dir, &live).await;
    assert_eq!(
        warmed, live,
        "the background fill warms the live objects and nothing the state retired"
    );

    // A forwarding-style read of one sequence window, now served from the
    // warmed cache.
    let forwarded = run_sql(
        &restarted,
        &format!("SELECT * FROM \"{TABLE}\" WHERE seq > 3 AND seq <= {last_seq} ORDER BY seq"),
    )
    .await;
    assert_eq!(seq_column(&forwarded), vec![4]);

    // A FetchLogs-style read: a key prefix, a cursor, and a limit.
    let fetched = run_sql(
        &restarted,
        &format!(
            "SELECT seq FROM \"{TABLE}\" WHERE seq > 1 AND prefix(worker_id, 'w-') ORDER BY seq LIMIT 2"
        ),
    )
    .await;
    assert_eq!(seq_column(&fetched), vec![2, 3]);
    assert_eq!(
        cached_data_objects(&cluster.data_dir),
        live,
        "warm reads never materialize a retired object"
    );

    // Compaction after a cold restart commits like any other.
    let recovered = invariants.check(&restarted).await;
    restarted.maintain_namespace(TABLE, true).await.unwrap();
    let compacted = invariants.check(&restarted).await;
    assert_eq!(compacted.seqs, vec![1, 2, 3, 4]);
    assert!(
        compacted.state.catalog_generation > recovered.state.catalog_generation,
        "the post-restart compaction publishes a new revision"
    );
    assert!(
        live_levels(&compacted.state).iter().all(|level| *level > 0),
        "the L0 written before the crash is promoted, not left behind: {:?}",
        live_levels(&compacted.state)
    );

    restarted.shutdown(Duration::from_secs(1)).await;
    cluster.cleanup();
}
