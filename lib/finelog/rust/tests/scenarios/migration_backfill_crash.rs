// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use finelog::store::compaction::config::CompactionConfig;
use finelog::store::object_store::{ObjectId, OBJECTS_PREFIX};
use finelog::store::table_state::{CommitError, TableRevision};
use finelog::test_support::{unique_dir, FaultAction, FaultGate, ObjectFault};

use crate::support::{
    assert_metadata_only_bootstrap, data_object_upload, live_objects, referenced_objects,
    register_v1, retargeted_spec, write_row, Cluster, Invariants, TABLE,
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
    // One source per backfill batch, so the crash can land between the first
    // source's checkpoint commit and the second source's output upload.
    store.tables().set_compaction_config(CompactionConfig {
        migration_batch_sources: 1,
        ..CompactionConfig::default()
    });
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
        // Each maintenance tick runs one single-source batch: the first
        // checkpoints, the second parks mid-upload.
        tokio::spawn(async move {
            loop {
                tables.maintain(TABLE, false).await.unwrap();
            }
        })
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
