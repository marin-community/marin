// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::support::{
    assert_metadata_only_bootstrap, cached_data_objects, live_levels, live_objects,
    localized_data_objects, register_v1, run_sql, seq_column, wait_for_cache_fill, write_row,
    Cluster, Invariants, TABLE,
};

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
