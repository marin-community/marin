// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the cross-cluster forwarder. Each drives a real source finelog
//! and the hub it forwards to, both served over loopback sockets.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

use crate::proto::finelog::logging::{FetchLogsRequest, LogEntry, MatchScope, PushLogsRequest};
use crate::proto::finelog::stats::ColumnType;
use crate::server::auth::{AuthIdentity, AuthPolicy};
use crate::server::test_support::{
    client, disk_store, serve, serve_rejecting, stats_client, TestTransport, PRIV_A,
    PRIV_UNTRUSTED, PUB_A,
};
use crate::store::policy::StoragePolicy;
use crate::store::schema::{Column, Schema};
use crate::store::store::LOG_NAMESPACE_NAME;
use crate::store::Store;

use super::*;

use crate::proto::finelog::logging::LogServiceClient;

const SOURCE_CLUSTER: &str = "cw-test";

fn jwt_policy(cluster: &str) -> AuthPolicy {
    AuthPolicy::parse(
        &serde_json::json!([
            {"type": "jwt", "keys": [{"cluster": cluster, "public_keys": [PUB_A]}]}
        ])
        .to_string(),
    )
    .unwrap()
}

/// A hub that verifies a bearer against the sending cluster's public key, and lets a
/// bearerless local client (this test, reading the result back) fall through to the
/// loopback rule.
///
/// Jwt sits first, inverting the cidr-first order deployed hubs use. Both ends are on
/// loopback here, so a cidr-first hub would admit every push on the network rule and
/// never reach the bearer that names the sending cluster.
fn hub_policy(cluster: &str) -> AuthPolicy {
    AuthPolicy::parse(
        &serde_json::json!([
            {"type": "jwt", "keys": [{"cluster": cluster, "public_keys": [PUB_A]}]},
            {"type": "cidr", "cidrs": ["127.0.0.0/8", "::1/128"]}
        ])
        .to_string(),
    )
    .unwrap()
}

/// A source store and the hub it forwards to, each served over a real socket: the hub
/// under [`hub_policy`], the source open to loopback.
struct Fixture {
    source: Arc<Store>,
    source_client: LogServiceClient<TestTransport>,
    /// The hub's store, kept for direct assertions. `None` when the hub is a stub that
    /// keeps no state (the rejecting hub).
    target_store: Option<Arc<Store>>,
    target_addr: SocketAddr,
    target_url: String,
    target_requests: Arc<AtomicUsize>,
}

impl Fixture {
    /// A hub that trusts [`SOURCE_CLUSTER`]'s public key, and a source that writes under
    /// it. Each namespace's watermark is unset: a forwarder started now seeds at the tip.
    /// Call [`Self::forward_from_start`] to drain what is already written.
    async fn new(tag: &str) -> Self {
        let target = disk_store(&format!("{tag}_target"));
        let (target_addr, target_requests) =
            serve(Arc::clone(&target), hub_policy(SOURCE_CLUSTER)).await;
        Self::with_hub(tag, Some(target), target_addr, target_requests).await
    }

    /// As [`Self::new`], but the hub refuses every request with `invalid_argument` and
    /// keeps no store, so only the source and the request count are observable.
    async fn with_rejecting_hub(tag: &str) -> Self {
        let (target_addr, target_requests) = serve_rejecting().await;
        Self::with_hub(tag, None, target_addr, target_requests).await
    }

    async fn with_hub(
        tag: &str,
        target_store: Option<Arc<Store>>,
        target_addr: SocketAddr,
        target_requests: Arc<AtomicUsize>,
    ) -> Self {
        let source = disk_store(&format!("{tag}_source"));
        let (source_addr, _) = serve(Arc::clone(&source), AuthPolicy::allow_localhost()).await;
        Self {
            source,
            source_client: client(source_addr),
            target_store,
            target_addr,
            target_url: format!("http://{target_addr}"),
            target_requests,
        }
    }

    /// Point `namespace`'s watermark below every row, so a forward drains it whole.
    fn forward_from_start(&self, namespace: &str) {
        self.source
            .set_forward_cursor(&self.target_url, namespace, 0)
            .unwrap();
    }

    /// A forwarder from this source to this hub, signing with `private_pem`.
    fn forwarder(&self, private_pem: &str) -> Forwarder<TestTransport> {
        let config = ForwardingConfig {
            target: self.target_url.clone(),
            cluster: SOURCE_CLUSTER.to_string(),
        };
        let minter = TokenMinter::new(private_pem, config.cluster.clone()).unwrap();
        Forwarder::with_client(
            Arc::clone(&self.source),
            config,
            minter,
            stats_client(self.target_addr),
        )
    }

    fn target_store(&self) -> &Arc<Store> {
        self.target_store.as_ref().expect("this hub keeps a store")
    }

    /// The last seq the source has made durable in `namespace`.
    fn tip(&self, namespace: &str) -> i64 {
        self.source.namespace_persisted_seq(namespace).unwrap()
    }

    fn cursor(&self, namespace: &str) -> Option<i64> {
        self.source
            .forward_cursor(&self.target_url, namespace)
            .unwrap()
    }

    fn requests(&self) -> usize {
        self.target_requests.load(Ordering::SeqCst)
    }

    /// Forward until `namespace`'s watermark settles at the source's current tip, then
    /// stop.
    async fn drain(&self, private_pem: &str, namespace: &str) {
        forward_until(
            self.forwarder(private_pem),
            &self.source,
            &self.target_url,
            namespace,
            self.tip(namespace),
        )
        .await;
    }

    /// Every log row the hub holds, as `(key, data)`.
    async fn hub_log_rows(&self) -> Vec<(String, String)> {
        read_all(&client(self.target_addr)).await
    }
}

/// Write `lines` under `key` into `store` through its own RPC surface, which returns
/// only once the rows are durable and therefore visible to a scan.
async fn push(client: &LogServiceClient<TestTransport>, key: &str, lines: &[&str]) {
    let entries = lines
        .iter()
        .map(|line| LogEntry::default().with_source("stdout").with_data(*line))
        .collect();
    let request = PushLogsRequest {
        entries,
        ..Default::default()
    }
    .with_key(key);
    client.push_logs(request).await.unwrap();
}

/// Every log row the server behind `client` holds, newest last, as `(key, data)` — read
/// back over the wire so the assertion sees exactly what a log reader would.
async fn read_all(client: &LogServiceClient<TestTransport>) -> Vec<(String, String)> {
    let response = client
        .fetch_logs(
            FetchLogsRequest {
                ..Default::default()
            }
            .with_source("/")
            .with_match_scope(MatchScope::MATCH_SCOPE_PREFIX)
            .with_max_lines(1000),
        )
        .await
        .unwrap();
    response
        .into_view()
        .entries
        .iter()
        .map(|e| {
            (
                e.key.unwrap_or("").to_string(),
                e.data.unwrap_or("").to_string(),
            )
        })
        .collect()
}

/// Poll `condition` until it holds, or fail after five seconds with `describe()`, so a
/// wedged forwarder fails the test rather than hanging it.
async fn poll_until(mut condition: impl FnMut() -> bool, describe: impl Fn() -> String) {
    for _ in 0..200 {
        if condition() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("{}", describe());
}

/// Wait for `store`'s watermark for `(target, namespace)` to reach `expected`. Reads
/// local state only, so a test can tell "the forwarder is done" without an RPC that
/// would perturb the target's request count.
async fn wait_for_cursor(store: &Store, target: &str, namespace: &str, expected: i64) {
    poll_until(
        || store.forward_cursor(target, namespace).unwrap() == Some(expected),
        || {
            format!(
                "watermark for {namespace:?} never reached {expected} (stuck at {:?})",
                store.forward_cursor(target, namespace).unwrap()
            )
        },
    )
    .await;
}

/// Poll `counter` until the hub has served at least `expected` requests. Lets a test
/// that asserts on the *absence* of an effect first wait for the attempt that would have
/// produced it.
async fn wait_for_requests(counter: &AtomicUsize, expected: usize) {
    poll_until(
        || counter.load(Ordering::SeqCst) >= expected,
        || {
            format!(
                "hub never served {expected} requests (saw {})",
                counter.load(Ordering::SeqCst)
            )
        },
    )
    .await;
}

/// A forwarder running on its own task, stopped and joined by [`Self::finish`].
struct RunningForwarder {
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl RunningForwarder {
    fn start(forwarder: Forwarder<TestTransport>) -> Self {
        let (stop, stop_rx) = watch::channel(false);
        Self {
            stop,
            task: spawn(forwarder, stop_rx),
        }
    }

    /// Latch the stop signal and join. Bounded, so a forwarder wedged in a backoff fails
    /// the test rather than hanging it.
    async fn finish(self) {
        self.stop.send(true).unwrap();
        tokio::time::timeout(Duration::from_secs(5), self.task)
            .await
            .expect("forwarder did not stop within 5s")
            .expect("forwarder task panicked");
    }
}

/// Run `forwarder` until `store`'s watermark for `namespace` reaches `expected`, then
/// stop it.
async fn forward_until(
    forwarder: Forwarder<TestTransport>,
    store: &Store,
    target: &str,
    namespace: &str,
    expected: i64,
) {
    let running = RunningForwarder::start(forwarder);
    wait_for_cursor(store, target, namespace, expected).await;
    running.finish().await;
}

// -------------------------------------------------------------------------------------
// Unit tests: the credential, config, and pure helpers.

#[test]
fn minted_bearer_is_accepted_by_a_hub_trusting_the_matching_public_key() {
    // The two halves of the trust config -- this server's private key and the public key
    // an operator pastes into the hub's `jwt` auth layer -- must agree.
    let minter = TokenMinter::new(PRIV_A, SOURCE_CLUSTER.to_string()).unwrap();
    let bearer = minter.bearer().unwrap();
    assert!(jwt_policy(SOURCE_CLUSTER)
        .admits(Some(&bearer), None)
        .is_some());
    // A hub that trusts some other cluster's key rejects it.
    let minter = TokenMinter::new(PRIV_UNTRUSTED, SOURCE_CLUSTER.to_string()).unwrap();
    assert!(jwt_policy(SOURCE_CLUSTER)
        .admits(Some(&minter.bearer().unwrap()), None)
        .is_none());
}

#[test]
fn the_key_names_the_forwarding_cluster_not_the_bearer() {
    // The hub binds an admitted identity to the key that verified the bearer. A sender
    // that mints under someone else's name still lands under the cluster the hub
    // configured for its key.
    let hub = jwt_policy("cw-rno2a");
    let admitted = Some(AuthIdentity::Jwt {
        cluster: "cw-rno2a".to_string(),
    });

    let honest = TokenMinter::new(PRIV_A, "cw-rno2a".to_string()).unwrap();
    assert_eq!(hub.admits(Some(&honest.bearer().unwrap()), None), admitted);

    // Same key, a bearer whose `iss`/`sub` claim another cluster entirely.
    let liar = TokenMinter::new(PRIV_A, "us-central2".to_string()).unwrap();
    assert_ne!(liar.bearer().unwrap(), honest.bearer().unwrap());
    assert_eq!(hub.admits(Some(&liar.bearer().unwrap()), None), admitted);
}

#[test]
fn a_cached_bearer_is_reused_until_it_nears_expiry() {
    // Asserting that two successive mints are equal would prove nothing: EdDSA is
    // deterministic and `iat`/`exp` are second-granular. Drive the cache directly.
    let minter = TokenMinter::new(PRIV_A, SOURCE_CLUSTER.to_string()).unwrap();

    let usable = Instant::now() + Duration::from_secs(600);
    *minter.cached.lock().unwrap() = Some(("cached-bearer".to_string(), usable));
    assert_eq!(minter.bearer().unwrap(), "cached-bearer");

    // Once the entry is inside the refresh margin, it is replaced rather than handed to
    // the hub, which would reject a token expiring mid-flight.
    *minter.cached.lock().unwrap() = Some(("stale-bearer".to_string(), Instant::now()));
    let fresh = minter.bearer().unwrap();
    assert_ne!(fresh, "stale-bearer");
    assert_eq!(
        jwt_policy(SOURCE_CLUSTER).admits(Some(&fresh), None),
        Some(AuthIdentity::Jwt {
            cluster: SOURCE_CLUSTER.to_string()
        }),
        "the re-minted bearer must still verify at the hub"
    );
}

#[test]
fn forwarding_config_rejects_a_target_that_would_expose_the_bearer() {
    assert!(ForwardingConfig::parse(r#"{"target":"http://hub","cluster":"a"}"#).is_err());
    assert!(ForwardingConfig::parse(r#"{"target":"https://hub","cluster":"a"}"#).is_ok());
}

#[test]
fn forwarding_config_rejects_an_empty_cluster() {
    assert!(ForwardingConfig::parse(r#"{"target":"https://hub","cluster":""}"#).is_err());
    assert!(ForwardingConfig::parse(r#"{"target":"https://hub","cluster":"a"}"#).is_ok());
}

#[test]
fn resume_after_eviction_reports_only_a_real_gap() {
    // Cursor at or above the oldest local row: nothing was lost.
    assert_eq!(resume_after_eviction(10, Some(11)), None);
    assert_eq!(resume_after_eviction(10, Some(5)), None);
    // Cursor below it: rows 11..=40 are archive-only, so resume just before 41.
    assert_eq!(resume_after_eviction(10, Some(41)), Some(40));
    // No local segments at all: nothing to be behind.
    assert_eq!(resume_after_eviction(10, None), None);
}

#[test]
fn chunk_by_bytes_splits_and_pairs_each_chunk_with_its_last_seq() {
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "data",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
    )
    .unwrap();
    let seqs = Int64Array::from(vec![10, 20, 30]);

    // A budget that fits the whole batch ships it in one chunk, cursor = last seq.
    let chunks = chunk_by_bytes(&batch, &seqs, 1 << 20).unwrap();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].1, 30);

    // A minimal budget forces one row per chunk; each chunk's cursor is its own row.
    let chunks = chunk_by_bytes(&batch, &seqs, 1).unwrap();
    let last_seqs: Vec<i64> = chunks.iter().map(|(_, seq)| *seq).collect();
    assert_eq!(last_seqs, vec![10, 20, 30]);
}

// -------------------------------------------------------------------------------------
// Integration tests: the `log` namespace end to end.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drains_a_startup_backlog_of_many_keys_in_one_request() {
    // A batch spanning many keys costs ONE WriteRows: forwarding throughput is
    // independent of how many distinct log keys are in flight, so a job fanned out over
    // 140 workers ships as fast as one that is not. The count is taken at the hub's HTTP
    // boundary, which is where the contract lives.
    let fx = Fixture::new("bulk").await;
    for key in ["/user/job/task-a", "/user/job/task-b", "/system/worker/1"] {
        push(&fx.source_client, key, &["first", "second"]).await;
    }
    fx.forward_from_start(LOG_NAMESPACE_NAME);
    let requests_before = fx.requests();

    fx.drain(PRIV_A, LOG_NAMESPACE_NAME).await;

    assert_eq!(
        fx.requests() - requests_before,
        1,
        "six rows across three keys must ship in a single request"
    );

    let mut forwarded = fx.hub_log_rows().await;
    forwarded.sort();
    assert_eq!(
        forwarded,
        vec![
            ("/system/worker/1".to_string(), "first".to_string()),
            ("/system/worker/1".to_string(), "second".to_string()),
            ("/user/job/task-a".to_string(), "first".to_string()),
            ("/user/job/task-a".to_string(), "second".to_string()),
            ("/user/job/task-b".to_string(), "first".to_string()),
            ("/user/job/task-b".to_string(), "second".to_string()),
        ],
        "every row lands under the key it was written with"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn forwarded_rows_carry_the_origin_cluster_of_the_store_that_sent_them() {
    // The hub selects logs by origin, and the forwarder stamps that origin into the
    // `cluster` column on the way out.
    let fx = Fixture::new("stamp").await;
    push(&fx.source_client, "/user/job/t", &["hello"]).await;
    fx.forward_from_start(LOG_NAMESPACE_NAME);
    fx.drain(PRIV_A, LOG_NAMESPACE_NAME).await;

    let entries = client(fx.target_addr)
        .fetch_logs(
            FetchLogsRequest {
                ..Default::default()
            }
            .with_source("/")
            .with_match_scope(MatchScope::MATCH_SCOPE_PREFIX)
            .with_cluster(SOURCE_CLUSTER),
        )
        .await
        .unwrap()
        .into_view()
        .entries
        .len();
    assert_eq!(
        entries, 1,
        "the row is readable only if `cluster` was stamped"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bearer_the_hub_does_not_trust_forwards_nothing_and_loses_nothing() {
    // The hub rejects the push, so the watermark must not advance: the rows are still
    // owed, and the local store still serves them.
    let fx = Fixture::new("reject").await;
    push(&fx.source_client, "/user/job/t", &["hello"]).await;
    fx.forward_from_start(LOG_NAMESPACE_NAME);

    let running = RunningForwarder::start(fx.forwarder(PRIV_UNTRUSTED));
    // Wait for the push to REACH the hub. Stopping on a timer instead would let a
    // forwarder that never pushed at all satisfy both assertions below.
    wait_for_requests(&fx.target_requests, 1).await;
    running.finish().await;

    assert_eq!(
        fx.cursor(LOG_NAMESPACE_NAME),
        Some(0),
        "a refused push leaves the watermark where it was, so the rows are retried"
    );
    assert!(fx.hub_log_rows().await.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_the_hub_calls_malformed_is_skipped_rather_than_retried_forever() {
    // invalid_argument means the hub refuses these bytes and always will. Retrying would
    // strand every row behind them, so the forwarder counts the batch as skipped and
    // moves its cursor past it. The rows are still in the local store.
    let fx = Fixture::with_rejecting_hub("poison").await;
    push(&fx.source_client, "/user/job/t", &["hello"]).await;
    fx.forward_from_start(LOG_NAMESPACE_NAME);
    let tip = fx.tip(LOG_NAMESPACE_NAME);
    fx.drain(PRIV_A, LOG_NAMESPACE_NAME).await;

    assert_eq!(
        fx.cursor(LOG_NAMESPACE_NAME),
        Some(tip),
        "the cursor must advance past a batch the hub will never accept"
    );
    assert_eq!(
        fx.requests(),
        1,
        "a permanently rejected batch is sent once, not retried"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seeding_at_the_tip_ships_new_rows_and_never_backfills() {
    let fx = Fixture::new("seed").await;
    push(&fx.source_client, "/user/job/t", &["before"]).await;

    // No watermark for the log namespace: seed at the tip, so "before" is never shipped.
    let running = RunningForwarder::start(fx.forwarder(PRIV_A));
    wait_for_cursor(
        &fx.source,
        &fx.target_url,
        LOG_NAMESPACE_NAME,
        fx.tip(LOG_NAMESPACE_NAME),
    )
    .await;

    push(&fx.source_client, "/user/job/t", &["after"]).await;
    wait_for_cursor(
        &fx.source,
        &fx.target_url,
        LOG_NAMESPACE_NAME,
        fx.tip(LOG_NAMESPACE_NAME),
    )
    .await;
    running.finish().await;

    assert_eq!(
        fx.hub_log_rows().await,
        vec![("/user/job/t".to_string(), "after".to_string())]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rows_that_already_carry_an_origin_cluster_are_never_re_forwarded() {
    // A row that arrived by forwarding already names an origin. Relaying it onward would
    // loop, so only rows this store's own writers produced are eligible.
    let fx = Fixture::new("loop").await;

    fx.source_client
        .push_logs(
            PushLogsRequest {
                entries: vec![LogEntry::default().with_data("relayed")],
                ..Default::default()
            }
            .with_key("/user/job/t")
            .with_cluster("some-other-cluster"),
        )
        .await
        .unwrap();
    push(&fx.source_client, "/user/job/t", &["local"]).await;
    fx.forward_from_start(LOG_NAMESPACE_NAME);
    fx.drain(PRIV_A, LOG_NAMESPACE_NAME).await;

    assert_eq!(
        fx.hub_log_rows().await,
        vec![("/user/job/t".to_string(), "local".to_string())]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_watermark_ahead_of_the_store_reseeds_at_the_tip() {
    // The volume was recreated: the stored cursor names a seq space that no longer
    // exists. Forwarding from it would mean forwarding nothing, forever.
    let fx = Fixture::new("ahead").await;
    push(&fx.source_client, "/user/job/t", &["one"]).await;
    fx.source
        .set_forward_cursor(&fx.target_url, LOG_NAMESPACE_NAME, 10_000)
        .unwrap();

    fx.drain(PRIV_A, LOG_NAMESPACE_NAME).await;

    assert!(fx.hub_log_rows().await.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_backlog_beyond_the_lag_cap_is_skipped_rather_than_drained() {
    // The store keeps every row; the hub copy is best effort. A forwarder too far behind
    // abandons the oldest part of the backlog and keeps the freshest `max_lag_seqs`.
    let fx = Fixture::new("cap").await;
    push(
        &fx.source_client,
        "/user/job/t",
        &["one", "two", "three", "four"],
    )
    .await;
    fx.forward_from_start(LOG_NAMESPACE_NAME);

    let mut forwarder = fx.forwarder(PRIV_A);
    forwarder.max_lag_seqs = 2;
    forward_until(
        forwarder,
        &fx.source,
        &fx.target_url,
        LOG_NAMESPACE_NAME,
        fx.tip(LOG_NAMESPACE_NAME),
    )
    .await;

    assert_eq!(
        fx.hub_log_rows().await,
        vec![
            ("/user/job/t".to_string(), "three".to_string()),
            ("/user/job/t".to_string(), "four".to_string()),
        ],
        "the two oldest rows are dropped; the freshest two still ship"
    );
}

// -------------------------------------------------------------------------------------
// Integration test: a non-log table forwards generically.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_non_log_table_is_registered_on_the_hub_and_forwarded() {
    // Forwarding is table-generic: a table the hub has never seen is created there with
    // RegisterTable, then its rows arrive through the same WriteRows path as logs. The
    // table has no origin column, so nothing is stamped -- it forwards verbatim.
    let fx = Fixture::new("generic").await;

    let schema = Schema::new(
        vec![Column::new("id", ColumnType::COLUMN_TYPE_STRING, false)],
        "id",
    );
    fx.source
        .register_table("events", schema, StoragePolicy::default())
        .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["e1", "e2"]))],
    )
    .unwrap();
    let ipc = encode_ipc(&batch.schema(), &[batch]).unwrap();
    let (_, last_seq) = fx.source.write_rows("events", &ipc).unwrap();
    // Seal the rows so the forwarder's durable watermark can reach them.
    fx.source
        .await_persisted("events", last_seq, Duration::from_secs(5))
        .await
        .unwrap();

    fx.forward_from_start("events");
    fx.drain(PRIV_A, "events").await;

    let stats = fx.target_store().list_namespaces_with_stats().unwrap();
    let events = stats
        .iter()
        .find(|(name, _, _, _)| name == "events")
        .expect("the hub created the events namespace from RegisterTable");
    assert_eq!(events.2.row_count, 2, "both rows landed on the hub");
}
