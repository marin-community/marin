// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Native cross-cluster log forwarding: tail this store's `log` namespace and
//! replicate its locally-written rows into a shared hub finelog.
//!
//! A federated cluster keeps its own finelog and also relays every log line to one
//! global store, so `iris --cluster=<hub> job logs` reads a job that ran anywhere.
//! The relay lives here, in the server that already owns the segment layout and the
//! durability watermark, rather than in a client tailing it over RPC: it ships a
//! whole multi-key batch in one [`PushLogsBulk`] round trip, so throughput no longer
//! scales with the number of distinct log keys in flight.
//!
//! # Credential
//!
//! The forwarder authenticates with a short-lived `aud="finelog"` EdDSA bearer minted
//! from *this server's own* Ed25519 key, which the hub pins in its `jwt` auth layer.
//! That key is distinct from the iris controller's signing key, which also mints
//! control-plane and federation tokens: a compromise of the log-ingest path therefore
//! grants log-plane authority only. The hub binds each pushed row's origin `cluster`
//! to the key that carried it, so this server can write logs under its own cluster
//! and no other.
//!
//! # Best effort by design
//!
//! The local store is the system of record — every row stays queryable here whether
//! or not the hub ever receives it. The forwarder therefore never fails the server
//! and never grows without bound:
//!
//! - It seeds at the current tip, so enabling it ships new logs rather than
//!   backfilling a retention window.
//! - It materializes at most [`FORWARD_BATCH_ROWS`] rows, and packs them into requests
//!   of at most [`FORWARD_BATCH_BYTES`]. A single row larger than that budget ships
//!   alone, in a request that exceeds it — splitting a log line is not an option.
//! - It advances its cursor past a batch only once that batch can never be sent again:
//!   the hub acked it, or the forwarder gave it up. A crash mid-batch re-forwards it
//!   (at-least-once; duplicate lines are bounded to the failure boundary and tolerable
//!   for logs).
//! - When it falls further behind than [`MAX_FORWARD_LAG_SEQS`], when eviction has
//!   already archived the segments its cursor points at, or when the hub refuses a
//!   batch outright, it skips forward and logs what it dropped. A counted, visible gap
//!   beats an unbounded queue.
//!
//! A push failure backs off and retries; nothing here can take the store down.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use connectrpc::client::{CallOptions, ClientConfig, ServiceTransport};
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::errors::StatsError;
use crate::proto::finelog::logging::{LogEntry, LogServiceClient, PushLogsBulkRequest, Timestamp};
use crate::query::provider::NamespaceProvider;
use crate::query::{fetch_log_rows, make_ctx};
use crate::server::auth::FINELOG_AUDIENCE;
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::log_read::LogRow;
use crate::store::Store;

/// Rows read from the local store per batch. Bounds the forwarder's working set:
/// it never holds more than this many rows in memory at once.
const FORWARD_BATCH_ROWS: i32 = 5_000;

/// Encoded-entry bytes per outbound request. A batch whose lines are large is
/// split into several requests rather than built into one huge message.
const FORWARD_BATCH_BYTES: usize = 8 << 20;

/// How far the cursor may trail the durability watermark before the forwarder
/// gives up on the backlog and jumps to `persisted - MAX_FORWARD_LAG_SEQS`,
/// keeping the freshest window rather than discarding it all.
///
/// Measured in `seq` positions, not rows: `seq` is the `log` namespace's own dense
/// counter, so the two coincide only when no earlier gap exists.
const MAX_FORWARD_LAG_SEQS: i64 = 2_000_000;

/// Delegation-bearer lifetime, and how early to re-mint. Short-lived because the
/// hub checks only signature + audience + expiry — it cannot reach a revocation
/// list, so a leaked bearer's blast radius is bounded by its TTL.
const TOKEN_TTL: Duration = Duration::from_secs(3600);
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// Deadline for one outbound push, and the retry backoff bounds.
const PUSH_TIMEOUT: Duration = Duration::from_secs(60);
const BACKOFF_MIN: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Emit a progress line at most this often, so a healthy forwarder is observable
/// without flooding the log it is forwarding.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(300);

/// Where this store forwards, and as whom. Parsed from the `FINELOG_FORWARDING`
/// JSON; the Ed25519 private key arrives separately (`FINELOG_SIGNING_KEY`) so it
/// never rides in an inline env value.
#[derive(Debug, Clone, Deserialize)]
pub struct ForwardingConfig {
    /// The hub finelog's base URL. `https://` only: the bearer is a credential.
    pub target: String,
    /// This store's cluster. Stamped on every forwarded row as its origin, and
    /// carried as the bearer's `iss`/`sub`.
    pub cluster: String,
}

impl ForwardingConfig {
    /// Parse the `FINELOG_FORWARDING` JSON, rejecting a config that could only fail
    /// later: an empty cluster has no identity to forward as, and a non-TLS target
    /// would put the bearer on the wire in the clear.
    pub fn parse(json: &str) -> Result<Self, String> {
        let config: ForwardingConfig =
            serde_json::from_str(json).map_err(|e| format!("invalid forwarding JSON: {e}"))?;
        if config.cluster.is_empty() {
            return Err("forwarding config has an empty cluster".to_string());
        }
        if !config.target.starts_with("https://") {
            return Err(format!(
                "forwarding target {:?} is not https:// (the delegation bearer must not travel in the clear)",
                config.target
            ));
        }
        Ok(config)
    }
}

/// The bearer claims. `iss`/`sub`/`aud`/`exp` are exactly the set the hub's
/// verifier requires; `iat` records when the token was minted.
#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

/// Mints and caches this server's `aud="finelog"` delegation bearer, re-minting
/// once the live one is within [`TOKEN_REFRESH_MARGIN`] of expiry.
pub struct TokenMinter {
    key: EncodingKey,
    cluster: String,
    /// The live bearer and the instant it stops being usable.
    cached: std::sync::Mutex<Option<(String, Instant)>>,
}

impl TokenMinter {
    /// Load the Ed25519 private key (PKCS#8 PEM). Fails on anything else, so a
    /// misconfigured secret stops the server at startup rather than at the first push.
    pub fn new(private_pem: &str, cluster: String) -> Result<Self, String> {
        let key = EncodingKey::from_ed_pem(private_pem.as_bytes())
            .map_err(|e| format!("signing key is not a valid Ed25519 private PEM: {e}"))?;
        Ok(Self {
            key,
            cluster,
            cached: std::sync::Mutex::new(None),
        })
    }

    /// The current bearer, minting a fresh one when none is cached or the cached
    /// one is close enough to expiry that the hub might reject it in flight.
    fn bearer(&self) -> Result<String, String> {
        let mut cached = self.cached.lock().unwrap();
        let now = Instant::now();
        if let Some((token, usable_until)) = cached.as_ref() {
            if now < *usable_until {
                return Ok(token.clone());
            }
        }
        let issued_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("system clock is before the unix epoch: {e}"))?
            .as_secs();
        let claims = Claims {
            iss: &self.cluster,
            sub: &self.cluster,
            aud: FINELOG_AUDIENCE,
            iat: issued_at,
            exp: issued_at + TOKEN_TTL.as_secs(),
        };
        let token = jsonwebtoken::encode(&Header::new(Algorithm::EdDSA), &claims, &self.key)
            .map_err(|e| format!("minting the finelog delegation bearer failed: {e}"))?;
        *cached = Some((token.clone(), now + TOKEN_TTL - TOKEN_REFRESH_MARGIN));
        Ok(token)
    }
}

type ClientBody = connectrpc::client::ClientBody;
type HttpsConnector =
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;

/// The forwarder's production transport: pooled HTTPS over hyper + rustls.
pub type HttpsTransport = ServiceTransport<HyperClient<HttpsConnector, ClientBody>>;

/// An HTTPS Connect client for the hub's `LogService`.
///
/// The trust anchors are compiled in (`webpki-roots`), so the container needs no CA
/// bundle, and the rustls crypto provider is passed explicitly rather than installed
/// process-wide — the store's other TLS users keep their own.
fn build_client(target: &str) -> Result<LogServiceClient<HttpsTransport>, String> {
    let uri: http::Uri = target
        .parse()
        .map_err(|e| format!("forwarding target {target:?} is not a URI: {e}"))?;

    let roots = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let tls = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("building the forwarder's TLS config failed: {e}"))?
        .with_root_certificates(roots)
        .with_no_client_auth();

    let mut http = hyper_util::client::legacy::connect::HttpConnector::new();
    http.enforce_http(false);
    http.set_nodelay(true);
    http.set_connect_timeout(Some(Duration::from_secs(10)));

    let https = hyper_rustls::HttpsConnectorBuilder::new()
        .with_tls_config(tls)
        .https_only()
        .enable_all_versions()
        .wrap_connector(http);

    let transport = ServiceTransport::new(HyperClient::builder(TokioExecutor::new()).build(https));
    Ok(LogServiceClient::new(transport, client_config(uri)))
}

fn client_config(uri: http::Uri) -> ClientConfig {
    ClientConfig::new(uri)
        .proto()
        .with_default_max_message_size(MAX_MESSAGE_BYTES)
}

/// Everything the forward loop needs, resolved once at startup.
///
/// Generic over the Connect transport so the same loop runs against the production
/// HTTPS client and, in tests, against a plaintext one pointed at an in-process hub.
pub struct Forwarder<T = HttpsTransport> {
    store: Arc<Store>,
    client: LogServiceClient<T>,
    minter: TokenMinter,
    config: ForwardingConfig,
    /// How far behind the durability mark the cursor may fall before the forwarder
    /// abandons the backlog and jumps to its freshest window. [`MAX_FORWARD_LAG_SEQS`]
    /// unless overridden.
    max_lag_seqs: i64,
}

impl Forwarder<HttpsTransport> {
    pub fn new(
        store: Arc<Store>,
        config: ForwardingConfig,
        signing_key_pem: &str,
    ) -> Result<Self, String> {
        let minter = TokenMinter::new(signing_key_pem, config.cluster.clone())?;
        let client = build_client(&config.target)?;
        Ok(Self::with_client(store, config, minter, client))
    }
}

impl<T> Forwarder<T>
where
    T: connectrpc::client::ClientTransport,
    <T::ResponseBody as http_body::Body>::Error: std::fmt::Display,
{
    fn with_client(
        store: Arc<Store>,
        config: ForwardingConfig,
        minter: TokenMinter,
        client: LogServiceClient<T>,
    ) -> Self {
        Self {
            store,
            client,
            minter,
            config,
            max_lag_seqs: MAX_FORWARD_LAG_SEQS,
        }
    }

    /// Run until `stop` latches. Errors are logged and retried; this never returns
    /// an error, because a store whose relay is broken must keep serving.
    pub async fn run(&self, mut stop: watch::Receiver<bool>) {
        let mut persisted_rx = match self.store.log_persisted_seq() {
            Ok(rx) => rx,
            Err(e) => {
                tracing::error!(error = %e, "finelog forwarder: no log namespace; not forwarding");
                return;
            }
        };

        let mut cursor = match self.seed(*persisted_rx.borrow()) {
            Ok(cursor) => cursor,
            Err(e) => {
                tracing::error!(error = %e, "finelog forwarder: cannot read its watermark; not forwarding");
                return;
            }
        };
        tracing::info!(
            target = %self.config.target,
            cluster = %self.config.cluster,
            cursor,
            "finelog forwarder: started"
        );

        let mut progress = Progress::new();
        loop {
            if *stop.borrow() {
                break;
            }
            // Read the CURRENT watermark before ever awaiting `changed()`: a fresh
            // receiver marks the present value as seen, so a restart with a backlog
            // would otherwise idle until the next flush.
            let persisted = *persisted_rx.borrow_and_update();
            cursor = self
                .drain_to(cursor, persisted, &mut progress, &mut stop)
                .await;
            progress.report(cursor, persisted);

            tokio::select! {
                _ = stop.changed() => break,
                changed = persisted_rx.changed() => {
                    if changed.is_err() {
                        break;  // the store is gone
                    }
                }
            }
        }
        tracing::info!(cursor, "finelog forwarder: stopped");
    }

    /// The cursor to start from: the stored watermark when it still names this
    /// target and points inside the store's seq space, otherwise the current tip.
    ///
    /// Reseeding rather than replaying is deliberate. A watermark for another target
    /// belongs to a foreign seq space, and one beyond `persisted` belongs to a store
    /// that no longer exists (a recreated volume). Both make forwarding *from* that
    /// cursor meaningless; forwarding from now on is what a log relay owes.
    fn seed(&self, persisted: i64) -> Result<i64, StatsError> {
        match self.store.forward_cursor(&self.config.target)? {
            Some(cursor) if cursor <= persisted => Ok(cursor),
            Some(cursor) => {
                tracing::warn!(
                    cursor,
                    persisted,
                    "finelog forwarder: watermark is ahead of the store; reseeding at the tip"
                );
                self.persist(persisted)?;
                Ok(persisted)
            }
            None => {
                tracing::info!(
                    persisted,
                    "finelog forwarder: no watermark for this target; seeding at the tip (new logs only)"
                );
                self.persist(persisted)?;
                Ok(persisted)
            }
        }
    }

    fn persist(&self, cursor: i64) -> Result<(), StatsError> {
        self.store.set_forward_cursor(&self.config.target, cursor)
    }

    /// Forward everything in `(cursor, persisted]`, returning the new cursor.
    /// A read or push failure returns the cursor it reached; the caller retries.
    async fn drain_to(
        &self,
        mut cursor: i64,
        persisted: i64,
        progress: &mut Progress,
        stop: &mut watch::Receiver<bool>,
    ) -> i64 {
        cursor = self.cap_lag(cursor, persisted, progress);
        while cursor < persisted && !*stop.borrow() {
            let batch = match self.read_batch(cursor, persisted).await {
                Ok(batch) => batch,
                Err(e) => {
                    tracing::warn!(cursor, error = %e, "finelog forwarder: read failed; retrying");
                    return cursor;
                }
            };
            // The scan reported its oldest locally-readable row. Anything below it was
            // archived to remote storage while we lagged, and no scan here can reach
            // it — jump the cursor and say how much was skipped.
            if let Some(evicted_to) = batch.evicted_below {
                let skipped = evicted_to - cursor;
                progress.skipped += skipped;
                tracing::warn!(
                    cursor,
                    skipped,
                    resume_at = evicted_to,
                    "finelog forwarder: rows evicted before they were forwarded; skipping ahead"
                );
                cursor = evicted_to;
                if !self.persist_cursor(cursor) {
                    return cursor;
                }
            }
            if batch.rows.is_empty() {
                // Every row up to `persisted` was filtered out by the scan: rows already
                // carrying a foreign origin cluster, or rows with no key. Advance, or the
                // loop rereads them forever. Safe against a concurrent writer: `persisted`
                // is a captured bound, and later rows arrive with a later watermark.
                cursor = persisted;
                self.persist_cursor(persisted);
                return cursor;
            }

            for chunk in split_by_bytes(batch.rows, FORWARD_BATCH_BYTES) {
                let last_seq = chunk
                    .last()
                    .expect("split_by_bytes yields no empty chunk")
                    .seq;
                let rows = chunk.len() as i64;
                match self.push(chunk, stop).await {
                    Ok(()) => progress.batches += 1,
                    Err(PushError::Stopping(e)) => {
                        tracing::warn!(cursor, error = %e, "finelog forwarder: push interrupted");
                        return cursor;
                    }
                    // The hub refused these bytes and always will, so retrying is a
                    // livelock that strands every later row too. Skip past them and
                    // count the gap: the rows remain queryable in this store.
                    Err(PushError::Rejected(e)) => {
                        progress.skipped += rows;
                        tracing::warn!(
                            cursor,
                            skipped = rows,
                            resume_at = last_seq,
                            error = %e,
                            "finelog forwarder: the hub rejected this batch as malformed; skipping it"
                        );
                    }
                }
                cursor = last_seq;
                if !self.persist_cursor(cursor) {
                    return cursor;
                }
            }
        }
        cursor
    }

    /// Record `cursor` as the durable watermark, reporting whether the write stuck.
    ///
    /// A failure is not data loss — the rows are already at the hub — but the caller
    /// stops draining, so the next round resumes from a cursor the catalog agrees
    /// with rather than racing further ahead of it.
    fn persist_cursor(&self, cursor: i64) -> bool {
        if let Err(e) = self.persist(cursor) {
            tracing::warn!(cursor, error = %e, "finelog forwarder: persisting the watermark failed");
            return false;
        }
        true
    }

    /// Abandon a backlog too large to be worth draining, keeping the freshest
    /// [`Self::max_lag_seqs`] of it.
    fn cap_lag(&self, cursor: i64, persisted: i64, progress: &mut Progress) -> i64 {
        if persisted - cursor <= self.max_lag_seqs {
            return cursor;
        }
        let resume_at = persisted - self.max_lag_seqs;
        let skipped = resume_at - cursor;
        progress.skipped += skipped;
        tracing::warn!(
            cursor,
            persisted,
            skipped,
            resume_at,
            "finelog forwarder: backlog exceeds the lag cap; skipping ahead"
        );
        self.persist_cursor(resume_at);
        resume_at
    }

    /// Read up to [`FORWARD_BATCH_ROWS`] locally-written rows in `(cursor, persisted]`.
    ///
    /// Holds the query-visibility read guard across BOTH the segment snapshot and the
    /// scan, so eviction (which takes the write side before unlinking) cannot remove a
    /// segment between the two. That is what lets `evicted_below` be trusted: it is
    /// computed from the very segment set the scan then reads.
    async fn read_batch(&self, cursor: i64, persisted: i64) -> Result<Batch, StatsError> {
        let _read_guard = self.store.query_visibility().read().await;

        let store = Arc::clone(&self.store);
        let snapshot = tokio::task::spawn_blocking(move || store.log_query_snapshot())
            .await
            .map_err(|e| StatsError::Internal(format!("log snapshot task panicked: {e}")))??;

        let evicted_below = resume_after_eviction(cursor, snapshot.min_seq);
        let read_from = evicted_below.unwrap_or(cursor);

        let provider = NamespaceProvider::build(snapshot.schema, &snapshot.paths)
            .map_err(|e| StatsError::Internal(format!("build log provider: {e}")))?;

        let where_parts = vec![
            format!("seq > {read_from}"),
            format!("seq <= {persisted}"),
            // Only rows this store's own writers produced. A row that already carries
            // an origin cluster arrived here by forwarding, and re-forwarding it would
            // loop. Segments predating the column store NULL, which is a local row.
            "(cluster IS NULL OR cluster = '')".to_string(),
            // `PushLogs` files a keyless push under the empty key, where no reader
            // looks. `PushLogsBulk` has no request-level key to fall back on and refuses
            // such an entry, so one unkeyed row would make the hub reject every row
            // batched with it. Excluding them here keeps the filter in SQL, where an
            // empty result still means "no forwardable rows in range" — the row limit
            // applies after the predicate, so the cursor never jumps a row it did not
            // consider.
            "key != ''".to_string(),
        ];
        let rows = fetch_log_rows(
            &make_ctx(),
            provider,
            &where_parts,
            /* include_key */ true,
            /* tail */ false,
            FORWARD_BATCH_ROWS,
        )
        .await
        .map_err(|e| StatsError::Internal(format!("log read failed: {e}")))?;

        Ok(Batch {
            rows,
            evicted_below,
        })
    }

    /// Ship one chunk and wait for the hub to durably ack it. Retries transient failures
    /// with an exponential backoff, returning only when the chunk lands, the hub rejects
    /// it outright, or `stop` latches — which interrupts the backoff, so a SIGTERM never
    /// waits one out.
    ///
    /// Every attempt takes the bearer afresh, and [`TokenMinter`] re-mints it once it
    /// nears expiry, so a chunk retried across a long backoff never presents a stale
    /// token. Auth failures are retried rather than skipped: the rows are safe in the
    /// local store, and dropping a cluster's whole log stream over a fixable typo in the
    /// hub's `jwt` layer would be far worse than stalling visibly until it is repaired.
    async fn push(
        &self,
        chunk: Vec<LogRow>,
        stop: &mut watch::Receiver<bool>,
    ) -> Result<(), PushError> {
        let request = PushLogsBulkRequest {
            entries: chunk.into_iter().map(log_row_to_entry).collect(),
            ..Default::default()
        }
        .with_cluster(self.config.cluster.clone());

        let mut backoff = BACKOFF_MIN;
        loop {
            let bearer = match self.minter.bearer() {
                Ok(bearer) => bearer,
                // The key parsed at startup, so this is not a config error we can
                // resolve by skipping the batch. Back off and try again.
                Err(e) => {
                    tracing::warn!(error = %e, "finelog forwarder: minting a bearer failed");
                    if wait_or_stop(backoff, stop).await {
                        return Err(PushError::Stopping(e));
                    }
                    backoff = (backoff * 2).min(BACKOFF_MAX);
                    continue;
                }
            };
            let options = CallOptions::default()
                .with_timeout(PUSH_TIMEOUT)
                .with_header("authorization", format!("Bearer {bearer}"));
            match self
                .client
                .push_logs_bulk_with_options(request.clone(), options)
                .await
            {
                Ok(_) => return Ok(()),
                Err(e) if is_permanent_rejection(&e) => {
                    return Err(PushError::Rejected(e.to_string()))
                }
                Err(e) if *stop.borrow() => return Err(PushError::Stopping(e.to_string())),
                Err(e) => {
                    tracing::warn!(error = %e, backoff_seconds = backoff.as_secs(), "finelog forwarder: push failed");
                    if wait_or_stop(backoff, stop).await {
                        return Err(PushError::Stopping(e.to_string()));
                    }
                    backoff = (backoff * 2).min(BACKOFF_MAX);
                }
            }
        }
    }
}

/// Why a push gave up.
#[derive(Debug)]
enum PushError {
    /// The forwarder is shutting down mid-retry. The chunk is still owed; the cursor
    /// stays where it is and the next process re-reads it.
    Stopping(String),
    /// The hub refused the chunk's *content*, so no retry of the same bytes can
    /// succeed. The caller skips past it rather than stalling the whole stream.
    Rejected(String),
}

/// Whether the hub's answer means "these bytes are unacceptable" rather than "not
/// right now".
///
/// Only `invalid_argument` qualifies: the hub returns it for a structurally bad entry
/// (e.g. one with an empty key), which is a poison pill — re-sending it forever would
/// strand every row behind it. Everything else, auth failures included, describes a
/// condition that can change without the chunk changing.
fn is_permanent_rejection(error: &connectrpc::ConnectError) -> bool {
    error.code == connectrpc::error::ErrorCode::InvalidArgument
}

/// Sleep for `backoff`, or wake early if `stop` latches. Returns whether it latched,
/// so shutdown never waits out a full backoff.
async fn wait_or_stop(backoff: Duration, stop: &mut watch::Receiver<bool>) -> bool {
    tokio::select! {
        _ = stop.changed() => true,
        _ = tokio::time::sleep(backoff) => false,
    }
}

/// One batch read from the local store: the rows to forward, and — when the cursor
/// had fallen below the oldest local row — the seq to resume from instead.
struct Batch {
    rows: Vec<LogRow>,
    evicted_below: Option<i64>,
}

/// The seq to resume from when the next row after `cursor` is already gone: `min_seq`
/// is the oldest row the local segments still hold, and anything below it lives only in
/// the remote archive, which no scan here reads. A forwarder that waits for those rows
/// waits forever.
///
/// `None` means nothing was lost — `cursor + 1` is still present, so the cursor may sit
/// one below `min_seq` — or there are no local segments at all (an empty store, whose
/// watermark cannot be behind anything).
fn resume_after_eviction(cursor: i64, min_seq: Option<i64>) -> Option<i64> {
    min_seq
        .filter(|min_seq| cursor + 1 < *min_seq)
        .map(|min_seq| min_seq - 1)
}

/// Cumulative forwarding counters, reported on a slow timer.
struct Progress {
    /// Requests the hub accepted. Counted in batches, not rows: one bulk request is
    /// the unit of forwarding work.
    batches: u64,
    /// Rows that will never reach the hub — evicted before they shipped, dropped by the
    /// lag cap, or in a batch the hub permanently refused.
    skipped: i64,
    last_report: Instant,
}

impl Progress {
    fn new() -> Self {
        Self {
            batches: 0,
            skipped: 0,
            last_report: Instant::now(),
        }
    }

    fn report(&mut self, cursor: i64, persisted: i64) {
        if self.last_report.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_report = Instant::now();
        tracing::info!(
            cursor,
            lag = persisted - cursor,
            batches = self.batches,
            skipped = self.skipped,
            "finelog forwarder: progress"
        );
    }
}

/// Split `rows` into chunks whose encoded entries stay under `max_bytes`.
///
/// A row larger than `max_bytes` on its own still gets its own chunk: dropping it
/// would stall the watermark behind a line that can never be sent, and the server's
/// message limit is far above any realistic log line.
fn split_by_bytes(rows: Vec<LogRow>, max_bytes: usize) -> Vec<Vec<LogRow>> {
    let mut chunks: Vec<Vec<LogRow>> = Vec::new();
    let mut current: Vec<LogRow> = Vec::new();
    let mut current_bytes = 0usize;
    for row in rows {
        let size = row_wire_size(&row);
        if !current.is_empty() && current_bytes + size > max_bytes {
            chunks.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
        current_bytes += size;
        current.push(row);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// A row's approximate encoded size. Only the variable-length fields matter; the
/// fixed ones are a constant few bytes each and the budget has ample headroom.
fn row_wire_size(row: &LogRow) -> usize {
    row.key.as_ref().map_or(0, |k| k.len()) + row.source.len() + row.data.len()
}

/// Rebuild a wire entry from a stored row. `seq` and `attempt_id` are omitted: the
/// hub assigns its own seq, and attempt is parsed from the key on read.
fn log_row_to_entry(row: LogRow) -> LogEntry {
    LogEntry {
        timestamp: buffa::MessageField::some(Timestamp {
            epoch_ms: Some(row.epoch_ms),
            ..Default::default()
        }),
        level: Some(buffa::EnumValue::from(row.level)),
        ..Default::default()
    }
    .with_key(row.key.unwrap_or_default())
    .with_source(row.source)
    .with_data(row.data)
}

/// Start the forward loop on the runtime, returning its handle. The caller latches
/// `stop` and awaits the handle at shutdown.
pub fn spawn<T>(forwarder: Forwarder<T>, stop: watch::Receiver<bool>) -> JoinHandle<()>
where
    T: connectrpc::client::ClientTransport,
    <T::ResponseBody as http_body::Body>::Error: std::fmt::Display,
{
    tokio::spawn(async move { forwarder.run(stop).await })
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::proto::finelog::logging::{FetchLogsRequest, MatchScope, PushLogsRequest};
    use crate::server::auth::AuthPolicy;
    use crate::server::test_support::{
        client, disk_store, serve, serve_rejecting, TestTransport, PRIV_A, PRIV_UNTRUSTED, PUB_A,
    };

    use super::*;

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

    /// What a hub finelog runs: a bearer is verified against the sender's public key,
    /// and a bearerless local client (this test, reading the result back) falls through
    /// to the loopback rule. Jwt sits first so a token is never bypassed by the network
    /// rule -- in production the sender is off-VPC and only the token admits it.
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

    fn forwarder(
        store: Arc<Store>,
        target: &SocketAddr,
        private_pem: &str,
    ) -> Forwarder<TestTransport> {
        let config = ForwardingConfig {
            target: format!("http://{target}"),
            cluster: SOURCE_CLUSTER.to_string(),
        };
        let minter = TokenMinter::new(private_pem, config.cluster.clone()).unwrap();
        Forwarder::with_client(store, config, minter, client(*target))
    }

    /// A source store and the hub it forwards to, each served over a real socket under
    /// the auth policy it runs in production.
    struct Fixture {
        source: Arc<Store>,
        source_client: LogServiceClient<TestTransport>,
        target_addr: SocketAddr,
        target_url: String,
        target_requests: Arc<AtomicUsize>,
    }

    impl Fixture {
        /// A hub that trusts [`SOURCE_CLUSTER`]'s public key, and a source that writes
        /// under it. The source's watermark is unset: a forwarder started now seeds at
        /// the tip. Call [`Self::forward_from_start`] to drain what is already written.
        async fn new(tag: &str) -> Self {
            let source = disk_store(&format!("{tag}_source"));
            let target = disk_store(&format!("{tag}_target"));
            let (source_addr, _) = serve(Arc::clone(&source), AuthPolicy::allow_localhost()).await;
            let (target_addr, target_requests) = serve(target, hub_policy(SOURCE_CLUSTER)).await;
            Self {
                source,
                source_client: client(source_addr),
                target_addr,
                target_url: format!("http://{target_addr}"),
                target_requests,
            }
        }

        /// As [`Self::new`], but the hub refuses every request with `invalid_argument`.
        /// It keeps no store, so only the source and the request count are observable.
        async fn with_rejecting_hub(tag: &str) -> Self {
            let source = disk_store(&format!("{tag}_source"));
            let (source_addr, _) = serve(Arc::clone(&source), AuthPolicy::allow_localhost()).await;
            let (target_addr, target_requests) = serve_rejecting().await;
            Self {
                source,
                source_client: client(source_addr),
                target_addr,
                target_url: format!("http://{target_addr}"),
                target_requests,
            }
        }

        /// Point the watermark below every row, so a forward drains the whole store.
        fn forward_from_start(&self) {
            self.source.set_forward_cursor(&self.target_url, 0).unwrap();
        }

        fn forwarder(&self, private_pem: &str) -> Forwarder<TestTransport> {
            forwarder(Arc::clone(&self.source), &self.target_addr, private_pem)
        }

        /// The last seq the source has made durable.
        fn tip(&self) -> i64 {
            *self.source.log_persisted_seq().unwrap().borrow()
        }

        fn cursor(&self) -> Option<i64> {
            self.source.forward_cursor(&self.target_url).unwrap()
        }

        fn requests(&self) -> usize {
            self.target_requests.load(Ordering::SeqCst)
        }

        /// Forward until the watermark settles at the source's current tip, then stop.
        async fn drain(&self, private_pem: &str) {
            forward_until(
                self.forwarder(private_pem),
                &self.source,
                &self.target_url,
                self.tip(),
            )
            .await;
        }

        /// Every row the hub holds, as `(key, data)`.
        async fn hub_rows(&self) -> Vec<(String, String)> {
            read_all(&client(self.target_addr)).await
        }
    }

    /// Write `entries` under `key` into `store` through its own RPC surface, which
    /// returns only once the rows are durable and therefore visible to a scan.
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

    /// Every row the server behind `client` holds, newest last, as `(key, data)` —
    /// read back over the wire so the assertion sees exactly what a log reader would.
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

    /// Poll `condition` until it holds, or fail after five seconds with `describe()`, so
    /// a wedged forwarder fails the test rather than hanging it.
    async fn poll_until(mut condition: impl FnMut() -> bool, describe: impl Fn() -> String) {
        for _ in 0..200 {
            if condition() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        panic!("{}", describe());
    }

    /// Wait for `store`'s watermark for `target` to reach `expected`. Reads local state
    /// only, so a test can tell "the forwarder is done" without issuing an RPC that
    /// would perturb the target's request count.
    async fn wait_for_cursor(store: &Store, target: &str, expected: i64) {
        poll_until(
            || store.forward_cursor(target).unwrap() == Some(expected),
            || {
                format!(
                    "watermark never reached {expected} (stuck at {:?})",
                    store.forward_cursor(target).unwrap()
                )
            },
        )
        .await;
    }

    /// Poll `counter` until the hub has served at least `expected` requests. Lets a
    /// test that asserts on the *absence* of an effect first wait for the attempt that
    /// would have produced it, so the assertion cannot pass merely because nothing has
    /// happened yet.
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

        /// Latch the stop signal and join. Bounded, so a forwarder wedged in a backoff
        /// fails the test rather than hanging it.
        async fn finish(self) {
            self.stop.send(true).unwrap();
            let _ = tokio::time::timeout(Duration::from_secs(5), self.task).await;
        }
    }

    /// Run `forwarder` until `store`'s watermark reaches `expected`, then stop it.
    async fn forward_until(
        forwarder: Forwarder<TestTransport>,
        store: &Store,
        target: &str,
        expected: i64,
    ) {
        let running = RunningForwarder::start(forwarder);
        wait_for_cursor(store, target, expected).await;
        running.finish().await;
    }

    #[test]
    fn minted_bearer_is_accepted_by_a_hub_trusting_the_matching_public_key() {
        // The two halves of the trust config -- this server's private key and the
        // public key an operator pastes into the hub's `jwt` auth layer -- must agree.
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
    fn minted_bearer_names_the_cluster_it_forwards_for() {
        // The hub binds a pushed row's origin cluster to the key that verified the
        // bearer, so the bearer must authenticate as this store's cluster.
        let minter = TokenMinter::new(PRIV_A, "cw-rno2a".to_string()).unwrap();
        let bearer = minter.bearer().unwrap();
        assert_eq!(
            jwt_policy("cw-rno2a").admits(Some(&bearer), None),
            Some(crate::server::auth::AuthIdentity::Jwt {
                cluster: "cw-rno2a".to_string()
            })
        );
    }

    #[test]
    fn a_cached_bearer_is_reused_until_it_nears_expiry() {
        // Asserting that two successive mints are equal would prove nothing: EdDSA is
        // deterministic and `iat`/`exp` are second-granular, so a cache-less minter
        // returns identical bytes too. Drive the cache directly instead.
        let minter = TokenMinter::new(PRIV_A, SOURCE_CLUSTER.to_string()).unwrap();

        // A live cache entry is handed back untouched — the minter does not re-sign
        // once per push.
        let usable = Instant::now() + Duration::from_secs(600);
        *minter.cached.lock().unwrap() = Some(("cached-bearer".to_string(), usable));
        assert_eq!(minter.bearer().unwrap(), "cached-bearer");

        // Once the entry is inside the refresh margin, it is replaced rather than
        // handed to the hub, which would reject a token expiring mid-flight.
        *minter.cached.lock().unwrap() = Some(("stale-bearer".to_string(), Instant::now()));
        let fresh = minter.bearer().unwrap();
        assert_ne!(fresh, "stale-bearer");
        assert_eq!(
            jwt_policy(SOURCE_CLUSTER).admits(Some(&fresh), None),
            Some(crate::server::auth::AuthIdentity::Jwt {
                cluster: SOURCE_CLUSTER.to_string()
            }),
            "the re-minted bearer must still verify at the hub"
        );
    }

    #[test]
    fn forwarding_config_rejects_a_target_that_would_expose_the_bearer() {
        assert!(ForwardingConfig::parse(r#"{"target":"http://hub","cluster":"a"}"#).is_err());
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
    fn split_by_bytes_bounds_a_chunk_but_never_drops_an_oversized_row() {
        let row = |data: &str| LogRow {
            seq: 1,
            key: Some("/k".to_string()),
            source: "stdout".to_string(),
            data: data.to_string(),
            epoch_ms: 0,
            level: 0,
        };
        let chunks = split_by_bytes(vec![row("aaaa"), row("bbbb"), row("cccc")], 24);
        assert_eq!(chunks.len(), 2, "each row is 12 bytes; two fit in 24");
        assert_eq!(chunks[0].len(), 2);
        assert_eq!(chunks[1].len(), 1);
        // A single row over budget still ships, rather than stalling the watermark.
        let chunks = split_by_bytes(vec![row(&"x".repeat(100))], 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drains_a_startup_backlog_of_many_keys_in_one_bulk_request() {
        // Two contracts at once, because they share a fixture.
        //
        // 1. A forwarder that starts with its watermark behind the store's durability
        //    mark must drain immediately. It subscribes to a watch channel whose
        //    current value counts as already seen, so a loop that awaited a change
        //    before reading would sit idle until the next flush -- for a quiet cluster,
        //    forever.
        // 2. A batch spanning many keys costs ONE request. That is what PushLogsBulk
        //    buys: forwarding throughput is independent of how many distinct log keys
        //    are in flight, so a job fanned out over 140 workers ships as fast as one
        //    that is not. The count is taken at the hub's HTTP boundary, which is where
        //    the contract lives.
        let fx = Fixture::new("bulk").await;
        for key in ["/user/job/task-a", "/user/job/task-b", "/system/worker/1"] {
            push(&fx.source_client, key, &["first", "second"]).await;
        }
        fx.forward_from_start();
        let requests_before = fx.requests();

        fx.drain(PRIV_A).await;

        assert_eq!(
            fx.requests() - requests_before,
            1,
            "six rows across three keys must ship in a single bulk request"
        );

        let mut forwarded = fx.hub_rows().await;
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
        // The hub namespaces logs by origin, and it takes that origin from the bearer's
        // cluster rather than the sender's word for it.
        let fx = Fixture::new("stamp").await;
        push(&fx.source_client, "/user/job/t", &["hello"]).await;
        fx.forward_from_start();
        fx.drain(PRIV_A).await;

        // Reading the hub restricted to this origin returns the row, which it can only
        // do if the `cluster` column was stamped.
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
        assert_eq!(entries, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_bearer_the_hub_does_not_trust_forwards_nothing_and_loses_nothing() {
        // The hub rejects the push, so the watermark must not advance: the rows are
        // still owed, and the local store still serves them.
        let fx = Fixture::new("reject").await;
        push(&fx.source_client, "/user/job/t", &["hello"]).await;
        fx.forward_from_start();

        // PRIV_UNTRUSTED is not the key the hub trusts.
        let running = RunningForwarder::start(fx.forwarder(PRIV_UNTRUSTED));
        // Wait for the push to REACH the hub. Stopping on a timer instead would let a
        // forwarder that never pushed at all satisfy both assertions below.
        wait_for_requests(&fx.target_requests, 1).await;
        running.finish().await;

        assert_eq!(
            fx.cursor(),
            Some(0),
            "a refused push leaves the watermark where it was, so the rows are retried"
        );
        assert!(fx.hub_rows().await.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_unkeyed_local_row_never_poisons_the_batch_it_would_ship_in() {
        // `PushLogs` files a keyless push under the empty key. The hub refuses such an
        // entry, and refuses the whole bulk request carrying it, so a forwarder that
        // shipped one would lose every row batched alongside it. It must never build
        // that request.
        let fx = Fixture::new("unkeyed").await;
        push(&fx.source_client, "", &["unkeyed"]).await;
        push(&fx.source_client, "/user/job/t", &["keyed"]).await;
        fx.forward_from_start();
        fx.drain(PRIV_A).await;

        assert_eq!(
            fx.hub_rows().await,
            vec![("/user/job/t".to_string(), "keyed".to_string())],
            "the keyed row ships; the unkeyed one is left behind rather than poisoning it"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_batch_the_hub_calls_malformed_is_skipped_rather_than_retried_forever() {
        // invalid_argument means the hub refuses these bytes and always will. Retrying
        // would strand every row behind them, so the forwarder counts the batch as
        // skipped and moves its cursor past it. The rows are still in the local store.
        let fx = Fixture::with_rejecting_hub("poison").await;
        push(&fx.source_client, "/user/job/t", &["hello"]).await;
        fx.forward_from_start();
        let tip = fx.tip();
        fx.drain(PRIV_A).await;

        assert_eq!(
            fx.cursor(),
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
    async fn seeding_at_the_tip_ships_new_logs_and_never_backfills() {
        let fx = Fixture::new("seed").await;
        push(&fx.source_client, "/user/job/t", &["before"]).await;

        // No watermark for this target: seed at the tip, so "before" is never shipped.
        let running = RunningForwarder::start(fx.forwarder(PRIV_A));
        wait_for_cursor(&fx.source, &fx.target_url, fx.tip()).await;

        push(&fx.source_client, "/user/job/t", &["after"]).await;
        wait_for_cursor(&fx.source, &fx.target_url, fx.tip()).await;
        running.finish().await;

        assert_eq!(
            fx.hub_rows().await,
            vec![("/user/job/t".to_string(), "after".to_string())]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rows_that_already_carry_an_origin_cluster_are_never_re_forwarded() {
        // A hub's own rows arrived by forwarding. Relaying them onward would loop, so
        // only rows this store's writers produced are eligible.
        let fx = Fixture::new("loop").await;

        // A relayed row (origin elsewhere) and a locally-written one.
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
        fx.forward_from_start();
        fx.drain(PRIV_A).await;

        assert_eq!(
            fx.hub_rows().await,
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
            .set_forward_cursor(&fx.target_url, 10_000)
            .unwrap();

        fx.drain(PRIV_A).await;

        assert!(fx.hub_rows().await.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_backlog_beyond_the_lag_cap_is_skipped_rather_than_drained() {
        // The store keeps every row; the hub copy is best effort. A forwarder too far
        // behind abandons the oldest part of the backlog instead of queueing without
        // bound, and keeps the freshest `max_lag_seqs` of it.
        let fx = Fixture::new("cap").await;
        push(
            &fx.source_client,
            "/user/job/t",
            &["one", "two", "three", "four"],
        )
        .await;
        fx.forward_from_start();

        let mut forwarder = fx.forwarder(PRIV_A);
        forwarder.max_lag_seqs = 2;
        forward_until(forwarder, &fx.source, &fx.target_url, fx.tip()).await;

        assert_eq!(
            fx.hub_rows().await,
            vec![
                ("/user/job/t".to_string(), "three".to_string()),
                ("/user/job/t".to_string(), "four".to_string()),
            ],
            "the two oldest rows are dropped; the freshest two still ship"
        );
    }
}
