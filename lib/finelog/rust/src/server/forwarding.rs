// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Native cross-cluster forwarding: replicate this store's locally-written rows,
//! table by table, into a shared hub finelog.
//!
//! A federated cluster keeps its own finelog and also ships every table it holds —
//! the `log` namespace and the `iris.*` stats tables alike — to one global store, so
//! `iris --cluster=<hub>` reads back a job that ran anywhere. The forwarder lives in
//! the server, which already owns the segment layout and the durability watermark: it
//! scans straight out of the sealed segments and ships each table's rows through the
//! generic [`WriteRows`] path, one Arrow batch per round trip.
//!
//! After waiting [`FORWARD_INTERVAL`], it lists the live namespaces and gives each one
//! batch-sized forwarding turn. It immediately repeats while any namespace has more
//! settled rows, so a busy stats table cannot delay the next `log` turn while the
//! forwarder still drains at full speed. A per-`(target, namespace)` cursor records
//! progress, and a namespace the hub lacks is created there first with [`RegisterTable`].
//!
//! # Credential
//!
//! The forwarder authenticates with a short-lived `aud="finelog"` EdDSA bearer minted
//! from *this server's own* Ed25519 key, which the hub pins in its `jwt` auth layer.
//! That key is distinct from the iris controller's signing key: a compromise of the
//! log-ingest path grants log-plane authority only.
//!
//! # Best effort by design
//!
//! The local store is the system of record — every row stays queryable here whether or
//! not the hub ever receives it. The forwarder therefore never fails the server and
//! never grows without bound:
//!
//! - Each namespace seeds at its current tip, so enabling forwarding ships new rows
//!   rather than backfilling a retention window.
//! - It materializes at most [`FORWARD_BATCH_ROWS`] rows per read, and packs them into
//!   requests of at most [`FORWARD_BATCH_BYTES`] unless one row alone exceeds the limit.
//! - It advances a namespace's cursor past a batch only once that batch can never be
//!   sent again: the hub acked it, or the forwarder gave it up. A crash mid-batch
//!   re-forwards it (at-least-once; tolerable for logs and append-only stats).
//! - A backlog is only a cursor into the source's already-bounded local retention, not a
//!   separate in-memory queue. The forwarder drains it without an age or row-count cap.
//!   It skips only when eviction has already removed the cursor's segments or when the
//!   hub permanently refuses a malformed batch.
//!
//! A push failure backs off and retries; nothing here can take the store down.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, AsArray, Int64Array, RecordBatch, StringArray};
use arrow::compute::concat_batches;
use arrow::datatypes::{Field, Int64Type, Schema as ArrowSchema};
use connectrpc::client::{CallOptions, ClientConfig, ServiceTransport};
use connectrpc::compression::{CompressionRegistry, GzipProvider, ZstdProvider};
use futures::StreamExt;
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::rt::TokioExecutor;
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{RegisterTableRequest, StatsServiceClient, WriteRowsRequest};
use crate::query::provider::NamespaceProvider;
use crate::query::{make_ctx, run_query_over, QueryResult, RegisteredProvider};
use crate::server::auth::FINELOG_AUDIENCE;
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::ipc::encode_ipc;
use crate::store::schema::{
    schema_to_proto_owned, Schema, IMPLICIT_CLUSTER_COLUMN, IMPLICIT_SEQ_COLUMN,
    MAX_WRITE_ROWS_BYTES,
};
use crate::store::store::LOG_NAMESPACE_NAME;
use crate::store::Store;

/// How long the forwarder waits after every namespace is caught up or unable to make
/// progress. Backlogged namespaces trigger another round immediately, so throughput
/// does not depend on this cadence.
const FORWARD_INTERVAL: Duration = Duration::from_secs(5);

/// Rows read from one namespace per batch. The hub durably acknowledges each outbound
/// request, so a small row cap turns its one-second flush-coalescing interval into a
/// throughput cap. A live telemetry scan showed that reading 200,000 rows was faster
/// than reading 50,000 because provider construction and planning dominate the bounded
/// scan. Byte chunking still bounds each request, and this stays below the store's
/// million-row write limit.
const FORWARD_BATCH_ROWS: i64 = 200_000;

/// Encoded bytes per outbound request. Keep one MiB below the receiving store's hard
/// Arrow IPC limit. [`chunk_by_bytes`] verifies the resulting size and adjusts each
/// chunk toward this budget.
const FORWARD_BATCH_BYTES: usize = MAX_WRITE_ROWS_BYTES - (1 << 20);

const FORWARD_REQUEST_COMPRESSION: &str = "zstd";
const FINELOG_ZSTD_LEVEL: i32 = 1;

/// Non-log chunks from one read turn may wait for durability concurrently. The hub
/// coalesces writes that arrive within its one-second flush window, so a telemetry-sized
/// turn split across several Arrow messages needs one durable flush instead of one per
/// message. Log chunks stay serial to preserve line order at the hub.
const FORWARD_CHUNK_CONCURRENCY: usize = 8;

/// Emit one warning when a namespace's cursor trails this far behind. This is an
/// observability threshold only: the backlog is still drained in full while its source
/// segments remain locally retained.
///
/// Measured in `seq` positions, not rows: `seq` is a namespace's own dense counter, so
/// the two coincide only when no earlier gap exists.
const FORWARD_LAG_WARNING_SEQS: i64 = 2_000_000;

/// Bearer lifetime, and how early to re-mint. Short-lived because the hub checks only
/// signature + audience + expiry — it cannot reach a revocation list, so a leaked
/// bearer's blast radius is bounded by its TTL.
const TOKEN_TTL: Duration = Duration::from_secs(3600);
const TOKEN_REFRESH_MARGIN: Duration = Duration::from_secs(300);

/// Deadline for one outbound request, and the retry backoff bounds.
const PUSH_TIMEOUT: Duration = Duration::from_secs(60);
const BACKOFF_MIN: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

/// Emit a progress line at most this often, so a healthy forwarder is observable
/// without flooding the logs it forwards.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(300);

/// Where this store forwards, and as whom. Parsed from the `FINELOG_FORWARDING` JSON;
/// the Ed25519 private key arrives separately (`FINELOG_SIGNING_KEY`) so it never rides
/// in an inline env value.
#[derive(Debug, Clone, Deserialize)]
pub struct ForwardingConfig {
    /// The hub finelog's base URL. `https://` only: the bearer is a credential.
    pub target: String,
    /// This store's cluster. Stamped on every forwarded row that has an origin column,
    /// and carried as the bearer's `iss`/`sub`.
    pub cluster: String,
}

impl ForwardingConfig {
    /// Parse the `FINELOG_FORWARDING` JSON, rejecting a config that could only fail
    /// later: an empty cluster has no identity to forward as, and a non-TLS target would
    /// put the bearer on the wire in the clear.
    pub fn parse(json: &str) -> Result<Self, String> {
        let config: ForwardingConfig =
            serde_json::from_str(json).map_err(|e| format!("invalid forwarding JSON: {e}"))?;
        if config.cluster.is_empty() {
            return Err("forwarding config has an empty cluster".to_string());
        }
        if !config.target.starts_with("https://") {
            return Err(format!(
                "forwarding target {:?} is not https:// (the bearer must not travel in the clear)",
                config.target
            ));
        }
        Ok(config)
    }
}

/// The bearer claims. `iss`/`sub`/`aud`/`exp` are exactly the set the hub's verifier
/// requires; `iat` records when the token was minted.
#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str,
    sub: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

/// Mints and caches this server's `aud="finelog"` bearer, re-minting once the live one
/// is within [`TOKEN_REFRESH_MARGIN`] of expiry.
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

    /// The current bearer, minting a fresh one when none is cached or the cached one is
    /// close enough to expiry that the hub might reject it in flight.
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
            .map_err(|e| format!("minting the finelog bearer failed: {e}"))?;
        *cached = Some((token.clone(), now + TOKEN_TTL - TOKEN_REFRESH_MARGIN));
        Ok(token)
    }
}

type ClientBody = connectrpc::client::ClientBody;
type HttpsConnector =
    hyper_rustls::HttpsConnector<hyper_util::client::legacy::connect::HttpConnector>;

/// The forwarder's production transport: pooled HTTPS over hyper + rustls.
pub type HttpsTransport = ServiceTransport<HyperClient<HttpsConnector, ClientBody>>;

/// An HTTPS Connect client for the hub's `StatsService`. Errors if `target` is not a URI.
fn build_client(target: &str) -> Result<StatsServiceClient<HttpsTransport>, String> {
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
    let config = forward_client_config(uri);
    Ok(StatsServiceClient::new(transport, config))
}

pub(super) fn forward_client_config(uri: http::Uri) -> ClientConfig {
    let compression = CompressionRegistry::new()
        .register(GzipProvider::default())
        .register(ZstdProvider::with_level(FINELOG_ZSTD_LEVEL));
    ClientConfig::new(uri)
        .proto()
        .with_compression(compression)
        .compress_requests(FORWARD_REQUEST_COMPRESSION)
        .with_default_max_message_size(MAX_MESSAGE_BYTES)
}

/// Everything the forward loop needs, resolved once at startup.
///
/// Generic over the Connect transport so the same loop runs against the production
/// HTTPS client and, in tests, against a plaintext one pointed at an in-process hub.
pub struct Forwarder<T = HttpsTransport> {
    store: Arc<Store>,
    client: StatsServiceClient<T>,
    minter: TokenMinter,
    config: ForwardingConfig,
    /// Backlog size that emits a warning. It never changes the cursor.
    lag_warning_seqs: i64,
    warned_lag: Mutex<HashSet<String>>,
    /// Namespaces already created on the hub this process, so `RegisterTable` runs at
    /// most once each.
    registered: Mutex<HashSet<String>>,
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
        client: StatsServiceClient<T>,
    ) -> Self {
        Self {
            store,
            client,
            minter,
            config,
            lag_warning_seqs: FORWARD_LAG_WARNING_SEQS,
            warned_lag: Mutex::new(HashSet::new()),
            registered: Mutex::new(HashSet::new()),
        }
    }

    /// Run until `stop` latches. Errors are logged and retried on the next tick; this
    /// never returns an error, because a store whose forwarding is broken must keep
    /// serving.
    pub async fn run(&self, mut stop: watch::Receiver<bool>) {
        tracing::info!(
            target = %self.config.target,
            cluster = %self.config.cluster,
            "finelog forwarder: started"
        );
        let mut progress = Progress::new();
        loop {
            if *stop.borrow() {
                break;
            }
            let turn = self.forward_round(&mut progress, &mut stop).await;

            // A drain's own `wait_or_stop` marks the value seen when it returns, so the
            // select below would wait for a second change that never comes. Read the
            // latched value first, which no `changed()` clears.
            if *stop.borrow() {
                break;
            }
            if turn == ForwardTurn::MoreRows {
                continue;
            }
            tokio::select! {
                _ = stop.changed() => break,
                _ = tokio::time::sleep(FORWARD_INTERVAL) => {}
            }
        }
        tracing::info!("finelog forwarder: stopped");
    }

    /// Give every live namespace one batch-sized turn. [`ForwardTurn::MoreRows`] means
    /// at least one namespace advanced successfully but remains behind its captured
    /// watermark; caught-up, failed, or interrupted rounds return [`ForwardTurn::Wait`].
    async fn forward_round(
        &self,
        progress: &mut Progress,
        stop: &mut watch::Receiver<bool>,
    ) -> ForwardTurn {
        let namespaces = match self.store.list_namespaces_with_stats() {
            Ok(namespaces) => namespaces,
            Err(e) => {
                tracing::warn!(error = %e, "finelog forwarder: cannot list namespaces; retrying");
                return ForwardTurn::Wait;
            }
        };
        let mut turn = ForwardTurn::Wait;
        for (name, schema, _stats, _policy) in namespaces {
            if *stop.borrow() {
                break;
            }
            if self.forward_namespace(&name, &schema, progress, stop).await == ForwardTurn::MoreRows
            {
                turn = ForwardTurn::MoreRows;
            }
        }
        progress.report();
        turn
    }

    /// Forward at most one read batch settled in `name` since its cursor. A read,
    /// register, or persist failure waits for the next timed sweep; remaining rows after
    /// a successful batch ask the caller to start another round immediately.
    async fn forward_namespace(
        &self,
        name: &str,
        schema: &Schema,
        progress: &mut Progress,
        stop: &mut watch::Receiver<bool>,
    ) -> ForwardTurn {
        let persisted = match self.store.namespace_persisted_seq(name) {
            Ok(persisted) => persisted,
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: no watermark; skipping");
                return ForwardTurn::Wait;
            }
        };
        let mut cursor = match self.seed(name, persisted) {
            Ok(cursor) => cursor,
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: cannot seed; skipping");
                return ForwardTurn::Wait;
            }
        };
        self.observe_lag(name, cursor, persisted);
        // The forwarder stamps this column with its own cluster on the way out and
        // forwards only rows that do not already carry a foreign origin, so a hub's
        // own relayed rows never loop back. A registered table always has it (added
        // implicitly at registration); a legacy namespace adopted from disk before
        // the column existed does not, and forwards unstamped until it is
        // re-registered.
        let has_origin = schema.column(IMPLICIT_CLUSTER_COLUMN).is_some();

        if cursor >= persisted || *stop.borrow() {
            return ForwardTurn::Wait;
        }
        let batch = match self.read_batch(name, cursor, persisted, has_origin).await {
            Ok(batch) => batch,
            Err(e) => {
                tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: read failed; retrying next tick");
                return ForwardTurn::Wait;
            }
        };
        // The scan reported its oldest locally-readable row. Anything below it was
        // archived to remote storage while we lagged, and no scan here can reach it —
        // jump the cursor and say how much was skipped.
        if let Some(resume_at) = batch.resume_at {
            let skipped = resume_at - cursor;
            progress.skipped_seqs += skipped;
            tracing::warn!(
                namespace = name,
                cursor,
                skipped,
                resume_at,
                "finelog forwarder: rows evicted before they were forwarded; skipping ahead"
            );
            cursor = resume_at;
            if !self.persist_cursor(name, cursor) {
                return ForwardTurn::Wait;
            }
        }
        let Some((ship, seqs)) = batch.rows else {
            // Every row up to `persisted` was filtered out by the scan (rows already
            // carrying a foreign origin cluster). Advance the cursor to `persisted`,
            // or the loop rereads them forever. Safe against a concurrent writer:
            // `persisted` is a captured bound, and later rows arrive with a later
            // watermark.
            self.persist_cursor(name, persisted);
            return ForwardTurn::Wait;
        };
        // The hub must hold the namespace before it can take rows for it.
        if !self.ensure_registered(name, schema).await {
            tracing::warn!(
                namespace = name,
                "finelog forwarder: hub not ready for namespace; retrying next tick"
            );
            return ForwardTurn::Wait;
        }

        let chunks = match chunk_by_bytes(&ship, &seqs, FORWARD_BATCH_BYTES) {
            Ok(chunks) => chunks,
            Err(e) => {
                tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: encoding a batch failed; retrying next tick");
                return ForwardTurn::Wait;
            }
        };
        let concurrency = if name == LOG_NAMESPACE_NAME {
            1
        } else {
            FORWARD_CHUNK_CONCURRENCY
        };
        let pushes = chunks.into_iter().map(|(ipc, last_seq)| {
            let mut chunk_stop = (*stop).clone();
            async move { (last_seq, self.push(name, ipc, &mut chunk_stop).await) }
        });
        // `buffered` polls several pushes at once but yields their results in input
        // order. The durable cursor therefore never advances past an earlier chunk that
        // is still retrying, even when a later chunk reaches the hub first.
        let mut pushes = futures::stream::iter(pushes).buffered(concurrency);
        while let Some((last_seq, result)) = pushes.next().await {
            match result {
                Ok(()) => progress.batches += 1,
                Err(PushError::Stopping(e)) => {
                    tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: push interrupted");
                    return ForwardTurn::Wait;
                }
                // The hub refused these bytes and always will, so retrying is a
                // livelock that strands every later row too. Skip past them and count
                // the gap: the rows remain queryable in this store.
                Err(PushError::Rejected(e)) => {
                    progress.skipped_seqs += last_seq - cursor;
                    tracing::warn!(
                        namespace = name,
                        cursor,
                        skipped = last_seq - cursor,
                        resume_at = last_seq,
                        error = %e,
                        "finelog forwarder: the hub rejected this batch as malformed; skipping it"
                    );
                }
            }
            cursor = last_seq;
            if !self.persist_cursor(name, cursor) {
                return ForwardTurn::Wait;
            }
        }
        if cursor < persisted {
            ForwardTurn::MoreRows
        } else {
            ForwardTurn::Wait
        }
    }

    /// The cursor to start `name` from: its stored watermark, or the current tip when
    /// there is none, or when the watermark sits beyond `persisted` and so names a seq
    /// space this store no longer has (a recreated volume).
    fn seed(&self, name: &str, persisted: i64) -> Result<i64, StatsError> {
        match self.store.forward_cursor(&self.config.target, name)? {
            Some(cursor) if cursor <= persisted => Ok(cursor),
            Some(cursor) => {
                tracing::warn!(
                    namespace = name,
                    cursor,
                    persisted,
                    "finelog forwarder: watermark is ahead of the store; reseeding at the tip"
                );
                self.persist(name, persisted)?;
                Ok(persisted)
            }
            None => {
                tracing::info!(
                    namespace = name,
                    persisted,
                    "finelog forwarder: no watermark for this target; seeding at the tip (new rows only)"
                );
                self.persist(name, persisted)?;
                Ok(persisted)
            }
        }
    }

    fn persist(&self, name: &str, cursor: i64) -> Result<(), StatsError> {
        self.store
            .set_forward_cursor(&self.config.target, name, cursor)
    }

    /// Record `cursor` as the durable watermark for `name`, reporting whether the write
    /// stuck. `false` is not data loss: every row stays queryable in this store, and the
    /// catalog still names an older cursor for the next round to resume from.
    fn persist_cursor(&self, name: &str, cursor: i64) -> bool {
        if let Err(e) = self.persist(name, cursor) {
            tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: persisting the watermark failed");
            return false;
        }
        true
    }

    /// Report a growing backlog once without changing its cursor. Clear the latch after
    /// catch-up so a later independent incident is visible too.
    fn observe_lag(&self, name: &str, cursor: i64, persisted: i64) {
        let lag = persisted - cursor;
        let mut warned = self.warned_lag.lock().unwrap();
        if lag > self.lag_warning_seqs {
            if warned.insert(name.to_string()) {
                tracing::warn!(
                    namespace = name,
                    cursor,
                    persisted,
                    lag,
                    "finelog forwarder: backlog exceeds the warning threshold; draining without skipping"
                );
            }
        } else {
            warned.remove(name);
        }
    }

    /// Up to [`FORWARD_BATCH_ROWS`] locally-written rows of `name` in `(cursor, persisted]`
    /// as a batch ready to ship (its `seq` column dropped, its origin column stamped),
    /// alongside the seqs those rows carry. When eviction has already archived the rows
    /// just above `cursor`, `resume_at` names the seq to resume from instead.
    async fn read_batch(
        &self,
        name: &str,
        cursor: i64,
        persisted: i64,
        has_origin: bool,
    ) -> Result<Batch, StatsError> {
        // One guard across both the snapshot and the scan: eviction takes the write side
        // before unlinking, so no segment can vanish between the two.
        let _read_guard = self.store.query_visibility().read().await;

        let store = Arc::clone(&self.store);
        let owned = name.to_string();
        let snapshot = tokio::task::spawn_blocking(move || store.query_snapshot(&owned))
            .await
            .map_err(|e| StatsError::Internal(format!("snapshot task panicked: {e}")))??;

        let resume_at = resume_after_eviction(cursor, snapshot.min_seq);
        let read_from = resume_at.unwrap_or(cursor);

        let provider =
            NamespaceProvider::build(snapshot.schema, &snapshot.paths, snapshot.index_cache)
                .map_err(|e| StatsError::Internal(format!("build provider {name:?}: {e}")))?;

        let table = quote_ident(name);
        let mut sql =
            format!("SELECT * FROM {table} WHERE seq > {read_from} AND seq <= {persisted}");
        if has_origin {
            // Only rows this store's own writers produced. A row that already carries an
            // origin cluster arrived here by forwarding, and re-forwarding it would loop.
            // Segments predating the column store NULL, which is a local row.
            sql.push_str(" AND (cluster IS NULL OR cluster = '')");
        }
        sql.push_str(&format!(" ORDER BY seq LIMIT {FORWARD_BATCH_ROWS}"));

        let providers = vec![RegisteredProvider {
            name: name.to_string(),
            provider,
        }];
        let result = run_query_over(&make_ctx(), providers, &sql)
            .await
            .map_err(|e| StatsError::Internal(format!("read {name:?} failed: {e}")))?;

        Ok(Batch {
            rows: self.ship_batch(result)?,
            resume_at,
        })
    }

    /// Turn a scan result into a batch ready for [`WriteRows`]: concatenate the collected
    /// batches, drop the server-assigned `seq` column (the hub assigns its own), and
    /// stamp the origin column with this store's cluster. `None` when the scan matched no
    /// rows. Returns the dropped seqs alongside, so the caller can advance its cursor.
    fn ship_batch(
        &self,
        result: QueryResult,
    ) -> Result<Option<(RecordBatch, Int64Array)>, StatsError> {
        if result.batches.is_empty() {
            return Ok(None);
        }
        let batch = concat_batches(&result.schema, &result.batches)
            .map_err(|e| StatsError::Internal(format!("concat batches: {e}")))?;
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(None);
        }

        let seq_idx = result
            .schema
            .index_of(IMPLICIT_SEQ_COLUMN)
            .map_err(|e| StatsError::Internal(format!("scan result has no seq column: {e}")))?;
        let seqs: Int64Array = batch.column(seq_idx).as_primitive::<Int64Type>().clone();

        let origin: ArrayRef = Arc::new(StringArray::from(vec![
            self.config.cluster.as_str();
            num_rows
        ]));
        let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns() - 1);
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() - 1);
        for (i, field) in result.schema.fields().iter().enumerate() {
            if i == seq_idx {
                continue;
            }
            fields.push(field.as_ref().clone());
            if field.name() == IMPLICIT_CLUSTER_COLUMN {
                columns.push(Arc::clone(&origin));
            } else {
                columns.push(Arc::clone(batch.column(i)));
            }
        }
        let ship_schema = Arc::new(ArrowSchema::new(fields));
        let ship = RecordBatch::try_new(ship_schema, columns)
            .map_err(|e| StatsError::Internal(format!("build ship batch: {e}")))?;
        Ok(Some((ship, seqs)))
    }

    /// Create `name` on the hub if this process has not already, so a later [`WriteRows`]
    /// resolves. The reserved `log` namespace exists on every finelog, so it is never
    /// registered. Returns whether the hub is ready to take rows for `name`.
    async fn ensure_registered(&self, name: &str, schema: &Schema) -> bool {
        if name == LOG_NAMESPACE_NAME || self.registered.lock().unwrap().contains(name) {
            return true;
        }
        let bearer = match self.minter.bearer() {
            Ok(bearer) => bearer,
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: minting a bearer for register failed");
                return false;
            }
        };
        // An empty storage policy leaves the hub's own retention untouched (re-register
        // with no policy is "no opinion").
        let request = RegisterTableRequest {
            schema: buffa::MessageField::some(schema_to_proto_owned(schema)),
            ..Default::default()
        }
        .with_namespace(name);
        let options = CallOptions::default()
            .with_timeout(PUSH_TIMEOUT)
            .with_header("authorization", format!("Bearer {bearer}"));
        match self
            .client
            .register_table_with_options(request, options)
            .await
        {
            Ok(_) => {
                self.registered.lock().unwrap().insert(name.to_string());
                true
            }
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: registering the namespace on the hub failed");
                false
            }
        }
    }

    /// Ship one encoded chunk to `name` on the hub and wait for it to durably ack,
    /// retrying transient failures — auth failures among them — with an exponential
    /// backoff.
    ///
    /// Returns [`PushError::Rejected`] only when the hub refuses the chunk's content, and
    /// [`PushError::Stopping`] when `stop` latches, which also interrupts the backoff so a
    /// SIGTERM never waits one out.
    async fn push(
        &self,
        name: &str,
        arrow_ipc: Vec<u8>,
        stop: &mut watch::Receiver<bool>,
    ) -> Result<(), PushError> {
        let request = WriteRowsRequest::default()
            .with_namespace(name)
            .with_arrow_ipc(arrow_ipc);

        let mut backoff = BACKOFF_MIN;
        loop {
            let bearer = match self.minter.bearer() {
                Ok(bearer) => bearer,
                // The key parsed at startup, so this is not a config error we can resolve
                // by skipping the batch. Back off and try again.
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
                .write_rows_with_options(request.clone(), options)
                .await
            {
                Ok(_) => {
                    tracing::debug!(namespace = name, "finelog forwarder: batch delivered");
                    return Ok(());
                }
                Err(e) if is_permanent_rejection(&e) => {
                    return Err(PushError::Rejected(e.to_string()))
                }
                Err(e) if *stop.borrow() => return Err(PushError::Stopping(e.to_string())),
                Err(e) => {
                    tracing::warn!(namespace = name, error = %e, backoff_seconds = backoff.as_secs(), "finelog forwarder: push failed");
                    if wait_or_stop(backoff, stop).await {
                        return Err(PushError::Stopping(e.to_string()));
                    }
                    backoff = (backoff * 2).min(BACKOFF_MAX);
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForwardTurn {
    /// Retry on the normal interval: work is caught up, unable to progress, or stopping.
    Wait,
    /// Start another round immediately after every other namespace receives a turn.
    MoreRows,
}

/// Why a push gave up.
#[derive(Debug)]
enum PushError {
    /// The forwarder is shutting down mid-retry. The chunk is still owed; the cursor
    /// stays where it is and the next process re-reads it.
    Stopping(String),
    /// The hub refused the chunk's *content*, so no retry of the same bytes can succeed.
    /// The caller skips past it rather than stalling the whole stream.
    Rejected(String),
}

/// Whether the hub's answer means "these bytes are unacceptable" rather than "not right
/// now".
///
/// Only `invalid_argument` qualifies: the hub returns it for a structurally bad batch
/// (a schema the namespace does not accept), which is a poison pill — re-sending it
/// forever would strand every row behind it. Everything else, auth failures included,
/// describes a condition that can change without the chunk changing.
fn is_permanent_rejection(error: &connectrpc::ConnectError) -> bool {
    error.code == connectrpc::error::ErrorCode::InvalidArgument
}

/// Sleep for `backoff`, or wake early if `stop` latches. Returns whether it latched, so
/// shutdown never waits out a full backoff.
async fn wait_or_stop(backoff: Duration, stop: &mut watch::Receiver<bool>) -> bool {
    tokio::select! {
        _ = stop.changed() => true,
        _ = tokio::time::sleep(backoff) => false,
    }
}

/// One batch read from a namespace: the rows to forward (already stripped of `seq` and
/// stamped with the origin cluster) and the seqs they carry, plus — when the cursor had
/// fallen below the oldest local row — the seq to resume from instead.
struct Batch {
    rows: Option<(RecordBatch, Int64Array)>,
    resume_at: Option<i64>,
}

/// The seq to resume from when the next row after `cursor` is already gone: `min_seq` is
/// the oldest row the local segments still hold, and anything below it lives only in the
/// remote archive, which no scan here reads. A forwarder that waits for those rows waits
/// forever.
///
/// `None` means nothing was lost — `cursor + 1` is still present, so the cursor may sit
/// one below `min_seq` — or there are no local segments at all (an empty namespace, whose
/// watermark cannot be behind anything).
fn resume_after_eviction(cursor: i64, min_seq: Option<i64>) -> Option<i64> {
    min_seq
        .filter(|min_seq| cursor + 1 < *min_seq)
        .map(|min_seq| min_seq - 1)
}

/// A double-quoted SQL identifier for `name`, so a dotted namespace (`iris.worker`)
/// resolves as one table and an embedded quote cannot break out of the identifier.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Encode `ship` into IPC chunks bounded by `max_bytes`, pairing each with the largest
/// `seq` it carries. A single row that exceeds the budget is emitted alone so it cannot
/// stall every later row behind the forwarding cursor.
fn chunk_by_bytes(
    ship: &RecordBatch,
    seqs: &Int64Array,
    max_bytes: usize,
) -> Result<Vec<(Vec<u8>, i64)>, StatsError> {
    let num_rows = ship.num_rows();
    let mem = ship.get_array_memory_size().max(1);
    let per_row = (mem / num_rows.max(1)).max(1);
    let rows_per_chunk = (max_bytes / per_row).max(1);
    let schema = ship.schema();

    let mut out = Vec::new();
    let mut start = 0;
    while start < num_rows {
        let remaining = num_rows - start;
        let mut len = rows_per_chunk.min(num_rows - start);
        let mut largest_fit: Option<(usize, Vec<u8>)> = None;
        let mut smallest_oversized: Option<usize> = None;

        let (len, ipc) = loop {
            let ipc = encode_ipc(&schema, &[ship.slice(start, len)])
                .map_err(|e| StatsError::Internal(format!("encode ship chunk: {e}")))?;

            if ipc.len() <= max_bytes {
                largest_fit = Some((len, ipc));
                let lower = len + 1;
                let upper = smallest_oversized.unwrap_or(remaining + 1);
                if lower >= upper {
                    break largest_fit.unwrap();
                }

                let proportional = len
                    .saturating_mul(max_bytes)
                    .checked_div(largest_fit.as_ref().unwrap().1.len().max(1))
                    .unwrap_or(len)
                    .max(lower);
                len = if smallest_oversized.is_some() {
                    lower + (upper - lower) / 2
                } else {
                    proportional.min(remaining)
                };
                continue;
            }

            if len == 1 {
                break (len, ipc);
            }

            smallest_oversized = Some(len);
            let lower = largest_fit.as_ref().map_or(1, |(fit, _)| fit + 1);
            let upper = len;
            if lower >= upper {
                break largest_fit.expect("an oversized one-row chunk returns above");
            }
            let proportional = len.saturating_mul(max_bytes) / ipc.len();
            len = proportional.clamp(lower, upper - 1);
        };
        out.push((ipc, seqs.value(start + len - 1)));
        start += len;
    }
    Ok(out)
}

/// Cumulative forwarding counters, reported on a slow timer.
struct Progress {
    /// Requests the hub accepted, across every namespace. Counted in batches, not rows.
    batches: u64,
    /// `seq` positions the forwarder passed over and will never send — evicted before
    /// they shipped or in a batch the hub permanently refused.
    /// An upper bound on rows lost: some positions held rows the scan would have filtered
    /// out anyway.
    skipped_seqs: i64,
    last_report: Instant,
}

impl Progress {
    fn new() -> Self {
        Self {
            batches: 0,
            skipped_seqs: 0,
            last_report: Instant::now(),
        }
    }

    fn report(&mut self) {
        if self.last_report.elapsed() < PROGRESS_INTERVAL {
            return;
        }
        self.last_report = Instant::now();
        tracing::info!(
            batches = self.batches,
            skipped_seqs = self.skipped_seqs,
            "finelog forwarder: progress"
        );
    }
}

/// Start the forward loop on the runtime, returning its handle. The caller latches
/// `stop` and awaits the handle at shutdown.
pub fn spawn<T>(forwarder: Forwarder<T>, stop: watch::Receiver<bool>) -> JoinHandle<()>
where
    T: connectrpc::client::ClientTransport + Send + Sync + 'static,
    <T::ResponseBody as http_body::Body>::Error: std::fmt::Display,
{
    tokio::spawn(async move { forwarder.run(stop).await })
}

#[cfg(test)]
#[path = "forwarding_tests.rs"]
mod tests;
