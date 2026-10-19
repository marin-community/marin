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
//! batch-sized forwarding turn, with bounded concurrency across namespaces. It
//! immediately repeats while any namespace has more
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
//! # Relay durability
//!
//! A forwarding deployment is a durable relay spool. Retryable hub failures never
//! advance its cursor, and object-native maintenance retains every unsettled segment.
//! Once the hub settles a whole segment, maintenance may retire it from the current
//! state; normal snapshot, rollback, and orphan windows delay physical collection.
//!
//! - A new legacy cursor seeds at its current tip, preserving the historical behavior
//!   used when a node-local version-0 store is rebuilt from its archive. A new
//!   object-native cursor starts at the beginning of the live object state.
//! - It materializes at most [`FORWARD_LOG_BATCH_ROWS`] log rows or
//!   [`FORWARD_BATCH_ROWS`] other rows per read, and packs them into
//!   requests of at most [`FORWARD_BATCH_BYTES`] unless one row alone exceeds the limit.
//! - It advances a namespace's cursor after the hub acks the batch or permanently
//!   rejects malformed content. A crash or retryable rejection re-forwards it
//!   (at-least-once; tolerable for logs and append-only stats).
//! - A backlog is only a cursor into durable table segments, not a separate in-memory
//!   queue. The forwarder drains it without an age or row-count cap. It also skips a
//!   batch the hub identifies as permanently invalid, so one poison row cannot wedge
//!   every later row in that namespace.
//! - Node-local deployments acknowledge writes after object-state publication. A
//!   replacement resumes the periodic loop from that state and the durable cursor;
//!   shutdown does not run a separate forwarding drain.
//!
//! A retryable push failure leaves the cursor in place and yields to the next namespace;
//! nothing here can take the store down.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
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
use crate::policies::storage_policy_for;
use crate::proto::finelog::stats::{
    RegisterTableRequest, ReportRelayNamespaceStatus, ReportRelayStatusRequest, StatsServiceClient,
    WriteRowsRequest,
};
use crate::server::auth::FINELOG_AUDIENCE;
use crate::server::telemetry::{counter_batch, telemetry_schema, CounterSample};
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::forwarding::{read_forwarding, ReadBudget};
use crate::store::ipc::encode_ipc;
use crate::store::schema::{
    schema_to_proto_owned, Schema, IMPLICIT_CLUSTER_COLUMN, IMPLICIT_SEQ_COLUMN,
    MAX_WRITE_ROWS_BYTES,
};
use crate::store::store::LOG_NAMESPACE_NAME;
use crate::store::Store;
use crate::telemetry_policy::{FINELOG_NAMESPACE, TELEMETRY_NAMESPACE};

/// How long the forwarder waits after every namespace is caught up or unable to make
/// progress. Backlogged namespaces trigger another round immediately, so throughput
/// does not depend on this cadence.
const FORWARD_INTERVAL: Duration = Duration::from_secs(5);

/// Relay liveness travels independently of row delivery. This cadence is short enough
/// for a two-minute missing-heartbeat alert to tolerate several dropped reports.
const RELAY_STATUS_INTERVAL: Duration = Duration::from_secs(30);
const RELAY_STATUS_TIMEOUT: Duration = Duration::from_secs(10);

/// A relay's bounded internal scan may need to open a cold set of remote L0s.
/// Keep its recovery budget independent of the public Query RPC deadline.
const FORWARD_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// Retained Arrow bytes per read, excluding decoder, ordering, and IPC working space.
const FORWARD_READ_BYTES: usize = 256 << 20;

/// Source sequence positions examined per non-log turn. Larger windows amortize
/// source setup and the hub's durable acknowledgement; byte limits bound memory
/// and outbound messages separately.
const FORWARD_BATCH_ROWS: i64 = 200_000;

/// Dense log backlogs amortize each sweep's scan and durable acknowledgement over
/// more rows. Requests retain the byte cap and log chunks still ship serially.
const FORWARD_LOG_BATCH_ROWS: i64 = 1_000_000;

/// Independent tables can share the hub's flush window instead of waiting for
/// every other table's durable acknowledgement before starting their next turn.
const FORWARD_NAMESPACE_CONCURRENCY: usize = 4;

/// Maximum number of ordered segment ranges one forwarding scan intentionally spans.
/// Sparse relay tables can seal thousands of tiny L0s while the hub is unavailable;
/// bounding by sequence span alone can still localize all of them. Overlapping
/// ranges needed to cover the prefix may exceed this count.
const FORWARD_BATCH_SEGMENTS: usize = 128;

/// Encoded bytes per outbound request. Keep ten MiB below the receiving store's hard
/// Arrow IPC limit. [`chunk_by_bytes`] halves oversized batches until they fit.
pub const FORWARD_BATCH_BYTES: usize = MAX_WRITE_ROWS_BYTES - (10 << 20);

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

/// Deadline for one outbound request. Transient failures receive a few local retries
/// before the namespace yields, bounding how long one broken target can delay others.
const PUSH_TIMEOUT: Duration = Duration::from_secs(60);
const PUSH_ATTEMPTS_PER_TURN: usize = 3;
const PUSH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// Emit a progress line at most this often, so a healthy forwarder is observable
/// without flooding the logs it forwards.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(300);
const SLOW_FORWARD_TURN: Duration = Duration::from_secs(5);

const OUTCOME_ACCEPTED: &str = "accepted";
const OUTCOME_PERMANENT_REJECTION: &str = "permanent_rejection";
const OUTCOME_RETENTION_EVICTION: &str = "retention_eviction";
const OUTCOME_RETRYABLE_FAILURE: &str = "retryable_failure";
const OUTCOME_SCHEMA_CONFLICT: &str = "schema_conflict";
const ATTRIBUTE_NAMESPACE: &str = "namespace";
const ATTRIBUTE_OUTCOME: &str = "outcome";

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

/// Bounds for one namespace turn; `rows` caps its source sequence span.
pub fn forward_read_budget(namespace: &str) -> ReadBudget {
    ReadBudget {
        rows: if namespace == LOG_NAMESPACE_NAME {
            FORWARD_LOG_BATCH_ROWS as usize
        } else {
            FORWARD_BATCH_ROWS as usize
        },
        bytes: FORWARD_READ_BYTES,
        segments: FORWARD_BATCH_SEGMENTS,
        timeout: FORWARD_READ_TIMEOUT,
    }
}

/// Configure the forwarding wire codec, compression, and message limits.
pub fn forward_client_config(uri: http::Uri) -> ClientConfig {
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
    /// The last source schema registered for each namespace. A local additive schema
    /// change must be sent to the hub before rows using its new columns are written.
    registered: Mutex<HashMap<String, Schema>>,
    boot_id: String,
    report_sequence: AtomicU64,
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
    pub fn config(&self) -> &ForwardingConfig {
        &self.config
    }

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
            registered: Mutex::new(HashMap::new()),
            boot_id: uuid::Uuid::new_v4().to_string(),
            report_sequence: AtomicU64::new(1),
        }
    }

    /// Run until `stop` latches. Errors are logged and retried on the next tick; this
    /// never returns an error, because a store whose forwarding is broken must keep
    /// serving.
    pub async fn run(&self, stop: watch::Receiver<bool>) {
        tracing::info!(
            target = %self.config.target,
            cluster = %self.config.cluster,
            "finelog forwarder: started"
        );
        let forwarding = self.run_forwarding(stop.clone());
        let status = self.run_status_reports(stop);
        tokio::join!(forwarding, status);
        tracing::info!("finelog forwarder: stopped");
    }

    async fn run_forwarding(&self, mut stop: watch::Receiver<bool>) {
        if let Err(error) = self.ensure_progress_namespace().await {
            tracing::warn!(error = %error, "finelog forwarder: registering progress telemetry failed");
        }
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
    }

    async fn run_status_reports(&self, mut stop: watch::Receiver<bool>) {
        let mut interval = tokio::time::interval_at(
            tokio::time::Instant::now() + RELAY_STATUS_INTERVAL,
            RELAY_STATUS_INTERVAL,
        );
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = stop.changed() => break,
                _ = interval.tick() => {
                    if let Err(error) = self.report_status().await {
                        tracing::warn!(%error, "finelog forwarder: relay status report failed");
                    }
                }
            }
        }
    }

    async fn report_status(&self) -> Result<(), String> {
        let mut namespaces = Vec::new();
        for (name, _schema, stats, _policy) in self
            .store
            .list_namespaces_with_stats()
            .map_err(|error| error.to_string())?
        {
            let published_high_water = self
                .store
                .namespace_published_seq(&name)
                .map_err(|error| error.to_string())?;
            let settled_cursor = self
                .store
                .forward_cursor(&self.config.target, &name)
                .map_err(|error| error.to_string())?;
            namespaces.push(ReportRelayNamespaceStatus {
                namespace: Some(name),
                visible_high_water: Some(stats.max_seq),
                published_high_water: Some(published_high_water),
                settled_cursor,
                ..Default::default()
            });
        }
        let request = ReportRelayStatusRequest {
            boot_id: Some(self.boot_id.clone()),
            report_sequence: Some(self.report_sequence.fetch_add(1, Ordering::Relaxed)),
            target: Some(self.config.target.clone()),
            namespaces,
            ..Default::default()
        };
        let bearer = self.minter.bearer()?;
        let options = CallOptions::default()
            .with_timeout(RELAY_STATUS_TIMEOUT)
            .with_header("authorization", format!("Bearer {bearer}"));
        self.client
            .report_relay_status_with_options(request, options)
            .await
            .map_err(|error| error.to_string())?;
        Ok(())
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
        let turns = namespaces
            .into_iter()
            .map(|(name, schema, _stats, _policy)| {
                let mut namespace_stop = stop.clone();
                async move {
                    let mut namespace_progress = Progress::new();
                    if *namespace_stop.borrow() {
                        return (ForwardTurn::Wait, namespace_progress);
                    }
                    let turn = self
                        .forward_namespace(
                            &name,
                            &schema,
                            &mut namespace_progress,
                            &mut namespace_stop,
                        )
                        .await;
                    (turn, namespace_progress)
                }
            });
        let mut turns =
            futures::stream::iter(turns).buffer_unordered(FORWARD_NAMESPACE_CONCURRENCY);
        let mut turn = ForwardTurn::Wait;
        while let Some((namespace_turn, namespace_progress)) = turns.next().await {
            progress.batches += namespace_progress.batches;
            progress.skipped_seqs += namespace_progress.skipped_seqs;
            progress.pending.merge(namespace_progress.pending);
            if namespace_turn == ForwardTurn::MoreRows {
                turn = ForwardTurn::MoreRows;
            }
        }
        if let Some(deltas) = progress.take_report() {
            if let Err(error) = self.emit_progress(&deltas).await {
                progress.restore(deltas);
                tracing::warn!(error = %error, "finelog forwarder: writing progress telemetry failed");
            }
        }
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
        let mut cursor = match self.seed(name, persisted).await {
            Ok(cursor) => cursor,
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: cannot seed; skipping");
                return ForwardTurn::Wait;
            }
        };
        let starting_cursor = cursor;
        let object_native = match self.store.namespace_uses_object_state(name) {
            Ok(value) => value,
            Err(error) => {
                tracing::warn!(namespace = name, %error, "finelog forwarder: cannot resolve storage state; skipping");
                return ForwardTurn::Wait;
            }
        };
        let turn_started = Instant::now();
        self.observe_lag(name, cursor, persisted);
        if cursor >= persisted || *stop.borrow() {
            return ForwardTurn::Wait;
        }
        let read_started = Instant::now();
        let batch = match self.read_batch(name, cursor, persisted).await {
            Ok(batch) => batch,
            Err(e) => {
                tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: read failed; retrying next tick");
                return ForwardTurn::Wait;
            }
        };
        let read_elapsed = read_started.elapsed();
        // The scan reported its oldest locally-readable row. Legacy retention
        // may require a cursor jump; object-native relays fail closed because
        // their live spool must retain every unsettled row.
        if let Some(resume_at) = batch.resume_at {
            if object_native {
                tracing::error!(
                    namespace = name,
                    cursor,
                    resume_at,
                    persisted,
                    "finelog forwarder: object-native cursor precedes the live spool; refusing to skip rows"
                );
                return ForwardTurn::Wait;
            }
            let skipped = resume_at - cursor;
            progress.skipped_seqs += skipped;
            progress.record_seq_positions(name, OUTCOME_RETENTION_EVICTION, skipped);
            tracing::warn!(
                namespace = name,
                cursor,
                skipped,
                resume_at,
                "finelog forwarder: rows evicted before they were forwarded; skipping ahead"
            );
            cursor = resume_at;
            if !self
                .persist_cursor(name, starting_cursor, cursor, turn_started)
                .await
            {
                return ForwardTurn::Wait;
            }
        }
        let Some((ship, seqs)) = batch.rows else {
            if batch.scanned_through <= cursor {
                return ForwardTurn::Wait;
            }
            // Every row up to `scanned_through` was filtered out by the scan (rows already
            // carrying a foreign origin cluster). Advance the cursor to `scanned_through`,
            // or the loop rereads them forever. Safe against a concurrent writer:
            // `scanned_through` is a captured bound, and later rows arrive with a later
            // watermark.
            if !self
                .persist_cursor(name, starting_cursor, batch.scanned_through, turn_started)
                .await
            {
                return ForwardTurn::Wait;
            }
            return if batch.scanned_through < persisted {
                ForwardTurn::MoreRows
            } else {
                ForwardTurn::Wait
            };
        };
        // The hub must hold the namespace before it can take rows for it.
        if !self.ensure_registered(name, schema).await {
            tracing::warn!(
                namespace = name,
                "finelog forwarder: hub not ready for namespace; retrying next tick"
            );
            return ForwardTurn::Wait;
        }

        let row_count = ship.num_rows();
        let encoding_started = Instant::now();
        let chunks = match chunk_by_bytes(&ship, &seqs, FORWARD_BATCH_BYTES) {
            Ok(chunks) => chunks,
            Err(e) => {
                tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: encoding a batch failed; retrying next tick");
                return ForwardTurn::Wait;
            }
        };
        let encoding_elapsed = encoding_started.elapsed();
        let chunk_count = chunks.len();
        let encoded_bytes: usize = chunks.iter().map(|(ipc, _)| ipc.len()).sum();
        drop(ship);
        drop(seqs);
        let concurrency = if name == LOG_NAMESPACE_NAME {
            1
        } else {
            FORWARD_CHUNK_CONCURRENCY
        };
        let push_started = Instant::now();
        let pushes = chunks.into_iter().map(|(ipc, last_seq)| {
            let chunk_stop = (*stop).clone();
            async move { (last_seq, self.push(name, ipc, chunk_stop).await) }
        });
        // `buffered` polls several pushes at once but yields their results in input
        // order. The durable cursor therefore never advances past an earlier chunk that
        // is still retrying, even when a later chunk reaches the hub first.
        let mut pushes = futures::stream::iter(pushes).buffered(concurrency);
        while let Some((last_seq, result)) = pushes.next().await {
            let advanced = last_seq - cursor;
            match result {
                Ok(()) => {
                    progress.batches += 1;
                    progress.record_batch(name, OUTCOME_ACCEPTED);
                    progress.record_seq_positions(name, OUTCOME_ACCEPTED, advanced);
                }
                Err(PushError::Stopping(e)) => {
                    tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: push interrupted");
                    self.persist_accepted_prefix(name, starting_cursor, cursor, turn_started)
                        .await;
                    return ForwardTurn::Wait;
                }
                Err(PushError::Retryable(e)) => {
                    progress.record_batch(name, OUTCOME_RETRYABLE_FAILURE);
                    tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: push failed; retrying next sweep");
                    self.persist_accepted_prefix(name, starting_cursor, cursor, turn_started)
                        .await;
                    return ForwardTurn::Wait;
                }
                Err(PushError::SchemaConflict(e)) => {
                    // The same bytes may become valid after the hub's registration is
                    // refreshed or its binary is upgraded. Keep the rows owed.
                    progress.record_batch(name, OUTCOME_SCHEMA_CONFLICT);
                    self.registered.lock().unwrap().remove(name);
                    tracing::warn!(
                        namespace = name,
                        cursor,
                        error = %e,
                        "finelog forwarder: batch conflicts with the hub schema; preserving the cursor and refreshing registration next sweep"
                    );
                    self.persist_accepted_prefix(name, starting_cursor, cursor, turn_started)
                        .await;
                    return ForwardTurn::Wait;
                }
                Err(PushError::Permanent(e)) => {
                    let skipped = advanced;
                    progress.skipped_seqs += skipped;
                    progress.record_batch(name, OUTCOME_PERMANENT_REJECTION);
                    progress.record_seq_positions(name, OUTCOME_PERMANENT_REJECTION, skipped);
                    tracing::warn!(
                        namespace = name,
                        cursor,
                        skipped,
                        resume_at = last_seq,
                        error = %e,
                        "finelog forwarder: hub permanently rejected the batch; dropping it"
                    );
                }
            }
            cursor = last_seq;
        }
        if turn_started.elapsed() >= SLOW_FORWARD_TURN {
            tracing::info!(
                namespace = name,
                previous_cursor = starting_cursor,
                cursor,
                rows = row_count,
                chunks = chunk_count,
                encoded_bytes,
                read_ms = read_elapsed.as_millis() as u64,
                chunk_encode_ms = encoding_elapsed.as_millis() as u64,
                push_ms = push_started.elapsed().as_millis() as u64,
                "finelog forwarder: slow batch stage breakdown"
            );
        }
        // The reader proves coverage through this boundary, including filtered rows.
        // Settle that trailing prefix only after every outbound chunk has completed.
        if cursor < batch.scanned_through {
            cursor = batch.scanned_through;
        }
        if !self
            .persist_cursor(name, starting_cursor, cursor, turn_started)
            .await
        {
            return ForwardTurn::Wait;
        }
        if cursor < persisted {
            ForwardTurn::MoreRows
        } else {
            ForwardTurn::Wait
        }
    }

    /// The cursor to start `name` from.
    ///
    /// A stored watermark wins when it remains in range. A watermark beyond
    /// `persisted` resets to the current tip. Without a watermark, an
    /// object-native relay starts immediately before its oldest live sequence;
    /// a legacy table starts at the current tip.
    async fn seed(&self, name: &str, persisted: i64) -> Result<i64, StatsError> {
        match self.store.forward_cursor(&self.config.target, name)? {
            Some(cursor) if cursor <= persisted => Ok(cursor),
            Some(cursor) if self.store.namespace_uses_object_state(name)? => {
                Err(StatsError::Internal(format!(
                    "object-native forwarding cursor {cursor} is ahead of durable high-water {persisted} for {name:?}"
                )))
            }
            Some(cursor) => {
                tracing::warn!(
                    namespace = name,
                    cursor,
                    persisted,
                    "finelog forwarder: watermark is ahead of the store; reseeding at the tip"
                );
                self.persist(name, persisted).await?;
                Ok(persisted)
            }
            None => {
                if self.store.namespace_uses_object_state(name)? {
                    let cursor = self
                        .store
                        .query_snapshot(name)?
                        .min_seq
                        .map(|minimum| minimum.saturating_sub(1))
                        .unwrap_or(persisted);
                    tracing::info!(
                        namespace = name,
                        persisted,
                        cursor,
                        "finelog forwarder: no watermark for object-native relay; starting at the beginning"
                    );
                    self.persist(name, cursor).await?;
                    return Ok(cursor);
                }
                tracing::info!(
                    namespace = name,
                    persisted,
                    "finelog forwarder: no watermark for this target; seeding at the tip (new rows only)"
                );
                self.persist(name, persisted).await?;
                Ok(persisted)
            }
        }
    }

    /// Persist one interval of delta counters locally. The ordinary forwarder carries
    /// `telemetry_v1.finelog` to the hub on a later sweep.
    async fn emit_progress(&self, deltas: &ForwardingDeltas) -> Result<(), StatsError> {
        self.ensure_progress_namespace().await?;
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| {
                StatsError::Internal(format!("system clock is before the unix epoch: {error}"))
            })?
            .as_millis() as i64;
        let batch = counter_batch("finelog", timestamp_ms, deltas.samples())?;
        let ipc = encode_ipc(&batch.schema(), &[batch]).map_err(|error| {
            StatsError::Internal(format!("encoding forwarding telemetry: {error}"))
        })?;
        let outcome = self
            .store
            .write_ingestion_rows(TELEMETRY_NAMESPACE, &ipc, None)?;
        for (namespace, last_seq) in outcome.persisted_targets {
            self.store
                .await_persisted(&namespace, last_seq, PUSH_TIMEOUT)
                .await?;
        }
        Ok(())
    }

    async fn ensure_progress_namespace(&self) -> Result<(), StatsError> {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || {
            store.register_managed_table(
                FINELOG_NAMESPACE,
                telemetry_schema(),
                storage_policy_for(FINELOG_NAMESPACE)?,
            )?;
            Ok(())
        })
        .await
        .map_err(|error| StatsError::Internal(format!("progress namespace task failed: {error}")))?
    }

    async fn persist(&self, name: &str, cursor: i64) -> Result<(), StatsError> {
        self.store
            .set_forward_cursor(&self.config.target, name, cursor)
            .await
    }

    /// Commit `cursor` and retire object-native relay segments it covers.
    ///
    /// `false` leaves the previous durable cursor selected, so the next sweep
    /// retries the same unsettled prefix.
    async fn persist_cursor(
        &self,
        name: &str,
        previous_cursor: i64,
        cursor: i64,
        turn_started: Instant,
    ) -> bool {
        let settlement_started = Instant::now();
        let settlement = match self
            .store
            .settle_forwarding(&self.config.target, name, cursor)
            .await
        {
            Ok(settlement) => settlement,
            Err(e) => {
                tracing::warn!(namespace = name, cursor, error = %e, "finelog forwarder: persisting the watermark failed");
                return false;
            }
        };
        let total = turn_started.elapsed();
        let settlement_elapsed = settlement_started.elapsed();
        tracing::debug!(
            namespace = name,
            previous_cursor,
            cursor = settlement.cursor,
            advanced_seqs = settlement.cursor.saturating_sub(previous_cursor),
            retired_segments = settlement.removed_paths.len(),
            retired_rows = settlement.removed_rows,
            retired_bytes = settlement.removed_bytes,
            settle_ms = settlement_elapsed.as_millis() as u64,
            total_ms = total.as_millis() as u64,
            "finelog forwarder: settled namespace turn"
        );
        if total >= SLOW_FORWARD_TURN {
            tracing::info!(
                namespace = name,
                previous_cursor,
                cursor = settlement.cursor,
                persisted = self.store.namespace_persisted_seq(name).unwrap_or(cursor),
                advanced_seqs = settlement.cursor.saturating_sub(previous_cursor),
                retired_segments = settlement.removed_paths.len(),
                retired_rows = settlement.removed_rows,
                retired_bytes = settlement.removed_bytes,
                settle_ms = settlement_elapsed.as_millis() as u64,
                total_ms = total.as_millis() as u64,
                "finelog forwarder: slow namespace turn completed"
            );
        }
        true
    }

    async fn persist_accepted_prefix(
        &self,
        name: &str,
        starting_cursor: i64,
        cursor: i64,
        turn_started: Instant,
    ) {
        if cursor > starting_cursor {
            self.persist_cursor(name, starting_cursor, cursor, turn_started)
                .await;
        }
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

    /// Read local-origin rows in `(cursor, persisted]` with their source sequences
    /// and examined boundary. `resume_at` identifies a legacy retention gap.
    async fn read_batch(
        &self,
        name: &str,
        cursor: i64,
        persisted: i64,
    ) -> Result<Batch, StatsError> {
        let read = read_forwarding(
            Arc::clone(&self.store),
            name,
            cursor,
            persisted,
            forward_read_budget(name),
        )
        .await?;
        Ok(Batch {
            rows: ship_batch(read.schema, read.batches, &self.config.cluster)?,
            resume_at: read.resume_at,
            scanned_through: read.scanned_through,
        })
    }

    /// Create or evolve `name` on the hub when this process has not registered its
    /// current source schema. The reserved `log` namespace exists on every finelog, so
    /// it is never registered. Returns whether the hub is ready to take rows for `name`.
    async fn ensure_registered(&self, name: &str, schema: &Schema) -> bool {
        if name == LOG_NAMESPACE_NAME || self.registered.lock().unwrap().get(name) == Some(schema) {
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
                self.registered
                    .lock()
                    .unwrap()
                    .insert(name.to_string(), schema.clone());
                true
            }
            Err(e) => {
                tracing::warn!(namespace = name, error = %e, "finelog forwarder: registering the namespace on the hub failed");
                false
            }
        }
    }

    /// Ship one encoded chunk to `name` on the hub and wait for it to durably ack.
    /// Retry transient failures at most [`PUSH_ATTEMPTS_PER_TURN`] times, then return the
    /// chunk still owed so the caller can give the next namespace its turn.
    ///
    /// Returns [`PushError::Permanent`] when the hub identifies invalid content,
    /// [`PushError::SchemaConflict`] when registration state may make the same bytes
    /// valid later, [`PushError::Retryable`] for other conditions that may clear, and
    /// [`PushError::Stopping`] when `stop` latches.
    async fn push(
        &self,
        name: &str,
        arrow_ipc: Vec<u8>,
        mut stop: watch::Receiver<bool>,
    ) -> Result<(), PushError> {
        let bytes = arrow_ipc.len();
        let request = WriteRowsRequest::default()
            .with_namespace(name)
            .with_arrow_ipc(arrow_ipc);

        let mut backoff = PUSH_RETRY_BACKOFF;
        for attempt in 1..=PUSH_ATTEMPTS_PER_TURN {
            if *stop.borrow() {
                return Err(PushError::Stopping("shutdown requested".to_string()));
            }
            let attempt_started = Instant::now();
            let result = match self.minter.bearer() {
                Ok(bearer) => {
                    let options = CallOptions::default()
                        .with_timeout(PUSH_TIMEOUT)
                        .with_header("authorization", format!("Bearer {bearer}"));
                    match self
                        .client
                        .write_rows_with_options(request.clone(), options)
                        .await
                    {
                        Ok(_) => {
                            let elapsed = attempt_started.elapsed();
                            if elapsed >= SLOW_FORWARD_TURN {
                                tracing::info!(
                                    namespace = name,
                                    attempt,
                                    bytes,
                                    push_ms = elapsed.as_millis() as u64,
                                    "finelog forwarder: slow hub durability acknowledgement completed"
                                );
                            } else {
                                tracing::debug!(
                                    namespace = name,
                                    attempt,
                                    bytes,
                                    push_ms = elapsed.as_millis() as u64,
                                    "finelog forwarder: batch delivered"
                                );
                            }
                            return Ok(());
                        }
                        Err(e) if is_permanent_rejection(&e) => {
                            Err(PushError::Permanent(e.to_string()))
                        }
                        Err(e) if is_schema_conflict(&e) => {
                            Err(PushError::SchemaConflict(e.to_string()))
                        }
                        Err(e) if *stop.borrow() => Err(PushError::Stopping(e.to_string())),
                        Err(e) => Err(PushError::Retryable(e.to_string())),
                    }
                }
                Err(e) => Err(PushError::Retryable(e)),
            };
            match result {
                Err(PushError::Retryable(error)) if attempt < PUSH_ATTEMPTS_PER_TURN => {
                    tracing::warn!(
                        namespace = name,
                        error = %error,
                        attempt,
                        max_attempts = PUSH_ATTEMPTS_PER_TURN,
                        backoff_seconds = backoff.as_secs(),
                        "finelog forwarder: push failed; retrying"
                    );
                    if wait_or_stop(backoff, &mut stop).await {
                        return Err(PushError::Stopping(error));
                    }
                    backoff *= 2;
                }
                result => return result,
            }
        }
        unreachable!("the bounded push loop always returns on its last attempt")
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
    /// The forwarder is shutting down. The chunk is still owed; the cursor stays where
    /// it is and the next process re-reads it.
    Stopping(String),
    /// The request may succeed later. The chunk is still owed, and the next namespace
    /// must receive its turn before this one retries.
    Retryable(String),
    /// The batch conflicts with the hub's current schema. The caller refreshes
    /// registration and preserves the cursor because the same bytes may become valid.
    SchemaConflict(String),
    /// The hub identified invalid content that cannot succeed when resent unchanged.
    Permanent(String),
}

fn is_permanent_rejection(error: &connectrpc::ConnectError) -> bool {
    error.code == connectrpc::error::ErrorCode::InvalidArgument
}

fn is_schema_conflict(error: &connectrpc::ConnectError) -> bool {
    error.code == connectrpc::error::ErrorCode::FailedPrecondition
}

/// Sleep between transient attempts. Returns `true` when shutdown latches or its sender
/// closes, and `false` when the delay elapses.
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
    /// Highest sequence position included in this physical scan window.
    scanned_through: i64,
}

/// Turn a scan result into a batch ready for [`WriteRows`]: concatenate the collected
/// batches, drop the server-assigned `seq` column (the hub assigns its own), and
/// stamp the origin column with this store's cluster. `None` when the scan matched no
/// rows. Returns the dropped seqs alongside, so the caller can advance its cursor.
pub fn ship_batch(
    schema: Arc<ArrowSchema>,
    mut batches: Vec<RecordBatch>,
    cluster: &str,
) -> Result<Option<(RecordBatch, Int64Array)>, StatsError> {
    if batches.is_empty() {
        return Ok(None);
    }
    let batch = if batches.len() == 1 {
        batches.pop().expect("one source batch")
    } else {
        concat_batches(&schema, &batches)
            .map_err(|e| StatsError::Internal(format!("concat batches: {e}")))?
    };
    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(None);
    }

    let seq_idx = schema
        .index_of(IMPLICIT_SEQ_COLUMN)
        .map_err(|e| StatsError::Internal(format!("scan result has no seq column: {e}")))?;
    let seqs: Int64Array = batch.column(seq_idx).as_primitive::<Int64Type>().clone();

    let origin: ArrayRef = Arc::new(StringArray::from(vec![cluster; num_rows]));
    let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns() - 1);
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns() - 1);
    for (i, field) in schema.fields().iter().enumerate() {
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

/// Encode `ship` into IPC chunks bounded by `max_bytes`, pairing each with the largest
/// `seq` it carries. A single row that exceeds the budget is emitted alone so it cannot
/// stall every later row behind the forwarding cursor.
pub fn chunk_by_bytes(
    ship: &RecordBatch,
    seqs: &Int64Array,
    max_bytes: usize,
) -> Result<Vec<(Vec<u8>, i64)>, StatsError> {
    if ship.num_rows() == 0 {
        return Ok(Vec::new());
    }
    let schema = ship.schema();
    let mut chunks = Vec::new();
    let mut pending = vec![(0, ship.num_rows())];
    while let Some((start, len)) = pending.pop() {
        let ipc = encode_ipc(&schema, &[ship.slice(start, len)])
            .map_err(|e| StatsError::Internal(format!("encode ship chunk: {e}")))?;
        if ipc.len() <= max_bytes || len == 1 {
            chunks.push((ipc, seqs.value(start + len - 1)));
            continue;
        }
        let half = len / 2;
        // Visit the left half first to preserve source sequence order.
        pending.push((start + half, len - half));
        pending.push((start, half));
    }
    Ok(chunks)
}

/// Cumulative forwarding counters, reported on a slow timer.
struct Progress {
    /// Requests the hub accepted, across every namespace. Counted in batches, not rows.
    batches: u64,
    /// `seq` positions the forwarder passed over because local retention evicted them or
    /// the hub permanently rejected their batch. This is an upper bound on rows lost:
    /// some positions held rows the scan would have filtered out anyway.
    skipped_seqs: i64,
    pending: ForwardingDeltas,
    last_report: Instant,
}

impl Progress {
    fn new() -> Self {
        Self {
            batches: 0,
            skipped_seqs: 0,
            pending: ForwardingDeltas::default(),
            last_report: Instant::now(),
        }
    }

    fn record_batch(&mut self, namespace: &str, outcome: &'static str) {
        *self
            .pending
            .batches
            .entry((namespace.to_string(), outcome))
            .or_default() += 1;
    }

    fn record_seq_positions(&mut self, namespace: &str, outcome: &'static str, count: i64) {
        let count = u64::try_from(count).expect("forwarding sequence deltas are nonnegative");
        *self
            .pending
            .seq_positions
            .entry((namespace.to_string(), outcome))
            .or_default() += count;
    }

    fn take_report(&mut self) -> Option<ForwardingDeltas> {
        if self.last_report.elapsed() < PROGRESS_INTERVAL {
            return None;
        }
        self.last_report = Instant::now();
        tracing::info!(
            batches = self.batches,
            skipped_seqs = self.skipped_seqs,
            "finelog forwarder: progress"
        );
        if self.pending.is_empty() {
            return None;
        }
        Some(std::mem::take(&mut self.pending))
    }

    fn restore(&mut self, deltas: ForwardingDeltas) {
        self.pending.merge(deltas);
    }
}

#[derive(Default)]
struct ForwardingDeltas {
    batches: BTreeMap<(String, &'static str), u64>,
    seq_positions: BTreeMap<(String, &'static str), u64>,
}

impl ForwardingDeltas {
    fn is_empty(&self) -> bool {
        self.batches.is_empty() && self.seq_positions.is_empty()
    }

    fn samples(&self) -> Vec<CounterSample> {
        self.batches
            .iter()
            .map(|((namespace, outcome), value)| CounterSample {
                name: "forwarding_batches".to_string(),
                value: *value as f64,
                unit: "batches".to_string(),
                attributes: forwarding_attributes(namespace, outcome),
            })
            .chain(
                self.seq_positions
                    .iter()
                    .map(|((namespace, outcome), value)| CounterSample {
                        name: "forwarding_seq_positions".to_string(),
                        value: *value as f64,
                        unit: "positions".to_string(),
                        attributes: forwarding_attributes(namespace, outcome),
                    }),
            )
            .collect()
    }

    fn merge(&mut self, other: Self) {
        for (key, value) in other.batches {
            *self.batches.entry(key).or_default() += value;
        }
        for (key, value) in other.seq_positions {
            *self.seq_positions.entry(key).or_default() += value;
        }
    }
}

fn forwarding_attributes(namespace: &str, outcome: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (ATTRIBUTE_NAMESPACE.to_string(), namespace.to_string()),
        (ATTRIBUTE_OUTCOME.to_string(), outcome.to_string()),
    ])
}

/// Start the forward loop, which runs until `stop` changes or closes.
pub fn spawn<T>(forwarder: Arc<Forwarder<T>>, stop: watch::Receiver<bool>) -> JoinHandle<()>
where
    T: connectrpc::client::ClientTransport + Send + Sync + 'static,
    <T::ResponseBody as http_body::Body>::Error: std::fmt::Display,
{
    tokio::spawn(async move { forwarder.run(stop).await })
}

#[cfg(test)]
#[path = "forwarding_tests.rs"]
mod tests;
