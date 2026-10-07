//! `LogService` trait impl: PushLogs and FetchLogs.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use buffa::{Enumeration, MessageField};
use connectrpc::{ConnectError, RequestContext, ServiceResult};

use crate::errors::StatsError;
use crate::proto::finelog::logging::{
    FetchLogsResponse, LogEntry, LogEntryView, LogLevel, LogService, MatchScope,
    OwnedFetchLogsRequestView, OwnedPushLogsRequestView, PushLogsResponse, Timestamp,
};
use crate::query::fetch_log_rows;
use crate::query::make_ctx;
use crate::query::{query_timeout, run_within_query_timeout, slow_query_log_ms};
use crate::server::auth::{request_identity, AuthIdentity};
use crate::store::log_read::{
    add_cluster_filter, add_common_filters, add_seq_upper_bound, build_log_predicates,
    shape_log_read_result, str_to_log_level, LogPredicates, LogRow, ShapedEntry,
};
use crate::store::store::{NamespaceSnapshot, LOG_NAMESPACE_NAME};
use crate::store::table::ingest::DEFAULT_PERSIST_TIMEOUT;
use crate::store::Store;

/// Server default for `max_lines` when the request leaves it unset/<=0.
const DEFAULT_MAX_LINES: i32 = 1000;

/// Process-local budget for repeated bounded tail reads. Cached rows are log
/// payloads, so a byte limit is more useful than an entry count.
const TAIL_CACHE_BUDGET_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TailCacheKey {
    scope: i32,
    source: String,
    since_ms: i64,
    substring: String,
    regex: String,
    min_level: i32,
    cluster: String,
    max_lines: i32,
}

#[derive(Clone)]
struct TailCacheEntry {
    rows_descending: Vec<LogRow>,
    scanned_through: i64,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct TailCacheState {
    entries: HashMap<TailCacheKey, TailCacheEntry>,
    bytes: usize,
    tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TailCacheStats {
    pub entries: usize,
    pub bytes: usize,
    pub budget_bytes: usize,
    pub hits: u64,
    pub delta_scans: u64,
    pub misses: u64,
    pub evictions: u64,
}

pub struct LogTailCache {
    state: Mutex<TailCacheState>,
    budget_bytes: usize,
    hits: AtomicU64,
    delta_scans: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

enum TailCacheLookup {
    Hit(Vec<LogRow>),
    Delta {
        scanned_through: i64,
        rows_descending: Vec<LogRow>,
    },
    Miss,
}

impl LogTailCache {
    pub(crate) fn new() -> Self {
        Self::with_budget_bytes(TAIL_CACHE_BUDGET_BYTES)
    }

    fn with_budget_bytes(budget_bytes: usize) -> Self {
        Self {
            state: Mutex::new(TailCacheState::default()),
            budget_bytes,
            hits: AtomicU64::new(0),
            delta_scans: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    fn lookup(&self, key: &TailCacheKey, minimum_seq: i64, maximum_seq: i64) -> TailCacheLookup {
        let mut state = self.state.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        let Some(scanned_through) = state.entries.get(key).map(|entry| entry.scanned_through)
        else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return TailCacheLookup::Miss;
        };
        if maximum_seq < scanned_through {
            let removed = state.entries.remove(key).expect("entry exists");
            state.bytes -= removed.bytes;
            self.misses.fetch_add(1, Ordering::Relaxed);
            return TailCacheLookup::Miss;
        }
        let entry = state.entries.get_mut(key).expect("entry exists");
        entry.last_used = tick;
        let retained_rows = entry
            .rows_descending
            .iter()
            .filter(|row| row.seq >= minimum_seq)
            .cloned()
            .collect();
        if maximum_seq == entry.scanned_through {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return TailCacheLookup::Hit(retained_rows);
        }
        self.delta_scans.fetch_add(1, Ordering::Relaxed);
        TailCacheLookup::Delta {
            scanned_through: entry.scanned_through,
            rows_descending: retained_rows,
        }
    }

    fn insert(&self, key: TailCacheKey, maximum_seq: i64, rows_descending: Vec<LogRow>) {
        let bytes = tail_cache_entry_bytes(&key, &rows_descending);
        if bytes > self.budget_bytes {
            return;
        }
        let mut state = self.state.lock().unwrap();
        if state
            .entries
            .get(&key)
            .is_some_and(|entry| entry.scanned_through > maximum_seq)
        {
            return;
        }
        state.tick += 1;
        let tick = state.tick;
        if let Some(old) = state.entries.remove(&key) {
            state.bytes -= old.bytes;
        }
        while state.bytes.saturating_add(bytes) > self.budget_bytes {
            let Some(victim) = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let removed = state.entries.remove(&victim).expect("victim exists");
            state.bytes -= removed.bytes;
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }
        state.bytes += bytes;
        state.entries.insert(
            key,
            TailCacheEntry {
                rows_descending,
                scanned_through: maximum_seq,
                bytes,
                last_used: tick,
            },
        );
    }

    pub(crate) fn stats(&self) -> TailCacheStats {
        let state = self.state.lock().unwrap();
        TailCacheStats {
            entries: state.entries.len(),
            bytes: state.bytes,
            budget_bytes: self.budget_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            delta_scans: self.delta_scans.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }
}

fn tail_cache_entry_bytes(key: &TailCacheKey, rows: &[LogRow]) -> usize {
    let key_bytes = key.source.len() + key.substring.len() + key.regex.len() + key.cluster.len();
    key_bytes
        + rows
            .iter()
            .map(|row| {
                row.key.as_ref().map_or(0, String::len)
                    + row.source.len()
                    + row.data.len()
                    + std::mem::size_of::<LogRow>()
            })
            .sum::<usize>()
}

struct TailCacheRequest<'a> {
    scope: MatchScope,
    source: &'a str,
    cursor: i64,
    until_cursor: i64,
    since_ms: i64,
    substring: &'a str,
    regex: &'a str,
    min_level: LogLevel,
    cluster: &'a str,
    tail: bool,
    max_lines: i32,
}

fn tail_cache_key(request: TailCacheRequest<'_>) -> Option<TailCacheKey> {
    if !request.tail || request.cursor != 0 || request.until_cursor != 0 {
        return None;
    }
    Some(TailCacheKey {
        scope: request.scope.to_i32(),
        source: request.source.to_string(),
        since_ms: request.since_ms,
        substring: request.substring.to_string(),
        regex: request.regex.to_string(),
        min_level: request.min_level.to_i32(),
        cluster: request.cluster.to_string(),
        max_lines: request.max_lines,
    })
}

struct PreparedFetchLogs {
    cursor: i64,
    tail: bool,
    max_lines: i32,
    predicates: LogPredicates,
    cache_key: Option<TailCacheKey>,
}

fn prepare_fetch_logs(
    request: OwnedFetchLogsRequestView,
) -> Result<PreparedFetchLogs, ConnectError> {
    // Wire UNSPECIFIED (and an unset field) maps to REGEX so clients that
    // encode a regex pattern in `source` without setting match_scope keep
    // working. New callers set EXACT/PREFIX explicitly.
    let scope = match request.match_scope.and_then(|value| value.as_known()) {
        Some(MatchScope::MATCH_SCOPE_UNSPECIFIED) | None => MatchScope::MATCH_SCOPE_REGEX,
        Some(scope) => scope,
    };
    let source = request.source.unwrap_or("");
    let cursor = request.cursor.unwrap_or(0);
    let until_cursor = request.until_cursor.unwrap_or(0);
    let since_ms = request.since_ms.unwrap_or(0);
    let substring = request.substring.unwrap_or("");
    let regex = request.regex.unwrap_or("");
    let tail = request.tail.unwrap_or(false);
    let min_level = str_to_log_level(request.min_level.unwrap_or(""));
    let cluster = request.cluster.unwrap_or("");
    let raw_max_lines = request.max_lines.unwrap_or(0);
    let max_lines = if raw_max_lines > 0 {
        raw_max_lines
    } else {
        DEFAULT_MAX_LINES
    };
    let cache_key = tail_cache_key(TailCacheRequest {
        scope,
        source,
        cursor,
        until_cursor,
        since_ms,
        substring,
        regex,
        min_level,
        cluster,
        tail,
        max_lines,
    });

    let mut predicates =
        build_log_predicates(source, cursor, scope).map_err(ConnectError::invalid_argument)?;
    add_seq_upper_bound(&mut predicates.where_parts, until_cursor);
    add_common_filters(
        &mut predicates.where_parts,
        since_ms,
        substring,
        regex,
        min_level,
    );
    add_cluster_filter(&mut predicates.where_parts, cluster);

    Ok(PreparedFetchLogs {
        cursor,
        tail,
        max_lines,
        predicates,
        cache_key,
    })
}

fn response_from_rows(
    rows: Vec<LogRow>,
    tail: bool,
    max_lines: i32,
    cursor: i64,
    include_key: bool,
    exact_key: Option<&str>,
) -> ServiceResult<FetchLogsResponse> {
    let shaped = shape_log_read_result(rows, tail, max_lines, cursor, include_key, exact_key);
    let entries: Vec<LogEntry> = shaped
        .entries
        .into_iter()
        .map(shaped_entry_to_proto)
        .collect();
    connectrpc::Response::ok(
        FetchLogsResponse {
            entries,
            ..Default::default()
        }
        .with_cursor(shaped.cursor),
    )
}

fn merge_tail_rows(
    mut delta_rows_descending: Vec<LogRow>,
    cached_rows_descending: Vec<LogRow>,
    max_lines: i32,
) -> Vec<LogRow> {
    delta_rows_descending.extend(cached_rows_descending);
    delta_rows_descending.truncate(max_lines as usize);
    delta_rows_descending
}

#[derive(Default)]
struct FetchLogsStageTimings {
    visibility_wait: Duration,
    snapshot: Duration,
    cache_lookup: Duration,
    provider: Duration,
    query: Duration,
    response: Duration,
}

fn fetch_logs_response(
    rows: Vec<LogRow>,
    request: &PreparedFetchLogs,
    request_started: Instant,
    mut timings: FetchLogsStageTimings,
    cache_outcome: &str,
) -> ServiceResult<FetchLogsResponse> {
    let row_count = rows.len();
    let response_started = Instant::now();
    let response = response_from_rows(
        rows,
        request.tail,
        request.max_lines,
        request.cursor,
        request.predicates.include_key,
        request.predicates.exact_key.as_deref(),
    );
    timings.response = response_started.elapsed();
    log_fetch_logs_stages(request_started.elapsed(), timings, cache_outcome, row_count);
    response
}

struct QueriedLogRows {
    rows: Vec<LogRow>,
    provider_elapsed: Duration,
    query_elapsed: Duration,
}

fn log_fetch_logs_stages(
    total: Duration,
    timings: FetchLogsStageTimings,
    cache_outcome: &str,
    rows: usize,
) {
    if total.as_millis() < slow_query_log_ms() {
        return;
    }
    tracing::warn!(
        total_ms = total.as_millis() as u64,
        visibility_wait_ms = timings.visibility_wait.as_millis() as u64,
        snapshot_ms = timings.snapshot.as_millis() as u64,
        tail_cache_lookup_ms = timings.cache_lookup.as_millis() as u64,
        provider_ms = timings.provider.as_millis() as u64,
        query_ms = timings.query.as_millis() as u64,
        response_ms = timings.response.as_millis() as u64,
        tail_cache = cache_outcome,
        rows,
        "slow FetchLogs RPC stage breakdown",
    );
}

/// The origin cluster to stamp on a push, bound to the credential that carried it.
///
/// A token names the one cluster its key authenticates, so a token-bearing writer
/// may claim that cluster or say nothing and have it filled in; claiming a
/// different one is `permission_denied`. Without that check any holder of any
/// trusted key could write rows attributed to any peer. A writer admitted by a
/// trusted network carries no per-writer identity and names its own origin (empty
/// for the ordinary local push).
fn authorized_cluster<'a>(
    ctx: &'a RequestContext,
    requested: &'a str,
) -> Result<&'a str, ConnectError> {
    match request_identity(ctx) {
        Some(AuthIdentity::Jwt { cluster }) => {
            if requested.is_empty() || requested == cluster {
                Ok(cluster)
            } else {
                Err(ConnectError::permission_denied(format!(
                    "finelog: token authenticates cluster {cluster:?}, not {requested:?}"
                )))
            }
        }
        Some(AuthIdentity::Network) => Ok(requested),
        None => Err(ConnectError::internal(
            "finelog: request reached a handler with no auth identity",
        )),
    }
}

/// Run a blocking store closure on the blocking pool, mapping a JoinError to an
/// internal ConnectError and a StatsError to its mapped code.
async fn run_blocking<T, F>(f: F) -> Result<T, ConnectError>
where
    F: FnOnce() -> Result<T, StatsError> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.into()),
        Err(join) => Err(ConnectError::internal(format!(
            "store task panicked: {join}"
        ))),
    }
}

pub struct LogServiceImpl {
    store: Arc<Store>,
    tail_cache: Arc<LogTailCache>,
}

impl LogServiceImpl {
    pub fn new(store: Arc<Store>, tail_cache: Arc<LogTailCache>) -> Self {
        Self { store, tail_cache }
    }

    async fn query_log_rows(
        &self,
        ctx: &RequestContext,
        snapshot: NamespaceSnapshot,
        request: &PreparedFetchLogs,
    ) -> Result<QueriedLogRows, ConnectError> {
        let table_bound = self.store.object_query_bound();
        let provider_started = Instant::now();
        let provider = self
            .store
            .namespace_provider(LOG_NAMESPACE_NAME, snapshot)
            .map_err(|error| ConnectError::internal(format!("build log provider: {error}")))?;
        let provider_elapsed = provider_started.elapsed();

        let query_ctx = make_ctx();
        let read = fetch_log_rows(
            &query_ctx,
            provider,
            &request.predicates.where_parts,
            request.predicates.include_key,
            request.tail,
            request.max_lines,
        );
        let query_started = Instant::now();
        let rows = run_within_query_timeout(
            query_timeout(ctx.time_remaining(), table_bound),
            read,
            |timeout| {
                ConnectError::deadline_exceeded(format!(
                    "log read exceeded deadline of {} ms",
                    timeout.as_millis()
                ))
            },
            |error| ConnectError::internal(format!("log read failed: {error}")),
        )
        .await?;
        Ok(QueriedLogRows {
            rows,
            provider_elapsed,
            query_elapsed: query_started.elapsed(),
        })
    }

    /// Append the prepared columns and return once they are durable, so a push
    /// acks only after the rows survive a crash.
    async fn append_and_persist(
        &self,
        columns: LogColumns,
        ctx: &RequestContext,
    ) -> Result<(), ConnectError> {
        let store = Arc::clone(&self.store);
        let last_seq = run_blocking(move || {
            store.append_log_columns(columns.columns, columns.num_rows, columns.byte_size)
        })
        .await?;

        let budget = ctx.time_remaining().unwrap_or(DEFAULT_PERSIST_TIMEOUT);
        self.store
            .await_persisted(LOG_NAMESPACE_NAME, last_seq, budget)
            .await?;
        Ok(())
    }
}

/// The six non-seq log columns built from pushed entries, plus their byte size.
/// Prepared before the namespace insertion lock is taken.
struct LogColumns {
    columns: Vec<ArrayRef>,
    num_rows: usize,
    byte_size: i64,
}

fn array_buffer_size(arr: &ArrayRef) -> i64 {
    arr.to_data().buffers().iter().map(|b| b.len() as i64).sum()
}

/// One pushed row's fields, borrowed from the request view.
struct EntryFields<'a> {
    key: &'a str,
    source: &'a str,
    data: &'a str,
    epoch_ms: i64,
    level: i32,
}

/// Project a wire entry onto the stored columns under `key`. `attempt_id` is not
/// among them: it is parsed back out of the key on read, never stored.
fn entry_fields<'a>(entry: &LogEntryView<'a>, key: &'a str) -> EntryFields<'a> {
    EntryFields {
        key,
        source: entry.source.unwrap_or(""),
        data: entry.data.unwrap_or(""),
        epoch_ms: entry
            .timestamp
            .as_option()
            .and_then(|t| t.epoch_ms)
            .unwrap_or(0),
        level: entry.level.map(|ev| ev.to_i32()).unwrap_or(0),
    }
}

/// Assemble the store's six log columns from `rows`, stamping every row with
/// `cluster` (one value for the whole push: it is the writer's identity, not a
/// per-entry field).
fn build_log_columns(rows: Vec<EntryFields<'_>>, cluster: &str) -> LogColumns {
    let num_rows = rows.len();
    let mut keys: Vec<&str> = Vec::with_capacity(num_rows);
    let mut sources: Vec<&str> = Vec::with_capacity(num_rows);
    let mut datas: Vec<&str> = Vec::with_capacity(num_rows);
    let mut epoch_ms: Vec<i64> = Vec::with_capacity(num_rows);
    let mut levels: Vec<i32> = Vec::with_capacity(num_rows);
    for row in rows {
        keys.push(row.key);
        sources.push(row.source);
        datas.push(row.data);
        epoch_ms.push(row.epoch_ms);
        levels.push(row.level);
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(keys)),
        Arc::new(StringArray::from(sources)),
        Arc::new(StringArray::from(datas)),
        Arc::new(Int64Array::from(epoch_ms)),
        Arc::new(Int32Array::from(levels)),
        Arc::new(StringArray::from(vec![cluster; num_rows])),
    ];
    let byte_size: i64 = columns.iter().map(array_buffer_size).sum();
    LogColumns {
        columns,
        num_rows,
        byte_size,
    }
}

// Naming the concrete `ServiceResult<T>` return type refines the trait's
// `impl Encodable<T> + Send`; that is intentional (see stats_service.rs).
#[allow(refining_impl_trait)]
impl LogService for LogServiceImpl {
    async fn push_logs(
        &self,
        ctx: RequestContext,
        request: OwnedPushLogsRequestView,
    ) -> ServiceResult<PushLogsResponse> {
        // Empty entries -> empty response, no append.
        if request.entries.is_empty() {
            return connectrpc::Response::ok(PushLogsResponse::default());
        }

        // Every ingested row is tagged with its origin cluster, so a global finelog
        // that collects pushes from many federated clusters can namespace them by
        // origin. The value is bound to the writer's credential.
        let cluster = authorized_cluster(&ctx, request.cluster.unwrap_or(""))?;

        let key = request.key.unwrap_or("");
        let rows = request
            .entries
            .iter()
            .map(|entry| entry_fields(entry, key))
            .collect();
        let log_columns = build_log_columns(rows, cluster);
        self.append_and_persist(log_columns, &ctx).await?;

        connectrpc::Response::ok(PushLogsResponse::default())
    }

    async fn fetch_logs(
        &self,
        ctx: RequestContext,
        request: OwnedFetchLogsRequestView,
    ) -> ServiceResult<FetchLogsResponse> {
        let request_started = Instant::now();
        let mut request = prepare_fetch_logs(request)?;

        // Hold the query-visibility READ guard across the whole scan: like
        // Query, DataFusion opens the snapshotted `log` parquet files lazily
        // during collect(), so the guard must outlive fetch_log_rows to keep a
        // concurrent structural mutation from unlinking a file mid-scan.
        let visibility_started = Instant::now();
        let _read_guard = self.store.query_visibility().read().await;
        let visibility_wait = visibility_started.elapsed();

        // Snapshot the sealed `log` segments (under the engine lock) on the
        // blocking pool, then build the provider over them.
        let store = Arc::clone(&self.store);
        let snapshot_started = Instant::now();
        let snapshot = run_blocking(move || store.query_snapshot(LOG_NAMESPACE_NAME)).await?;
        let snapshot_elapsed = snapshot_started.elapsed();
        let minimum_seq = snapshot
            .seq_bounds
            .values()
            .map(|(minimum, _)| *minimum)
            .min()
            .unwrap_or(0);
        let maximum_seq = snapshot
            .seq_bounds
            .values()
            .map(|(_, maximum)| *maximum)
            .max()
            .unwrap_or(0);
        let cache_lookup_started = Instant::now();
        let cache_lookup = request
            .cache_key
            .as_ref()
            .map(|key| self.tail_cache.lookup(key, minimum_seq, maximum_seq));
        let cache_lookup_elapsed = cache_lookup_started.elapsed();
        let (cached_rows, cache_outcome) = match cache_lookup {
            Some(TailCacheLookup::Hit(rows)) => {
                return fetch_logs_response(
                    rows,
                    &request,
                    request_started,
                    FetchLogsStageTimings {
                        visibility_wait,
                        snapshot: snapshot_elapsed,
                        cache_lookup: cache_lookup_elapsed,
                        ..Default::default()
                    },
                    "hit",
                );
            }
            Some(TailCacheLookup::Delta {
                scanned_through,
                rows_descending,
            }) => {
                request
                    .predicates
                    .where_parts
                    .push(format!("seq > {scanned_through}"));
                (Some(rows_descending), "delta")
            }
            Some(TailCacheLookup::Miss) => (None, "miss"),
            None => (None, "bypass"),
        };
        let queried = self.query_log_rows(&ctx, snapshot, &request).await?;
        let mut rows = queried.rows;
        if let Some(cached_rows) = cached_rows {
            rows = merge_tail_rows(rows, cached_rows, request.max_lines);
        }
        if let Some(key) = request.cache_key.take() {
            self.tail_cache.insert(key, maximum_seq, rows.clone());
        }
        fetch_logs_response(
            rows,
            &request,
            request_started,
            FetchLogsStageTimings {
                visibility_wait,
                snapshot: snapshot_elapsed,
                cache_lookup: cache_lookup_elapsed,
                provider: queried.provider_elapsed,
                query: queried.query_elapsed,
                ..Default::default()
            },
            cache_outcome,
        )
    }
}

/// Convert a shaped log entry into the wire `LogEntry`. `attempt_id` and `key`
/// are populated per the scope's shaping rules.
fn shaped_entry_to_proto(e: ShapedEntry) -> LogEntry {
    let mut entry = LogEntry::default()
        .with_seq(e.seq)
        .with_source(e.source)
        .with_data(e.data)
        .with_attempt_id(e.attempt_id);
    entry = LogEntry {
        timestamp: MessageField::some(Timestamp {
            epoch_ms: Some(e.epoch_ms),
            ..Default::default()
        }),
        // `level` is an OPEN enum: preserve the raw stored int verbatim
        // (`Known` if it matches a variant, `Unknown(raw)` otherwise) so an
        // out-of-range level round-trips exactly rather than collapsing to
        // UNKNOWN.
        level: Some(buffa::EnumValue::<LogLevel>::from(e.level)),
        ..entry
    };
    if let Some(key) = e.key {
        entry = entry.with_key(key);
    }
    entry
}

#[cfg(test)]
mod tests {
    use axum::http::{Extensions, HeaderMap};

    use super::*;

    fn ctx_with(identity: Option<AuthIdentity>) -> RequestContext {
        let mut extensions = Extensions::new();
        if let Some(identity) = identity {
            extensions.insert(identity);
        }
        RequestContext::new(HeaderMap::new()).with_extensions(extensions)
    }

    fn jwt(cluster: &str) -> RequestContext {
        ctx_with(Some(AuthIdentity::Jwt {
            cluster: cluster.to_string(),
        }))
    }

    fn cache_key(source: &str) -> TailCacheKey {
        TailCacheKey {
            scope: MatchScope::MATCH_SCOPE_PREFIX.to_i32(),
            source: source.to_string(),
            since_ms: 0,
            substring: "routes:".to_string(),
            regex: String::new(),
            min_level: LogLevel::LOG_LEVEL_UNKNOWN.to_i32(),
            cluster: String::new(),
            max_lines: 100,
        }
    }

    fn row(seq: i64) -> LogRow {
        LogRow {
            seq,
            key: Some("/muchanem/glm53-relay-e/worker".to_string()),
            source: "stdout".to_string(),
            data: format!("row {seq} routes:"),
            epoch_ms: seq,
            level: LogLevel::LOG_LEVEL_INFO.to_i32(),
        }
    }

    #[test]
    fn unchanged_tail_snapshot_is_served_from_cache_with_current_retention() {
        let cache = LogTailCache::with_budget_bytes(1024 * 1024);
        let key = cache_key("/muchanem/glm53-relay-e/");
        cache.insert(key.clone(), 10, vec![row(10), row(5)]);

        let TailCacheLookup::Hit(retained) = cache.lookup(&key, 6, 10) else {
            panic!("unchanged snapshot should hit");
        };
        assert_eq!(retained, vec![row(10)]);

        // Filtering belongs to the snapshot lookup. It does not mutate the
        // stored result while another in-flight snapshot may still need it.
        let TailCacheLookup::Hit(all_rows) = cache.lookup(&key, 0, 10) else {
            panic!("unchanged snapshot should hit");
        };
        assert_eq!(all_rows, vec![row(10), row(5)]);
    }

    #[test]
    fn advanced_tail_snapshot_scans_only_the_delta_and_rejects_stale_replacement() {
        let cache = LogTailCache::with_budget_bytes(1024 * 1024);
        let key = cache_key("/muchanem/glm53-relay-e/");
        cache.insert(key.clone(), 10, vec![row(10)]);

        let TailCacheLookup::Delta {
            scanned_through,
            rows_descending,
        } = cache.lookup(&key, 0, 20)
        else {
            panic!("advanced snapshot should request a delta scan");
        };
        assert_eq!(scanned_through, 10);
        assert_eq!(rows_descending, vec![row(10)]);

        cache.insert(key.clone(), 20, vec![row(20), row(10)]);
        cache.insert(key.clone(), 15, vec![row(15), row(10)]);
        let TailCacheLookup::Hit(rows) = cache.lookup(&key, 0, 20) else {
            panic!("newer concurrent insertion must win");
        };
        assert_eq!(rows, vec![row(20), row(10)]);
    }

    #[test]
    fn repeated_tail_reads_cover_unchanged_nonmatching_and_matching_updates() {
        let cache = LogTailCache::with_budget_bytes(1024 * 1024);
        let key = cache_key("/muchanem/glm53-relay-e/");
        cache.insert(key.clone(), 10, vec![row(10)]);

        assert!(matches!(
            cache.lookup(&key, 0, 10),
            TailCacheLookup::Hit(rows) if rows == vec![row(10)]
        ));

        let TailCacheLookup::Delta {
            rows_descending, ..
        } = cache.lookup(&key, 0, 11)
        else {
            panic!("new nonmatching row should require a delta scan");
        };
        let after_nonmatch = merge_tail_rows(Vec::new(), rows_descending, 100);
        cache.insert(key.clone(), 11, after_nonmatch);
        assert!(matches!(
            cache.lookup(&key, 0, 11),
            TailCacheLookup::Hit(rows) if rows == vec![row(10)]
        ));

        let TailCacheLookup::Delta {
            rows_descending, ..
        } = cache.lookup(&key, 0, 12)
        else {
            panic!("new matching row should require a delta scan");
        };
        let after_match = merge_tail_rows(vec![row(12)], rows_descending, 100);
        cache.insert(key.clone(), 12, after_match);
        assert!(matches!(
            cache.lookup(&key, 0, 12),
            TailCacheLookup::Hit(rows) if rows == vec![row(12), row(10)]
        ));
    }

    #[test]
    fn sequence_reset_invalidates_a_tail_cache_entry() {
        let cache = LogTailCache::with_budget_bytes(1024 * 1024);
        let key = cache_key("/muchanem/glm53-relay-e/");
        cache.insert(key.clone(), 20, vec![row(20)]);

        assert!(matches!(cache.lookup(&key, 0, 5), TailCacheLookup::Miss));
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn tail_cache_is_byte_bounded_and_evicts_least_recently_used_filter() {
        let first = cache_key("/first/");
        let one_entry_budget = tail_cache_entry_bytes(&first, &[row(1)]);
        let cache = LogTailCache::with_budget_bytes(one_entry_budget);
        cache.insert(first.clone(), 1, vec![row(1)]);
        let second = cache_key("/second");
        cache.insert(second.clone(), 2, vec![row(2)]);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.evictions, 1);
        assert!(matches!(cache.lookup(&first, 0, 1), TailCacheLookup::Miss));
        assert!(matches!(
            cache.lookup(&second, 0, 2),
            TailCacheLookup::Hit(_)
        ));
    }

    #[test]
    fn a_token_may_only_write_logs_under_the_cluster_it_authenticates() {
        // Every cluster in a hub's jwt layer admits equally, so without this binding
        // any trusted key could file its rows under a peer's name.
        assert_eq!(
            authorized_cluster(&jwt("cw-rno2a"), "cw-rno2a").unwrap(),
            "cw-rno2a"
        );
        assert!(authorized_cluster(&jwt("cw-rno2a"), "marin").is_err());
    }

    #[test]
    fn a_token_that_names_no_cluster_has_one_stamped_from_its_key() {
        assert_eq!(
            authorized_cluster(&jwt("cw-rno2a"), "").unwrap(),
            "cw-rno2a",
            "an omitted origin is filled in from the credential, never left empty"
        );
    }

    #[test]
    fn a_writer_on_a_trusted_network_names_its_own_origin() {
        // The local single-cluster push: no per-writer credential, so the request's
        // value stands (empty, for a store writing its own logs).
        let network = ctx_with(Some(AuthIdentity::Network));
        assert_eq!(authorized_cluster(&network, "").unwrap(), "");
        assert_eq!(
            authorized_cluster(&network, "anything").unwrap(),
            "anything"
        );
    }

    #[test]
    fn a_push_with_no_auth_identity_is_refused() {
        // Unreachable through the interceptor, which admits nothing without recording
        // an identity. Refusing rather than defaulting keeps it that way.
        assert!(authorized_cluster(&ctx_with(None), "").is_err());
    }
}
