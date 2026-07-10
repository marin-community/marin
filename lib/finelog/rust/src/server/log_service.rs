//! `LogService` trait impl: PushLogs and FetchLogs.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray};
use buffa::MessageField;
use connectrpc::{ConnectError, RequestContext, ServiceResult};

use crate::errors::StatsError;
use crate::proto::finelog::logging::{
    FetchLogsResponse, LogEntry, LogEntryView, LogLevel, LogService, MatchScope,
    OwnedFetchLogsRequestView, OwnedPushLogsRequestView, PushLogsResponse, Timestamp,
};
use crate::query::fetch_log_rows;
use crate::query::make_ctx;
use crate::query::provider::NamespaceProvider;
use crate::server::auth::{request_identity, AuthIdentity};
use crate::store::log_read::{
    add_cluster_filter, add_common_filters, add_seq_upper_bound, build_log_predicates,
    shape_log_read_result, str_to_log_level, ShapedEntry,
};
use crate::store::namespace::DEFAULT_PERSIST_TIMEOUT;
use crate::store::store::LOG_NAMESPACE_NAME;
use crate::store::Store;

/// Server default for `max_lines` when the request leaves it unset/<=0.
const DEFAULT_MAX_LINES: i32 = 1000;

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
}

impl LogServiceImpl {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
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
        _ctx: RequestContext,
        request: OwnedFetchLogsRequestView,
    ) -> ServiceResult<FetchLogsResponse> {
        // Wire UNSPECIFIED (and an unset field) maps to REGEX so clients that
        // encode a regex pattern in `source` without setting match_scope keep
        // working. New callers set EXACT/PREFIX explicitly.
        let scope = match request.match_scope.and_then(|ev| ev.as_known()) {
            Some(MatchScope::MATCH_SCOPE_UNSPECIFIED) | None => MatchScope::MATCH_SCOPE_REGEX,
            Some(s) => s,
        };
        let source = request.source.unwrap_or("");
        let cursor = request.cursor.unwrap_or(0);
        let until_cursor = request.until_cursor.unwrap_or(0);
        let since_ms = request.since_ms.unwrap_or(0);
        let substring = request.substring.unwrap_or("");
        let tail = request.tail.unwrap_or(false);
        let min_level: LogLevel = str_to_log_level(request.min_level.unwrap_or(""));
        // max_lines <= 0 -> server default 1000.
        let raw_max_lines = request.max_lines.unwrap_or(0);
        let max_lines = if raw_max_lines > 0 {
            raw_max_lines
        } else {
            DEFAULT_MAX_LINES
        };

        // Build predicates (pure). Empty PREFIX source -> invalid_argument.
        let mut predicates =
            build_log_predicates(source, cursor, scope).map_err(ConnectError::invalid_argument)?;
        // Bracket the scope's `seq > cursor` from above so a reader can page
        // backwards from a row it names (`until_cursor` + `tail`).
        add_seq_upper_bound(&mut predicates.where_parts, until_cursor);
        add_common_filters(&mut predicates.where_parts, since_ms, substring, min_level);
        // Restrict to one origin cluster when the caller asks (the federated read
        // path filters `cluster = <peer>`); empty = unfiltered, so a local
        // single-cluster read behaves exactly as before.
        add_cluster_filter(&mut predicates.where_parts, request.cluster.unwrap_or(""));

        // Hold the query-visibility READ guard across the whole scan: like
        // Query, DataFusion opens the snapshotted `log` parquet files lazily
        // during collect(), so the guard must outlive fetch_log_rows to keep a
        // concurrent structural mutation from unlinking a file mid-scan.
        let _read_guard = self.store.query_visibility().read().await;

        // Snapshot the sealed `log` segments (under the engine lock) on the
        // blocking pool, then build the provider over them.
        let store = Arc::clone(&self.store);
        let snapshot = run_blocking(move || store.query_snapshot(LOG_NAMESPACE_NAME)).await?;
        let provider = NamespaceProvider::build(snapshot.schema, &snapshot.paths)
            .map_err(|e| ConnectError::internal(format!("build log provider: {e}")))?;

        // Run the read (DataFusion schedules its own CPU tasks; await directly).
        let ctx = make_ctx();
        let rows = fetch_log_rows(
            &ctx,
            provider,
            &predicates.where_parts,
            predicates.include_key,
            tail,
            max_lines,
        )
        .await
        .map_err(|e| ConnectError::internal(format!("log read failed: {e}")))?;

        let shaped = shape_log_read_result(
            rows,
            tail,
            max_lines,
            cursor,
            predicates.include_key,
            predicates.exact_key.as_deref(),
        );

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
