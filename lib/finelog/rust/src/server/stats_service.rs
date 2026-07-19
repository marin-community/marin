//! `finelog.stats.StatsService` trait impl.
//!
//! Handlers return OWNED response messages (JSON-codec safety) and wrap the
//! blocking rusqlite-backed `Store` calls in `spawn_blocking`. The store call
//! result is decoded into proto messages back on the async side. The wire
//! `effective_schema` strips the implicit `seq` column.

use std::sync::Arc;

use buffa::MessageField;
use connectrpc::{ConnectError, RequestContext, ServiceResult};
use datafusion::error::DataFusionError;

use crate::errors::StatsError;
use crate::proto::finelog::stats::{
    DropTableResponse, GetTableSchemaResponse, ListNamespacesResponse, NamespaceInfo,
    OwnedDropTableRequestView, OwnedGetTableSchemaRequestView, OwnedListNamespacesRequestView,
    OwnedQueryRequestView, OwnedRegisterTableRequestView, OwnedWriteRowsRequestView, QueryResponse,
    RegisterTableResponse, StatsService, WriteRowsResponse,
};
use crate::query::{make_ctx, query_timeout, run_query_over, truncate_sql_for_log};
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::ipc::encode_ipc;
use crate::store::namespace::DEFAULT_PERSIST_TIMEOUT;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_from_proto_view, schema_to_proto_owned, Schema};
use crate::store::Store;

pub struct StatsServiceImpl {
    store: Arc<Store>,
}

impl StatsServiceImpl {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }
}

/// Run a blocking store closure on the blocking pool, mapping a JoinError to
/// an internal ConnectError and a StatsError to its mapped code.
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

/// Map a DataFusion query error to the right Connect status: a SQL parse /
/// plan / schema / catalog / unsupported fault is a client error ->
/// `invalid_argument`; `ResourcesExhausted` -> `resource_exhausted`; anything
/// else (IO, execution-time, internal) is a server fault -> `internal`. Reading
/// a server bug back as `invalid_argument` would wrongly blame the client.
///
/// `find_root` unwraps `Context`/`External`/`Shared` wrappers so a plan error
/// buried under context still classifies as `invalid_argument`.
fn map_query_error(e: DataFusionError) -> ConnectError {
    let msg = format!("query failed: {e}");
    match e.find_root() {
        DataFusionError::SQL(..)
        | DataFusionError::Plan(_)
        | DataFusionError::SchemaError(..)
        | DataFusionError::NotImplemented(_) => ConnectError::invalid_argument(msg),
        DataFusionError::ResourcesExhausted(_) => ConnectError::resource_exhausted(msg),
        _ => ConnectError::internal(msg),
    }
}

// Naming the concrete `ServiceResult<T>` return type (per the connectrpc
// cookbook) refines the trait's `impl Encodable<T> + Send`; that is intentional.
#[allow(refining_impl_trait)]
impl StatsService for StatsServiceImpl {
    async fn register_table(
        &self,
        _ctx: RequestContext,
        request: OwnedRegisterTableRequestView,
    ) -> ServiceResult<RegisterTableResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        let schema_view = request
            .schema
            .as_option()
            .ok_or_else(|| ConnectError::invalid_argument("schema required"))?;
        let schema: Schema = schema_from_proto_view(schema_view)?;
        let policy = StoragePolicy::from_proto_view(request.storage_policy.as_option());

        let store = Arc::clone(&self.store);
        let ns = namespace.clone();
        let (effective, effective_policy) = run_blocking(move || {
            let effective = store.register_table(&ns, schema, policy)?;
            let effective_policy = store.get_policy(&ns)?;
            Ok((effective, effective_policy))
        })
        .await?;

        connectrpc::Response::ok(RegisterTableResponse {
            effective_schema: MessageField::some(schema_to_proto_owned(&effective)),
            effective_policy: MessageField::some(effective_policy.to_proto_owned()),
            ..Default::default()
        })
    }

    async fn write_rows(
        &self,
        ctx: RequestContext,
        request: OwnedWriteRowsRequestView,
    ) -> ServiceResult<WriteRowsResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        // Copy the IPC bytes out of the borrowed request so the blocking decode
        // owns them across the spawn_blocking boundary.
        let arrow_ipc: Vec<u8> = request.arrow_ipc.unwrap_or(&[]).to_vec();

        // Decode + validate + align + append on the blocking pool; the size/row
        // caps and IPC decode live in `Store::write_rows`.
        let store = Arc::clone(&self.store);
        let ns = namespace.clone();
        let (rows_written, last_seq) =
            run_blocking(move || store.write_rows(&ns, &arrow_ipc)).await?;

        // The server does not auto-cancel on the client deadline; enforce the
        // durability await ourselves, bounded by the remaining budget (falling
        // back to DEFAULT_PERSIST_TIMEOUT).
        let budget = ctx.time_remaining().unwrap_or(DEFAULT_PERSIST_TIMEOUT);
        self.store
            .await_persisted(&namespace, last_seq, budget)
            .await?;

        connectrpc::Response::ok(WriteRowsResponse::default().with_rows_written(rows_written))
    }

    async fn query(
        &self,
        _ctx: RequestContext,
        request: OwnedQueryRequestView,
    ) -> ServiceResult<QueryResponse> {
        let sql = request.sql.unwrap_or("").to_string();

        // Hold the query-visibility READ guard across the WHOLE scan. DataFusion
        // opens the snapshotted parquet files LAZILY during collect(), so the
        // guard must outlive run_query_over (not just query_providers) to keep a
        // concurrent drop_table / compaction from unlinking a file mid-scan.
        let _read_guard = self.store.query_visibility().read().await;

        // Snapshot every live namespace (schema + sealed-segment paths) under
        // the engine locks on the blocking pool.
        let store = Arc::clone(&self.store);
        let providers = run_blocking(move || store.query_providers()).await?;

        // DataFusion schedules its own CPU tasks; await sql()/collect() directly
        // (no spawn_blocking). Errors map by variant: parse/plan/schema/catalog
        // faults are client errors, IO/execution faults are server errors.
        //
        // Bound execution by the server-side wall-clock deadline: on elapse the
        // query future is dropped (aborting the scan) and the caller gets a
        // clean deadline_exceeded, so one pathological query can't run unbounded.
        let ctx = make_ctx();
        let query = run_query_over(&ctx, providers, &sql);
        let result = match query_timeout() {
            Some(deadline) => match tokio::time::timeout(deadline, query).await {
                Ok(r) => r.map_err(map_query_error)?,
                Err(_elapsed) => {
                    tracing::warn!(
                        deadline_ms = deadline.as_millis() as u64,
                        sql = %truncate_sql_for_log(&sql),
                        "query aborted: exceeded server-side deadline",
                    );
                    return Err(ConnectError::deadline_exceeded(format!(
                        "query exceeded server-side deadline of {} ms",
                        deadline.as_millis()
                    )));
                }
            },
            None => query.await.map_err(map_query_error)?,
        };

        let row_count: i64 = result.batches.iter().map(|b| b.num_rows() as i64).sum();
        // The schema is captured from the planned DataFrame, so an empty result
        // still emits the correct typed schema (the typed-empty contract).
        let buf = encode_ipc(&result.schema, &result.batches)
            .map_err(|e| ConnectError::internal(format!("encode query result: {e}")))?;
        // No server-side row cap; the only result bound is the 64MB transport
        // message limit -> resource_exhausted.
        if buf.len() > MAX_MESSAGE_BYTES {
            return Err(ConnectError::resource_exhausted(format!(
                "query result {} bytes exceeds {MAX_MESSAGE_BYTES} message limit",
                buf.len()
            )));
        }
        connectrpc::Response::ok(
            QueryResponse::default()
                .with_arrow_ipc(buf)
                .with_row_count(row_count),
        )
    }

    async fn drop_table(
        &self,
        _ctx: RequestContext,
        request: OwnedDropTableRequestView,
    ) -> ServiceResult<DropTableResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        // Structural mutation: take the query-visibility WRITE guard so no
        // in-flight query/FetchLogs scan is reading the segment files we are
        // about to unlink. New readers block until the drop completes.
        let _write_guard = self.store.query_visibility().write().await;
        let store = Arc::clone(&self.store);
        run_blocking(move || store.drop_table(&namespace)).await?;
        connectrpc::Response::ok(DropTableResponse::default())
    }

    async fn list_namespaces(
        &self,
        _ctx: RequestContext,
        _request: OwnedListNamespacesRequestView,
    ) -> ServiceResult<ListNamespacesResponse> {
        let store = Arc::clone(&self.store);
        let entries = run_blocking(move || store.list_namespaces_with_stats()).await?;
        let namespaces: Vec<NamespaceInfo> = entries
            .into_iter()
            .map(|(name, schema, stats, policy)| {
                let info = NamespaceInfo::default()
                    .with_namespace(name)
                    .with_row_count(stats.row_count)
                    .with_byte_size(stats.byte_size)
                    .with_min_seq(stats.min_seq)
                    .with_max_seq(stats.max_seq)
                    .with_segment_count(stats.segment_count);
                NamespaceInfo {
                    schema: MessageField::some(schema_to_proto_owned(&schema)),
                    storage_policy: MessageField::some(policy.to_proto_owned()),
                    ..info
                }
            })
            .collect();
        connectrpc::Response::ok(ListNamespacesResponse {
            namespaces,
            ..Default::default()
        })
    }

    async fn get_table_schema(
        &self,
        _ctx: RequestContext,
        request: OwnedGetTableSchemaRequestView,
    ) -> ServiceResult<GetTableSchemaResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        let store = Arc::clone(&self.store);
        let schema = run_blocking(move || store.get_table_schema(&namespace)).await?;
        connectrpc::Response::ok(GetTableSchemaResponse {
            schema: MessageField::some(schema_to_proto_owned(&schema)),
            ..Default::default()
        })
    }
}
