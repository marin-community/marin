//! `finelog.stats.StatsService` trait impl.
//!
//! Handlers return OWNED response messages (JSON-codec safety) and wrap the
//! blocking rusqlite-backed `Store` calls in `spawn_blocking`. The store call
//! result is decoded into proto messages back on the async side. The wire
//! `effective_schema` strips the implicit `seq` column.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use buffa::MessageField;
use connectrpc::{ConnectError, RequestContext, ServiceResult};
use datafusion::error::DataFusionError;

use crate::errors::StatsError;
use crate::policies::{managed_storage_policy_for, registration_namespace_for};
use crate::proto::finelog::stats::{
    AbortTableMigrationResponse, DropTableResponse, GetTableSchemaResponse, GetTableStatusResponse,
    ListNamespacesResponse, NamespaceInfo, OwnedAbortTableMigrationRequestView,
    OwnedDropTableRequestView, OwnedGetTableSchemaRequestView, OwnedGetTableStatusRequestView,
    OwnedListNamespacesRequestView, OwnedQueryRequestView, OwnedRegisterTableRequestView,
    OwnedWriteRowsRequestView, QueryResponse, RegisterTableResponse, StatsService,
    WriteRowsResponse,
};
use crate::query::{make_ctx, query_deadline, run_query_over, truncate_sql_for_log};
use crate::server::auth::{request_identity, AuthIdentity};
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::catalog::TableSpecStatus;
use crate::store::ipc::encode_ipc;
use crate::store::namespace::DEFAULT_PERSIST_TIMEOUT;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{
    ignored_forwarded_schema_columns, schema_from_proto_view, schema_to_proto_owned, Schema,
};
use crate::store::store::ForwardedWrite;
use crate::store::table_spec::ValidatedTableSpec;
use crate::store::Store;
use crate::telemetry_policy::{is_forwarded_telemetry_namespace, TELEMETRY_NAMESPACE};

pub struct StatsServiceImpl {
    store: Arc<Store>,
    ignored_forwarded_telemetry_columns: Mutex<HashSet<String>>,
}

struct RegistrationOutcome {
    namespace: String,
    schema: Schema,
    policy: StoragePolicy,
    ignored_columns: Vec<String>,
    table_spec_status: TableSpecStatus,
    object_backed: bool,
}

impl StatsServiceImpl {
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            store,
            ignored_forwarded_telemetry_columns: Mutex::new(HashSet::new()),
        }
    }

    fn report_ignored_forwarded_telemetry_columns(&self, columns: Vec<String>) {
        let mut seen = self.ignored_forwarded_telemetry_columns.lock().unwrap();
        let new_columns: Vec<String> = columns
            .into_iter()
            .filter(|column| seen.insert(column.clone()))
            .collect();
        if !new_columns.is_empty() {
            tracing::warn!(
                namespace = TELEMETRY_NAMESPACE,
                columns = ?new_columns,
                "finelog hub: ignoring candidate-only nullable telemetry columns",
            );
        }
    }
}

fn is_forwarded_telemetry(ctx: &RequestContext, namespace: &str) -> bool {
    is_forwarded_telemetry_namespace(namespace)
        && matches!(request_identity(ctx), Some(AuthIdentity::Jwt { .. }))
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

/// The origin cluster to stamp on a WriteRows batch, bound to the credential
/// that carried it — the stats-plane analogue of the log plane's
/// `authorized_cluster`. A forwarding JWT names exactly one cluster, so its rows
/// are attributed to it regardless of what the batch carried; this is what makes
/// cross-cluster attribution independent of whether the sender's local schema
/// held the implicit origin column (the federation gap where forwarded `iris.*`
/// stats from a finelog whose namespace predates that column arrive unstamped).
/// A trusted-network writer carries no per-writer identity and names its own
/// origin (empty for a local write), so its batch is left as supplied.
fn write_origin_cluster(ctx: &RequestContext) -> Result<Option<String>, ConnectError> {
    match request_identity(ctx) {
        Some(AuthIdentity::Jwt { cluster }) => Ok(Some(cluster.clone())),
        Some(AuthIdentity::Network) => Ok(None),
        None => Err(ConnectError::internal(
            "finelog: request reached a handler with no auth identity",
        )),
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
        ctx: RequestContext,
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
        let requested_policy = StoragePolicy::from_proto_view(request.storage_policy.as_option());
        let validated_table_spec = request
            .table_spec
            .as_option()
            .map(|view| ValidatedTableSpec::from_view(view, &schema, &requested_policy))
            .transpose()?;
        let forwarded_telemetry = is_forwarded_telemetry(&ctx, &namespace);

        let store = Arc::clone(&self.store);
        let requested_namespace = namespace.clone();
        let outcome = run_blocking(move || {
            let ns = registration_namespace_for(&requested_namespace)?;
            let policy = managed_storage_policy_for(&ns)?.unwrap_or(requested_policy);
            if forwarded_telemetry {
                if validated_table_spec.is_some() {
                    return Err(StatsError::SchemaValidation(
                        "forwarded telemetry registration cannot change table_spec".to_string(),
                    ));
                }
                match store.get_table_schema(&ns) {
                    Ok(effective) => {
                        let ignored = ignored_forwarded_schema_columns(&schema, &effective)?;
                        let effective_policy = store.get_policy(&ns)?;
                        let table_spec_status = store.table_spec_status(&ns)?;
                        return Ok(RegistrationOutcome {
                            namespace: ns,
                            schema: effective,
                            policy: effective_policy,
                            ignored_columns: ignored,
                            table_spec_status,
                            object_backed: false,
                        });
                    }
                    Err(StatsError::NamespaceNotFound(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            if let Some(validated) = validated_table_spec {
                if validated.cache_policy != policy {
                    return Err(StatsError::SchemaValidation(
                        "table_spec local_cache does not match the server-managed storage policy"
                            .to_string(),
                    ));
                }
                let registration = store.register_versioned_table(&ns, validated)?;
                return Ok(RegistrationOutcome {
                    namespace: ns,
                    schema: registration.schema,
                    policy: registration.policy,
                    ignored_columns: Vec::new(),
                    table_spec_status: registration.table_spec_status,
                    object_backed: registration.object_backed,
                });
            }
            let effective = store.register_table(&ns, schema, policy)?;
            let effective_policy = store.get_policy(&ns)?;
            let table_spec_status = store.table_spec_status(&ns)?;
            Ok(RegistrationOutcome {
                namespace: ns,
                schema: effective,
                policy: effective_policy,
                ignored_columns: Vec::new(),
                table_spec_status,
                object_backed: false,
            })
        })
        .await?;
        self.report_ignored_forwarded_telemetry_columns(outcome.ignored_columns);
        if outcome.object_backed {
            self.store
                .publish_object_catalog(&outcome.namespace)
                .await?;
        }

        connectrpc::Response::ok(RegisterTableResponse {
            effective_schema: MessageField::some(schema_to_proto_owned(&outcome.schema)),
            effective_policy: MessageField::some(outcome.policy.to_proto_owned()),
            active_table_spec_version: Some(outcome.table_spec_status.active_version()),
            desired_table_spec_version: Some(outcome.table_spec_status.desired_version()),
            transition_phase: Some(outcome.table_spec_status.phase.into()),
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
        // Resolve the origin cluster from the credential before the blocking
        // hop so a forwarding writer's rows are stamped with its cluster.
        let origin_cluster = write_origin_cluster(&ctx)?;
        let forwarded_telemetry =
            origin_cluster.is_some() && is_forwarded_telemetry_namespace(&namespace);

        let store = Arc::clone(&self.store);
        let ns = namespace.clone();
        let outcome = run_blocking(move || {
            if forwarded_telemetry {
                return store.write_forwarded_telemetry_rows(
                    &ns,
                    &arrow_ipc,
                    origin_cluster
                        .as_deref()
                        .expect("forwarded telemetry has an origin cluster"),
                );
            }
            store.write_ingestion_rows(&ns, &arrow_ipc, origin_cluster.as_deref())
        })
        .await?;
        let ForwardedWrite {
            rows_written,
            persisted_targets,
            ignored_columns,
        } = outcome;
        self.report_ignored_forwarded_telemetry_columns(ignored_columns);

        // The server does not auto-cancel on the client deadline; enforce the
        // durability await ourselves, bounded by the remaining budget (falling
        // back to DEFAULT_PERSIST_TIMEOUT).
        for (destination, last_seq) in persisted_targets {
            let budget = ctx.time_remaining().unwrap_or(DEFAULT_PERSIST_TIMEOUT);
            self.store
                .await_persisted(&destination, last_seq, budget)
                .await?;
        }

        connectrpc::Response::ok(WriteRowsResponse::default().with_rows_written(rows_written))
    }

    async fn query(
        &self,
        ctx: RequestContext,
        request: OwnedQueryRequestView,
    ) -> ServiceResult<QueryResponse> {
        let sql = request.sql.unwrap_or("").to_string();

        // Hold the query-visibility READ guard across the WHOLE scan. DataFusion
        // opens the snapshotted parquet files LAZILY during collect(), so the
        // guard must outlive run_query_over (not just query_providers) to keep a
        // concurrent drop_table / compaction from unlinking a file mid-scan.
        let _read_guard = self.store.query_visibility().read().await;

        // Plan every live namespace from its pinned state (schema, bounds,
        // partitions, exact object references) on the blocking pool. Objects are
        // localized later, and only for the segments the scan selects.
        let store = Arc::clone(&self.store);
        let providers = run_blocking(move || store.query_providers()).await?;
        // Object-backed tables bound the read themselves; that bound cannot be
        // configured away.
        let table_bound = self.store.object_query_bound();

        // DataFusion schedules its own CPU tasks; await sql()/collect() directly
        // (no spawn_blocking). Errors map by variant: parse/plan/schema/catalog
        // faults are client errors, IO/execution faults are server errors.
        //
        // Bound execution by the earlier of the server ceiling and the caller's
        // remaining budget. On elapse the query future is dropped (aborting the
        // scan), so a timed-out caller cannot leave CPU work behind.
        let query_ctx = make_ctx();
        let query = run_query_over(&query_ctx, providers, &sql);
        let result = match query_deadline(ctx.time_remaining(), table_bound) {
            Some(deadline) => match tokio::time::timeout(deadline, query).await {
                Ok(r) => r.map_err(map_query_error)?,
                Err(_elapsed) => {
                    tracing::warn!(
                        deadline_ms = deadline.as_millis() as u64,
                        sql = %truncate_sql_for_log(&sql),
                        "query aborted: exceeded deadline",
                    );
                    return Err(ConnectError::deadline_exceeded(format!(
                        "query exceeded deadline of {} ms",
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

    async fn get_table_status(
        &self,
        _ctx: RequestContext,
        request: OwnedGetTableStatusRequestView,
    ) -> ServiceResult<GetTableStatusResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        let store = Arc::clone(&self.store);
        let status = run_blocking(move || store.table_spec_status(&namespace)).await?;
        connectrpc::Response::ok(GetTableStatusResponse {
            active_table_spec: status
                .active
                .map(MessageField::some)
                .unwrap_or_else(MessageField::none),
            desired_table_spec: status
                .desired
                .map(MessageField::some)
                .unwrap_or_else(MessageField::none),
            migration: status
                .migration
                .map(MessageField::some)
                .unwrap_or_else(MessageField::none),
            catalog_generation: Some(status.catalog_generation),
            ..Default::default()
        })
    }

    async fn abort_table_migration(
        &self,
        _ctx: RequestContext,
        request: OwnedAbortTableMigrationRequestView,
    ) -> ServiceResult<AbortTableMigrationResponse> {
        let namespace = request
            .namespace
            .ok_or_else(|| ConnectError::invalid_argument("namespace required"))?
            .to_string();
        let status = self.store.abort_table_migration(&namespace).await?;
        connectrpc::Response::ok(AbortTableMigrationResponse {
            catalog_generation: Some(status.catalog_generation),
            active_table_spec_version: Some(status.active_version()),
            ..Default::default()
        })
    }
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
    fn a_forwarding_jwt_stamps_the_cluster_its_key_authenticates() {
        // The stats-plane twin of the log path's `authorized_cluster`: a WriteRows
        // batch's origin is bound to the credential that carried it. The function reads
        // no caller-supplied cluster at all, so a spoofed origin in the batch cannot
        // influence the stamp — `stamp_cluster_column` then overwrites the batch column
        // with this value. That is what makes attribution independent of whether the
        // sender's local schema even held the origin column (the cross-cluster
        // forwarding gap, where a forwarded batch arrives without the column).
        assert_eq!(
            write_origin_cluster(&jwt("cw-rno2a")).unwrap(),
            Some("cw-rno2a".to_string())
        );
    }

    #[test]
    fn a_trusted_network_writer_stamps_nothing() {
        // A local write (admitted by the loopback/VPC cidr rule) carries no per-writer
        // identity, so its batch is left as supplied — empty for a store writing its own
        // rows. Nothing is stamped, so an empty/NULL origin denotes the local cluster.
        let network = ctx_with(Some(AuthIdentity::Network));
        assert_eq!(write_origin_cluster(&network).unwrap(), None);
    }

    #[test]
    fn a_write_with_no_auth_identity_is_refused() {
        // Unreachable through the interceptor, which admits nothing without recording an
        // identity. Refusing rather than defaulting fails closed: an unauthenticated
        // write can never silently become a hub-local row.
        assert!(write_origin_cluster(&ctx_with(None)).is_err());
    }
}
