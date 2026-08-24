//! Bounded JSON telemetry ingestion backed by the ordinary `Store::write_rows` path.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::error::Error;
use std::future::Future;
use std::sync::Arc;

use arrow::array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::header::{ACCEPT_ENCODING, CONTENT_TYPE, RETRY_AFTER};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::LengthLimitError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OnceCell, OwnedSemaphorePermit, Semaphore};
use tower_http::decompression::RequestDecompressionLayer;
use tower_http::set_header::SetResponseHeaderLayer;
use uuid::Uuid;

use crate::errors::StatsError;
use crate::ingestion::route_ingestion_batch;
use crate::ingestion_policy::IngestionBatchSource;
use crate::proto::finelog::stats::ColumnType;
use crate::server::auth::{auth_gate, AuthIdentity, AuthPolicy};
use crate::server::ingest_health::IngestHealth;
use crate::store::group_extrema::GroupExtremaConfig;
use crate::store::ipc::encode_ipc;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_arrow, Column, CoveringProjection, Schema};
use crate::store::Store;
use crate::telemetry_policy::{
    matches_telemetry_namespace, storage_max_bytes, TELEMETRY_NAMESPACE, TELEMETRY_STORAGE_SHARDS,
};

pub const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 8;
pub const DEFAULT_DEDUPE_CAPACITY: usize = 10_000;

const MAX_BODY_BYTES: usize = 4 << 20;
const MAX_NORMALIZED_BYTES: usize = 16 << 20;
const MAX_RECORDS: usize = 10_000;
const MAX_ATTRIBUTES: usize = 64;
const MAX_STRING_BYTES: usize = 4_096;
const MAX_JSON_DEPTH: usize = 32;
const NORMALIZED_ROW_OVERHEAD: usize = 128;
pub(crate) const TELEMETRY_MAX_ROW_GROUP_ROWS: u32 = 128 * 1024;
const TELEMETRY_VERSION: u32 = 1;
const ERROR_CODE_INTERNAL: &str = "internal";
const RUN_ID_COLUMN: &str = "run_id";
const JOB_ID_COLUMN: &str = "job_id";
const EXECUTION_UID_COLUMN: &str = "execution_uid";
const REGION_COLUMN: &str = "region";
const NODE_NAME_COLUMN: &str = "node_name";
const PROCESS_INDEX_COLUMN: &str = "process_index";
const PRIMARY_PROCESS_INDEX: &str = "0";

fn telemetry_storage_namespaces() -> impl Iterator<Item = &'static str> {
    TELEMETRY_STORAGE_SHARDS
        .iter()
        .map(|shard| shard.storage_namespace)
}
const TRAINING_STATUS_NAMES: [&str; 3] = ["phase", "progress_time_seconds", "step"];
const TRAINING_LOSS_NAMES: [&str; 1] = ["train_loss"];
const TRAINING_RUN_NAMES: [&str; 1] = ["global_step"];
const HOST_METRIC_NAMES: [&str; 7] = [
    "node_cpu_utilization_percent",
    "node_disk_total_bytes",
    "node_disk_used_bytes",
    "node_memory_total_bytes",
    "node_memory_used_bytes",
    "node_network_receive_bytes",
    "node_network_transmit_bytes",
];
const ACCELERATOR_METRIC_NAMES: [&str; 16] = [
    "gpu_memory_temperature_celsius",
    "gpu_memory_total_bytes",
    "gpu_memory_used_bytes",
    "gpu_nvlink_receive_bytes_per_second",
    "gpu_nvlink_transmit_bytes_per_second",
    "gpu_pcie_replay_errors",
    "gpu_pcie_receive_bytes_per_second",
    "gpu_pcie_transmit_bytes_per_second",
    "gpu_power_watts",
    "gpu_row_remap_failures",
    "gpu_sm_active_ratio",
    "gpu_temperature_celsius",
    "gpu_tensor_active_ratio",
    "gpu_utilization_percent",
    "gpu_xid_error_code",
    "hardware_inventory",
];
const DEVICE_METRIC_PROJECTION_COLUMNS: [&str; 7] = [
    "timestamp_ms",
    "service",
    NODE_NAME_COLUMN,
    "name",
    "value",
    "attributes_json",
    "cluster",
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryBatch {
    version: u32,
    batch_id: String,
    #[serde(default)]
    namespace: Option<String>,
    resource: Resource,
    records: Vec<TelemetryRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Resource {
    service: String,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    job_id: Option<String>,
    #[serde(default)]
    execution_uid: Option<String>,
    #[serde(default)]
    region: Option<String>,
    #[serde(default)]
    node_name: Option<String>,
    #[serde(default)]
    process_index: Option<String>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Clone, Copy)]
struct ResourceDimensions<'a> {
    run_id: Option<&'a str>,
    job_id: Option<&'a str>,
    execution_uid: Option<&'a str>,
    region: Option<&'a str>,
    node_name: Option<&'a str>,
    process_index: Option<&'a str>,
}

impl Resource {
    fn dimensions(&self) -> ResourceDimensions<'_> {
        ResourceDimensions {
            run_id: self.explicit_or_attribute(self.run_id.as_deref(), RUN_ID_COLUMN),
            job_id: self.explicit_or_attribute(self.job_id.as_deref(), JOB_ID_COLUMN),
            execution_uid: self
                .explicit_or_attribute(self.execution_uid.as_deref(), EXECUTION_UID_COLUMN),
            region: self.explicit_or_attribute(self.region.as_deref(), REGION_COLUMN),
            node_name: self.explicit_or_attribute(self.node_name.as_deref(), NODE_NAME_COLUMN),
            process_index: self
                .explicit_or_attribute(self.process_index.as_deref(), PROCESS_INDEX_COLUMN),
        }
    }

    fn explicit_or_attribute<'a>(
        &'a self,
        explicit: Option<&'a str>,
        attribute_name: &str,
    ) -> Option<&'a str> {
        explicit.or_else(|| self.attributes.get(attribute_name).map(String::as_str))
    }

    fn normalized_attributes(&self) -> BTreeMap<String, String> {
        let mut attributes = self.attributes.clone();
        for (name, value) in [
            (RUN_ID_COLUMN, self.run_id.as_ref()),
            (JOB_ID_COLUMN, self.job_id.as_ref()),
            (EXECUTION_UID_COLUMN, self.execution_uid.as_ref()),
            (REGION_COLUMN, self.region.as_ref()),
            (NODE_NAME_COLUMN, self.node_name.as_ref()),
            (PROCESS_INDEX_COLUMN, self.process_index.as_ref()),
        ] {
            if let Some(value) = value {
                attributes.insert(name.to_string(), value.clone());
            }
        }
        attributes
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RecordKind {
    Counter,
    Gauge,
    Histogram,
    Event,
}

impl RecordKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
            Self::Event => "event",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryRecord {
    timestamp_ms: i64,
    kind: RecordKind,
    name: String,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    body: Option<Value>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    attributes: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct TelemetryAck {
    version: u32,
    batch_id: String,
    status: &'static str,
    deduplicated: bool,
    record_count: usize,
}

#[derive(Debug, Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    retry_after: bool,
}

impl ApiError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            retry_after: matches!(
                status,
                StatusCode::TOO_MANY_REQUESTS | StatusCode::SERVICE_UNAVAILABLE
            ),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
    }

    fn too_large(message: impl Into<String>) -> Self {
        Self::new(StatusCode::PAYLOAD_TOO_LARGE, "request_too_large", message)
    }
}

fn request_body_error(error: axum::Error) -> ApiError {
    let mut cause = error.source();
    while let Some(current) = cause {
        if current.is::<LengthLimitError>() {
            return ApiError::too_large(format!("request body exceeds {MAX_BODY_BYTES} bytes"));
        }
        cause = current.source();
    }
    ApiError::bad_request("request body could not be decoded")
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    code: self.code,
                    message: self.message,
                },
            }),
        )
            .into_response();
        if self.retry_after {
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

#[derive(Clone)]
struct CachedBatch {
    digest: [u8; 32],
    ack: TelemetryAck,
}

struct DedupeCache {
    capacity: usize,
    entries: HashMap<Uuid, CachedBatch>,
    order: VecDeque<Uuid>,
}

impl DedupeCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, batch_id: &Uuid, digest: &[u8; 32]) -> Result<Option<TelemetryAck>, ApiError> {
        let Some(cached) = self.entries.get(batch_id) else {
            return Ok(None);
        };
        if cached.digest != *digest {
            return Err(ApiError::new(
                StatusCode::CONFLICT,
                "idempotency_conflict",
                "batch_id was already used with different content",
            ));
        }
        let mut ack = cached.ack.clone();
        ack.deduplicated = true;
        Ok(Some(ack))
    }

    fn insert(&mut self, batch_id: Uuid, batch: CachedBatch) {
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(batch_id);
        self.entries.insert(batch_id, batch);
    }
}

struct TelemetryState {
    store: Arc<Store>,
    admission: Arc<Semaphore>,
    dedupe: Mutex<DedupeCache>,
    namespace_registrations: Mutex<HashMap<String, Arc<OnceCell<()>>>>,
    health: Arc<IngestHealth>,
}

struct PreparedBatch {
    batch_id: Uuid,
    batch_id_text: String,
    digest: [u8; 32],
    shards: Vec<PreparedShard>,
    record_count: usize,
}

struct PreparedShard {
    namespace: String,
    ipc: Vec<u8>,
}

impl TelemetryState {
    async fn ensure_namespace_registered(&self, namespace: &str) -> Result<(), ApiError> {
        let registration = {
            let mut registrations = self.namespace_registrations.lock().await;
            Arc::clone(
                registrations
                    .entry(namespace.to_string())
                    .or_insert_with(|| Arc::new(OnceCell::new())),
            )
        };
        self.health.declare_owned(namespace);
        let result = registration
            .get_or_try_init(|| async {
                let store = Arc::clone(&self.store);
                let namespace = namespace.to_string();
                let registered_namespace = namespace.clone();
                match tokio::task::spawn_blocking(move || {
                    let policy = StoragePolicy {
                        max_bytes: Some(
                            storage_max_bytes(&namespace)
                                .expect("only telemetry storage namespaces are registered"),
                        ),
                        ..StoragePolicy::default()
                    };
                    store.register_table(&namespace, telemetry_schema(), policy)
                })
                .await
                {
                    Ok(Ok(_)) => {
                        self.health.record_registered(&registered_namespace);
                        Ok(())
                    }
                    Ok(Err(error)) => Err(registration_error(&registered_namespace, error)),
                    Err(join) => Err(ApiError::new(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "storage_unavailable",
                        format!("telemetry namespace task failed: {join}"),
                    )),
                }
            })
            .await;
        if let Err(error) = result {
            self.health.record_failure(namespace, &error.message);
            return Err(error);
        }
        Ok(())
    }
}

fn registration_error(namespace: &str, error: StatsError) -> ApiError {
    match error {
        StatsError::SchemaConflict(message)
        | StatsError::SchemaValidation(message)
        | StatsError::InvalidNamespace(message) => ApiError::bad_request(format!(
            "telemetry namespace {namespace:?} is incompatible: {message}"
        )),
        error => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "storage_unavailable",
            format!("telemetry namespace {namespace:?} is unavailable: {error}"),
        ),
    }
}

/// Return an authenticated `POST /v1/telemetry` router with bounded admission and
/// process-local retry deduplication.
pub fn router(
    store: Arc<Store>,
    auth: Arc<AuthPolicy>,
    max_concurrent: usize,
    dedupe_capacity: usize,
    health: Arc<IngestHealth>,
) -> Router {
    let mut startup_namespaces = telemetry_storage_namespaces().collect::<Vec<_>>();
    if store.get_table_schema(TELEMETRY_NAMESPACE).is_ok() {
        startup_namespaces.push(TELEMETRY_NAMESPACE);
    }
    for namespace in &startup_namespaces {
        health.declare_owned(namespace);
    }
    let state = Arc::new(TelemetryState {
        store,
        admission: Arc::new(Semaphore::new(max_concurrent)),
        dedupe: Mutex::new(DedupeCache::new(dedupe_capacity)),
        namespace_registrations: Mutex::new(HashMap::new()),
        health,
    });
    // The schema is server-owned, so apply additive index-policy evolution at
    // startup even when telemetry reaches this store through StatsService or a
    // forwarder instead of the HTTP endpoint below. Requests share the OnceCell
    // and wait for this same registration if they arrive while it is running.
    for namespace in startup_namespaces {
        let startup_registration = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(error) = startup_registration
                .ensure_namespace_registered(namespace)
                .await
            {
                tracing::warn!(namespace, error = %error.message, "telemetry namespace startup registration failed");
            }
        });
    }
    Router::new()
        .route("/v1/telemetry", post(post_telemetry))
        .with_state(state)
        .layer(RequestDecompressionLayer::new())
        .layer(SetResponseHeaderLayer::if_not_present(
            ACCEPT_ENCODING,
            HeaderValue::from_static("zstd"),
        ))
        .layer(from_fn_with_state(auth, auth_gate))
}

async fn post_telemetry(
    State(state): State<Arc<TelemetryState>>,
    request: Request<Body>,
) -> Result<Json<TelemetryAck>, ApiError> {
    let permit = Arc::clone(&state.admission)
        .try_acquire_owned()
        .map_err(|_| {
            ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "admission_limited",
                "too many telemetry requests are active",
            )
        })?;
    let batch_id_header = required_header(request.headers(), "idempotency-key", 64)?;
    let content_type = required_header(request.headers(), CONTENT_TYPE.as_str(), 128)?;
    if content_type
        .split(';')
        .next()
        .map(str::trim)
        .filter(|value| value.eq_ignore_ascii_case("application/json"))
        .is_none()
    {
        return Err(ApiError::bad_request(
            "Content-Type must be application/json",
        ));
    }
    let origin_cluster = match request.extensions().get::<AuthIdentity>() {
        Some(AuthIdentity::Jwt { cluster }) => Some(cluster.clone()),
        Some(AuthIdentity::Network) => None,
        None => {
            return Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ERROR_CODE_INTERNAL,
                "authenticated identity is missing",
            ));
        }
    };
    let body = to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(request_body_error)?;
    let work = complete_request(
        Arc::clone(&state),
        permit,
        batch_id_header,
        body,
        origin_cluster,
    );
    let ack = run_to_completion(work).await.map_err(|join| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ERROR_CODE_INTERNAL,
            format!("telemetry request task failed: {join}"),
        )
    })??;
    Ok(Json(ack))
}

async fn complete_request(
    state: Arc<TelemetryState>,
    _permit: OwnedSemaphorePermit,
    batch_id_header: String,
    body: bytes::Bytes,
    origin_cluster: Option<String>,
) -> Result<TelemetryAck, ApiError> {
    let prepared = tokio::task::spawn_blocking(move || prepare_batch(&batch_id_header, &body))
        .await
        .map_err(|join| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ERROR_CODE_INTERNAL,
                format!("telemetry normalization task failed: {join}"),
            )
        })??;
    for shard in &prepared.shards {
        state.ensure_namespace_registered(&shard.namespace).await?;
    }

    // Holding this small process-local mutex through the append makes concurrent
    // same-ID requests share one decision. It is not durable state and vanishes
    // on restart by design.
    let mut dedupe = state.dedupe.lock().await;
    if let Some(ack) = dedupe.get(&prepared.batch_id, &prepared.digest)? {
        return Ok(ack);
    }
    let store = Arc::clone(&state.store);
    let shards = prepared.shards;
    tokio::task::spawn_blocking(move || {
        for shard in shards {
            store.write_rows(&shard.namespace, &shard.ipc, origin_cluster.as_deref())?;
        }
        Ok::<(), StatsError>(())
    })
    .await
    .map_err(|join| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ERROR_CODE_INTERNAL,
            format!("telemetry storage task failed: {join}"),
        )
    })?
    .map_err(store_error)?;

    let ack = TelemetryAck {
        version: TELEMETRY_VERSION,
        batch_id: prepared.batch_id_text,
        status: "accepted",
        deduplicated: false,
        record_count: prepared.record_count,
    };
    dedupe.insert(
        prepared.batch_id,
        CachedBatch {
            digest: prepared.digest,
            ack: ack.clone(),
        },
    );
    Ok(ack)
}

/// Await work in its own task. Dropping the waiter detaches rather than cancels
/// the work, so resources moved into the future stay owned through completion.
pub(super) async fn run_to_completion<F, T>(work: F) -> Result<T, tokio::task::JoinError>
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    tokio::spawn(work).await
}

fn required_header(headers: &HeaderMap, name: &str, max_bytes: usize) -> Result<String, ApiError> {
    let values = headers.get_all(name);
    if values.iter().count() != 1 {
        return Err(ApiError::bad_request(format!(
            "exactly one {name} header is required"
        )));
    }
    let value = values
        .iter()
        .next()
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| ApiError::bad_request(format!("{name} header is not valid text")))?;
    if value.len() > max_bytes {
        return Err(ApiError::too_large(format!(
            "{name} header exceeds {max_bytes} bytes"
        )));
    }
    Ok(value.to_string())
}

fn prepare_batch(idempotency_key: &str, body: &[u8]) -> Result<PreparedBatch, ApiError> {
    let header_id = Uuid::parse_str(idempotency_key)
        .map_err(|_| ApiError::bad_request("Idempotency-Key must be a UUID"))?;
    let batch: TelemetryBatch = serde_json::from_slice(body)
        .map_err(|error| ApiError::bad_request(format!("invalid telemetry JSON: {error}")))?;
    let body_id = Uuid::parse_str(&batch.batch_id)
        .map_err(|_| ApiError::bad_request("batch_id must be a UUID"))?;
    if header_id != body_id {
        return Err(ApiError::bad_request(
            "Idempotency-Key must equal the body batch_id",
        ));
    }
    let logical_namespace = batch
        .namespace
        .clone()
        .unwrap_or_else(|| TELEMETRY_NAMESPACE.to_string());
    if !matches_telemetry_namespace(&logical_namespace) {
        return Err(ApiError::bad_request(
            "namespace is not in the telemetry schema family",
        ));
    }
    validate_batch(&batch)?;
    let records = batch.records.iter().enumerate().collect::<Vec<_>>();
    let normalized = normalize_record_batch(&batch, &records)?;
    let shards = route_ingestion_batch(
        IngestionBatchSource::Declared(&logical_namespace),
        &normalized,
    )
    .map_err(telemetry_policy_error)?
    .into_iter()
    .map(|partition| {
        let schema = partition.batch.schema();
        Ok(PreparedShard {
            namespace: partition.destination.physical_namespace,
            ipc: encode_ipc(&schema, &[partition.batch]).map_err(|error| {
                ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ERROR_CODE_INTERNAL,
                    format!("could not encode telemetry: {error}"),
                )
            })?,
        })
    })
    .collect::<Result<Vec<_>, ApiError>>()?;
    Ok(PreparedBatch {
        batch_id: body_id,
        batch_id_text: batch.batch_id,
        digest: Sha256::digest(body).into(),
        shards,
        record_count: batch.records.len(),
    })
}

fn telemetry_policy_error(error: StatsError) -> ApiError {
    match error {
        StatsError::SchemaConflict(message)
        | StatsError::SchemaValidation(message)
        | StatsError::InvalidNamespace(message) => ApiError::bad_request(message),
        error => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            ERROR_CODE_INTERNAL,
            format!("telemetry policy failed: {error}"),
        ),
    }
}

#[cfg(test)]
pub(super) fn normalize_test_batch(idempotency_key: &str, body: &[u8]) -> Vec<u8> {
    let mut prepared = prepare_batch(idempotency_key, body)
        .unwrap_or_else(|error| panic!("could not normalize test telemetry: {}", error.message));
    assert_eq!(prepared.shards.len(), 1);
    prepared.shards.remove(0).ipc
}

fn validate_batch(batch: &TelemetryBatch) -> Result<(), ApiError> {
    if batch.version != TELEMETRY_VERSION {
        return Err(ApiError::bad_request(format!(
            "version must be {TELEMETRY_VERSION}"
        )));
    }
    validate_string(&batch.resource.service, "resource.service", false)?;
    validate_attributes(&batch.resource.attributes, "resource.attributes")?;
    for (field, value) in [
        ("resource.run_id", batch.resource.run_id.as_deref()),
        ("resource.job_id", batch.resource.job_id.as_deref()),
        (
            "resource.execution_uid",
            batch.resource.execution_uid.as_deref(),
        ),
        ("resource.region", batch.resource.region.as_deref()),
        ("resource.node_name", batch.resource.node_name.as_deref()),
        (
            "resource.process_index",
            batch.resource.process_index.as_deref(),
        ),
    ] {
        if let Some(value) = value {
            validate_string(value, field, false)?;
        }
    }
    if batch.records.is_empty() {
        return Err(ApiError::bad_request("records must not be empty"));
    }
    if batch.records.len() > MAX_RECORDS {
        return Err(ApiError::too_large(format!(
            "record count exceeds {MAX_RECORDS}"
        )));
    }
    for (index, record) in batch.records.iter().enumerate() {
        if record.timestamp_ms < 0 {
            return Err(ApiError::bad_request(format!(
                "records[{index}].timestamp_ms must be nonnegative"
            )));
        }
        validate_string(&record.name, &format!("records[{index}].name"), false)?;
        validate_attributes(&record.attributes, &format!("records[{index}].attributes"))?;
        if let Some(unit) = &record.unit {
            validate_string(unit, &format!("records[{index}].unit"), true)?;
        }
        match record.kind {
            RecordKind::Event
                if record.body.is_some() && record.value.is_none() && record.unit.is_none() =>
            {
                validate_json(record.body.as_ref().unwrap(), index)?;
            }
            RecordKind::Event => {
                return Err(ApiError::bad_request(format!(
                    "records[{index}] event requires body and forbids value/unit"
                )));
            }
            _ if record.value.is_some_and(f64::is_finite) && record.body.is_none() => {}
            _ => {
                return Err(ApiError::bad_request(format!(
                    "records[{index}] metric requires a finite value and forbids body"
                )));
            }
        }
    }
    Ok(())
}

fn validate_attributes(attributes: &BTreeMap<String, String>, field: &str) -> Result<(), ApiError> {
    if attributes.len() > MAX_ATTRIBUTES {
        return Err(ApiError::too_large(format!(
            "{field} contains more than {MAX_ATTRIBUTES} entries"
        )));
    }
    for (key, value) in attributes {
        validate_string(key, &format!("{field} key"), false)?;
        validate_string(value, &format!("{field}[{key}]"), false)?;
    }
    Ok(())
}

fn validate_string(value: &str, field: &str, allow_empty: bool) -> Result<(), ApiError> {
    if !allow_empty && value.is_empty() {
        return Err(ApiError::bad_request(format!("{field} must not be empty")));
    }
    if value.len() > MAX_STRING_BYTES {
        return Err(ApiError::too_large(format!(
            "{field} exceeds {MAX_STRING_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_json(root: &Value, record_index: usize) -> Result<(), ApiError> {
    let mut pending = vec![(root, 0_usize)];
    while let Some((value, depth)) = pending.pop() {
        if depth > MAX_JSON_DEPTH {
            return Err(ApiError::too_large(format!(
                "records[{record_index}].body exceeds JSON depth {MAX_JSON_DEPTH}"
            )));
        }
        match value {
            Value::String(text) => validate_string(text, "event body string", true)?,
            Value::Array(values) => pending.extend(values.iter().map(|value| (value, depth + 1))),
            Value::Object(values) => {
                for (key, value) in values {
                    validate_string(key, "event body key", false)?;
                    pending.push((value, depth + 1));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn normalize_record_batch(
    batch: &TelemetryBatch,
    records: &[(usize, &TelemetryRecord)],
) -> Result<RecordBatch, ApiError> {
    let resource_attributes = serde_json::to_string(&batch.resource.normalized_attributes())
        .map_err(|error| ApiError::bad_request(format!("invalid resource attributes: {error}")))?;
    let mut kinds = Vec::with_capacity(records.len());
    let mut names = Vec::with_capacity(records.len());
    let mut values = Vec::with_capacity(records.len());
    let mut bodies = Vec::with_capacity(records.len());
    let mut units = Vec::with_capacity(records.len());
    let mut attributes = Vec::with_capacity(records.len());
    let mut normalized_bytes = 0_usize;
    let dimensions = batch.resource.dimensions();
    let dimension_bytes = [
        dimensions.run_id,
        dimensions.job_id,
        dimensions.execution_uid,
        dimensions.region,
        dimensions.node_name,
        dimensions.process_index,
    ]
    .into_iter()
    .flatten()
    .map(str::len)
    .sum::<usize>();
    for (_, record) in records {
        let body = record
            .body
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|error| ApiError::bad_request(format!("invalid event body: {error}")))?;
        let attrs = serde_json::to_string(&record.attributes).map_err(|error| {
            ApiError::bad_request(format!("invalid record attributes: {error}"))
        })?;
        normalized_bytes = normalized_bytes
            .checked_add(NORMALIZED_ROW_OVERHEAD)
            .and_then(|size| size.checked_add(batch.batch_id.len()))
            .and_then(|size| size.checked_add(batch.resource.service.len()))
            .and_then(|size| size.checked_add(resource_attributes.len()))
            .and_then(|size| size.checked_add(dimension_bytes))
            .and_then(|size| size.checked_add(record.name.len()))
            .and_then(|size| size.checked_add(record.unit.as_deref().map_or(0, str::len)))
            .and_then(|size| size.checked_add(body.as_deref().map_or(0, str::len)))
            .and_then(|size| size.checked_add(attrs.len()))
            .ok_or_else(|| ApiError::too_large("normalized telemetry size overflow"))?;
        if normalized_bytes > MAX_NORMALIZED_BYTES {
            return Err(ApiError::too_large(format!(
                "normalized telemetry exceeds {MAX_NORMALIZED_BYTES} bytes"
            )));
        }
        kinds.push(record.kind.as_str());
        names.push(record.name.as_str());
        values.push(record.value);
        bodies.push(body);
        units.push(record.unit.as_deref());
        attributes.push(attrs);
    }
    let row_count = records.len();
    let schema = telemetry_schema();
    let arrow_schema = schema_to_arrow(&schema);
    RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int32Array::from(vec![TELEMETRY_VERSION as i32; row_count])),
            Arc::new(Int64Array::from(
                records
                    .iter()
                    .map(|(_, record)| record.timestamp_ms)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec![batch.batch_id.as_str(); row_count])),
            Arc::new(Int64Array::from_iter_values(
                records.iter().map(|(index, _)| *index as i64),
            )),
            Arc::new(StringArray::from(vec![
                batch.resource.service.as_str();
                row_count
            ])),
            Arc::new(StringArray::from(vec![dimensions.run_id; row_count])),
            Arc::new(StringArray::from(vec![dimensions.job_id; row_count])),
            Arc::new(StringArray::from(vec![dimensions.execution_uid; row_count])),
            Arc::new(StringArray::from(vec![dimensions.region; row_count])),
            Arc::new(StringArray::from(vec![dimensions.node_name; row_count])),
            Arc::new(StringArray::from(vec![dimensions.process_index; row_count])),
            Arc::new(StringArray::from(kinds)),
            Arc::new(StringArray::from(names)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from_iter(
                bodies.iter().map(|value| value.as_deref()),
            )),
            Arc::new(StringArray::from(units)),
            Arc::new(StringArray::from(vec![
                resource_attributes.as_str();
                row_count
            ])),
            Arc::new(StringArray::from_iter_values(
                attributes.iter().map(String::as_str),
            )),
        ],
    )
    .map_err(|error| ApiError::bad_request(format!("could not normalize telemetry: {error}")))
}

pub(crate) fn telemetry_schema() -> Schema {
    Schema::new(
        vec![
            Column::new("schema_version", ColumnType::COLUMN_TYPE_INT32, false),
            Column::new("timestamp_ms", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("batch_id", ColumnType::COLUMN_TYPE_STRING, false),
            Column::new("record_index", ColumnType::COLUMN_TYPE_INT64, false),
            Column::new("service", ColumnType::COLUMN_TYPE_STRING, false).with_value_counts(),
            Column::new(RUN_ID_COLUMN, ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(JOB_ID_COLUMN, ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(EXECUTION_UID_COLUMN, ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(REGION_COLUMN, ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(NODE_NAME_COLUMN, ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(PROCESS_INDEX_COLUMN, ColumnType::COLUMN_TYPE_STRING, true)
                .with_exact_values([PRIMARY_PROCESS_INDEX]),
            Column::new("kind", ColumnType::COLUMN_TYPE_STRING, false).with_value_counts(),
            // Metric names are the primary substring-search target
            // (`name LIKE '%nccl%'`), so this column carries the trigram index.
            Column::new("name", ColumnType::COLUMN_TYPE_STRING, false)
                .with_trigram_index()
                .with_exact_values(
                    TRAINING_STATUS_NAMES
                        .into_iter()
                        .chain(TRAINING_LOSS_NAMES)
                        .chain(TRAINING_RUN_NAMES)
                        .chain(HOST_METRIC_NAMES)
                        .chain(ACCELERATOR_METRIC_NAMES),
                )
                .with_value_counts(),
            Column::new("value", ColumnType::COLUMN_TYPE_FLOAT64, true),
            Column::new("body_json", ColumnType::COLUMN_TYPE_STRING, true),
            Column::new("unit", ColumnType::COLUMN_TYPE_STRING, true),
            Column::new(
                "resource_attributes_json",
                ColumnType::COLUMN_TYPE_STRING,
                false,
            ),
            Column::new("attributes_json", ColumnType::COLUMN_TYPE_STRING, false),
        ],
        "timestamp_ms",
    )
    .with_sort_columns(["service", RUN_ID_COLUMN, "name", "timestamp_ms"])
    .with_max_row_group_rows(TELEMETRY_MAX_ROW_GROUP_ROWS)
    .with_covering_projection(CoveringProjection::new(
        "training-status",
        "name",
        TRAINING_STATUS_NAMES,
        [
            "seq",
            "timestamp_ms",
            "service",
            RUN_ID_COLUMN,
            JOB_ID_COLUMN,
            "name",
            "value",
            "resource_attributes_json",
            "attributes_json",
            "cluster",
        ],
    ))
    .with_covering_projection(CoveringProjection::new(
        "training-process-zero",
        PROCESS_INDEX_COLUMN,
        [PRIMARY_PROCESS_INDEX],
        [
            "seq",
            "timestamp_ms",
            "service",
            RUN_ID_COLUMN,
            EXECUTION_UID_COLUMN,
            PROCESS_INDEX_COLUMN,
            "name",
            "value",
            "cluster",
        ],
    ))
    .with_covering_projection(CoveringProjection::new(
        "training-run-attribution",
        "name",
        TRAINING_RUN_NAMES,
        [
            "timestamp_ms",
            "service",
            RUN_ID_COLUMN,
            JOB_ID_COLUMN,
            NODE_NAME_COLUMN,
            PROCESS_INDEX_COLUMN,
            "name",
            "resource_attributes_json",
            "cluster",
        ],
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-power",
        "name",
        ["gpu_power_watts"],
        [
            "timestamp_ms",
            "service",
            NODE_NAME_COLUMN,
            "name",
            "value",
            "resource_attributes_json",
            "attributes_json",
            "cluster",
        ],
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-memory",
        "name",
        ["gpu_memory_total_bytes", "gpu_memory_used_bytes"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-faults",
        "name",
        [
            "gpu_pcie_replay_errors",
            "gpu_row_remap_failures",
            "gpu_xid_error_code",
        ],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-interconnect",
        "name",
        [
            "gpu_nvlink_receive_bytes_per_second",
            "gpu_nvlink_transmit_bytes_per_second",
            "gpu_pcie_receive_bytes_per_second",
            "gpu_pcie_transmit_bytes_per_second",
        ],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-inventory",
        "name",
        ["hardware_inventory"],
        [
            "timestamp_ms",
            "service",
            NODE_NAME_COLUMN,
            "name",
            "attributes_json",
            "cluster",
        ],
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-sm-activity",
        "name",
        ["gpu_sm_active_ratio"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-temperature",
        "name",
        ["gpu_memory_temperature_celsius", "gpu_temperature_celsius"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-tensor-activity",
        "name",
        ["gpu_tensor_active_ratio"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "accelerator-utilization",
        "name",
        ["gpu_utilization_percent"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "node-host-network",
        "name",
        ["node_network_receive_bytes", "node_network_transmit_bytes"],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_covering_projection(CoveringProjection::new(
        "node-host-utilization",
        "name",
        [
            "node_cpu_utilization_percent",
            "node_disk_total_bytes",
            "node_disk_used_bytes",
            "node_memory_total_bytes",
            "node_memory_used_bytes",
        ],
        DEVICE_METRIC_PROJECTION_COLUMNS,
    ))
    .with_grouped_extrema(GroupExtremaConfig::new(
        "service",
        "resource_attributes_json",
        JOB_ID_COLUMN,
        "timestamp_ms",
    ))
}

fn store_error(error: StatsError) -> ApiError {
    ApiError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "storage_unavailable",
        format!("telemetry write failed: {error}"),
    )
}
