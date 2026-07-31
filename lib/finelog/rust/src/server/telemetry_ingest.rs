// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

//! Dependency-light REST telemetry ingestion and durable acknowledgement.
//!
//! Request media type also selects the success/schema-response media type.
//! Custom endpoint failures use one stable JSON `{code,message}` envelope; OTLP
//! failures use `google.rpc.Status` in the request media type. Admission wraps
//! body polling, decode, normalization, and storage under one deadline. Heavy
//! work runs on a bounded blocking pool. A durable batch intent reserves the
//! global key/digest before children; the visible batch completion marker is
//! persisted only after every child is durable.

use std::cell::Cell as LocalCell;
use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{
    ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, ListArray, MapBuilder,
    MapFieldNames, RecordBatch, StringArray, StringBuilder,
};
use arrow::datatypes::{Float64Type, Int64Type};
use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{Extension, Request, State};
use axum::http::header::{CONTENT_ENCODING, CONTENT_TYPE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use buffa::Message;
use flate2::read::GzDecoder;
use opentelemetry_proto::tonic::collector::logs::v1::{
    ExportLogsPartialSuccess, ExportLogsServiceRequest, ExportLogsServiceResponse,
};
use opentelemetry_proto::tonic::collector::metrics::v1::{
    ExportMetricsPartialSuccess, ExportMetricsServiceRequest, ExportMetricsServiceResponse,
};
use opentelemetry_proto::tonic::common::v1::{any_value, KeyValue};
use opentelemetry_proto::tonic::logs::v1::LogRecord;
use opentelemetry_proto::tonic::metrics::v1::{
    metric, number_data_point, AggregationTemporality, HistogramDataPoint, Metric, NumberDataPoint,
};
use opentelemetry_proto::tonic::resource::v1::Resource as OtlpResource;
use prost::Message as ProstMessage;
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tokio::time::{Duration, Instant};

use crate::errors::StatsError;
use crate::proto::finelog::stats::ColumnType;
use crate::proto::finelog::telemetry::{
    TelemetryArtifactV1, TelemetryBatchV1, TelemetryCommitV1, TelemetryEventV1, TelemetryLogV1,
    TelemetryMetricV1, TelemetryRecordV1, TelemetryResourceV1, TelemetryValidationErrorV1,
    TelemetryWriteAckV1,
};
use crate::server::auth::AuthIdentity;
use crate::server::MAX_MESSAGE_BYTES;
use crate::store::ipc::encode_ipc;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{schema_to_arrow, Schema, MAX_WRITE_ROWS_BYTES};
use crate::store::store::{log_registered_schema, LOG_NAMESPACE_NAME};
use crate::store::telemetry_catalog::{
    catalog, schema_for_namespace, ARTIFACT_NAMESPACE, BATCH_INTENT_NAMESPACE, BATCH_NAMESPACE,
    CATALOG_VERSION, EVENT_NAMESPACE,
};
use crate::store::types::{ReceiptState, WriteRowsResult};
use crate::store::Store;

const CUSTOM_ENDPOINT: &str = "/v1/telemetry";
const OTLP_METRICS_ENDPOINT: &str = "/v1/metrics";
const OTLP_LOGS_ENDPOINT: &str = "/v1/logs";
const REST_DIGEST_DOMAIN: &[u8] = b"finelog.telemetry.rest.v1\0";
const SUB_BATCH_ID_DOMAIN: &[u8] = b"finelog.telemetry.sub_batch.v1\0";
const INTERNAL_DELIVERY_CLASS: &str = "internal";
const SCHEMA_VERSION: i32 = 1;
const MAX_IDEMPOTENCY_KEY_BYTES: usize = 128;
const MAX_REST_RECORDS: usize = 10_000;
const MAX_NORMALIZED_BYTES: usize = 32 * 1024 * 1024;
const MAX_STRING_BYTES: usize = 64 * 1024;
const MAX_STRUCTURAL_ITEMS: usize = MAX_REST_RECORDS * 64;
const MAX_STRUCTURAL_DEPTH: usize = 32;
const MAX_VALIDATION_ERRORS: usize = 64;
const MAX_CONCURRENT_REQUESTS: usize = 4;
const MAX_CONCURRENT_BLOCKING_TASKS: usize = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const RETRY_AFTER_SECONDS: &str = "1";
const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
const ZSTD_WINDOW_LOG_MAX: u32 = 23;

#[derive(Clone, Copy)]
enum WireFormat {
    Json,
    Protobuf,
}

impl WireFormat {
    fn from_headers(headers: &HeaderMap) -> Result<Self, ApiError> {
        let value = headers
            .get(CONTENT_TYPE)
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "missing_content_type",
                    "content-type is required",
                )
            })?
            .to_str()
            .map_err(|_| {
                ApiError::new(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_content_type",
                    "content-type must be valid ASCII",
                )
            })?
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match value.as_str() {
            "application/json" => Ok(Self::Json),
            "application/x-protobuf" | "application/protobuf" => Ok(Self::Protobuf),
            _ => Err(ApiError::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_content_type",
                "content-type must be application/json or application/x-protobuf",
            )),
        }
    }

    fn content_type(self) -> &'static str {
        match self {
            Self::Json => "application/json",
            Self::Protobuf => "application/x-protobuf",
        }
    }

    fn response_from_headers(headers: &HeaderMap) -> Self {
        let Some(value) = headers
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
        else {
            return Self::Protobuf;
        };
        if value
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
        {
            Self::Json
        } else {
            Self::Protobuf
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    Custom,
    OtlpMetrics,
    OtlpLogs,
}

impl Endpoint {
    fn from_path(path: &str) -> Option<Self> {
        match path {
            CUSTOM_ENDPOINT => Some(Self::Custom),
            OTLP_METRICS_ENDPOINT => Some(Self::OtlpMetrics),
            OTLP_LOGS_ENDPOINT => Some(Self::OtlpLogs),
            _ => None,
        }
    }

    fn is_otlp(self) -> bool {
        self != Self::Custom
    }
}

#[derive(Debug, Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    message: String,
}

#[derive(Debug)]
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
            retry_after: false,
        }
    }

    fn retryable_overload(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "write_buffer_full",
            message: message.into(),
            retry_after: true,
        }
    }

    fn retryable(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        debug_assert!(matches!(
            status,
            StatusCode::TOO_MANY_REQUESTS
                | StatusCode::BAD_GATEWAY
                | StatusCode::SERVICE_UNAVAILABLE
                | StatusCode::GATEWAY_TIMEOUT
        ));
        Self {
            status,
            code,
            message: message.into(),
            retry_after: true,
        }
    }

    fn from_store(error: StatsError) -> Self {
        match error {
            StatsError::IdempotencyConflict(message) => {
                Self::new(StatusCode::CONFLICT, "idempotency_conflict", message)
            }
            StatsError::SchemaValidation(message) | StatsError::InvalidNamespace(message) => {
                Self::new(StatusCode::BAD_REQUEST, "invalid_request", message)
            }
            StatsError::WriteBufferFull(message) => Self::retryable_overload(message),
            StatsError::DeadlineExceeded(message) => {
                Self::retryable(StatusCode::GATEWAY_TIMEOUT, "durability_timeout", message)
            }
            StatsError::Internal(message) => Self::retryable(
                StatusCode::SERVICE_UNAVAILABLE,
                "store_unavailable",
                message,
            ),
            StatsError::SchemaConflict(message) | StatsError::NamespaceNotFound(message) => {
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
            }
            StatsError::QueryResultTooLarge(message) => {
                Self::new(StatusCode::PAYLOAD_TOO_LARGE, "response_too_large", message)
            }
        }
    }

    fn into_endpoint_response(self, endpoint: Endpoint, format: WireFormat) -> Response {
        if endpoint.is_otlp() {
            self.into_otlp_response(format)
        } else {
            self.into_custom_response()
        }
    }

    fn into_custom_response(self) -> Response {
        let status = self.status;
        let body = serde_json::to_vec(&ErrorBody {
            code: self.code,
            message: self.message.clone(),
        })
        .expect("REST error serialization is infallible");
        Self::finish_response(response(status, "application/json", body), self.retry_after)
    }

    fn into_otlp_response(self, format: WireFormat) -> Response {
        let status_code = self.status;
        let status = GoogleRpcStatus {
            code: google_rpc_code(status_code),
            message: format!("{}: {}", self.code, self.message),
        };
        let body = match format {
            WireFormat::Json => serde_json::to_vec(&serde_json::json!({
                "code": status.code,
                "message": status.message,
                "details": [],
            }))
            .expect("OTLP error JSON serialization is infallible"),
            WireFormat::Protobuf => status.encode_to_vec(),
        };
        Self::finish_response(
            response(status_code, format.content_type(), body),
            self.retry_after,
        )
    }

    fn finish_response(mut response: Response, retry_after: bool) -> Response {
        if retry_after {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                HeaderValue::from_static(RETRY_AFTER_SECONDS),
            );
        }
        response
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        self.into_custom_response()
    }
}

#[derive(Clone, PartialEq, prost::Message)]
struct GoogleRpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
}

fn google_rpc_code(status: StatusCode) -> i32 {
    match status {
        StatusCode::BAD_REQUEST | StatusCode::UNSUPPORTED_MEDIA_TYPE => 3,
        StatusCode::GATEWAY_TIMEOUT => 4,
        StatusCode::CONFLICT => 6,
        StatusCode::PAYLOAD_TOO_LARGE | StatusCode::TOO_MANY_REQUESTS => 8,
        StatusCode::UNAUTHORIZED => 16,
        StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE => 14,
        _ => 13,
    }
}

pub(crate) fn auth_error_response(request: &Request) -> Option<Response> {
    let endpoint = Endpoint::from_path(request.uri().path())?;
    let format = WireFormat::response_from_headers(request.headers());
    Some(
        ApiError::new(
            StatusCode::UNAUTHORIZED,
            "unauthenticated",
            "finelog: unauthorized",
        )
        .into_endpoint_response(endpoint, format),
    )
}

#[derive(Clone, Debug)]
enum Cell {
    String(String),
    Int32(i32),
    Int64(i64),
    Float64(f64),
    Float64List(Vec<f64>),
    Int64List(Vec<i64>),
    Map(BTreeMap<String, String>),
}

type Row = BTreeMap<String, Cell>;

#[derive(Clone, Debug)]
struct RoutedRecord {
    namespace: String,
    delivery_class: String,
    row: Row,
}

struct NamespaceCommit {
    namespace: String,
    result: WriteRowsResult,
}

struct PreparedNamespace {
    namespace: String,
    batch_id: String,
    arrow_ipc: Vec<u8>,
}

struct CommitOutcome {
    commits: Vec<NamespaceCommit>,
    parent_deduplicated: bool,
}

struct IngestState {
    store: Arc<Store>,
    requests: Arc<Semaphore>,
    blocking_tasks: Arc<Semaphore>,
    schema_registration: Arc<tokio::sync::Mutex<()>>,
    request_timeout: Duration,
    #[cfg(test)]
    fail_after_child_append: std::sync::Mutex<Option<String>>,
}

/// Routes mounted ahead of the SPA/connect fallback.
pub fn router(store: Arc<Store>) -> Router {
    router_with_state(Arc::new(IngestState {
        store,
        requests: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
        blocking_tasks: Arc::new(Semaphore::new(MAX_CONCURRENT_BLOCKING_TASKS)),
        schema_registration: Arc::new(tokio::sync::Mutex::new(())),
        request_timeout: REQUEST_TIMEOUT,
        #[cfg(test)]
        fail_after_child_append: std::sync::Mutex::new(None),
    }))
}

fn router_with_state(state: Arc<IngestState>) -> Router {
    Router::new()
        .route(CUSTOM_ENDPOINT, post(write_telemetry))
        .route(OTLP_METRICS_ENDPOINT, post(write_otlp_metrics))
        .route(OTLP_LOGS_ENDPOINT, post(write_otlp_logs))
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&state),
            admission_gate,
        ))
        .with_state(state)
}

#[derive(Clone, Copy)]
struct RequestDeadline(Instant);

async fn admission_gate(
    State(state): State<Arc<IngestState>>,
    mut request: Request,
    next: Next,
) -> Response {
    let endpoint = Endpoint::from_path(request.uri().path()).expect("telemetry route path");
    let format = WireFormat::response_from_headers(request.headers());
    let permit = match Arc::clone(&state.requests).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            return ApiError::retryable_overload(format!(
                "at most {MAX_CONCURRENT_REQUESTS} telemetry requests may be admitted"
            ))
            .into_endpoint_response(endpoint, format);
        }
    };
    let deadline = Instant::now() + state.request_timeout;
    request.extensions_mut().insert(RequestDeadline(deadline));
    let response = tokio::time::timeout_at(deadline, next.run(request)).await;
    drop(permit);
    match response {
        Ok(response) => response,
        Err(_) => request_deadline().into_endpoint_response(endpoint, format),
    }
}

async fn write_telemetry(
    State(state): State<Arc<IngestState>>,
    Extension(identity): Extension<AuthIdentity>,
    Extension(RequestDeadline(deadline)): Extension<RequestDeadline>,
    request: Request,
) -> Response {
    let format = WireFormat::response_from_headers(request.headers());
    match write_telemetry_inner(state, identity, deadline, request).await {
        Ok(response) => response,
        Err(error) => error.into_endpoint_response(Endpoint::Custom, format),
    }
}

async fn write_telemetry_inner(
    state: Arc<IngestState>,
    identity: AuthIdentity,
    deadline: Instant,
    request: Request,
) -> Result<Response, ApiError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let format = WireFormat::from_headers(&headers)?;
    let idempotency_key = required_idempotency_key(&headers)?;
    let body = read_wire_body(body).await?;
    let body = decode_body(&state, &headers, body, deadline).await?;
    let (body, batch) = run_blocking(&state, deadline, move || {
        preflight_structure(Endpoint::Custom, format, &body)?;
        let batch = decode_custom_batch(format, &body)?;
        check_custom_admission(&batch)?;
        Ok((body, batch))
    })
    .await?;
    let batch_id = batch.batch_id.clone().unwrap_or_default();
    if idempotency_key != batch_id {
        return Ok(validation_response(
            format,
            &batch_id,
            vec![validation_error(
                -1,
                "batch_id",
                "must equal the Idempotency-Key header",
            )],
        ));
    }

    let validation_identity = identity.clone();
    let accepted_records = batch.records.len();
    let records = match run_blocking(&state, deadline, move || {
        Ok(validate_custom_batch(&batch, &validation_identity))
    })
    .await?
    {
        Ok(records) => records,
        Err(errors) => return Ok(validation_response(format, &batch_id, errors)),
    };
    let digest = request_digest(&identity, CUSTOM_ENDPOINT, format.content_type(), &body);
    let outcome = commit_records(
        Arc::clone(&state),
        identity_origin(&identity),
        batch_id.clone(),
        format.content_type().to_string(),
        digest,
        records,
        deadline,
    )
    .await?;
    let ack = success_ack(&batch_id, accepted_records, &outcome);
    let status = if outcome.parent_deduplicated {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok(encode_custom_response(status, format, &ack))
}

async fn write_otlp_metrics(
    State(state): State<Arc<IngestState>>,
    Extension(identity): Extension<AuthIdentity>,
    Extension(RequestDeadline(deadline)): Extension<RequestDeadline>,
    request: Request,
) -> Response {
    let format = WireFormat::response_from_headers(request.headers());
    match write_otlp_metrics_inner(state, identity, deadline, request).await {
        Ok(response) => response,
        Err(error) => error.into_endpoint_response(Endpoint::OtlpMetrics, format),
    }
}

async fn write_otlp_metrics_inner(
    state: Arc<IngestState>,
    identity: AuthIdentity,
    deadline: Instant,
    request: Request,
) -> Result<Response, ApiError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let format = WireFormat::from_headers(&headers)?;
    let explicit_batch_id = optional_idempotency_key(&headers)?;
    let body = read_wire_body(body).await?;
    let body = decode_body(&state, &headers, body, deadline).await?;
    let (body, request): (Vec<u8>, ExportMetricsServiceRequest) =
        run_blocking(&state, deadline, move || {
            preflight_structure(Endpoint::OtlpMetrics, format, &body)?;
            let request = decode_otlp(format, &body, "ExportMetricsServiceRequest")?;
            check_otlp_metric_admission(&request)?;
            Ok((body, request))
        })
        .await?;
    let digest = request_digest(
        &identity,
        OTLP_METRICS_ENDPOINT,
        format.content_type(),
        &body,
    );
    let batch_id = explicit_batch_id.unwrap_or_else(|| format!("otlp-{digest}"));
    let normalize_batch_id = batch_id.clone();
    let normalize_identity = identity.clone();
    let (records, rejected, reasons) = run_blocking(&state, deadline, move || {
        Ok(normalize_otlp_metrics(
            &request,
            &normalize_batch_id,
            &normalize_identity,
        ))
    })
    .await?;
    commit_records(
        Arc::clone(&state),
        identity_origin(&identity),
        batch_id,
        format.content_type().to_string(),
        digest,
        records,
        deadline,
    )
    .await?;
    let response_message = ExportMetricsServiceResponse {
        partial_success: (rejected > 0).then(|| ExportMetricsPartialSuccess {
            rejected_data_points: rejected,
            error_message: partial_success_message(&reasons),
        }),
    };
    Ok(encode_otlp_response(format, &response_message))
}

async fn write_otlp_logs(
    State(state): State<Arc<IngestState>>,
    Extension(identity): Extension<AuthIdentity>,
    Extension(RequestDeadline(deadline)): Extension<RequestDeadline>,
    request: Request,
) -> Response {
    let format = WireFormat::response_from_headers(request.headers());
    match write_otlp_logs_inner(state, identity, deadline, request).await {
        Ok(response) => response,
        Err(error) => error.into_endpoint_response(Endpoint::OtlpLogs, format),
    }
}

async fn write_otlp_logs_inner(
    state: Arc<IngestState>,
    identity: AuthIdentity,
    deadline: Instant,
    request: Request,
) -> Result<Response, ApiError> {
    let (parts, body) = request.into_parts();
    let headers = parts.headers;
    let format = WireFormat::from_headers(&headers)?;
    let explicit_batch_id = optional_idempotency_key(&headers)?;
    let body = read_wire_body(body).await?;
    let body = decode_body(&state, &headers, body, deadline).await?;
    let (body, request): (Vec<u8>, ExportLogsServiceRequest) =
        run_blocking(&state, deadline, move || {
            preflight_structure(Endpoint::OtlpLogs, format, &body)?;
            let request = decode_otlp(format, &body, "ExportLogsServiceRequest")?;
            check_otlp_log_admission(&request)?;
            Ok((body, request))
        })
        .await?;
    let digest = request_digest(&identity, OTLP_LOGS_ENDPOINT, format.content_type(), &body);
    let batch_id = explicit_batch_id.unwrap_or_else(|| format!("otlp-{digest}"));
    let normalize_batch_id = batch_id.clone();
    let normalize_identity = identity.clone();
    let (records, rejected, reasons) = run_blocking(&state, deadline, move || {
        Ok(normalize_otlp_logs(
            &request,
            &normalize_batch_id,
            &normalize_identity,
        ))
    })
    .await?;
    commit_records(
        Arc::clone(&state),
        identity_origin(&identity),
        batch_id,
        format.content_type().to_string(),
        digest,
        records,
        deadline,
    )
    .await?;
    let response_message = ExportLogsServiceResponse {
        partial_success: (rejected > 0).then(|| ExportLogsPartialSuccess {
            rejected_log_records: rejected,
            error_message: partial_success_message(&reasons),
        }),
    };
    Ok(encode_otlp_response(format, &response_message))
}

struct StructuralQuota {
    items: LocalCell<usize>,
    records: LocalCell<usize>,
    violation: LocalCell<Option<&'static str>>,
}

impl StructuralQuota {
    fn new() -> Self {
        Self {
            items: LocalCell::new(0),
            records: LocalCell::new(0),
            violation: LocalCell::new(None),
        }
    }

    fn repeated(&self, local_count: &mut usize, field: &'static str) -> Result<(), ApiError> {
        *local_count = local_count
            .checked_add(1)
            .ok_or_else(|| structural_limit(field))?;
        if *local_count > MAX_REST_RECORDS {
            return Err(structural_limit(field));
        }
        self.item(field)
    }

    fn record(&self, local_count: &mut usize, field: &'static str) -> Result<(), ApiError> {
        self.repeated(local_count, field)?;
        let records = self
            .records
            .get()
            .checked_add(1)
            .ok_or_else(|| structural_limit(field))?;
        if records > MAX_REST_RECORDS {
            return Err(structural_limit(field));
        }
        self.records.set(records);
        Ok(())
    }

    fn item(&self, field: &'static str) -> Result<(), ApiError> {
        let items = self
            .items
            .get()
            .checked_add(1)
            .ok_or_else(|| structural_limit(field))?;
        if items > MAX_STRUCTURAL_ITEMS {
            return Err(structural_limit(field));
        }
        self.items.set(items);
        Ok(())
    }

    fn json_item(&self, local_count: &mut usize, field: &'static str) -> bool {
        match self.repeated(local_count, field) {
            Ok(()) => true,
            Err(_) => {
                self.violation.set(Some(field));
                false
            }
        }
    }
}

fn structural_limit(field: &'static str) -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "structural_limit_exceeded",
        format!(
            "{field} exceeds the pre-decode structural quota: {MAX_REST_RECORDS} per container and {MAX_STRUCTURAL_ITEMS} total items"
        ),
    )
}

fn preflight_structure(
    endpoint: Endpoint,
    format: WireFormat,
    body: &[u8],
) -> Result<(), ApiError> {
    match format {
        WireFormat::Json => preflight_json(body),
        WireFormat::Protobuf => {
            let quota = StructuralQuota::new();
            match endpoint {
                Endpoint::Custom => scan_custom_batch(body, &quota),
                Endpoint::OtlpMetrics => scan_otlp_metrics_request(body, &quota),
                Endpoint::OtlpLogs => scan_otlp_logs_request(body, &quota),
            }
        }
    }
}

fn invalid_protobuf_structure(message: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, "invalid_protobuf", message)
}

fn read_protobuf_varint(input: &[u8], position: &mut usize) -> Result<u64, ApiError> {
    let mut value = 0_u64;
    for shift in (0..=63).step_by(7) {
        let byte = *input.get(*position).ok_or_else(|| {
            invalid_protobuf_structure("truncated protobuf varint during structural preflight")
        })?;
        *position += 1;
        if shift == 63 && byte > 1 {
            return Err(invalid_protobuf_structure(
                "protobuf varint overflow during structural preflight",
            ));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(invalid_protobuf_structure(
        "protobuf varint overflow during structural preflight",
    ))
}

enum ProtobufWireValue<'a> {
    Varint,
    Fixed64,
    LengthDelimited(&'a [u8]),
    Fixed32,
}

fn scan_protobuf_wire_fields(
    input: &[u8],
    quota: &StructuralQuota,
    mut field_value: impl FnMut(u32, ProtobufWireValue<'_>) -> Result<(), ApiError>,
) -> Result<(), ApiError> {
    let mut position = 0;
    while position < input.len() {
        let key = read_protobuf_varint(input, &mut position)?;
        let field = u32::try_from(key >> 3)
            .ok()
            .filter(|field| *field > 0)
            .ok_or_else(|| invalid_protobuf_structure("protobuf field number is invalid"))?;
        quota.item("protobuf wire fields")?;
        match key & 0x07 {
            0 => {
                read_protobuf_varint(input, &mut position)?;
                field_value(field, ProtobufWireValue::Varint)?;
            }
            1 => {
                position = position.checked_add(8).ok_or_else(|| {
                    invalid_protobuf_structure("protobuf fixed64 offset overflow")
                })?;
                field_value(field, ProtobufWireValue::Fixed64)?;
            }
            2 => {
                let length = usize::try_from(read_protobuf_varint(input, &mut position)?)
                    .map_err(|_| invalid_protobuf_structure("protobuf field length overflow"))?;
                let end = position
                    .checked_add(length)
                    .ok_or_else(|| invalid_protobuf_structure("protobuf field length overflow"))?;
                let value = input.get(position..end).ok_or_else(|| {
                    invalid_protobuf_structure(
                        "truncated length-delimited protobuf field during structural preflight",
                    )
                })?;
                position = end;
                field_value(field, ProtobufWireValue::LengthDelimited(value))?;
            }
            3 => {
                skip_protobuf_group(input, &mut position, field, quota, 1)?;
            }
            4 => {
                return Err(invalid_protobuf_structure(format!(
                    "unexpected protobuf end group for field {field}"
                )));
            }
            5 => {
                position = position.checked_add(4).ok_or_else(|| {
                    invalid_protobuf_structure("protobuf fixed32 offset overflow")
                })?;
                field_value(field, ProtobufWireValue::Fixed32)?;
            }
            _ => {
                return Err(invalid_protobuf_structure(
                    "invalid protobuf wire type during structural preflight",
                ));
            }
        }
        if position > input.len() {
            return Err(invalid_protobuf_structure(
                "truncated fixed-width protobuf field during structural preflight",
            ));
        }
    }
    Ok(())
}

fn skip_protobuf_group(
    input: &[u8],
    position: &mut usize,
    group_field: u32,
    quota: &StructuralQuota,
    depth: usize,
) -> Result<(), ApiError> {
    if depth > MAX_STRUCTURAL_DEPTH {
        return Err(structural_limit("protobuf group nesting"));
    }

    let mut fields = 0;
    while *position < input.len() {
        let key = read_protobuf_varint(input, position)?;
        let field = u32::try_from(key >> 3)
            .ok()
            .filter(|field| *field > 0)
            .ok_or_else(|| invalid_protobuf_structure("protobuf field number is invalid"))?;
        let wire_type = key & 0x07;
        if wire_type == 4 {
            if field == group_field {
                return Ok(());
            }
            return Err(invalid_protobuf_structure(format!(
                "protobuf end group for field {field} does not match start field {group_field}"
            )));
        }

        quota.repeated(&mut fields, "protobuf group fields")?;
        match wire_type {
            0 => {
                read_protobuf_varint(input, position)?;
            }
            1 => {
                *position = position.checked_add(8).ok_or_else(|| {
                    invalid_protobuf_structure("protobuf fixed64 offset overflow")
                })?;
            }
            2 => {
                let length = usize::try_from(read_protobuf_varint(input, position)?)
                    .map_err(|_| invalid_protobuf_structure("protobuf field length overflow"))?;
                *position = position
                    .checked_add(length)
                    .ok_or_else(|| invalid_protobuf_structure("protobuf field length overflow"))?;
            }
            3 => {
                skip_protobuf_group(input, position, field, quota, depth + 1)?;
            }
            5 => {
                *position = position.checked_add(4).ok_or_else(|| {
                    invalid_protobuf_structure("protobuf fixed32 offset overflow")
                })?;
            }
            _ => {
                return Err(invalid_protobuf_structure(
                    "invalid protobuf wire type during structural preflight",
                ));
            }
        }
        if *position > input.len() {
            return Err(invalid_protobuf_structure(
                "truncated protobuf group field during structural preflight",
            ));
        }
    }

    Err(invalid_protobuf_structure(format!(
        "protobuf start group for field {group_field} has no matching end group"
    )))
}

fn scan_protobuf_fields(
    input: &[u8],
    quota: &StructuralQuota,
    mut length_delimited: impl FnMut(u32, &[u8]) -> Result<(), ApiError>,
) -> Result<(), ApiError> {
    scan_protobuf_wire_fields(input, quota, |field, value| {
        if let ProtobufWireValue::LengthDelimited(value) = value {
            length_delimited(field, value)?;
        }
        Ok(())
    })
}

fn check_wire_string(value: &[u8], field: &'static str) -> Result<(), ApiError> {
    if value.len() > MAX_STRING_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "string_limit_exceeded",
            format!(
                "{field} is {} bytes; limit is {MAX_STRING_BYTES}",
                value.len()
            ),
        ));
    }
    Ok(())
}

fn scan_custom_batch(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut records = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        2 => check_wire_string(value, "catalog_version"),
        3 => check_wire_string(value, "batch_id"),
        4 => {
            quota.record(&mut records, "custom records")?;
            scan_custom_record(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_custom_record(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    scan_protobuf_fields(input, quota, |field, value| match field {
        2 => check_wire_string(value, "record.signal"),
        5 => scan_custom_resource(value, quota),
        6 => scan_custom_metric(value, quota),
        7 => scan_custom_event(value, quota),
        8 => scan_custom_log(value, quota),
        9 => scan_custom_artifact(value, quota),
        10 => check_wire_string(value, "record.delivery_class"),
        _ => Ok(()),
    })
}

fn scan_custom_resource(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    scan_protobuf_fields(input, quota, |_, value| {
        check_wire_string(value, "resource string")
    })
}

fn scan_string_map(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => check_wire_string(value, "attribute key"),
        2 => check_wire_string(value, "attribute value"),
        _ => Ok(()),
    })
}

fn scan_packed_varints(input: &[u8]) -> Result<usize, ApiError> {
    let mut position = 0;
    let mut count = 0_usize;
    while position < input.len() {
        read_protobuf_varint(input, &mut position)?;
        count = count
            .checked_add(1)
            .ok_or_else(|| structural_limit("packed values"))?;
    }
    Ok(count)
}

fn add_packed_items(
    quota: &StructuralQuota,
    local_count: &mut usize,
    count: usize,
    field: &'static str,
) -> Result<(), ApiError> {
    for _ in 0..count {
        quota.repeated(local_count, field)?;
    }
    Ok(())
}

fn scan_custom_metric(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut bounds = 0;
    let mut buckets = 0;
    let mut attributes = 0;
    scan_protobuf_wire_fields(input, quota, |field, value| match (field, value) {
        (1..=10 | 18..=19, ProtobufWireValue::LengthDelimited(value)) => {
            check_wire_string(value, "metric string")
        }
        (14, ProtobufWireValue::Fixed64) => quota.repeated(&mut bounds, "metric explicit bounds"),
        (14, ProtobufWireValue::LengthDelimited(value)) => {
            if value.len() % 8 != 0 {
                return Err(invalid_protobuf_structure(
                    "packed metric bounds are not fixed64-aligned",
                ));
            }
            add_packed_items(
                quota,
                &mut bounds,
                value.len() / 8,
                "metric explicit bounds",
            )
        }
        (15, ProtobufWireValue::Varint) => quota.repeated(&mut buckets, "metric bucket counts"),
        (15, ProtobufWireValue::LengthDelimited(value)) => add_packed_items(
            quota,
            &mut buckets,
            scan_packed_varints(value)?,
            "metric bucket counts",
        ),
        (16, ProtobufWireValue::LengthDelimited(value)) => {
            quota.repeated(&mut attributes, "metric attributes")?;
            scan_string_map(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_custom_event(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 3..=7 | 9..=12 | 14 => check_wire_string(value, "event string"),
        8 => {
            quota.repeated(&mut attributes, "event attributes")?;
            scan_string_map(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_custom_log(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 2 | 4 | 5 | 7 | 8 => check_wire_string(value, "log string"),
        6 => {
            quota.repeated(&mut attributes, "log attributes")?;
            scan_string_map(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_custom_artifact(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 2 | 5 | 7..=9 => check_wire_string(value, "artifact string"),
        10 => {
            quota.repeated(&mut attributes, "artifact attributes")?;
            scan_string_map(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_metrics_request(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut resources = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.repeated(&mut resources, "OTLP resource metrics")?;
            scan_otlp_resource_metrics(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_resource_metrics(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut scopes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => scan_otlp_resource(value, quota),
        2 => {
            quota.repeated(&mut scopes, "OTLP scope metrics")?;
            scan_otlp_scope_metrics(value, quota)
        }
        3 => check_wire_string(value, "resource metrics schema URL"),
        _ => Ok(()),
    })
}

fn scan_otlp_scope_metrics(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut metrics = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => scan_otlp_scope(value, quota),
        2 => {
            quota.repeated(&mut metrics, "OTLP metrics")?;
            scan_otlp_metric(value, quota)
        }
        3 => check_wire_string(value, "scope metrics schema URL"),
        _ => Ok(()),
    })
}

fn scan_otlp_metric(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut metadata = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1..=3 => check_wire_string(value, "metric descriptor string"),
        5 | 7 => scan_otlp_number_aggregate(value, quota),
        9 => scan_otlp_histogram(value, quota),
        10 => scan_otlp_exponential_histogram(value, quota),
        11 => scan_otlp_summary(value, quota),
        12 => {
            quota.repeated(&mut metadata, "metric metadata")?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_number_aggregate(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut points = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.record(&mut points, "OTLP metric data points")?;
            scan_otlp_number_point(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_number_point(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    let mut exemplars = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        5 => {
            quota.repeated(&mut exemplars, "OTLP exemplars")?;
            scan_otlp_exemplar(value, quota)
        }
        7 => {
            quota.repeated(&mut attributes, "OTLP point attributes")?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_histogram(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut points = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.record(&mut points, "OTLP histogram points")?;
            scan_otlp_histogram_point(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_histogram_point(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut bucket_counts = 0;
    let mut bounds = 0;
    let mut exemplars = 0;
    let mut attributes = 0;
    scan_protobuf_wire_fields(input, quota, |field, value| match (field, value) {
        (6, ProtobufWireValue::Fixed64) => {
            quota.repeated(&mut bucket_counts, "OTLP histogram bucket counts")
        }
        (7, ProtobufWireValue::Fixed64) => quota.repeated(&mut bounds, "OTLP histogram bounds"),
        (6 | 7, ProtobufWireValue::LengthDelimited(value)) if value.len() % 8 == 0 => {
            let (count, name) = if field == 6 {
                (&mut bucket_counts, "OTLP histogram bucket counts")
            } else {
                (&mut bounds, "OTLP histogram bounds")
            };
            add_packed_items(quota, count, value.len() / 8, name)
        }
        (6 | 7, ProtobufWireValue::LengthDelimited(_)) => Err(invalid_protobuf_structure(
            "packed OTLP histogram fixed64 values are misaligned",
        )),
        (8, ProtobufWireValue::LengthDelimited(value)) => {
            quota.repeated(&mut exemplars, "OTLP histogram exemplars")?;
            scan_otlp_exemplar(value, quota)
        }
        (9, ProtobufWireValue::LengthDelimited(value)) => {
            quota.repeated(&mut attributes, "OTLP histogram attributes")?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_exponential_histogram(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut points = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.record(&mut points, "OTLP exponential histogram points")?;
            scan_otlp_exponential_point(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_exponential_point(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    let mut exemplars = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => {
            quota.repeated(&mut attributes, "OTLP exponential histogram attributes")?;
            scan_otlp_key_value(value, quota)
        }
        8 | 9 => scan_otlp_exponential_buckets(value, quota),
        11 => {
            quota.repeated(&mut exemplars, "OTLP exponential histogram exemplars")?;
            scan_otlp_exemplar(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_exemplar(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut filtered_attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        4 | 5 => check_wire_string(value, "OTLP exemplar trace identity"),
        7 => {
            quota.repeated(
                &mut filtered_attributes,
                "OTLP exemplar filtered attributes",
            )?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_exponential_buckets(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut buckets = 0;
    scan_protobuf_wire_fields(input, quota, |field, value| {
        if field != 2 {
            return Ok(());
        }
        match value {
            ProtobufWireValue::Varint => {
                quota.repeated(&mut buckets, "OTLP exponential bucket counts")?
            }
            ProtobufWireValue::LengthDelimited(value) => add_packed_items(
                quota,
                &mut buckets,
                scan_packed_varints(value)?,
                "OTLP exponential bucket counts",
            )?,
            _ => {}
        }
        Ok(())
    })
}

fn scan_otlp_summary(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut points = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.record(&mut points, "OTLP summary points")?;
            scan_otlp_summary_point(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_summary_point(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut quantiles = 0;
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        6 => quota.repeated(&mut quantiles, "OTLP summary quantiles"),
        7 => {
            quota.repeated(&mut attributes, "OTLP summary attributes")?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_logs_request(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut resources = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.repeated(&mut resources, "OTLP resource logs")?;
            scan_otlp_resource_logs(value, quota)?;
        }
        Ok(())
    })
}

fn scan_otlp_resource_logs(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut scopes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => scan_otlp_resource(value, quota),
        2 => {
            quota.repeated(&mut scopes, "OTLP scope logs")?;
            scan_otlp_scope_logs(value, quota)
        }
        3 => check_wire_string(value, "resource logs schema URL"),
        _ => Ok(()),
    })
}

fn scan_otlp_scope_logs(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut records = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => scan_otlp_scope(value, quota),
        2 => {
            quota.record(&mut records, "OTLP log records")?;
            scan_otlp_log_record(value, quota)
        }
        3 => check_wire_string(value, "scope logs schema URL"),
        _ => Ok(()),
    })
}

fn scan_otlp_log_record(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        3 => check_wire_string(value, "OTLP severity text"),
        5 => scan_otlp_any_value(value, quota, 0),
        6 => {
            quota.repeated(&mut attributes, "OTLP log attributes")?;
            scan_otlp_key_value(value, quota)
        }
        9 | 10 => check_wire_string(value, "OTLP trace identity"),
        12 => check_wire_string(value, "OTLP event name"),
        _ => Ok(()),
    })
}

fn scan_otlp_resource(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    let mut entity_refs = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => {
            quota.repeated(&mut attributes, "OTLP resource attributes")?;
            scan_otlp_key_value(value, quota)
        }
        3 => {
            quota.repeated(&mut entity_refs, "OTLP resource entity references")?;
            scan_otlp_entity_ref(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_entity_ref(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut id_keys = 0;
    let mut description_keys = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 2 => check_wire_string(value, "OTLP entity reference string"),
        3 => {
            quota.repeated(&mut id_keys, "OTLP entity reference ID keys")?;
            check_wire_string(value, "OTLP entity reference ID key")
        }
        4 => {
            quota.repeated(
                &mut description_keys,
                "OTLP entity reference description keys",
            )?;
            check_wire_string(value, "OTLP entity reference description key")
        }
        _ => Ok(()),
    })
}

fn scan_otlp_scope(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    let mut attributes = 0;
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 2 => check_wire_string(value, "OTLP scope string"),
        3 => {
            quota.repeated(&mut attributes, "OTLP scope attributes")?;
            scan_otlp_key_value(value, quota)
        }
        _ => Ok(()),
    })
}

fn scan_otlp_key_value(input: &[u8], quota: &StructuralQuota) -> Result<(), ApiError> {
    scan_otlp_key_value_at_depth(input, quota, 0)
}

fn scan_otlp_key_value_at_depth(
    input: &[u8],
    quota: &StructuralQuota,
    depth: usize,
) -> Result<(), ApiError> {
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 => check_wire_string(value, "OTLP attribute key"),
        2 => scan_otlp_any_value(value, quota, depth),
        _ => Ok(()),
    })
}

fn scan_otlp_any_value(
    input: &[u8],
    quota: &StructuralQuota,
    depth: usize,
) -> Result<(), ApiError> {
    if depth > MAX_STRUCTURAL_DEPTH {
        return Err(structural_limit("protobuf nesting"));
    }
    scan_protobuf_fields(input, quota, |field, value| match field {
        1 | 7 => check_wire_string(value, "OTLP attribute value"),
        5 => scan_otlp_array_value(value, quota, depth + 1),
        6 => scan_otlp_key_value_list(value, quota, depth + 1),
        _ => Ok(()),
    })
}

fn scan_otlp_array_value(
    input: &[u8],
    quota: &StructuralQuota,
    depth: usize,
) -> Result<(), ApiError> {
    let mut values = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.repeated(&mut values, "OTLP array values")?;
            scan_otlp_any_value(value, quota, depth)?;
        }
        Ok(())
    })
}

fn scan_otlp_key_value_list(
    input: &[u8],
    quota: &StructuralQuota,
    depth: usize,
) -> Result<(), ApiError> {
    let mut values = 0;
    scan_protobuf_fields(input, quota, |field, value| {
        if field == 1 {
            quota.repeated(&mut values, "OTLP key-value list")?;
            scan_otlp_key_value_at_depth(value, quota, depth)?;
        }
        Ok(())
    })
}

struct JsonStructuralSeed<'a> {
    quota: &'a StructuralQuota,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for JsonStructuralSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        if self.depth > MAX_STRUCTURAL_DEPTH {
            self.quota.violation.set(Some("JSON nesting"));
            return Err(serde::de::Error::custom(
                "JSON nesting exceeds structural quota",
            ));
        }
        deserializer.deserialize_any(JsonStructuralVisitor {
            quota: self.quota,
            depth: self.depth,
        })
    }
}

struct JsonStructuralVisitor<'a> {
    quota: &'a StructuralQuota,
    depth: usize,
}

impl<'de> Visitor<'de> for JsonStructuralVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("bounded telemetry JSON")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_some<D>(self, deserializer: D) -> Result<(), D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        JsonStructuralSeed {
            quota: self.quota,
            depth: self.depth + 1,
        }
        .deserialize(deserializer)
    }

    fn visit_str<E>(self, value: &str) -> Result<(), E>
    where
        E: serde::de::Error,
    {
        if value.len() > MAX_STRING_BYTES {
            self.quota.violation.set(Some("JSON string"));
            return Err(E::custom("JSON string exceeds structural quota"));
        }
        Ok(())
    }

    fn visit_string<E>(self, value: String) -> Result<(), E>
    where
        E: serde::de::Error,
    {
        self.visit_str(&value)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut elements = 0;
        while sequence
            .next_element_seed(JsonStructuralSeed {
                quota: self.quota,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            if !self.quota.json_item(&mut elements, "JSON array elements") {
                return Err(serde::de::Error::custom(
                    "JSON array exceeds structural quota",
                ));
            }
        }
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = 0;
        while map
            .next_key_seed(JsonStructuralSeed {
                quota: self.quota,
                depth: self.depth + 1,
            })?
            .is_some()
        {
            map.next_value_seed(JsonStructuralSeed {
                quota: self.quota,
                depth: self.depth + 1,
            })?;
            if !self.quota.json_item(&mut entries, "JSON object entries") {
                return Err(serde::de::Error::custom(
                    "JSON object exceeds structural quota",
                ));
            }
        }
        Ok(())
    }
}

fn preflight_json(input: &[u8]) -> Result<(), ApiError> {
    let quota = StructuralQuota::new();
    let mut deserializer = serde_json::Deserializer::from_slice(input);
    let result = JsonStructuralSeed {
        quota: &quota,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .and_then(|()| deserializer.end());
    match result {
        Ok(()) => Ok(()),
        Err(error) => match quota.violation.get() {
            Some(field) => Err(structural_limit(field)),
            None => Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                format!("invalid telemetry JSON during structural preflight: {error}"),
            )),
        },
    }
}

fn decode_otlp<T>(format: WireFormat, body: &[u8], name: &str) -> Result<T, ApiError>
where
    T: ProstMessage + Default + serde::de::DeserializeOwned,
{
    match format {
        WireFormat::Json => serde_json::from_slice(body).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                format!("invalid {name} JSON: {error}"),
            )
        }),
        WireFormat::Protobuf => T::decode(body).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_protobuf",
                format!("invalid {name} protobuf: {error}"),
            )
        }),
    }
}

fn encode_otlp_response<T>(format: WireFormat, message: &T) -> Response
where
    T: ProstMessage + Serialize,
{
    let body = match format {
        WireFormat::Json => serde_json::to_vec(message).expect("OTLP response JSON is infallible"),
        WireFormat::Protobuf => message.encode_to_vec(),
    };
    response(StatusCode::OK, format.content_type(), body)
}

fn optional_idempotency_key(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let Some(value) = headers.get("idempotency-key") else {
        return Ok(None);
    };
    let key = value.to_str().map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_idempotency_key",
            "Idempotency-Key must be valid ASCII",
        )
    })?;
    if !valid_idempotency_key(key) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_idempotency_key",
            "Idempotency-Key must contain 1..=128 safe ASCII characters",
        ));
    }
    Ok(Some(key.to_string()))
}

fn check_custom_admission(batch: &TelemetryBatchV1) -> Result<(), ApiError> {
    check_record_count(batch.records.len())?;
    let estimated = batch
        .records
        .iter()
        .fold(batch.records.len() * 1024, |total, record| {
            total.saturating_add(estimate_custom_record(record))
        });
    check_normalized_bytes(estimated)
}

fn check_otlp_metric_admission(request: &ExportMetricsServiceRequest) -> Result<(), ApiError> {
    let mut records = 0_usize;
    let mut estimated = request.encoded_len();
    for resource_metrics in &request.resource_metrics {
        check_string(&resource_metrics.schema_url, "resource_metrics.schema_url")?;
        let resource_points = resource_metrics
            .scope_metrics
            .iter()
            .flat_map(|scope| &scope.metrics)
            .try_fold(0_usize, |total, metric| {
                checked_add(total, metric_point_count(metric))
            })?;
        let resource_size = estimate_otlp_resource(resource_metrics.resource.as_ref())?;
        estimated = checked_add(
            estimated,
            checked_mul(resource_size.saturating_add(256), resource_points)?,
        )?;
        for scope_metrics in &resource_metrics.scope_metrics {
            check_string(&scope_metrics.schema_url, "scope_metrics.schema_url")?;
            let scope_points = scope_metrics
                .metrics
                .iter()
                .try_fold(0_usize, |total, metric| {
                    checked_add(total, metric_point_count(metric))
                })?;
            let scope_size = estimate_otlp_scope(scope_metrics.scope.as_ref())?;
            estimated = checked_add(
                estimated,
                checked_mul(scope_size.saturating_add(128), scope_points)?,
            )?;
            for metric in &scope_metrics.metrics {
                let points = metric_point_count(metric);
                let metric_size = checked_strings([
                    (&metric.name, "metric.name"),
                    (&metric.description, "metric.description"),
                    (&metric.unit, "metric.unit"),
                ])?;
                estimated = checked_add(
                    estimated,
                    checked_mul(metric_size.saturating_add(1024), points)?,
                )?;
                estimated = checked_add(estimated, estimate_otlp_metric_point_attributes(metric)?)?;
                records = checked_add(records, points)?;
            }
        }
    }
    check_record_count(records)?;
    check_normalized_bytes(estimated)
}

fn check_otlp_log_admission(request: &ExportLogsServiceRequest) -> Result<(), ApiError> {
    let mut records = 0_usize;
    let mut estimated = request.encoded_len();
    for resource_logs in &request.resource_logs {
        check_string(&resource_logs.schema_url, "resource_logs.schema_url")?;
        let resource_records = resource_logs
            .scope_logs
            .iter()
            .try_fold(0_usize, |total, scope| {
                checked_add(total, scope.log_records.len())
            })?;
        let resource_size = estimate_otlp_resource(resource_logs.resource.as_ref())?;
        estimated = checked_add(
            estimated,
            checked_mul(resource_size.saturating_add(256), resource_records)?,
        )?;
        for scope_logs in &resource_logs.scope_logs {
            check_string(&scope_logs.schema_url, "scope_logs.schema_url")?;
            let scope_size = estimate_otlp_scope(scope_logs.scope.as_ref())?;
            estimated = checked_add(
                estimated,
                checked_mul(
                    scope_size.saturating_add(1024),
                    scope_logs.log_records.len(),
                )?,
            )?;
            for log in &scope_logs.log_records {
                check_string(&log.severity_text, "log.severity_text")?;
                check_string(&log.event_name, "log.event_name")?;
                estimated = checked_add(estimated, estimate_otlp_key_values(&log.attributes)?)?;
                if let Some(body) = &log.body {
                    estimated = checked_add(estimated, estimate_otlp_any_value(body)?)?;
                }
            }
            records = checked_add(records, scope_logs.log_records.len())?;
        }
    }
    check_record_count(records)?;
    check_normalized_bytes(estimated)
}

fn estimate_otlp_resource(resource: Option<&OtlpResource>) -> Result<usize, ApiError> {
    resource.map_or(Ok(0), |resource| {
        estimate_otlp_key_values(&resource.attributes)
    })
}

fn estimate_otlp_scope(
    scope: Option<&opentelemetry_proto::tonic::common::v1::InstrumentationScope>,
) -> Result<usize, ApiError> {
    let Some(scope) = scope else {
        return Ok(0);
    };
    checked_add(
        checked_strings([
            (&scope.name, "scope.name"),
            (&scope.version, "scope.version"),
        ])?,
        estimate_otlp_key_values(&scope.attributes)?,
    )
}

fn estimate_otlp_metric_point_attributes(metric: &Metric) -> Result<usize, ApiError> {
    match metric.data.as_ref() {
        Some(metric::Data::Gauge(gauge)) => gauge.data_points.iter().try_fold(0, |total, point| {
            checked_add(total, estimate_otlp_key_values(&point.attributes)?)
        }),
        Some(metric::Data::Sum(sum)) => sum.data_points.iter().try_fold(0, |total, point| {
            checked_add(total, estimate_otlp_key_values(&point.attributes)?)
        }),
        Some(metric::Data::Histogram(histogram)) => {
            histogram.data_points.iter().try_fold(0, |total, point| {
                checked_add(total, estimate_otlp_key_values(&point.attributes)?)
            })
        }
        Some(metric::Data::ExponentialHistogram(histogram)) => {
            histogram.data_points.iter().try_fold(0, |total, point| {
                checked_add(total, estimate_otlp_key_values(&point.attributes)?)
            })
        }
        Some(metric::Data::Summary(summary)) => {
            summary.data_points.iter().try_fold(0, |total, point| {
                checked_add(total, estimate_otlp_key_values(&point.attributes)?)
            })
        }
        None => Ok(0),
    }
}

fn estimate_otlp_key_values(values: &[KeyValue]) -> Result<usize, ApiError> {
    values.iter().try_fold(0, |total, pair| {
        check_string(&pair.key, "attribute.key")?;
        let value_size = pair.value.as_ref().map_or(Ok(0), estimate_otlp_any_value)?;
        checked_add(
            total,
            pair.key.len().saturating_add(value_size).saturating_add(64),
        )
    })
}

fn estimate_otlp_any_value(
    value: &opentelemetry_proto::tonic::common::v1::AnyValue,
) -> Result<usize, ApiError> {
    match value.value.as_ref() {
        Some(any_value::Value::StringValue(value)) => {
            check_string(value, "attribute.value")?;
            Ok(value.len())
        }
        Some(any_value::Value::BytesValue(value)) => {
            check_bytes(value, "attribute.value")?;
            Ok(value.len().saturating_mul(2))
        }
        Some(any_value::Value::ArrayValue(values)) => {
            values.values.iter().try_fold(0, |total, value| {
                checked_add(total, estimate_otlp_any_value(value)?)
            })
        }
        Some(any_value::Value::KvlistValue(values)) => estimate_otlp_key_values(&values.values),
        _ => Ok(16),
    }
}

fn checked_strings<const N: usize>(
    values: [(&String, &'static str); N],
) -> Result<usize, ApiError> {
    values.into_iter().try_fold(0, |total, (value, field)| {
        check_string(value, field)?;
        checked_add(total, value.len())
    })
}

fn check_string(value: &str, field: &'static str) -> Result<(), ApiError> {
    check_bytes(value.as_bytes(), field)
}

fn check_bytes(value: &[u8], field: &'static str) -> Result<(), ApiError> {
    if value.len() > MAX_STRING_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "string_limit_exceeded",
            format!(
                "{field} is {} bytes; limit is {MAX_STRING_BYTES}",
                value.len()
            ),
        ));
    }
    Ok(())
}

fn checked_add(left: usize, right: usize) -> Result<usize, ApiError> {
    left.checked_add(right).ok_or_else(normalized_size_overflow)
}

fn checked_mul(left: usize, right: usize) -> Result<usize, ApiError> {
    left.checked_mul(right).ok_or_else(normalized_size_overflow)
}

fn normalized_size_overflow() -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "normalized_body_too_large",
        "estimated normalized telemetry size overflowed the admission bound",
    )
}

fn check_record_count(records: usize) -> Result<(), ApiError> {
    if records > MAX_REST_RECORDS {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "record_limit_exceeded",
            format!("REST telemetry request has {records} records; limit is {MAX_REST_RECORDS}"),
        ));
    }
    Ok(())
}

fn check_normalized_bytes(estimated: usize) -> Result<(), ApiError> {
    if estimated > MAX_NORMALIZED_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "normalized_body_too_large",
            format!(
                "estimated normalized telemetry size {estimated} bytes exceeds {MAX_NORMALIZED_BYTES} limit"
            ),
        ));
    }
    Ok(())
}

fn estimate_custom_record(record: &TelemetryRecordV1) -> usize {
    let mut total = optional_strings_size([&record.signal, &record.delivery_class]);
    if let Some(resource) = record.resource.as_option() {
        total = total.saturating_add(estimate_resource(resource));
    }
    if let Some(metric) = record.metric.as_option() {
        total = total
            .saturating_add(optional_strings_size([
                &metric.scope,
                &metric.scope_version,
                &metric.name,
                &metric.description,
                &metric.unit,
                &metric.instrument_kind,
                &metric.temporality,
                &metric.reset_id,
                &metric.series_id,
                &metric.device_uid,
                &metric.device_type,
            ]))
            .saturating_add(map_size(&metric.attributes))
            .saturating_add(metric.explicit_bounds.len() * size_of::<f64>())
            .saturating_add(metric.bucket_counts.len() * size_of::<i64>());
    }
    if let Some(event) = record.event.as_option() {
        total = total
            .saturating_add(optional_strings_size([
                &event.event_name,
                &event.severity_text,
                &event.outcome,
                &event.phase,
                &event.error_type,
                &event.body,
                &event.trace_id,
                &event.span_id,
                &event.evidence_uri,
                &event.result_uri,
                &event.probe_status,
            ]))
            .saturating_add(map_size(&event.attributes));
    }
    if let Some(log) = record.log.as_option() {
        total = total
            .saturating_add(optional_strings_size([
                &log.source,
                &log.body,
                &log.severity_text,
                &log.event_name,
                &log.trace_id,
                &log.span_id,
            ]))
            .saturating_add(map_size(&log.attributes));
    }
    if let Some(artifact) = record.artifact.as_option() {
        total = total
            .saturating_add(optional_strings_size([
                &artifact.capture_type,
                &artifact.trigger,
                &artifact.outcome,
                &artifact.sha256,
                &artifact.uri,
                &artifact.summary,
            ]))
            .saturating_add(map_size(&artifact.attributes));
    }
    total
}

fn estimate_resource(resource: &TelemetryResourceV1) -> usize {
    optional_strings_size([
        &resource.service_name,
        &resource.service_instance_id,
        &resource.role,
        &resource.root_run_uid,
        &resource.service_version,
        &resource.run_id_alias,
        &resource.iris_job_id,
        &resource.iris_task_id,
        &resource.attempt_uid,
        &resource.worker_id,
        &resource.node_id,
        &resource.pod_uid,
        &resource.container_id,
        &resource.actor_id,
        &resource.engine_id,
        &resource.repository,
        &resource.git_revision,
        &resource.image_digest,
        &resource.model_id,
        &resource.model_revision,
        &resource.owner,
        &resource.cluster,
        &resource.entity_authority,
        &resource.entity_type,
        &resource.entity_uid,
    ])
}

fn optional_strings_size<const N: usize>(values: [&Option<String>; N]) -> usize {
    values
        .into_iter()
        .filter_map(Option::as_ref)
        .fold(0, |total, value| total.saturating_add(value.len()))
}

fn map_size(values: &HashMap<String, String>) -> usize {
    values.iter().fold(0, |total, (key, value)| {
        total
            .saturating_add(key.len())
            .saturating_add(value.len())
            .saturating_add(64)
    })
}

fn normalize_otlp_metrics(
    request: &ExportMetricsServiceRequest,
    batch_id: &str,
    identity: &AuthIdentity,
) -> (Vec<RoutedRecord>, i64, Vec<String>) {
    let mut records = Vec::new();
    let mut rejected = 0_i64;
    let mut reasons = Vec::new();
    let mut point_index = 0_i32;
    for resource_metrics in &request.resource_metrics {
        let resource = otlp_resource(resource_metrics.resource.as_ref())
            .map(|resource| authoritative_resource(&resource, identity));
        for scope_metrics in &resource_metrics.scope_metrics {
            let scope_name = scope_metrics
                .scope
                .as_ref()
                .map(|scope| scope.name.as_str())
                .unwrap_or("");
            let scope_version = scope_metrics
                .scope
                .as_ref()
                .map(|scope| scope.version.as_str())
                .unwrap_or("");
            for metric in &scope_metrics.metrics {
                let normalized = normalize_otlp_metric(
                    batch_id,
                    point_index,
                    resource.as_ref().map_err(|reason| *reason),
                    scope_name,
                    scope_version,
                    metric,
                );
                let point_count = metric_point_count(metric) as i64;
                match normalized {
                    Ok(metric_records) => {
                        for record in metric_records {
                            match record {
                                Ok(record) => records.push(record),
                                Err(reason) => {
                                    rejected += 1;
                                    push_reason(&mut reasons, reason);
                                }
                            }
                        }
                    }
                    Err(reason) => {
                        rejected += point_count;
                        push_reason(&mut reasons, reason);
                    }
                }
                point_index += point_count as i32;
            }
        }
    }
    (records, rejected, reasons)
}

fn normalize_otlp_metric(
    batch_id: &str,
    first_index: i32,
    resource: Result<&TelemetryResourceV1, &str>,
    scope_name: &str,
    scope_version: &str,
    metric: &Metric,
) -> Result<Vec<Result<RoutedRecord, String>>, String> {
    let resource = resource.map_err(str::to_string)?;
    validate_resource(resource, first_index).map_err(validation_reason)?;
    let descriptor = catalog().metric(scope_name, &metric.name).ok_or_else(|| {
        format!(
            "metric {scope_name}.{} is not in telemetry catalog",
            metric.name
        )
    })?;
    if metric.description != descriptor.description || metric.unit != descriptor.unit {
        return Err(format!(
            "metric {scope_name}.{} description or unit differs from telemetry catalog",
            metric.name
        ));
    }

    match metric.data.as_ref() {
        Some(metric::Data::Gauge(gauge)) => {
            if descriptor.instrument_kind != "gauge" || descriptor.temporality != "unspecified" {
                return Err(format!(
                    "metric {scope_name}.{} aggregation differs from telemetry catalog",
                    metric.name
                ));
            }
            Ok(gauge
                .data_points
                .iter()
                .enumerate()
                .map(|(offset, point)| {
                    normalize_number_point(
                        batch_id,
                        first_index + offset as i32,
                        resource,
                        scope_name,
                        scope_version,
                        metric,
                        point,
                        "gauge",
                        "unspecified",
                    )
                })
                .collect())
        }
        Some(metric::Data::Sum(sum)) => {
            let temporality = otlp_temporality(sum.aggregation_temporality)?;
            if descriptor.instrument_kind != "counter" || descriptor.temporality != temporality {
                return Err(format!(
                    "metric {scope_name}.{} aggregation differs from telemetry catalog",
                    metric.name
                ));
            }
            if !sum.is_monotonic {
                return Err(format!(
                    "catalog counter {scope_name}.{} requires a monotonic OTLP sum",
                    metric.name
                ));
            }
            Ok(sum
                .data_points
                .iter()
                .enumerate()
                .map(|(offset, point)| {
                    normalize_number_point(
                        batch_id,
                        first_index + offset as i32,
                        resource,
                        scope_name,
                        scope_version,
                        metric,
                        point,
                        "counter",
                        temporality,
                    )
                })
                .collect())
        }
        Some(metric::Data::Histogram(histogram)) => {
            let temporality = otlp_temporality(histogram.aggregation_temporality)?;
            if descriptor.instrument_kind != "histogram" || descriptor.temporality != temporality {
                return Err(format!(
                    "metric {scope_name}.{} aggregation differs from telemetry catalog",
                    metric.name
                ));
            }
            Ok(histogram
                .data_points
                .iter()
                .enumerate()
                .map(|(offset, point)| {
                    normalize_histogram_point(
                        batch_id,
                        first_index + offset as i32,
                        resource,
                        scope_name,
                        scope_version,
                        metric,
                        point,
                        temporality,
                    )
                })
                .collect())
        }
        Some(metric::Data::ExponentialHistogram(_)) => {
            Err("exponential histograms are not accepted by telemetry catalog v1".to_string())
        }
        Some(metric::Data::Summary(_)) => {
            Err("summary metrics are not accepted by telemetry catalog v1".to_string())
        }
        None => Err("metric has no data".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn normalize_number_point(
    batch_id: &str,
    index: i32,
    resource: &TelemetryResourceV1,
    scope_name: &str,
    scope_version: &str,
    metric: &Metric,
    point: &NumberDataPoint,
    instrument_kind: &str,
    temporality: &str,
) -> Result<RoutedRecord, String> {
    let event_ts = checked_timestamp(point.time_unix_nano)?;
    let attributes = otlp_attributes(&point.attributes)?;
    let value = match point.value {
        Some(number_data_point::Value::AsDouble(value)) if value.is_finite() => value,
        Some(number_data_point::Value::AsInt(value))
            if value.unsigned_abs() <= MAX_EXACT_F64_INTEGER =>
        {
            value as f64
        }
        Some(number_data_point::Value::AsInt(_)) => {
            return Err("integer metric value cannot be represented exactly as f64".to_string());
        }
        _ => return Err("metric point must contain one finite numeric value".to_string()),
    };
    let (start_ts, reset_id) = reset_fields(point.start_time_unix_nano, temporality)?;
    let series_id =
        canonical_series_id(scope_name, &metric.name, resource, None, None, &attributes);
    let custom_metric = TelemetryMetricV1 {
        scope: Some(scope_name.to_string()),
        scope_version: (!scope_version.is_empty()).then(|| scope_version.to_string()),
        name: Some(metric.name.clone()),
        description: Some(metric.description.clone()),
        unit: Some(metric.unit.clone()),
        instrument_kind: Some(instrument_kind.to_string()),
        temporality: Some(temporality.to_string()),
        start_ts_unix_nano: start_ts,
        reset_id,
        series_id: Some(series_id),
        value: Some(value),
        attributes,
        ..Default::default()
    };
    metric_routed_record(
        batch_id,
        index,
        event_ts,
        event_ts,
        resource,
        &custom_metric,
    )
}

#[allow(clippy::too_many_arguments)]
fn normalize_histogram_point(
    batch_id: &str,
    index: i32,
    resource: &TelemetryResourceV1,
    scope_name: &str,
    scope_version: &str,
    metric: &Metric,
    point: &HistogramDataPoint,
    temporality: &str,
) -> Result<RoutedRecord, String> {
    let event_ts = checked_timestamp(point.time_unix_nano)?;
    let attributes = otlp_attributes(&point.attributes)?;
    let bucket_counts = point
        .bucket_counts
        .iter()
        .map(|count| i64::try_from(*count).map_err(|_| "histogram bucket count exceeds i64"))
        .collect::<Result<Vec<_>, _>>()
        .map_err(str::to_string)?;
    let count = i64::try_from(point.count).map_err(|_| "histogram count exceeds i64")?;
    let (start_ts, reset_id) = reset_fields(point.start_time_unix_nano, temporality)?;
    let custom_metric = TelemetryMetricV1 {
        scope: Some(scope_name.to_string()),
        scope_version: (!scope_version.is_empty()).then(|| scope_version.to_string()),
        name: Some(metric.name.clone()),
        description: Some(metric.description.clone()),
        unit: Some(metric.unit.clone()),
        instrument_kind: Some("histogram".to_string()),
        temporality: Some(temporality.to_string()),
        start_ts_unix_nano: start_ts,
        reset_id,
        series_id: Some(canonical_series_id(
            scope_name,
            &metric.name,
            resource,
            None,
            None,
            &attributes,
        )),
        count: Some(count),
        sum: point.sum,
        explicit_bounds: point.explicit_bounds.clone(),
        bucket_counts,
        attributes,
        ..Default::default()
    };
    metric_routed_record(
        batch_id,
        index,
        event_ts,
        event_ts,
        resource,
        &custom_metric,
    )
}

fn metric_routed_record(
    batch_id: &str,
    index: i32,
    event_ts: i64,
    observed_ts: i64,
    resource: &TelemetryResourceV1,
    metric: &TelemetryMetricV1,
) -> Result<RoutedRecord, String> {
    let batch = TelemetryBatchV1 {
        schema_version: Some(SCHEMA_VERSION),
        catalog_version: Some(CATALOG_VERSION.to_string()),
        batch_id: Some(batch_id.to_string()),
        ..Default::default()
    };
    let mut row = base_row(&batch, index, event_ts, observed_ts, resource);
    let descriptor = catalog()
        .metric(
            metric.scope.as_deref().unwrap_or(""),
            metric.name.as_deref().unwrap_or(""),
        )
        .ok_or_else(|| "metric is not in telemetry catalog".to_string())?;
    let delivery_class = descriptor.delivery_class.clone();
    let namespace =
        validate_metric(metric, &delivery_class, index, &mut row).map_err(validation_reason)?;
    Ok(RoutedRecord {
        namespace,
        delivery_class,
        row,
    })
}

fn normalize_otlp_logs(
    request: &ExportLogsServiceRequest,
    batch_id: &str,
    identity: &AuthIdentity,
) -> (Vec<RoutedRecord>, i64, Vec<String>) {
    let mut records = Vec::new();
    let mut rejected = 0_i64;
    let mut reasons = Vec::new();
    let mut index = 0_i32;
    for resource_logs in &request.resource_logs {
        let resource = otlp_resource(resource_logs.resource.as_ref())
            .map(|resource| authoritative_resource(&resource, identity));
        for scope_logs in &resource_logs.scope_logs {
            let source = scope_logs
                .scope
                .as_ref()
                .map(|scope| scope.name.as_str())
                .filter(|name| !name.is_empty())
                .unwrap_or("otlp");
            for log in &scope_logs.log_records {
                match normalize_otlp_log(
                    batch_id,
                    index,
                    resource.as_ref().map_err(|reason| *reason),
                    source,
                    log,
                ) {
                    Ok(record) => records.push(record),
                    Err(reason) => {
                        rejected += 1;
                        push_reason(&mut reasons, reason);
                    }
                }
                index += 1;
            }
        }
    }
    (records, rejected, reasons)
}

fn normalize_otlp_log(
    batch_id: &str,
    index: i32,
    resource: Result<&TelemetryResourceV1, &str>,
    source: &str,
    log: &LogRecord,
) -> Result<RoutedRecord, String> {
    let resource = resource.map_err(str::to_string)?;
    validate_resource(resource, index).map_err(validation_reason)?;
    let event_ts = checked_timestamp(if log.time_unix_nano == 0 {
        log.observed_time_unix_nano
    } else {
        log.time_unix_nano
    })?;
    let observed_ts = if log.observed_time_unix_nano == 0 {
        event_ts
    } else {
        checked_timestamp(log.observed_time_unix_nano)?
    };
    let attributes = otlp_attributes(&log.attributes)?;
    let body = log
        .body
        .as_ref()
        .and_then(|body| body.value.as_ref())
        .map(otlp_scalar)
        .transpose()?
        .unwrap_or_default();
    let event_name = if log.event_name.is_empty() {
        attributes.get("event.name").cloned()
    } else {
        Some(log.event_name.clone())
    };
    let custom_log = TelemetryLogV1 {
        source: Some(source.to_string()),
        body: Some(body),
        severity_number: Some(log.severity_number),
        severity_text: Some(log.severity_text.clone()),
        event_name,
        attributes,
        trace_id: valid_trace_id(&log.trace_id, 16),
        span_id: valid_trace_id(&log.span_id, 8),
        ..Default::default()
    };
    let batch = TelemetryBatchV1 {
        schema_version: Some(SCHEMA_VERSION),
        catalog_version: Some(CATALOG_VERSION.to_string()),
        batch_id: Some(batch_id.to_string()),
        ..Default::default()
    };
    let mut row = base_row(&batch, index, event_ts, observed_ts, resource);
    validate_log(&custom_log, index, event_ts, resource, &mut row).map_err(validation_reason)?;
    Ok(RoutedRecord {
        namespace: LOG_NAMESPACE_NAME.to_string(),
        delivery_class: "buffered".to_string(),
        row,
    })
}

fn otlp_resource(resource: Option<&OtlpResource>) -> Result<TelemetryResourceV1, &'static str> {
    let attributes = resource
        .map(|resource| otlp_attributes(&resource.attributes))
        .transpose()
        .map_err(|_| "resource attributes must be unique scalar values")?
        .unwrap_or_default();
    let service_name = attributes.get("service.name").cloned().unwrap_or_default();
    let service_instance_id = attributes
        .get("service.instance.id")
        .cloned()
        .unwrap_or_default();
    if service_name.is_empty() || service_instance_id.is_empty() {
        return Err("resource requires service.name and service.instance.id");
    }
    let mut result = TelemetryResourceV1 {
        service_name: Some(service_name),
        service_instance_id: Some(service_instance_id),
        ..Default::default()
    };
    for (key, target) in [
        ("service.version", &mut result.service_version),
        ("marin.role", &mut result.role),
        ("marin.root_run_uid", &mut result.root_run_uid),
        ("marin.run_id_alias", &mut result.run_id_alias),
        ("iris.job_id", &mut result.iris_job_id),
        ("iris.task_id", &mut result.iris_task_id),
        ("marin.attempt_uid", &mut result.attempt_uid),
        ("marin.worker_id", &mut result.worker_id),
        ("host.id", &mut result.node_id),
        ("k8s.pod.uid", &mut result.pod_uid),
        ("container.id", &mut result.container_id),
        ("marin.actor_id", &mut result.actor_id),
        ("marin.engine_id", &mut result.engine_id),
        ("vcs.repository.url.full", &mut result.repository),
        ("vcs.ref.head.revision", &mut result.git_revision),
        ("container.image.id", &mut result.image_digest),
        ("gen_ai.request.model", &mut result.model_id),
        ("marin.model_revision", &mut result.model_revision),
        ("marin.owner", &mut result.owner),
        ("marin.cluster", &mut result.cluster),
        ("marin.entity.authority", &mut result.entity_authority),
        ("marin.entity.type", &mut result.entity_type),
        ("marin.entity.uid", &mut result.entity_uid),
    ] {
        *target = attributes.get(key).cloned();
    }
    result.task_index = parsed_attribute(&attributes, "iris.task_index");
    result.attempt_id = parsed_attribute(&attributes, "marin.attempt_id");
    result.rank = parsed_attribute(&attributes, "process.rank");
    result.process_index = parsed_attribute(&attributes, "process.index");
    result.experiment_issue = parsed_attribute(&attributes, "marin.experiment_issue");
    result.policy_step = parsed_attribute(&attributes, "marin.policy_step");
    Ok(result)
}

fn parsed_attribute<T: std::str::FromStr>(
    attributes: &HashMap<String, String>,
    key: &str,
) -> Option<T> {
    attributes.get(key).and_then(|value| value.parse().ok())
}

fn otlp_attributes(attributes: &[KeyValue]) -> Result<HashMap<String, String>, String> {
    let mut result = HashMap::with_capacity(attributes.len());
    for attribute in attributes {
        if attribute.key.is_empty() {
            return Err("attribute key must not be empty".to_string());
        }
        let value = attribute
            .value
            .as_ref()
            .and_then(|value| value.value.as_ref())
            .ok_or_else(|| format!("attribute {:?} has no value", attribute.key))
            .and_then(otlp_scalar)?;
        if result.insert(attribute.key.clone(), value).is_some() {
            return Err(format!("attribute {:?} is duplicated", attribute.key));
        }
    }
    Ok(result)
}

fn otlp_scalar(value: &any_value::Value) -> Result<String, String> {
    match value {
        any_value::Value::StringValue(value) => Ok(value.clone()),
        any_value::Value::BoolValue(value) => Ok(value.to_string()),
        any_value::Value::IntValue(value) => Ok(value.to_string()),
        any_value::Value::DoubleValue(value) if value.is_finite() => Ok(value.to_string()),
        any_value::Value::BytesValue(value) => Ok(hex(value)),
        _ => Err("telemetry attributes and log bodies must be scalar".to_string()),
    }
}

fn metric_point_count(metric: &Metric) -> usize {
    match metric.data.as_ref() {
        Some(metric::Data::Gauge(gauge)) => gauge.data_points.len(),
        Some(metric::Data::Sum(sum)) => sum.data_points.len(),
        Some(metric::Data::Histogram(histogram)) => histogram.data_points.len(),
        Some(metric::Data::ExponentialHistogram(histogram)) => histogram.data_points.len(),
        Some(metric::Data::Summary(summary)) => summary.data_points.len(),
        None => 0,
    }
}

fn otlp_temporality(value: i32) -> Result<&'static str, String> {
    match AggregationTemporality::try_from(value) {
        Ok(AggregationTemporality::Cumulative) => Ok("cumulative"),
        Ok(AggregationTemporality::Delta) => Ok("delta"),
        _ => Err("aggregation temporality must be delta or cumulative".to_string()),
    }
}

fn checked_timestamp(value: u64) -> Result<i64, String> {
    i64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "timestamp must be positive and fit in i64".to_string())
}

fn reset_fields(
    start_time_unix_nano: u64,
    temporality: &str,
) -> Result<(Option<i64>, Option<String>), String> {
    if temporality != "cumulative" {
        return Ok((None, None));
    }
    let start = checked_timestamp(start_time_unix_nano)?;
    Ok((
        Some(start),
        Some(stable_id(
            "finelog.telemetry.reset.v1",
            &[&start.to_string()],
        )),
    ))
}

pub(crate) fn canonical_series_id(
    scope: &str,
    name: &str,
    resource: &TelemetryResourceV1,
    device_uid: Option<&str>,
    device_type: Option<&str>,
    attributes: &HashMap<String, String>,
) -> String {
    let mut fields = vec![
        scope.to_string(),
        name.to_string(),
        resource.cluster.clone().unwrap_or_default(),
        resource.entity_authority.clone().unwrap_or_default(),
        resource.entity_type.clone().unwrap_or_default(),
        resource.entity_uid.clone().unwrap_or_default(),
        resource.service_name.clone().unwrap_or_default(),
        resource.service_instance_id.clone().unwrap_or_default(),
        resource.attempt_uid.clone().unwrap_or_default(),
        resource.actor_id.clone().unwrap_or_default(),
        resource.engine_id.clone().unwrap_or_default(),
        resource
            .process_index
            .map_or_else(String::new, |value| value.to_string()),
        resource
            .rank
            .map_or_else(String::new, |value| value.to_string()),
        device_uid.unwrap_or_default().to_string(),
        device_type.unwrap_or_default().to_string(),
    ];
    let mut attributes: Vec<_> = attributes.iter().collect();
    attributes.sort();
    fields.extend(
        attributes
            .into_iter()
            .map(|(key, value)| format!("{key}={value}")),
    );
    let fields: Vec<_> = fields.iter().map(String::as_str).collect();
    stable_id("finelog.telemetry.series.v1", &fields)
}

fn stable_id(domain: &str, fields: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain.as_bytes());
    digest.update([0]);
    for field in fields {
        digest_field(&mut digest, field.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn valid_trace_id(value: &[u8], expected_len: usize) -> Option<String> {
    (value.len() == expected_len && value.iter().any(|byte| *byte != 0)).then(|| hex(value))
}

fn hex(value: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(value.len() * 2);
    for byte in value {
        result.push(DIGITS[(byte >> 4) as usize] as char);
        result.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    result
}

fn validation_reason(error: TelemetryValidationErrorV1) -> String {
    format!(
        "{}: {}",
        error.field.as_deref().unwrap_or("record"),
        error.reason.as_deref().unwrap_or("invalid")
    )
}

fn push_reason(reasons: &mut Vec<String>, reason: String) {
    if reasons.len() < 3 && !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn partial_success_message(reasons: &[String]) -> String {
    if reasons.is_empty() {
        String::new()
    } else {
        format!("rejected telemetry: {}", reasons.join("; "))
    }
}

fn decode_custom_batch(format: WireFormat, body: &[u8]) -> Result<TelemetryBatchV1, ApiError> {
    match format {
        WireFormat::Json => serde_json::from_slice(body).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_json",
                format!("invalid TelemetryBatchV1 JSON: {error}"),
            )
        }),
        WireFormat::Protobuf => TelemetryBatchV1::decode_from_slice(body).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_protobuf",
                format!("invalid TelemetryBatchV1 protobuf: {error}"),
            )
        }),
    }
}

fn validate_custom_batch(
    batch: &TelemetryBatchV1,
    identity: &AuthIdentity,
) -> Result<Vec<RoutedRecord>, Vec<TelemetryValidationErrorV1>> {
    let mut errors = Vec::new();
    if batch.schema_version != Some(SCHEMA_VERSION) {
        errors.push(validation_error(-1, "schema_version", "must equal 1"));
    }
    if batch.catalog_version.as_deref() != Some(CATALOG_VERSION) {
        errors.push(validation_error(
            -1,
            "catalog_version",
            &format!("must equal {CATALOG_VERSION}"),
        ));
    }
    if batch
        .batch_id
        .as_deref()
        .is_none_or(|batch_id| !valid_idempotency_key(batch_id))
    {
        errors.push(validation_error(
            -1,
            "batch_id",
            "must be a valid nonempty idempotency key",
        ));
    }
    if batch.records.is_empty() {
        errors.push(validation_error(
            -1,
            "records",
            "must contain at least one record",
        ));
    }
    if batch.records.len() > MAX_REST_RECORDS {
        errors.push(validation_error(
            -1,
            "records",
            &format!("must contain at most {MAX_REST_RECORDS} records"),
        ));
    }

    let mut routed = Vec::with_capacity(batch.records.len());
    for (position, record) in batch.records.iter().enumerate() {
        if errors.len() >= MAX_VALIDATION_ERRORS {
            break;
        }
        match validate_custom_record(batch, record, position, identity) {
            Ok(record) => routed.push(record),
            Err(error) => errors.push(error),
        }
    }
    if errors.is_empty() {
        Ok(routed)
    } else {
        Err(errors)
    }
}

pub(crate) fn validate_canonical_agent_batch(batch: &TelemetryBatchV1) -> Result<(), String> {
    let cluster = batch
        .records
        .first()
        .and_then(|record| record.resource.as_option())
        .and_then(|resource| resource.cluster.clone())
        .ok_or_else(|| "canonical agent batch requires a stamped cluster".to_string())?;
    match validate_custom_batch(batch, &AuthIdentity::TrustedCollector { cluster }) {
        Ok(_) => Ok(()),
        Err(errors) => Err(errors
            .into_iter()
            .map(|error| {
                format!(
                    "record {} {}: {}",
                    error.record_index.unwrap_or(-1),
                    error.field.as_deref().unwrap_or("unknown"),
                    error.reason.as_deref().unwrap_or("invalid"),
                )
            })
            .collect::<Vec<_>>()
            .join("; ")),
    }
}

fn validate_custom_record(
    batch: &TelemetryBatchV1,
    record: &TelemetryRecordV1,
    position: usize,
    identity: &AuthIdentity,
) -> Result<RoutedRecord, TelemetryValidationErrorV1> {
    let index = record.record_index.unwrap_or(-1);
    if index != position as i32 {
        return Err(validation_error(
            index,
            "record_index",
            "must equal the record position",
        ));
    }
    let event_ts = positive_i64(record.event_ts_unix_nano, index, "event_ts_unix_nano")?;
    let observed_ts = positive_i64(record.observed_ts_unix_nano, index, "observed_ts_unix_nano")?;
    let resource = record
        .resource
        .as_option()
        .ok_or_else(|| validation_error(index, "resource", "must contain resource identity"))?;
    let resource = authoritative_resource(resource, identity);
    validate_resource(&resource, index)?;

    let mut row = base_row(batch, index, event_ts, observed_ts, &resource);
    let signal = record.signal.as_deref().unwrap_or("");
    let delivery_class = required_string(&record.delivery_class, index, "delivery_class")?;
    let populated = [
        record.metric.as_option().is_some(),
        record.event.as_option().is_some(),
        record.log.as_option().is_some(),
        record.artifact.as_option().is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    if populated != 1 {
        return Err(validation_error(
            index,
            "signal",
            "must select exactly one populated signal payload",
        ));
    }

    let namespace = match signal {
        "metric" => {
            let metric = record.metric.as_option().ok_or_else(|| {
                validation_error(index, "metric", "is required for signal metric")
            })?;
            let canonical = canonical_series_id(
                metric.scope.as_deref().unwrap_or(""),
                metric.name.as_deref().unwrap_or(""),
                &resource,
                metric.device_uid.as_deref(),
                metric.device_type.as_deref(),
                &metric.attributes,
            );
            if metric.series_id.as_deref() != Some(canonical.as_str()) {
                return Err(validation_error(
                    index,
                    "metric.series_id",
                    "must equal the server canonical series ID for the authoritative resource",
                ));
            }
            validate_metric(metric, delivery_class, index, &mut row)?
        }
        "event" => {
            validate_event(
                record.event.as_option().ok_or_else(|| {
                    validation_error(index, "event", "is required for signal event")
                })?,
                delivery_class,
                index,
                &mut row,
            )?;
            EVENT_NAMESPACE.to_string()
        }
        "log" => {
            validate_delivery_class(delivery_class, "buffered", index)?;
            validate_log(
                record
                    .log
                    .as_option()
                    .ok_or_else(|| validation_error(index, "log", "is required for signal log"))?,
                index,
                event_ts,
                &resource,
                &mut row,
            )?;
            LOG_NAMESPACE_NAME.to_string()
        }
        "artifact" => {
            validate_delivery_class(delivery_class, "durable", index)?;
            validate_artifact(
                record.artifact.as_option().ok_or_else(|| {
                    validation_error(index, "artifact", "is required for signal artifact")
                })?,
                index,
                &mut row,
            )?;
            ARTIFACT_NAMESPACE.to_string()
        }
        _ => {
            return Err(validation_error(
                index,
                "signal",
                "must be metric, event, log, or artifact",
            ));
        }
    };
    Ok(RoutedRecord {
        namespace,
        delivery_class: delivery_class.to_string(),
        row,
    })
}

fn validate_delivery_class(
    actual: &str,
    expected: &str,
    index: i32,
) -> Result<(), TelemetryValidationErrorV1> {
    if actual == expected {
        return Ok(());
    }
    Err(validation_error(
        index,
        "delivery_class",
        &format!("must match catalog value {expected:?}"),
    ))
}

fn validate_resource(
    resource: &TelemetryResourceV1,
    index: i32,
) -> Result<(), TelemetryValidationErrorV1> {
    required_string(&resource.service_name, index, "resource.service_name")?;
    required_string(
        &resource.service_instance_id,
        index,
        "resource.service_instance_id",
    )?;
    let identity_fields = [
        resource.entity_authority.as_deref(),
        resource.entity_type.as_deref(),
        resource.entity_uid.as_deref(),
    ];
    let populated = identity_fields
        .into_iter()
        .filter(|value| value.is_some_and(|value| !value.is_empty()))
        .count();
    if populated != 0 && populated != identity_fields.len() {
        return Err(validation_error(
            index,
            "resource.entity_uid",
            "entity_authority, entity_type, and entity_uid must be set together",
        ));
    }
    Ok(())
}

fn validate_metric(
    metric: &TelemetryMetricV1,
    delivery_class: &str,
    index: i32,
    row: &mut Row,
) -> Result<String, TelemetryValidationErrorV1> {
    let scope = required_string(&metric.scope, index, "metric.scope")?;
    let name = required_string(&metric.name, index, "metric.name")?;
    let descriptor = catalog().metric(scope, name).ok_or_else(|| {
        validation_error(
            index,
            "metric.name",
            "descriptor is not in the checked-in catalog",
        )
    })?;
    for (field, actual, expected) in [
        (
            "metric.description",
            metric.description.as_deref().unwrap_or(""),
            descriptor.description.as_str(),
        ),
        (
            "metric.unit",
            metric.unit.as_deref().unwrap_or(""),
            descriptor.unit.as_str(),
        ),
        (
            "metric.instrument_kind",
            metric.instrument_kind.as_deref().unwrap_or(""),
            descriptor.instrument_kind.as_str(),
        ),
        (
            "metric.temporality",
            metric.temporality.as_deref().unwrap_or(""),
            descriptor.temporality.as_str(),
        ),
    ] {
        if actual != expected {
            return Err(validation_error(
                index,
                field,
                &format!("must match catalog value {expected:?}"),
            ));
        }
    }
    validate_delivery_class(delivery_class, &descriptor.delivery_class, index)?;
    validate_attributes(
        &metric.attributes,
        &descriptor.attributes,
        index,
        "metric.attributes",
    )?;
    let series_id = required_string(&metric.series_id, index, "metric.series_id")?;

    if descriptor.instrument_kind == "histogram" {
        let bucket_total = metric
            .bucket_counts
            .iter()
            .try_fold(0_i64, |total, count| total.checked_add(*count));
        if metric.value.is_some()
            || metric.count.is_none()
            || metric.sum.is_none()
            || metric.explicit_bounds != descriptor.buckets
            || metric.bucket_counts.len() != metric.explicit_bounds.len() + 1
            || metric.bucket_counts.iter().any(|count| *count < 0)
            || bucket_total.is_none()
            || metric.count != bucket_total
            || metric.sum.is_some_and(|sum| !sum.is_finite())
        {
            return Err(validation_error(
                index,
                "metric",
                "histogram count, sum, catalog bounds, and bucket counts are inconsistent",
            ));
        }
    } else if metric.value.is_none_or(|value| !value.is_finite())
        || metric.count.is_some()
        || metric.sum.is_some()
        || !metric.explicit_bounds.is_empty()
        || !metric.bucket_counts.is_empty()
    {
        return Err(validation_error(
            index,
            "metric.value",
            "scalar metrics require one finite value and no histogram fields",
        ));
    }
    if descriptor.temporality == "cumulative"
        && (metric.start_ts_unix_nano.is_none_or(|value| value <= 0)
            || metric.reset_id.as_deref().is_none_or(str::is_empty))
    {
        return Err(validation_error(
            index,
            "metric.reset_id",
            "cumulative metrics require positive start_ts_unix_nano and reset_id",
        ));
    }
    if descriptor.namespace == crate::store::telemetry_catalog::HARDWARE_METRIC_NAMESPACE
        && (metric.device_uid.as_deref().is_none_or(str::is_empty)
            || metric.device_type.as_deref().is_none_or(str::is_empty))
    {
        return Err(validation_error(
            index,
            "metric.device_uid",
            "hardware metrics require device_uid and device_type",
        ));
    }

    insert_string(row, "scope", scope);
    insert_optional_string(row, "scope_version", &metric.scope_version);
    insert_string(row, "name", name);
    insert_string(row, "description", &descriptor.description);
    insert_string(row, "unit", &descriptor.unit);
    insert_string(row, "instrument_kind", &descriptor.instrument_kind);
    insert_string(row, "temporality", &descriptor.temporality);
    insert_optional_i64(row, "start_ts_unix_nano", metric.start_ts_unix_nano);
    insert_optional_string(row, "reset_id", &metric.reset_id);
    insert_string(row, "series_id", series_id);
    insert_optional_f64(row, "value", metric.value);
    insert_optional_i64(row, "count", metric.count);
    insert_optional_f64(row, "sum", metric.sum);
    if !metric.explicit_bounds.is_empty() {
        row.insert(
            "explicit_bounds".to_string(),
            Cell::Float64List(metric.explicit_bounds.clone()),
        );
    }
    if !metric.bucket_counts.is_empty() {
        row.insert(
            "bucket_counts".to_string(),
            Cell::Int64List(metric.bucket_counts.clone()),
        );
    }
    row.insert(
        "attributes".to_string(),
        Cell::Map(
            metric
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
    );
    insert_string(row, "delivery_class", &descriptor.delivery_class);
    insert_optional_string(row, "device_uid", &metric.device_uid);
    insert_optional_string(row, "device_type", &metric.device_type);
    Ok(descriptor.namespace.clone())
}

fn validate_event(
    event: &TelemetryEventV1,
    delivery_class: &str,
    index: i32,
    row: &mut Row,
) -> Result<(), TelemetryValidationErrorV1> {
    let event_name = required_string(&event.event_name, index, "event.event_name")?;
    let descriptor = catalog().event(event_name).ok_or_else(|| {
        validation_error(
            index,
            "event.event_name",
            "event is not in the checked-in catalog",
        )
    })?;
    validate_delivery_class(delivery_class, &descriptor.delivery_class, index)?;
    let allowed: BTreeMap<String, Vec<String>> = descriptor
        .attributes
        .iter()
        .map(|name| (name.clone(), Vec::new()))
        .collect();
    validate_attributes(&event.attributes, &allowed, index, "event.attributes")?;
    if event
        .severity_number
        .is_none_or(|severity| !(1..=24).contains(&severity))
    {
        return Err(validation_error(
            index,
            "event.severity_number",
            "must be an OpenTelemetry severity number from 1 through 24",
        ));
    }
    required_string(&event.severity_text, index, "event.severity_text")?;
    if event
        .probe_status
        .as_deref()
        .is_some_and(|status| !["success", "failed", "unsupported"].contains(&status))
    {
        return Err(validation_error(
            index,
            "event.probe_status",
            "must be success, failed, or unsupported",
        ));
    }

    insert_string(row, "event_name", event_name);
    insert_optional_i32(row, "severity_number", event.severity_number);
    insert_optional_string(row, "severity_text", &event.severity_text);
    insert_optional_string(row, "outcome", &event.outcome);
    insert_optional_string(row, "phase", &event.phase);
    insert_optional_string(row, "error_type", &event.error_type);
    insert_optional_string(row, "body", &event.body);
    row.insert(
        "attributes".to_string(),
        Cell::Map(
            event
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
    );
    insert_optional_string(row, "trace_id", &event.trace_id);
    insert_optional_string(row, "span_id", &event.span_id);
    insert_optional_string(row, "evidence_uri", &event.evidence_uri);
    insert_optional_string(row, "result_uri", &event.result_uri);
    insert_string(row, "delivery_class", &descriptor.delivery_class);
    insert_optional_string(row, "probe_status", &event.probe_status);
    Ok(())
}

fn validate_log(
    log: &TelemetryLogV1,
    index: i32,
    event_ts: i64,
    resource: &TelemetryResourceV1,
    row: &mut Row,
) -> Result<(), TelemetryValidationErrorV1> {
    let source = required_string(&log.source, index, "log.source")?;
    let body = required_string(&log.body, index, "log.body")?;
    if log
        .severity_number
        .is_none_or(|severity| !(0..=24).contains(&severity))
    {
        return Err(validation_error(
            index,
            "log.severity_number",
            "must be an OpenTelemetry severity number from 0 through 24",
        ));
    }
    insert_string(
        row,
        "key",
        resource.service_instance_id.as_deref().unwrap_or(source),
    );
    insert_string(row, "source", source);
    insert_string(row, "data", body);
    row.insert("epoch_ms".to_string(), Cell::Int64(event_ts / 1_000_000));
    insert_optional_i32(row, "level", log.severity_number);
    insert_optional_string(row, "event_name", &log.event_name);
    insert_optional_string(row, "severity_text", &log.severity_text);
    row.insert(
        "attributes".to_string(),
        Cell::Map(
            log.attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
    );
    insert_optional_string(row, "trace_id", &log.trace_id);
    insert_optional_string(row, "span_id", &log.span_id);
    Ok(())
}

fn validate_artifact(
    artifact: &TelemetryArtifactV1,
    index: i32,
    row: &mut Row,
) -> Result<(), TelemetryValidationErrorV1> {
    let capture_type = required_string(&artifact.capture_type, index, "artifact.capture_type")?;
    if !catalog().artifact(capture_type) {
        return Err(validation_error(
            index,
            "artifact.capture_type",
            "capture type is not in the checked-in catalog",
        ));
    }
    let trigger = required_string(&artifact.trigger, index, "artifact.trigger")?;
    let outcome = required_string(&artifact.outcome, index, "artifact.outcome")?;
    if artifact.size_bytes.is_some_and(|size| size < 0) {
        return Err(validation_error(
            index,
            "artifact.size_bytes",
            "must be non-negative",
        ));
    }
    insert_string(row, "capture_type", capture_type);
    insert_string(row, "trigger", trigger);
    insert_optional_i64(
        row,
        "capture_start_ts_unix_nano",
        artifact.start_ts_unix_nano,
    );
    insert_optional_i64(row, "capture_end_ts_unix_nano", artifact.end_ts_unix_nano);
    insert_string(row, "outcome", outcome);
    insert_optional_i64(row, "size_bytes", artifact.size_bytes);
    insert_optional_string(row, "sha256", &artifact.sha256);
    insert_optional_string(row, "uri", &artifact.uri);
    insert_optional_string(row, "summary", &artifact.summary);
    row.insert(
        "attributes".to_string(),
        Cell::Map(
            artifact
                .attributes
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
    );
    Ok(())
}

fn base_row(
    batch: &TelemetryBatchV1,
    index: i32,
    event_ts: i64,
    observed_ts: i64,
    resource: &TelemetryResourceV1,
) -> Row {
    let batch_id = batch.batch_id.as_deref().unwrap_or("");
    let mut row = Row::new();
    row.insert("schema_version".to_string(), Cell::Int32(SCHEMA_VERSION));
    insert_string(&mut row, "catalog_version", CATALOG_VERSION);
    insert_string(&mut row, "batch_id", batch_id);
    row.insert("record_index".to_string(), Cell::Int32(index));
    insert_string(&mut row, "point_id", &format!("{batch_id}:{index}"));
    row.insert("event_ts_unix_nano".to_string(), Cell::Int64(event_ts));
    row.insert(
        "observed_ts_unix_nano".to_string(),
        Cell::Int64(observed_ts),
    );
    insert_resource(&mut row, resource);
    row
}

fn insert_resource(row: &mut Row, resource: &TelemetryResourceV1) {
    for (name, value) in [
        ("service_name", &resource.service_name),
        ("service_instance_id", &resource.service_instance_id),
        ("role", &resource.role),
        ("root_run_uid", &resource.root_run_uid),
        ("service_version", &resource.service_version),
        ("run_id_alias", &resource.run_id_alias),
        ("iris_job_id", &resource.iris_job_id),
        ("iris_task_id", &resource.iris_task_id),
        ("attempt_uid", &resource.attempt_uid),
        ("worker_id", &resource.worker_id),
        ("node_id", &resource.node_id),
        ("pod_uid", &resource.pod_uid),
        ("container_id", &resource.container_id),
        ("actor_id", &resource.actor_id),
        ("engine_id", &resource.engine_id),
        ("repository", &resource.repository),
        ("git_revision", &resource.git_revision),
        ("image_digest", &resource.image_digest),
        ("model_id", &resource.model_id),
        ("model_revision", &resource.model_revision),
        ("owner", &resource.owner),
        ("cluster", &resource.cluster),
        ("entity_authority", &resource.entity_authority),
        ("entity_type", &resource.entity_type),
        ("entity_uid", &resource.entity_uid),
    ] {
        insert_optional_string(row, name, value);
    }
    for (name, value) in [
        ("task_index", resource.task_index),
        ("attempt_id", resource.attempt_id),
        ("rank", resource.rank),
        ("process_index", resource.process_index),
        ("experiment_issue", resource.experiment_issue),
    ] {
        insert_optional_i32(row, name, value);
    }
    insert_optional_i64(row, "policy_step", resource.policy_step);
}

fn validate_attributes(
    actual: &HashMap<String, String>,
    allowed: &BTreeMap<String, Vec<String>>,
    index: i32,
    field: &str,
) -> Result<(), TelemetryValidationErrorV1> {
    let mut attributes: Vec<_> = actual.iter().collect();
    attributes.sort_by(|left, right| left.0.cmp(right.0));
    for (name, value) in attributes {
        let allowed_values = allowed.get(name).ok_or_else(|| {
            validation_error(index, field, &format!("attribute {name:?} is not declared"))
        })?;
        if !allowed_values.is_empty() && !allowed_values.contains(value) {
            return Err(validation_error(
                index,
                field,
                &format!("attribute {name:?} value {value:?} is not allowed"),
            ));
        }
    }
    Ok(())
}

async fn commit_records(
    state: Arc<IngestState>,
    origin_cluster: Option<String>,
    batch_id: String,
    content_type: String,
    payload_sha256: String,
    records: Vec<RoutedRecord>,
    deadline: Instant,
) -> Result<CommitOutcome, ApiError> {
    let record_count = records.len();
    let mut rows_by_route: BTreeMap<(String, String), Vec<Row>> = BTreeMap::new();
    for record in records {
        rows_by_route
            .entry((record.namespace, record.delivery_class))
            .or_default()
            .push(record.row);
    }
    let receipt_row = batch_receipt_row(&batch_id, &payload_sha256, record_count, &content_type);
    let prepare_store = Arc::clone(&state.store);
    let prepare_batch_id = batch_id.clone();
    let prepare_digest = payload_sha256.clone();
    let prepare_origin = origin_cluster.clone();
    let (intent, children, completion) = {
        let _schema_registration =
            tokio::time::timeout_at(deadline, state.schema_registration.lock())
                .await
                .map_err(|_| request_deadline())?;
        run_blocking(&state, deadline, move || {
            let intent = prepare_namespace(
                &prepare_store,
                BATCH_INTENT_NAMESPACE,
                vec![receipt_row.clone()],
                &prepare_batch_id,
                prepare_origin.as_deref(),
                &prepare_digest,
                INTERNAL_DELIVERY_CLASS,
            )?;
            let mut children = Vec::with_capacity(rows_by_route.len());
            for ((namespace, delivery_class), rows) in rows_by_route {
                children.push(prepare_namespace(
                    &prepare_store,
                    &namespace,
                    rows,
                    &prepare_batch_id,
                    prepare_origin.as_deref(),
                    &prepare_digest,
                    &delivery_class,
                )?);
            }
            let completion = prepare_namespace(
                &prepare_store,
                BATCH_NAMESPACE,
                vec![receipt_row],
                &prepare_batch_id,
                prepare_origin.as_deref(),
                &prepare_digest,
                INTERNAL_DELIVERY_CLASS,
            )?;
            Ok((intent, children, completion))
        })
        .await?
    };

    let intent_commit = append_prepared(
        &state,
        intent,
        origin_cluster.clone(),
        payload_sha256.clone(),
        deadline,
    )
    .await?;
    await_commit(&state.store, &intent_commit, deadline).await?;

    let append_state = Arc::clone(&state);
    let append_origin = origin_cluster.clone();
    let append_digest = payload_sha256.clone();
    let child_commits = run_blocking(&state, deadline, move || {
        children
            .into_iter()
            .map(|prepared| {
                append_prepared_sync(
                    &append_state,
                    prepared,
                    append_origin.as_deref(),
                    &append_digest,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(ApiError::from_store)
    })
    .await?;
    for commit in &child_commits {
        await_commit(&state.store, commit, deadline).await?;
    }

    let completion_commit =
        append_prepared(&state, completion, origin_cluster, payload_sha256, deadline).await?;
    await_commit(&state.store, &completion_commit, deadline).await?;
    let parent_deduplicated = completion_commit.result.deduplicated;
    Ok(CommitOutcome {
        commits: child_commits,
        parent_deduplicated,
    })
}

fn prepare_namespace(
    store: &Store,
    namespace: &str,
    rows: Vec<Row>,
    parent_batch_id: &str,
    origin_cluster: Option<&str>,
    payload_sha256: &str,
    delivery_class: &str,
) -> Result<PreparedNamespace, ApiError> {
    let schema = namespace_schema(namespace).ok_or_else(|| {
        ApiError::from_store(StatsError::Internal(format!(
            "telemetry catalog routed to unapproved namespace {namespace:?}"
        )))
    })?;
    store
        .register_table(namespace, schema.clone(), StoragePolicy::default())
        .map_err(ApiError::from_store)?;
    let batch = rows_to_batch(&schema, &rows).map_err(ApiError::from_store)?;
    let arrow_ipc = encode_ipc(&batch.schema(), &[batch]).map_err(|error| {
        ApiError::from_store(StatsError::Internal(format!(
            "encode telemetry IPC: {error}"
        )))
    })?;
    if arrow_ipc.len() > MAX_WRITE_ROWS_BYTES {
        return Err(ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "normalized_namespace_too_large",
            format!(
                "normalized namespace {namespace:?} is {} bytes; limit is {MAX_WRITE_ROWS_BYTES}",
                arrow_ipc.len()
            ),
        ));
    }
    let batch_id = sub_batch_id(parent_batch_id, namespace, delivery_class);
    store
        .preflight_rows(
            namespace,
            &batch_id,
            &arrow_ipc,
            origin_cluster,
            payload_sha256,
        )
        .map_err(ApiError::from_store)?;
    Ok(PreparedNamespace {
        namespace: namespace.to_string(),
        batch_id,
        arrow_ipc,
    })
}

async fn append_prepared(
    state: &Arc<IngestState>,
    prepared: PreparedNamespace,
    origin_cluster: Option<String>,
    payload_sha256: String,
    deadline: Instant,
) -> Result<NamespaceCommit, ApiError> {
    let state = Arc::clone(state);
    let task_state = Arc::clone(&state);
    run_blocking(&state, deadline, move || {
        append_prepared_sync(
            &task_state,
            prepared,
            origin_cluster.as_deref(),
            &payload_sha256,
        )
        .map_err(ApiError::from_store)
    })
    .await
}

fn append_prepared_sync(
    state: &IngestState,
    prepared: PreparedNamespace,
    origin_cluster: Option<&str>,
    payload_sha256: &str,
) -> Result<NamespaceCommit, StatsError> {
    let result = state.store.write_rows_with_payload_digest(
        &prepared.namespace,
        &prepared.batch_id,
        &prepared.arrow_ipc,
        origin_cluster,
        payload_sha256,
    )?;
    #[cfg(test)]
    state.fail_after_child_append(&prepared.namespace)?;
    Ok(NamespaceCommit {
        namespace: prepared.namespace,
        result,
    })
}

#[cfg(test)]
impl IngestState {
    fn inject_failure_after_child_append(&self, namespace: &str) {
        *self.fail_after_child_append.lock().unwrap() = Some(namespace.to_string());
    }

    fn fail_after_child_append(&self, namespace: &str) -> Result<(), StatsError> {
        let mut target = self.fail_after_child_append.lock().unwrap();
        if target.as_deref() == Some(namespace) {
            *target = None;
            return Err(StatsError::Internal(
                "injected failure after telemetry child append".to_string(),
            ));
        }
        Ok(())
    }
}

async fn await_commit(
    store: &Store,
    commit: &NamespaceCommit,
    deadline: Instant,
) -> Result<(), ApiError> {
    if commit.result.receipt_state == ReceiptState::Durable {
        return Ok(());
    }
    let budget = remaining(deadline)?;
    store
        .await_persisted(&commit.namespace, commit.result.receipt.last_seq, budget)
        .await
        .map_err(ApiError::from_store)
}

fn namespace_schema(namespace: &str) -> Option<Schema> {
    if namespace == LOG_NAMESPACE_NAME {
        Some(log_registered_schema())
    } else {
        schema_for_namespace(namespace)
    }
}

fn batch_receipt_row(
    batch_id: &str,
    payload_sha256: &str,
    record_count: usize,
    content_type: &str,
) -> Row {
    let mut row = Row::new();
    row.insert("schema_version".to_string(), Cell::Int32(SCHEMA_VERSION));
    insert_string(&mut row, "catalog_version", CATALOG_VERSION);
    insert_string(&mut row, "batch_id", batch_id);
    insert_string(&mut row, "payload_sha256", payload_sha256);
    row.insert("record_count".to_string(), Cell::Int32(record_count as i32));
    row.insert(
        "received_ts_unix_nano".to_string(),
        Cell::Int64(now_unix_nano()),
    );
    insert_string(&mut row, "content_type", content_type);
    row
}

fn rows_to_batch(schema: &Schema, rows: &[Row]) -> Result<RecordBatch, StatsError> {
    let mut arrays = Vec::with_capacity(schema.columns.len());
    for column in &schema.columns {
        let array: ArrayRef = match column.r#type {
            ColumnType::COLUMN_TYPE_STRING => Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| string_cell(row.get(&column.name)))
                    .collect::<Vec<_>>(),
            )),
            ColumnType::COLUMN_TYPE_INT32 => Arc::new(Int32Array::from(
                rows.iter()
                    .map(|row| int32_cell(row.get(&column.name)))
                    .collect::<Vec<_>>(),
            )),
            ColumnType::COLUMN_TYPE_INT64 => Arc::new(Int64Array::from(
                rows.iter()
                    .map(|row| int64_cell(row.get(&column.name)))
                    .collect::<Vec<_>>(),
            )),
            ColumnType::COLUMN_TYPE_FLOAT64 => Arc::new(Float64Array::from(
                rows.iter()
                    .map(|row| float64_cell(row.get(&column.name)))
                    .collect::<Vec<_>>(),
            )),
            ColumnType::COLUMN_TYPE_BOOL => Arc::new(BooleanArray::new_null(rows.len())),
            ColumnType::COLUMN_TYPE_LIST_FLOAT64 => {
                Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(
                    rows.iter().map(|row| match row.get(&column.name) {
                        Some(Cell::Float64List(values)) => {
                            Some(values.iter().copied().map(Some).collect::<Vec<_>>())
                        }
                        _ => None,
                    }),
                ))
            }
            ColumnType::COLUMN_TYPE_LIST_INT64 => {
                Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
                    rows.iter().map(|row| match row.get(&column.name) {
                        Some(Cell::Int64List(values)) => {
                            Some(values.iter().copied().map(Some).collect::<Vec<_>>())
                        }
                        _ => None,
                    }),
                ))
            }
            ColumnType::COLUMN_TYPE_MAP => map_array(rows, &column.name),
            unsupported => {
                return Err(StatsError::Internal(format!(
                    "telemetry schema column {:?} has unsupported type {unsupported:?}",
                    column.name
                )));
            }
        };
        arrays.push(array);
    }
    RecordBatch::try_new(schema_to_arrow(schema), arrays)
        .map_err(|error| StatsError::Internal(format!("build telemetry record batch: {error}")))
}

fn map_array(rows: &[Row], column: &str) -> ArrayRef {
    let names = MapFieldNames {
        entry: "entries".to_string(),
        key: "key".to_string(),
        value: "value".to_string(),
    };
    let mut builder = MapBuilder::new(Some(names), StringBuilder::new(), StringBuilder::new());
    for row in rows {
        match row.get(column) {
            Some(Cell::Map(values)) => {
                for (key, value) in values {
                    builder.keys().append_value(key);
                    builder.values().append_value(value);
                }
                builder.append(true).expect("map offsets remain in range");
            }
            _ => builder.append(false).expect("map offsets remain in range"),
        }
    }
    Arc::new(builder.finish())
}

fn string_cell(cell: Option<&Cell>) -> Option<&str> {
    match cell {
        Some(Cell::String(value)) => Some(value),
        _ => None,
    }
}

fn int32_cell(cell: Option<&Cell>) -> Option<i32> {
    match cell {
        Some(Cell::Int32(value)) => Some(*value),
        _ => None,
    }
}

fn int64_cell(cell: Option<&Cell>) -> Option<i64> {
    match cell {
        Some(Cell::Int64(value)) => Some(*value),
        _ => None,
    }
}

fn float64_cell(cell: Option<&Cell>) -> Option<f64> {
    match cell {
        Some(Cell::Float64(value)) => Some(*value),
        _ => None,
    }
}

async fn decode_body(
    state: &Arc<IngestState>,
    headers: &HeaderMap,
    body: Bytes,
    deadline: Instant,
) -> Result<Vec<u8>, ApiError> {
    let encoding = match headers.get(CONTENT_ENCODING) {
        Some(value) => value
            .to_str()
            .map_err(|_| {
                ApiError::new(
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "unsupported_content_encoding",
                    "content-encoding must be valid ASCII",
                )
            })?
            .trim()
            .to_ascii_lowercase(),
        None => "identity".to_string(),
    };
    run_blocking(state, deadline, move || decode_body_sync(&encoding, body)).await
}

async fn read_wire_body(body: Body) -> Result<Bytes, ApiError> {
    to_bytes(body, MAX_MESSAGE_BYTES)
        .await
        .map_err(|_| body_too_large(MAX_MESSAGE_BYTES + 1))
}

fn decode_body_sync(encoding: &str, body: Bytes) -> Result<Vec<u8>, ApiError> {
    if matches!(encoding, "" | "identity") {
        if body.len() > MAX_MESSAGE_BYTES {
            return Err(body_too_large(body.len()));
        }
        return Ok(body.to_vec());
    }
    let mut decoded = Vec::new();
    let limit = (MAX_MESSAGE_BYTES + 1) as u64;
    match encoding {
        "gzip" => {
            GzDecoder::new(std::io::Cursor::new(body))
                .take(limit)
                .read_to_end(&mut decoded)
                .map_err(|error| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_compression",
                        format!("invalid gzip body: {error}"),
                    )
                })?;
        }
        "zstd" => {
            let mut decoder = zstd::stream::read::Decoder::new(std::io::Cursor::new(body))
                .map_err(|error| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_compression",
                        format!("invalid zstd body: {error}"),
                    )
                })?;
            decoder
                .window_log_max(ZSTD_WINDOW_LOG_MAX)
                .map_err(|error| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_compression",
                        format!("invalid zstd window: {error}"),
                    )
                })?;
            decoder
                .take(limit)
                .read_to_end(&mut decoded)
                .map_err(|error| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_compression",
                        format!("invalid zstd body: {error}"),
                    )
                })?;
        }
        _ => {
            return Err(ApiError::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_content_encoding",
                "content-encoding must be identity, gzip, or zstd",
            ));
        }
    }
    if decoded.len() > MAX_MESSAGE_BYTES {
        return Err(body_too_large(decoded.len()));
    }
    Ok(decoded)
}

fn body_too_large(size: usize) -> ApiError {
    ApiError::new(
        StatusCode::PAYLOAD_TOO_LARGE,
        "body_too_large",
        format!("uncompressed request body {size} bytes exceeds {MAX_MESSAGE_BYTES} limit"),
    )
}

async fn run_blocking<T, F>(
    state: &Arc<IngestState>,
    deadline: Instant,
    task: F,
) -> Result<T, ApiError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ApiError> + Send + 'static,
{
    let permit =
        tokio::time::timeout_at(deadline, Arc::clone(&state.blocking_tasks).acquire_owned())
            .await
            .map_err(|_| request_deadline())?
            .map_err(|_| {
                ApiError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_stopping",
                    "telemetry ingestion is stopping",
                )
            })?;
    tokio::time::timeout_at(
        deadline,
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            task()
        }),
    )
    .await
    .map_err(|_| request_deadline())?
    .map_err(|error| {
        ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("telemetry blocking task failed: {error}"),
        )
    })?
}

fn remaining(deadline: Instant) -> Result<Duration, ApiError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(request_deadline)
}

fn request_deadline() -> ApiError {
    ApiError::retryable(
        StatusCode::GATEWAY_TIMEOUT,
        "request_deadline",
        "telemetry request exceeded its 30 second deadline",
    )
}

fn required_idempotency_key(headers: &HeaderMap) -> Result<String, ApiError> {
    let key = headers
        .get("idempotency-key")
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "missing_idempotency_key",
                "Idempotency-Key header is required",
            )
        })?
        .to_str()
        .map_err(|_| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_idempotency_key",
                "Idempotency-Key must be valid ASCII",
            )
        })?;
    if !valid_idempotency_key(key) {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_idempotency_key",
            "Idempotency-Key must contain 1..=128 safe ASCII characters",
        ));
    }
    Ok(key.to_string())
}

fn valid_idempotency_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= MAX_IDEMPOTENCY_KEY_BYTES
        && key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
}

fn request_digest(
    identity: &AuthIdentity,
    endpoint: &str,
    content_type: &str,
    body: &[u8],
) -> String {
    let mut digest = Sha256::new();
    digest.update(REST_DIGEST_DOMAIN);
    match identity {
        AuthIdentity::Jwt { cluster } => {
            digest.update([1]);
            digest_field(&mut digest, cluster.as_bytes());
        }
        AuthIdentity::TrustedCollector { cluster } => {
            digest.update([2]);
            digest_field(&mut digest, cluster.as_bytes());
        }
        AuthIdentity::Network => digest.update([0]),
    }
    digest_field(&mut digest, endpoint.as_bytes());
    digest_field(&mut digest, content_type.as_bytes());
    digest_field(&mut digest, body);
    format!("{:x}", digest.finalize())
}

fn digest_field(digest: &mut Sha256, value: &[u8]) {
    digest.update((value.len() as u64).to_be_bytes());
    digest.update(value);
}

fn sub_batch_id(parent_batch_id: &str, namespace: &str, delivery_class: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(SUB_BATCH_ID_DOMAIN);
    digest_field(&mut digest, parent_batch_id.as_bytes());
    digest_field(&mut digest, namespace.as_bytes());
    digest_field(&mut digest, delivery_class.as_bytes());
    format!("telemetry-{:x}", digest.finalize())
}

fn identity_origin(identity: &AuthIdentity) -> Option<String> {
    match identity {
        AuthIdentity::Jwt { cluster } | AuthIdentity::TrustedCollector { cluster } => {
            Some(cluster.clone())
        }
        AuthIdentity::Network => None,
    }
}

fn authoritative_resource(
    resource: &TelemetryResourceV1,
    identity: &AuthIdentity,
) -> TelemetryResourceV1 {
    let mut resource = resource.clone();
    if let AuthIdentity::TrustedCollector { cluster } = identity {
        resource.cluster = Some(cluster.clone());
        return resource;
    }
    resource.cluster = match identity {
        AuthIdentity::Jwt { cluster } => Some(cluster.clone()),
        AuthIdentity::Network => None,
        AuthIdentity::TrustedCollector { .. } => unreachable!(),
    };
    resource.iris_job_id = None;
    resource.iris_task_id = None;
    resource.task_index = None;
    resource.attempt_id = None;
    resource.attempt_uid = None;
    resource.worker_id = None;
    resource.node_id = None;
    resource.pod_uid = None;
    resource.container_id = None;
    resource.entity_authority = None;
    resource.entity_type = None;
    resource.entity_uid = None;
    resource
}

fn success_ack(batch_id: &str, accepted: usize, outcome: &CommitOutcome) -> TelemetryWriteAckV1 {
    TelemetryWriteAckV1 {
        schema_version: Some(SCHEMA_VERSION),
        batch_id: Some(batch_id.to_string()),
        status: Some(if outcome.parent_deduplicated {
            "duplicate".to_string()
        } else {
            "accepted".to_string()
        }),
        durability: Some("finelog_local".to_string()),
        accepted_records: Some(accepted as i32),
        rejected_records: Vec::new(),
        commits: outcome
            .commits
            .iter()
            .filter(|commit| commit.namespace != BATCH_NAMESPACE)
            .map(|commit| TelemetryCommitV1 {
                namespace: Some(commit.namespace.clone()),
                first_seq: Some(commit.result.receipt.first_seq),
                last_seq: Some(commit.result.receipt.last_seq),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    }
}

fn validation_response(
    format: WireFormat,
    batch_id: &str,
    errors: Vec<TelemetryValidationErrorV1>,
) -> Response {
    let ack = TelemetryWriteAckV1 {
        schema_version: Some(SCHEMA_VERSION),
        batch_id: (!batch_id.is_empty()).then(|| batch_id.to_string()),
        status: Some("rejected".to_string()),
        durability: Some("none".to_string()),
        accepted_records: Some(0),
        rejected_records: errors,
        commits: Vec::new(),
        ..Default::default()
    };
    encode_custom_response(StatusCode::BAD_REQUEST, format, &ack)
}

fn encode_custom_response(
    status: StatusCode,
    format: WireFormat,
    ack: &TelemetryWriteAckV1,
) -> Response {
    let body = match format {
        WireFormat::Json => {
            serde_json::to_vec(ack).expect("TelemetryWriteAckV1 JSON serialization is infallible")
        }
        WireFormat::Protobuf => ack.encode_to_vec(),
    };
    response(status, format.content_type(), body)
}

fn response(status: StatusCode, content_type: &'static str, body: Vec<u8>) -> Response {
    let mut response = (status, body).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
}

fn validation_error(record_index: i32, field: &str, reason: &str) -> TelemetryValidationErrorV1 {
    TelemetryValidationErrorV1 {
        record_index: Some(record_index),
        field: Some(field.to_string()),
        reason: Some(reason.to_string()),
        ..Default::default()
    }
}

fn required_string<'a>(
    value: &'a Option<String>,
    record_index: i32,
    field: &str,
) -> Result<&'a str, TelemetryValidationErrorV1> {
    value
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| validation_error(record_index, field, "is required"))
}

fn positive_i64(
    value: Option<i64>,
    record_index: i32,
    field: &str,
) -> Result<i64, TelemetryValidationErrorV1> {
    value
        .filter(|value| *value > 0)
        .ok_or_else(|| validation_error(record_index, field, "must be positive"))
}

fn insert_string(row: &mut Row, name: &str, value: &str) {
    row.insert(name.to_string(), Cell::String(value.to_string()));
}

fn insert_optional_string(row: &mut Row, name: &str, value: &Option<String>) {
    if let Some(value) = value {
        row.insert(name.to_string(), Cell::String(value.clone()));
    }
}

fn insert_optional_i32(row: &mut Row, name: &str, value: Option<i32>) {
    if let Some(value) = value {
        row.insert(name.to_string(), Cell::Int32(value));
    }
}

fn insert_optional_i64(row: &mut Row, name: &str, value: Option<i64>) {
    if let Some(value) = value {
        row.insert(name.to_string(), Cell::Int64(value));
    }
}

fn insert_optional_f64(row: &mut Row, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        row.insert(name.to_string(), Cell::Float64(value));
    }
}

fn now_unix_nano() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(i64::MAX as u128) as i64
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::convert::Infallible;
    use std::io::Write;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    use arrow::array::{Array, StringArray};
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use buffa::MessageField;
    use datafusion::common::TableReference;
    use http_body::Frame;
    use jsonwebtoken::{Algorithm, EncodingKey};
    use opentelemetry_proto::tonic::common::v1::{AnyValue, InstrumentationScope};
    use opentelemetry_proto::tonic::logs::v1::{ResourceLogs, ScopeLogs};
    use opentelemetry_proto::tonic::metrics::v1::{Gauge, ResourceMetrics, ScopeMetrics};
    use tower::ServiceExt;

    use super::*;
    use crate::query::{make_ctx, run_query_over};
    use crate::server::auth::{AuthPolicy, FINELOG_AUDIENCE};
    use crate::server::test_support::{PRIV_A, PUB_A};
    use crate::test_support::unique_dir;

    fn resource() -> TelemetryResourceV1 {
        TelemetryResourceV1 {
            service_name: Some("test-service".to_string()),
            service_instance_id: Some("instance-1".to_string()),
            iris_job_id: Some("forged-job".to_string()),
            iris_task_id: Some("forged-task".to_string()),
            worker_id: Some("forged-worker".to_string()),
            node_id: Some("forged-node".to_string()),
            pod_uid: Some("forged-pod".to_string()),
            container_id: Some("forged-container".to_string()),
            entity_authority: Some("forged-authority".to_string()),
            entity_type: Some("worker".to_string()),
            entity_uid: Some("forged-uid".to_string()),
            ..Default::default()
        }
    }

    fn log_batch(batch_id: &str, body: &str) -> TelemetryBatchV1 {
        TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some(batch_id.to_string()),
            records: vec![TelemetryRecordV1 {
                record_index: Some(0),
                signal: Some("log".to_string()),
                event_ts_unix_nano: Some(1_000_000_000),
                observed_ts_unix_nano: Some(1_000_000_001),
                delivery_class: Some("buffered".to_string()),
                resource: MessageField::some(resource()),
                log: MessageField::some(TelemetryLogV1 {
                    source: Some("test.logger".to_string()),
                    body: Some(body.to_string()),
                    severity_number: Some(9),
                    severity_text: Some("INFO".to_string()),
                    event_name: Some("test.ready".to_string()),
                    attributes: HashMap::from([("phase".to_string(), "ready".to_string())]),
                    trace_id: Some("00112233445566778899aabbccddeeff".to_string()),
                    span_id: Some("0011223344556677".to_string()),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn event_batch(batch_id: &str, severity_text: Option<&str>) -> TelemetryBatchV1 {
        TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some(batch_id.to_string()),
            records: vec![TelemetryRecordV1 {
                record_index: Some(0),
                signal: Some("event".to_string()),
                event_ts_unix_nano: Some(1_000_000_000),
                observed_ts_unix_nano: Some(1_000_000_001),
                delivery_class: Some("durable".to_string()),
                resource: MessageField::some(resource()),
                event: MessageField::some(TelemetryEventV1 {
                    event_name: Some("telemetry.runtime.gap".to_string()),
                    severity_number: Some(17),
                    severity_text: severity_text.map(str::to_string),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn metric_resource() -> TelemetryResourceV1 {
        TelemetryResourceV1 {
            service_name: Some("test-service".to_string()),
            service_instance_id: Some("instance-1".to_string()),
            ..Default::default()
        }
    }

    fn metric_batch(batch_id: &str, spoof_series_id: bool) -> TelemetryBatchV1 {
        let resource = metric_resource();
        let attributes = HashMap::new();
        let series_id = if spoof_series_id {
            "producer-controlled".to_string()
        } else {
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &resource,
                None,
                None,
                &attributes,
            )
        };
        TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some(batch_id.to_string()),
            records: vec![TelemetryRecordV1 {
                record_index: Some(0),
                signal: Some("metric".to_string()),
                event_ts_unix_nano: Some(1_000_000_000),
                observed_ts_unix_nano: Some(1_000_000_001),
                delivery_class: Some("coalescing".to_string()),
                resource: MessageField::some(resource),
                metric: MessageField::some(TelemetryMetricV1 {
                    scope: Some("telemetry.runtime".to_string()),
                    scope_version: Some("1".to_string()),
                    name: Some("queue_records".to_string()),
                    description: Some(
                        "Telemetry records waiting for background export".to_string(),
                    ),
                    unit: Some("{record}".to_string()),
                    instrument_kind: Some("gauge".to_string()),
                    temporality: Some("unspecified".to_string()),
                    series_id: Some(series_id),
                    value: Some(2.0),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn otlp_metric_request(points: Vec<NumberDataPoint>) -> ExportMetricsServiceRequest {
        ExportMetricsServiceRequest {
            resource_metrics: vec![ResourceMetrics {
                resource: Some(OtlpResource {
                    attributes: vec![
                        KeyValue {
                            key: "service.name".to_string(),
                            value: Some(AnyValue {
                                value: Some(any_value::Value::StringValue(
                                    "test-service".to_string(),
                                )),
                            }),
                            ..Default::default()
                        },
                        KeyValue {
                            key: "service.instance.id".to_string(),
                            value: Some(AnyValue {
                                value: Some(any_value::Value::StringValue(
                                    "instance-1".to_string(),
                                )),
                            }),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                }),
                scope_metrics: vec![ScopeMetrics {
                    scope: Some(InstrumentationScope {
                        name: "telemetry.runtime".to_string(),
                        version: "1".to_string(),
                        ..Default::default()
                    }),
                    metrics: vec![Metric {
                        name: "queue_records".to_string(),
                        description: "Telemetry records waiting for background export".to_string(),
                        unit: "{record}".to_string(),
                        data: Some(metric::Data::Gauge(Gauge {
                            data_points: points,
                        })),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
        }
    }

    async fn send_custom(
        store: Arc<Store>,
        identity: AuthIdentity,
        batch: &TelemetryBatchV1,
    ) -> Response {
        send_custom_to(router(store), identity, batch).await
    }

    async fn send_custom_to(
        app: Router,
        identity: AuthIdentity,
        batch: &TelemetryBatchV1,
    ) -> Response {
        let batch_id = batch.batch_id.as_deref().unwrap_or("missing");
        let request = Request::builder()
            .method("POST")
            .uri(CUSTOM_ENDPOINT)
            .header(CONTENT_TYPE, "application/json")
            .header("idempotency-key", batch_id)
            .body(Body::from(serde_json::to_vec(batch).unwrap()))
            .unwrap();
        app.layer(Extension(identity))
            .oneshot(request)
            .await
            .unwrap()
    }

    fn test_state(store: Arc<Store>) -> Arc<IngestState> {
        test_state_with_timeout(store, REQUEST_TIMEOUT)
    }

    fn test_state_with_timeout(store: Arc<Store>, request_timeout: Duration) -> Arc<IngestState> {
        Arc::new(IngestState {
            store,
            requests: Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS)),
            blocking_tasks: Arc::new(Semaphore::new(MAX_CONCURRENT_BLOCKING_TASKS)),
            schema_registration: Arc::new(tokio::sync::Mutex::new(())),
            request_timeout,
            fail_after_child_append: std::sync::Mutex::new(None),
        })
    }

    fn push_test_varint(output: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            output.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    fn protobuf_message_field(field: u32, value: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(value.len() + 8);
        push_test_varint(&mut output, u64::from((field << 3) | 2));
        push_test_varint(&mut output, value.len() as u64);
        output.extend_from_slice(value);
        output
    }

    fn push_test_key(output: &mut Vec<u8>, field: u32, wire_type: u8) {
        push_test_varint(output, u64::from((field << 3) | u32::from(wire_type)));
    }

    fn protobuf_group(field: u32, contents: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(contents.len() + 8);
        push_test_key(&mut output, field, 3);
        output.extend_from_slice(contents);
        push_test_key(&mut output, field, 4);
        output
    }

    struct PendingBody {
        polls: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
    }

    impl http_body::Body for PendingBody {
        type Data = Bytes;
        type Error = Infallible;

        fn poll_frame(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            Poll::Pending
        }
    }

    impl Drop for PendingBody {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    fn namespace_rows(store: &Store, namespace: &str) -> i64 {
        store
            .list_namespaces_with_stats()
            .unwrap()
            .into_iter()
            .find(|(name, _, _, _)| name == namespace)
            .map_or(0, |(_, _, stats, _)| stats.row_count)
    }

    async fn query_namespace(store: &Store, namespace: &str, sql: &str) -> Vec<RecordBatch> {
        let references = BTreeSet::from([TableReference::bare(namespace)]);
        let providers = store.query_providers_for(&references).unwrap();
        run_query_over(&make_ctx(), providers, sql)
            .await
            .unwrap()
            .batches
    }

    #[test]
    fn request_digest_binds_origin_endpoint_content_type_and_body() {
        let local = AuthIdentity::Network;
        let remote = AuthIdentity::Jwt {
            cluster: "cluster-a".to_string(),
        };
        let baseline = request_digest(&local, CUSTOM_ENDPOINT, "application/json", b"body");
        assert_eq!(
            baseline,
            request_digest(&local, CUSTOM_ENDPOINT, "application/json", b"body")
        );
        assert_ne!(
            baseline,
            request_digest(&remote, CUSTOM_ENDPOINT, "application/json", b"body")
        );
        assert_ne!(
            baseline,
            request_digest(&local, "/v1/logs", "application/json", b"body")
        );
        assert_ne!(
            baseline,
            request_digest(&local, CUSTOM_ENDPOINT, "application/x-protobuf", b"body")
        );
        assert_ne!(
            baseline,
            request_digest(&local, CUSTOM_ENDPOINT, "application/json", b"changed")
        );
    }

    #[test]
    fn canonical_series_id_binds_authoritative_identity_and_attempt() {
        let mut resource = metric_resource();
        resource.cluster = Some("cluster-a".to_string());
        resource.entity_authority = Some("iris".to_string());
        resource.entity_type = Some("task_attempt".to_string());
        resource.entity_uid = Some("entity-1".to_string());
        resource.attempt_uid = Some("attempt-1".to_string());
        let attributes = HashMap::from([("outcome".to_string(), "success".to_string())]);
        let baseline = canonical_series_id(
            "telemetry.runtime",
            "queue_records",
            &resource,
            None,
            None,
            &attributes,
        );
        assert_eq!(
            baseline,
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &resource,
                None,
                None,
                &attributes,
            )
        );

        let mut changed = resource.clone();
        changed.cluster = Some("cluster-b".to_string());
        assert_ne!(
            baseline,
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &changed,
                None,
                None,
                &attributes,
            )
        );
        changed = resource.clone();
        changed.entity_uid = Some("entity-2".to_string());
        assert_ne!(
            baseline,
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &changed,
                None,
                None,
                &attributes,
            )
        );
        changed = resource.clone();
        changed.attempt_uid = Some("attempt-2".to_string());
        assert_ne!(
            baseline,
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &changed,
                None,
                None,
                &attributes,
            )
        );
        assert_ne!(
            baseline,
            canonical_series_id(
                "telemetry.runtime",
                "queue_records",
                &resource,
                Some("accelerator-1"),
                Some("tpu"),
                &attributes,
            )
        );
    }

    #[tokio::test]
    async fn custom_metric_rejects_a_producer_controlled_series_id() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let response = send_custom(
            Arc::clone(&store),
            AuthIdentity::Network,
            &metric_batch("spoofed-series", true),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            namespace_rows(
                &store,
                crate::store::telemetry_catalog::SERVICE_METRIC_NAMESPACE
            ),
            0
        );
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
    }

    #[test]
    fn custom_and_otlp_metrics_derive_the_same_series_id() {
        let resource = metric_resource();
        let attributes = HashMap::new();
        let expected = canonical_series_id(
            "telemetry.runtime",
            "queue_records",
            &resource,
            None,
            None,
            &attributes,
        );
        let request = otlp_metric_request(vec![NumberDataPoint {
            time_unix_nano: 1_000_000_000,
            value: Some(number_data_point::Value::AsInt(2)),
            ..Default::default()
        }]);
        let (records, rejected, _) =
            normalize_otlp_metrics(&request, "series-parity", &AuthIdentity::Network);
        assert_eq!(rejected, 0);
        assert_eq!(
            string_cell(records[0].row.get("series_id")),
            Some(expected.as_str())
        );
        assert_eq!(
            metric_batch("series-parity", false).records[0]
                .metric
                .as_option()
                .unwrap()
                .series_id
                .as_deref(),
            Some(expected.as_str())
        );
    }

    #[test]
    fn otlp_shared_resource_amplification_and_individual_strings_are_bounded() {
        let mut request = otlp_metric_request(
            (0..MAX_REST_RECORDS)
                .map(|offset| NumberDataPoint {
                    time_unix_nano: 1_000_000_000 + offset as u64,
                    value: Some(number_data_point::Value::AsInt(1)),
                    ..Default::default()
                })
                .collect(),
        );
        let resource = request.resource_metrics[0].resource.as_mut().unwrap();
        resource.attributes[0].value = Some(AnyValue {
            value: Some(any_value::Value::StringValue("s".repeat(4 * 1024))),
        });
        let error = check_otlp_metric_admission(&request).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error.code, "normalized_body_too_large");

        let logs = ExportLogsServiceRequest {
            resource_logs: vec![ResourceLogs {
                scope_logs: vec![ScopeLogs {
                    log_records: vec![LogRecord {
                        event_name: "e".repeat(MAX_STRING_BYTES + 1),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
        };
        let error = check_otlp_log_admission(&logs).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error.code, "string_limit_exceeded");
    }

    #[test]
    fn integer_metric_values_must_round_trip_exactly_through_f64() {
        let request = otlp_metric_request(vec![NumberDataPoint {
            time_unix_nano: 1_000_000_000,
            value: Some(number_data_point::Value::AsInt(
                MAX_EXACT_F64_INTEGER as i64 + 1,
            )),
            ..Default::default()
        }]);
        let (records, rejected, reasons) =
            normalize_otlp_metrics(&request, "large-int", &AuthIdentity::Network);
        assert!(records.is_empty());
        assert_eq!(rejected, 1);
        assert!(reasons[0].contains("represented exactly"));
    }

    #[test]
    fn invalid_zero_point_metric_does_not_invent_a_rejected_point() {
        let mut request = otlp_metric_request(Vec::new());
        request.resource_metrics[0].scope_metrics[0].metrics[0].data = None;
        let (records, rejected, _) =
            normalize_otlp_metrics(&request, "zero-point", &AuthIdentity::Network);
        assert!(records.is_empty());
        assert_eq!(rejected, 0);
    }

    #[test]
    fn namespace_sub_batch_ids_are_stable_and_distinct() {
        assert_eq!(
            sub_batch_id("batch-a", "namespace-a", "buffered"),
            sub_batch_id("batch-a", "namespace-a", "buffered")
        );
        assert_ne!(
            sub_batch_id("batch-a", "namespace-a", "buffered"),
            sub_batch_id("batch-b", "namespace-a", "buffered")
        );
        assert_ne!(
            sub_batch_id("batch-a", "namespace-a", "buffered"),
            sub_batch_id("batch-a", "namespace-b", "buffered")
        );
        assert_ne!(
            sub_batch_id("batch-a", "namespace-a", "buffered"),
            sub_batch_id("batch-a", "namespace-a", "coalescing")
        );
    }

    #[test]
    fn histogram_bucket_count_overflow_is_rejected_without_panicking() {
        let mut row = Row::new();
        let metric = TelemetryMetricV1 {
            scope: Some("telemetry.runtime".to_string()),
            name: Some("export_duration".to_string()),
            description: Some("Telemetry export request duration".to_string()),
            unit: Some("s".to_string()),
            instrument_kind: Some("histogram".to_string()),
            temporality: Some("cumulative".to_string()),
            start_ts_unix_nano: Some(1),
            reset_id: Some("reset".to_string()),
            series_id: Some("series".to_string()),
            count: Some(i64::MAX),
            sum: Some(1.0),
            explicit_bounds: vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            bucket_counts: vec![i64::MAX, 1, 0, 0, 0, 0, 0, 0, 0],
            attributes: HashMap::from([("outcome".to_string(), "success".to_string())]),
            ..Default::default()
        };
        let error = validate_metric(&metric, "buffered", 0, &mut row).unwrap_err();
        assert_eq!(error.field.as_deref(), Some("metric"));
    }

    #[tokio::test]
    async fn validation_and_namespace_size_failures_leave_no_durable_rows() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let response = send_custom(
            Arc::clone(&store),
            AuthIdentity::Network,
            &event_batch("invalid-event", None),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);
        assert_eq!(namespace_rows(&store, EVENT_NAMESPACE), 0);

        let response = send_custom(
            Arc::clone(&store),
            AuthIdentity::Network,
            &log_batch("oversized-log", &"x".repeat(MAX_WRITE_ROWS_BYTES)),
        )
        .await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);
        assert_eq!(namespace_rows(&store, LOG_NAMESPACE_NAME), 0);
    }

    #[tokio::test]
    async fn partial_child_failure_keeps_completion_absent_and_global_intent_fences_payload() {
        let directory = unique_dir("telemetry_partial_child");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let state = test_state(Arc::clone(&store));
        let app = router_with_state(Arc::clone(&state));
        let original = event_batch("partial-batch", Some("ERROR"));
        state.inject_failure_after_child_append(EVENT_NAMESPACE);
        let response = send_custom_to(app.clone(), AuthIdentity::Network, &original).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 1);
        assert_eq!(namespace_rows(&store, EVENT_NAMESPACE), 1);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);

        let changed = log_batch("partial-batch", "changed route");
        let response = send_custom_to(app.clone(), AuthIdentity::Network, &changed).await;
        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(namespace_rows(&store, LOG_NAMESPACE_NAME), 0);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);

        let response = send_custom_to(app.clone(), AuthIdentity::Network, &original).await;
        assert_eq!(response.status(), StatusCode::CREATED);
        let accepted_body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let accepted: TelemetryWriteAckV1 = serde_json::from_slice(&accepted_body).unwrap();
        assert_eq!(namespace_rows(&store, EVENT_NAMESPACE), 1);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 1);

        let response = send_custom_to(app, AuthIdentity::Network, &original).await;
        assert_eq!(response.status(), StatusCode::OK);
        let duplicate_body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let duplicate: TelemetryWriteAckV1 = serde_json::from_slice(&duplicate_body).unwrap();
        assert_eq!(duplicate.status.as_deref(), Some("duplicate"));
        assert_eq!(duplicate.durability.as_deref(), Some("finelog_local"));
        assert_eq!(duplicate.commits, accepted.commits);
    }

    #[tokio::test]
    async fn concurrent_same_payload_requests_complete_once() {
        let directory = unique_dir("telemetry_concurrent_request");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let batch = event_batch("concurrent-batch", Some("ERROR"));
        let first = send_custom(Arc::clone(&store), AuthIdentity::Network, &batch);
        let second = send_custom(Arc::clone(&store), AuthIdentity::Network, &batch);
        let (first, second) = tokio::join!(first, second);
        let mut statuses = [first.status(), second.status()];
        statuses.sort();
        assert_eq!(statuses, [StatusCode::OK, StatusCode::CREATED]);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 1);
        assert_eq!(namespace_rows(&store, EVENT_NAMESPACE), 1);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 1);
    }

    #[tokio::test]
    async fn jwt_identity_is_stamped_and_log_severity_text_round_trips() {
        let directory = unique_dir("telemetry_rest_identity");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let response = send_custom(
            Arc::clone(&store),
            AuthIdentity::Jwt {
                cluster: "trusted-cluster".to_string(),
            },
            &log_batch("identity-log", "hello"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::CREATED);

        let batches = query_namespace(
            &store,
            LOG_NAMESPACE_NAME,
            r#"SELECT cluster, iris_job_id, node_id, pod_uid, container_id,
                       entity_authority, entity_uid, severity_text
                FROM "log""#,
        )
        .await;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        let batch = &batches[0];
        let string = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        };
        assert_eq!(string("cluster").value(0), "trusted-cluster");
        for name in [
            "iris_job_id",
            "node_id",
            "pod_uid",
            "container_id",
            "entity_authority",
            "entity_uid",
        ] {
            assert!(string(name).is_null(0), "{name} was not cleared");
        }
        assert_eq!(string("severity_text").value(0), "INFO");
    }

    #[tokio::test]
    async fn policy_composition_preserves_collector_identity_and_clears_network_identity() {
        let directory = unique_dir("telemetry_rest_collector_identity");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let policy = AuthPolicy::parse(
            &serde_json::json!([
                {"type": "cidr", "cidrs": ["10.0.0.0/8"]},
                {"type": "jwt", "keys": [{
                    "cluster": "collector-cluster",
                    "role": "trusted_collector",
                    "public_keys": [PUB_A]
                }]}
            ])
            .to_string(),
        )
        .unwrap();
        let claims = serde_json::json!({
            "iss": "collector",
            "aud": FINELOG_AUDIENCE,
            "sub": "agent",
            "exp": now_unix_nano() / 1_000_000_000 + 300,
        });
        let token = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(Algorithm::EdDSA),
            &claims,
            &EncodingKey::from_ed_pem(PRIV_A.as_bytes()).unwrap(),
        )
        .unwrap();
        let peer = "10.1.2.3".parse().unwrap();
        let collector_identity = policy.admits(Some(&token), Some(peer)).unwrap();
        let network_identity = policy.admits(None, Some(peer)).unwrap();

        for (batch_id, identity) in [
            ("collector-log", collector_identity),
            ("network-log", network_identity),
        ] {
            let mut batch = log_batch(batch_id, "hello");
            batch.records[0].resource.modify(|resource| {
                resource.iris_job_id = Some("signed-job".to_string());
                resource.iris_task_id = Some("signed-task".to_string());
                resource.worker_id = Some("signed-worker".to_string());
                resource.node_id = Some("signed-node".to_string());
                resource.pod_uid = None;
                resource.container_id = None;
                resource.entity_authority = None;
                resource.entity_type = None;
                resource.entity_uid = None;
            });
            let response = send_custom(Arc::clone(&store), identity, &batch).await;
            assert_eq!(response.status(), StatusCode::CREATED);
        }

        let batches = query_namespace(
            &store,
            LOG_NAMESPACE_NAME,
            r#"SELECT batch_id, "cluster", iris_job_id, iris_task_id, worker_id, node_id,
                       pod_uid, container_id, entity_authority, entity_type, entity_uid
                FROM "log" ORDER BY batch_id"#,
        )
        .await;
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
        let string = |name: &str| {
            batch
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        };
        assert_eq!(string("batch_id").value(0), "collector-log");
        assert_eq!(string("cluster").value(0), "collector-cluster");
        for (name, expected) in [
            ("iris_job_id", "signed-job"),
            ("iris_task_id", "signed-task"),
            ("worker_id", "signed-worker"),
            ("node_id", "signed-node"),
        ] {
            let column = string(name);
            assert_eq!(column.value(0), expected);
            assert!(column.is_null(1), "{name} network value was not cleared");
        }
        for name in [
            "pod_uid",
            "container_id",
            "entity_authority",
            "entity_type",
            "entity_uid",
        ] {
            assert!(string(name).is_null(0));
            assert!(string(name).is_null(1));
        }
        assert_eq!(string("batch_id").value(1), "network-log");
        assert!(string("cluster").is_null(1));
    }

    #[tokio::test]
    async fn compact_requests_over_the_rest_record_cap_are_rejected() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let batch = TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some("too-many".to_string()),
            records: vec![TelemetryRecordV1::default(); MAX_REST_RECORDS + 1],
            ..Default::default()
        };
        let request = Request::builder()
            .method("POST")
            .uri(CUSTOM_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .header("idempotency-key", "too-many")
            .body(Body::from(batch.encode_to_vec()))
            .unwrap();
        let response = router(Arc::clone(&store))
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
    }

    #[tokio::test]
    async fn detailed_validation_errors_are_capped() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let batch = TelemetryBatchV1 {
            schema_version: Some(SCHEMA_VERSION),
            catalog_version: Some(CATALOG_VERSION.to_string()),
            batch_id: Some("invalid-many".to_string()),
            records: vec![TelemetryRecordV1::default(); MAX_VALIDATION_ERRORS * 2],
            ..Default::default()
        };
        let response = send_custom(store, AuthIdentity::Network, &batch).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let ack: TelemetryWriteAckV1 = serde_json::from_slice(&body).unwrap();
        assert_eq!(ack.rejected_records.len(), MAX_VALIDATION_ERRORS);
    }

    #[test]
    fn unknown_groups_match_protobuf_decoder_wire_compatibility() {
        let mut group_contents = Vec::new();
        push_test_key(&mut group_contents, 102, 0);
        push_test_varint(&mut group_contents, 7);
        let nested_group = protobuf_group(101, &group_contents);
        let outer_group = protobuf_group(100, &nested_group);

        let mut custom_body = log_batch("unknown-group", "ready").encode_to_vec();
        custom_body.extend_from_slice(&outer_group);
        preflight_structure(
            Endpoint::Custom,
            WireFormat::Protobuf,
            custom_body.as_slice(),
        )
        .unwrap();
        TelemetryBatchV1::decode_from_slice(custom_body.as_slice()).unwrap();

        preflight_structure(
            Endpoint::OtlpMetrics,
            WireFormat::Protobuf,
            outer_group.as_slice(),
        )
        .unwrap();
        ExportMetricsServiceRequest::decode(outer_group.as_slice()).unwrap();
    }

    #[test]
    fn unbalanced_and_over_quota_unknown_groups_are_rejected() {
        let mut unexpected_end = Vec::new();
        push_test_key(&mut unexpected_end, 100, 4);
        let mut missing_end = Vec::new();
        push_test_key(&mut missing_end, 100, 3);
        let mut mismatched_end = missing_end.clone();
        push_test_key(&mut mismatched_end, 101, 4);

        for body in [unexpected_end, missing_end, mismatched_end] {
            let error =
                preflight_structure(Endpoint::Custom, WireFormat::Protobuf, &body).unwrap_err();
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.code, "invalid_protobuf");
        }

        let mut too_deep = Vec::new();
        for field in 100..100 + u32::try_from(MAX_STRUCTURAL_DEPTH + 1).unwrap() {
            too_deep = protobuf_group(field, &too_deep);
        }
        let error =
            preflight_structure(Endpoint::Custom, WireFormat::Protobuf, &too_deep).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);

        let mut too_many_group_fields = Vec::new();
        for _ in 0..=MAX_REST_RECORDS {
            push_test_key(&mut too_many_group_fields, 101, 0);
            push_test_varint(&mut too_many_group_fields, 0);
        }
        let body = protobuf_group(100, &too_many_group_fields);
        let error = preflight_structure(Endpoint::Custom, WireFormat::Protobuf, &body).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn custom_unknown_wire_fields_hit_global_quota_before_decode() {
        let mut body = Vec::with_capacity((MAX_STRUCTURAL_ITEMS + 1) * 2);
        for _ in 0..=MAX_STRUCTURAL_ITEMS {
            push_test_key(&mut body, 15, 0);
            push_test_varint(&mut body, 0);
        }

        let error = preflight_structure(Endpoint::Custom, WireFormat::Protobuf, &body).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(error.code, "structural_limit_exceeded");
    }

    #[tokio::test]
    async fn repeated_empty_protobuf_messages_hit_structural_quota_before_decode() {
        for (endpoint, repeated_tag) in [
            (OTLP_METRICS_ENDPOINT, 0x0a),
            (OTLP_LOGS_ENDPOINT, 0x0a),
            (CUSTOM_ENDPOINT, 0x22),
        ] {
            let store = Arc::new(Store::new(None, String::new()).unwrap());
            let mut body = Vec::with_capacity((MAX_REST_RECORDS + 1) * 2);
            for _ in 0..=MAX_REST_RECORDS {
                body.extend_from_slice(&[repeated_tag, 0x00]);
            }
            let mut request = Request::builder()
                .method("POST")
                .uri(endpoint)
                .header(CONTENT_TYPE, "application/x-protobuf");
            if endpoint == CUSTOM_ENDPOINT {
                request = request.header("idempotency-key", "structural-protobuf");
            }
            let response = router(Arc::clone(&store))
                .layer(Extension(AuthIdentity::Network))
                .oneshot(request.body(Body::from(body)).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
            assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
            assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);
        }
    }

    #[tokio::test]
    async fn nested_empty_exemplar_attributes_hit_structural_quota_before_decode() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let exemplar = [0x3a, 0x00].repeat(MAX_REST_RECORDS + 1);
        let number_point = protobuf_message_field(5, &exemplar);
        let gauge = protobuf_message_field(1, &number_point);
        let metric = protobuf_message_field(5, &gauge);
        let scope_metrics = protobuf_message_field(2, &metric);
        let resource_metrics = protobuf_message_field(2, &scope_metrics);
        let request_body = protobuf_message_field(1, &resource_metrics);
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(request_body))
            .unwrap();
        let response = router(Arc::clone(&store))
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);
        assert_eq!(
            namespace_rows(
                &store,
                crate::store::telemetry_catalog::SERVICE_METRIC_NAMESPACE
            ),
            0
        );
    }

    #[tokio::test]
    async fn resource_entity_refs_hit_structural_quotas_before_decode() {
        let repeated_entity_refs = [0x1a, 0x00].repeat(MAX_REST_RECORDS + 1);
        let repeated_id_keys = [0x1a, 0x00].repeat(MAX_REST_RECORDS + 1);
        let repeated_description_keys = [0x22, 0x00].repeat(MAX_REST_RECORDS + 1);
        let oversized_string = vec![b'x'; MAX_STRING_BYTES + 1];
        let cases = [
            ("entity_refs", repeated_entity_refs),
            (
                "entity_ref.id_keys",
                protobuf_message_field(3, &repeated_id_keys),
            ),
            (
                "entity_ref.description_keys",
                protobuf_message_field(3, &repeated_description_keys),
            ),
            (
                "entity_ref.schema_url",
                protobuf_message_field(3, &protobuf_message_field(1, &oversized_string)),
            ),
            (
                "entity_ref.type",
                protobuf_message_field(3, &protobuf_message_field(2, &oversized_string)),
            ),
        ];

        for (case, resource) in cases {
            let store = Arc::new(Store::new(None, String::new()).unwrap());
            let resource_metrics = protobuf_message_field(1, &resource);
            let request_body = protobuf_message_field(1, &resource_metrics);
            let request = Request::builder()
                .method("POST")
                .uri(OTLP_METRICS_ENDPOINT)
                .header(CONTENT_TYPE, "application/x-protobuf")
                .body(Body::from(request_body))
                .unwrap();
            let response = router(Arc::clone(&store))
                .layer(Extension(AuthIdentity::Network))
                .oneshot(request)
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE, "{case}");
            assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0, "{case}");
            assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0, "{case}");
        }
    }

    #[tokio::test]
    async fn repeated_empty_json_objects_hit_structural_quota_before_decode() {
        let objects = (0..=MAX_REST_RECORDS)
            .map(|_| "{}")
            .collect::<Vec<_>>()
            .join(",");
        for (endpoint, field) in [
            (OTLP_METRICS_ENDPOINT, "resourceMetrics"),
            (OTLP_LOGS_ENDPOINT, "resourceLogs"),
            (CUSTOM_ENDPOINT, "records"),
        ] {
            let store = Arc::new(Store::new(None, String::new()).unwrap());
            let body = format!(r#"{{"{field}":[{objects}]}}"#);
            let mut request = Request::builder()
                .method("POST")
                .uri(endpoint)
                .header(CONTENT_TYPE, "application/json");
            if endpoint == CUSTOM_ENDPOINT {
                request = request.header("idempotency-key", "structural-json");
            }
            let response = router(Arc::clone(&store))
                .layer(Extension(AuthIdentity::Network))
                .oneshot(request.body(Body::from(body)).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
            assert_eq!(namespace_rows(&store, BATCH_INTENT_NAMESPACE), 0);
            assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 0);
        }
    }

    #[tokio::test]
    async fn high_ratio_compressed_body_stops_at_the_uncompressed_limit() {
        let body = vec![b'x'; MAX_MESSAGE_BYTES + 1];
        let compressed = zstd::stream::encode_all(body.as_slice(), 1).unwrap();
        let error = decode_body_sync("zstd", Bytes::from(compressed.clone())).unwrap_err();
        assert_eq!(error.status, StatusCode::PAYLOAD_TOO_LARGE);

        let request = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .header(CONTENT_ENCODING, "zstd")
            .body(Body::from(compressed))
            .unwrap();
        let response = router(Arc::new(Store::new(None, String::new()).unwrap()))
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .is_none());
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status = GoogleRpcStatus::decode(body).unwrap();
        assert_eq!(status.code, 8);
        assert!(status.message.contains("body_too_large"));
    }

    #[test]
    fn zstd_frame_window_is_bounded_before_decompression() {
        let body = vec![b'x'; (1 << ZSTD_WINDOW_LOG_MAX) + 1];
        let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 1).unwrap();
        encoder.window_log(ZSTD_WINDOW_LOG_MAX + 1).unwrap();
        encoder
            .set_pledged_src_size(Some(body.len() as u64))
            .unwrap();
        encoder.write_all(&body).unwrap();
        let compressed = encoder.finish().unwrap();
        let error = decode_body_sync("zstd", Bytes::from(compressed)).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "invalid_compression");
        assert!(
            error.message.contains("window") || error.message.contains("memory"),
            "{}",
            error.message
        );
    }

    #[tokio::test]
    async fn overloaded_request_is_rejected_without_polling_its_body() {
        let state = test_state(Arc::new(Store::new(None, String::new()).unwrap()));
        let held = Arc::clone(&state.requests)
            .acquire_many_owned(MAX_CONCURRENT_REQUESTS as u32)
            .await
            .unwrap();
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::new(PendingBody {
                polls: Arc::clone(&polls),
                dropped: Arc::clone(&dropped),
            }))
            .unwrap();
        let response = router_with_state(state)
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&HeaderValue::from_static(RETRY_AFTER_SECONDS))
        );
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status = GoogleRpcStatus::decode(body).unwrap();
        assert_eq!(status.code, 8);
        drop(held);
    }

    #[tokio::test]
    async fn stalled_body_times_out_and_releases_admission_permit() {
        let timeout = Duration::from_millis(25);
        let state =
            test_state_with_timeout(Arc::new(Store::new(None, String::new()).unwrap()), timeout);
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_LOGS_ENDPOINT)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::new(PendingBody {
                polls: Arc::clone(&polls),
                dropped: Arc::clone(&dropped),
            }))
            .unwrap();
        let app = router_with_state(Arc::clone(&state)).layer(Extension(AuthIdentity::Network));
        let task = tokio::spawn(async move { app.oneshot(request).await.unwrap() });
        tokio::task::yield_now().await;
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        let response = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("stalled body exceeded the outer test budget")
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(state.requests.available_permits(), MAX_CONCURRENT_REQUESTS);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&HeaderValue::from_static(RETRY_AFTER_SECONDS))
        );
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["code"], 4);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blocking_work_does_not_occupy_the_async_executor() {
        let state = test_state(Arc::new(Store::new(None, String::new()).unwrap()));
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let task_barrier = Arc::clone(&barrier);
        let (progress_tx, progress_rx) = std::sync::mpsc::channel();
        let observer = std::thread::spawn(move || {
            let progressed = progress_rx
                .recv_timeout(std::time::Duration::from_secs(1))
                .is_ok();
            barrier.wait();
            progressed
        });
        let blocking = run_blocking(&state, Instant::now() + REQUEST_TIMEOUT, move || {
            task_barrier.wait();
            Ok(())
        });
        let progress = async {
            tokio::task::yield_now().await;
            progress_tx.send(()).unwrap();
        };
        let (blocking, ()) = tokio::join!(blocking, progress);
        blocking.unwrap();
        assert!(observer.join().unwrap());
    }

    #[tokio::test]
    async fn otlp_metrics_keep_valid_points_and_report_protocol_partial_success() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let resource = OtlpResource {
            attributes: vec![
                KeyValue {
                    key: "service.name".to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("test-service".to_string())),
                    }),
                    ..Default::default()
                },
                KeyValue {
                    key: "service.instance.id".to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("instance-1".to_string())),
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let request_message = ExportMetricsServiceRequest {
            resource_metrics: vec![ResourceMetrics {
                resource: Some(resource),
                scope_metrics: vec![ScopeMetrics {
                    scope: Some(InstrumentationScope {
                        name: "telemetry.runtime".to_string(),
                        version: "1".to_string(),
                        ..Default::default()
                    }),
                    metrics: vec![Metric {
                        name: "queue_records".to_string(),
                        description: "Telemetry records waiting for background export".to_string(),
                        unit: "{record}".to_string(),
                        data: Some(metric::Data::Gauge(Gauge {
                            data_points: vec![
                                NumberDataPoint {
                                    time_unix_nano: 1_000_000_000,
                                    value: Some(number_data_point::Value::AsInt(2)),
                                    ..Default::default()
                                },
                                NumberDataPoint {
                                    time_unix_nano: 1_000_000_001,
                                    value: None,
                                    ..Default::default()
                                },
                            ],
                        })),
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
        };
        let body = request_message.encode_to_vec();
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(body.clone()))
            .unwrap();
        let app = router(Arc::clone(&store)).layer(Extension(AuthIdentity::Network));
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE).unwrap(),
            "application/x-protobuf"
        );
        let response_body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let response_message =
            <ExportMetricsServiceResponse as ProstMessage>::decode(response_body).unwrap();
        let partial = response_message.partial_success.unwrap();
        assert_eq!(partial.rejected_data_points, 1);
        assert!(partial.error_message.contains("finite numeric value"));
        assert_eq!(
            namespace_rows(
                &store,
                crate::store::telemetry_catalog::SERVICE_METRIC_NAMESPACE
            ),
            1
        );
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 1);

        let retry = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(body))
            .unwrap();
        assert_eq!(app.oneshot(retry).await.unwrap().status(), StatusCode::OK);
        assert_eq!(
            namespace_rows(
                &store,
                crate::store::telemetry_catalog::SERVICE_METRIC_NAMESPACE
            ),
            1
        );
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 1);
    }

    #[tokio::test]
    async fn otlp_logs_keep_valid_records_and_report_protocol_partial_success() {
        let directory = unique_dir("telemetry_otlp_log_roundtrip");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let resource = OtlpResource {
            attributes: vec![
                KeyValue {
                    key: "service.name".to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("test-service".to_string())),
                    }),
                    ..Default::default()
                },
                KeyValue {
                    key: "service.instance.id".to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue("instance-1".to_string())),
                    }),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let request_message = ExportLogsServiceRequest {
            resource_logs: vec![ResourceLogs {
                resource: Some(resource),
                scope_logs: vec![ScopeLogs {
                    scope: Some(InstrumentationScope {
                        name: "test.logger".to_string(),
                        ..Default::default()
                    }),
                    log_records: vec![
                        LogRecord {
                            time_unix_nano: 1_000_000_000,
                            observed_time_unix_nano: 1_000_000_001,
                            severity_number: 9,
                            severity_text: "INFO".to_string(),
                            event_name: "test.ready".to_string(),
                            attributes: vec![KeyValue {
                                key: "event.name".to_string(),
                                value: Some(AnyValue {
                                    value: Some(any_value::Value::StringValue(
                                        "legacy.fallback".to_string(),
                                    )),
                                }),
                                ..Default::default()
                            }],
                            body: Some(AnyValue {
                                value: Some(any_value::Value::StringValue("hello".to_string())),
                            }),
                            ..Default::default()
                        },
                        LogRecord {
                            time_unix_nano: 1_000_000_002,
                            severity_number: 9,
                            severity_text: "INFO".to_string(),
                            body: Some(AnyValue {
                                value: Some(any_value::Value::ArrayValue(Default::default())),
                            }),
                            ..Default::default()
                        },
                    ],
                    ..Default::default()
                }],
                ..Default::default()
            }],
        };
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_LOGS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(request_message.encode_to_vec()))
            .unwrap();
        let response = router(Arc::clone(&store))
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response_body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let response_message =
            <ExportLogsServiceResponse as ProstMessage>::decode(response_body).unwrap();
        assert_eq!(
            response_message
                .partial_success
                .unwrap()
                .rejected_log_records,
            1
        );
        assert_eq!(namespace_rows(&store, LOG_NAMESPACE_NAME), 1);
        assert_eq!(namespace_rows(&store, BATCH_NAMESPACE), 1);
        let batches = query_namespace(
            &store,
            LOG_NAMESPACE_NAME,
            r#"SELECT event_name FROM "log""#,
        )
        .await;
        let event_names = batches[0]
            .column_by_name("event_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(event_names.value(0), "test.ready");
    }

    #[tokio::test]
    async fn otlp_malformed_requests_use_google_status_in_request_format() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let protobuf = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(vec![0xff]))
            .unwrap();
        let response = router(Arc::clone(&store))
            .layer(Extension(AuthIdentity::Network))
            .oneshot(protobuf)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers().get(CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/x-protobuf"))
        );
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status = GoogleRpcStatus::decode(body).unwrap();
        assert_eq!(status.code, 3);
        assert!(status.message.contains("invalid_protobuf"));

        let json = Request::builder()
            .method("POST")
            .uri(OTLP_LOGS_ENDPOINT)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from("{"))
            .unwrap();
        let response = router(store)
            .layer(Extension(AuthIdentity::Network))
            .oneshot(json)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["code"], 3);
        assert!(status["message"].as_str().unwrap().contains("invalid_json"));
        assert_eq!(status["details"], serde_json::json!([]));
    }

    #[tokio::test]
    async fn otlp_auth_failure_uses_google_status_without_polling_body() {
        let state = test_state(Arc::new(Store::new(None, String::new()).unwrap()));
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_LOGS_ENDPOINT)
            .header(CONTENT_TYPE, "application/json")
            .body(Body::new(PendingBody {
                polls: Arc::clone(&polls),
                dropped: Arc::clone(&dropped),
            }))
            .unwrap();
        let policy = Arc::new(crate::server::auth::AuthPolicy::allow_localhost());
        let response = router_with_state(state)
            .layer(axum::middleware::from_fn_with_state(
                policy,
                crate::server::auth::auth_gate,
            ))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert!(dropped.load(Ordering::SeqCst));
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(status["code"], 16);
        assert!(status["message"]
            .as_str()
            .unwrap()
            .contains("unauthenticated"));
    }

    #[tokio::test]
    async fn transient_otlp_store_failure_is_retryable_google_status() {
        let directory = unique_dir("telemetry_otlp_store_failure");
        let store = Arc::new(Store::new(Some(directory), String::new()).unwrap());
        store.bootstrap_maintenance();
        let state = test_state(store);
        state.inject_failure_after_child_append(
            crate::store::telemetry_catalog::SERVICE_METRIC_NAMESPACE,
        );
        let message = otlp_metric_request(vec![NumberDataPoint {
            time_unix_nano: 1_000_000_000,
            value: Some(number_data_point::Value::AsInt(2)),
            ..Default::default()
        }]);
        let request = Request::builder()
            .method("POST")
            .uri(OTLP_METRICS_ENDPOINT)
            .header(CONTENT_TYPE, "application/x-protobuf")
            .body(Body::from(message.encode_to_vec()))
            .unwrap();
        let response = router_with_state(state)
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&HeaderValue::from_static(RETRY_AFTER_SECONDS))
        );
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let status = GoogleRpcStatus::decode(body).unwrap();
        assert_eq!(status.code, 14);
        assert!(status.message.contains("store_unavailable"));
    }

    #[test]
    fn nonretryable_otlp_internal_error_has_no_retry_after() {
        let response = ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            "invariant failed",
        )
        .into_endpoint_response(Endpoint::OtlpMetrics, WireFormat::Protobuf);
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(response
            .headers()
            .get(axum::http::header::RETRY_AFTER)
            .is_none());
    }

    #[tokio::test]
    async fn idempotency_header_is_checked_before_compressed_body_decode() {
        let store = Arc::new(Store::new(None, String::new()).unwrap());
        let request = Request::builder()
            .method("POST")
            .uri(CUSTOM_ENDPOINT)
            .header(CONTENT_TYPE, "application/json")
            .header(CONTENT_ENCODING, "gzip")
            .body(Body::from("not gzip"))
            .unwrap();
        let response = router(store)
            .layer(Extension(AuthIdentity::Network))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), MAX_MESSAGE_BYTES)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["code"], "missing_idempotency_key");
    }

    #[test]
    fn overload_errors_include_retry_after() {
        let response =
            ApiError::from_store(StatsError::WriteBufferFull("full".to_string())).into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            response.headers().get(axum::http::header::RETRY_AFTER),
            Some(&HeaderValue::from_static(RETRY_AFTER_SECONDS))
        );
    }
}
