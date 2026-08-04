//! Authenticated CRUD API for shared SQL dashboard definitions.

use std::collections::HashSet;
use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Path, Request, State};
use axum::http::{header::CONTENT_TYPE, StatusCode};
use axum::middleware::from_fn_with_state;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::server::auth::{auth_gate, AuthPolicy};
use crate::store::catalog::{DashboardDeleteOutcome, SavedDashboard};
use crate::store::Store;

const MAX_BODY_BYTES: usize = 256 << 10;
const MAX_DASHBOARD_ID_BYTES: usize = 64;
const MAX_TITLE_BYTES: usize = 200;
const MAX_DESCRIPTION_BYTES: usize = 2_000;
const MAX_VARIABLES: usize = 16;
const MAX_VARIABLE_VALUE_BYTES: usize = 4_096;
const MAX_PANELS: usize = 32;
const MAX_SQL_BYTES: usize = 128 << 10;
const DASHBOARD_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct DashboardDefinition {
    version: u32,
    id: String,
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    variables: Vec<DashboardVariable>,
    #[serde(default)]
    panels: Vec<DashboardPanel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct DashboardVariable {
    name: String,
    label: String,
    default: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct DashboardPanel {
    id: String,
    title: String,
    kind: PanelKind,
    #[serde(default)]
    description: String,
    sql: String,
    width: PanelWidth,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum PanelKind {
    Table,
    Stat,
    Timeseries,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum PanelWidth {
    Half,
    Full,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DashboardResponse {
    definition: DashboardDefinition,
    created_at_ms: i64,
    updated_at_ms: i64,
}

#[derive(Debug, Serialize)]
struct DashboardListResponse {
    dashboards: Vec<DashboardResponse>,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_dashboard",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.into(),
        }
    }

    fn not_found(dashboard_id: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "dashboard_not_found",
            message: format!("dashboard {dashboard_id:?} was not found"),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": {"code": self.code, "message": self.message}
            })),
        )
            .into_response()
    }
}

fn valid_id(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_DASHBOARD_ID_BYTES
        && bytes[0].is_ascii_lowercase()
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
}

fn valid_variable_name(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && value.len() <= MAX_DASHBOARD_ID_BYTES
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn require_nonempty(value: &str, field: &str, max_bytes: usize) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Err(ApiError::bad_request(format!("{field} must not be empty")));
    }
    if value.len() > max_bytes {
        return Err(ApiError::bad_request(format!(
            "{field} exceeds {max_bytes} bytes"
        )));
    }
    Ok(())
}

fn validate_definition(path_id: &str, definition: &DashboardDefinition) -> Result<(), ApiError> {
    if definition.version != DASHBOARD_VERSION {
        return Err(ApiError::bad_request(format!(
            "version must be {DASHBOARD_VERSION}"
        )));
    }
    if !valid_id(path_id) || !valid_id(&definition.id) {
        return Err(ApiError::bad_request(
            "dashboard id must start with a lowercase letter and contain only lowercase letters, digits, or hyphens",
        ));
    }
    if path_id != definition.id {
        return Err(ApiError::bad_request(format!(
            "path id {path_id:?} does not match definition id {:?}",
            definition.id
        )));
    }
    require_nonempty(&definition.title, "title", MAX_TITLE_BYTES)?;
    if definition.description.len() > MAX_DESCRIPTION_BYTES {
        return Err(ApiError::bad_request(format!(
            "description exceeds {MAX_DESCRIPTION_BYTES} bytes"
        )));
    }
    if definition.variables.len() > MAX_VARIABLES {
        return Err(ApiError::bad_request(format!(
            "dashboard has more than {MAX_VARIABLES} variables"
        )));
    }
    if definition.panels.len() > MAX_PANELS {
        return Err(ApiError::bad_request(format!(
            "dashboard has more than {MAX_PANELS} panels"
        )));
    }

    let mut variable_names = HashSet::new();
    for variable in &definition.variables {
        if !valid_variable_name(&variable.name) {
            return Err(ApiError::bad_request(format!(
                "variable name {:?} must be a SQL-style identifier",
                variable.name
            )));
        }
        if !variable_names.insert(&variable.name) {
            return Err(ApiError::bad_request(format!(
                "duplicate variable name {:?}",
                variable.name
            )));
        }
        require_nonempty(&variable.label, "variable label", MAX_TITLE_BYTES)?;
        if variable.default.len() > MAX_VARIABLE_VALUE_BYTES {
            return Err(ApiError::bad_request(format!(
                "variable {:?} default exceeds {MAX_VARIABLE_VALUE_BYTES} bytes",
                variable.name
            )));
        }
    }

    let mut panel_ids = HashSet::new();
    for panel in &definition.panels {
        if !valid_id(&panel.id) {
            return Err(ApiError::bad_request(format!(
                "panel id {:?} is invalid",
                panel.id
            )));
        }
        if !panel_ids.insert(&panel.id) {
            return Err(ApiError::bad_request(format!(
                "duplicate panel id {:?}",
                panel.id
            )));
        }
        require_nonempty(&panel.title, "panel title", MAX_TITLE_BYTES)?;
        if panel.description.len() > MAX_DESCRIPTION_BYTES {
            return Err(ApiError::bad_request(format!(
                "panel {:?} description exceeds {MAX_DESCRIPTION_BYTES} bytes",
                panel.id
            )));
        }
        require_nonempty(&panel.sql, "panel SQL", MAX_SQL_BYTES)?;
    }
    Ok(())
}

fn response_from_saved(saved: SavedDashboard) -> Result<DashboardResponse, ApiError> {
    let definition = serde_json::from_str(&saved.definition_json).map_err(|error| {
        ApiError::internal(format!(
            "saved dashboard {:?} is invalid: {error}",
            saved.dashboard_id
        ))
    })?;
    Ok(DashboardResponse {
        definition,
        created_at_ms: saved.created_at_ms,
        updated_at_ms: saved.updated_at_ms,
    })
}

async fn list_dashboards(
    State(store): State<Arc<Store>>,
) -> Result<Json<DashboardListResponse>, ApiError> {
    let rows = tokio::task::spawn_blocking(move || store.list_dashboards())
        .await
        .map_err(|error| ApiError::internal(format!("list dashboards task failed: {error}")))?
        .map_err(|error| ApiError::internal(error.to_string()))?;
    let dashboards = rows
        .into_iter()
        .map(response_from_saved)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Json(DashboardListResponse { dashboards }))
}

async fn get_dashboard(
    State(store): State<Arc<Store>>,
    Path(dashboard_id): Path<String>,
) -> Result<Json<DashboardResponse>, ApiError> {
    let requested_id = dashboard_id.clone();
    let saved = tokio::task::spawn_blocking(move || store.dashboard(&dashboard_id))
        .await
        .map_err(|error| ApiError::internal(format!("get dashboard task failed: {error}")))?
        .map_err(|error| ApiError::internal(error.to_string()))?
        .ok_or_else(|| ApiError::not_found(&requested_id))?;
    Ok(Json(response_from_saved(saved)?))
}

async fn put_dashboard(
    State(store): State<Arc<Store>>,
    Path(dashboard_id): Path<String>,
    request: Request,
) -> Result<Json<DashboardResponse>, ApiError> {
    let content_type = request
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    if !content_type.is_some_and(|value| value.eq_ignore_ascii_case("application/json")) {
        return Err(ApiError::bad_request(
            "Content-Type must be application/json",
        ));
    }
    let body = to_bytes(request.into_body(), MAX_BODY_BYTES)
        .await
        .map_err(|_| ApiError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            code: "dashboard_too_large",
            message: format!("request body exceeds {MAX_BODY_BYTES} bytes"),
        })?;
    let definition: DashboardDefinition = serde_json::from_slice(&body)
        .map_err(|error| ApiError::bad_request(format!("invalid dashboard JSON: {error}")))?;
    validate_definition(&dashboard_id, &definition)?;
    let definition_json = serde_json::to_string(&definition)
        .map_err(|error| ApiError::internal(format!("encode dashboard: {error}")))?;
    let saved = tokio::task::spawn_blocking(move || {
        store.upsert_dashboard(&dashboard_id, &definition_json)
    })
    .await
    .map_err(|error| ApiError::internal(format!("save dashboard task failed: {error}")))?
    .map_err(|error| ApiError::internal(error.to_string()))?;
    Ok(Json(response_from_saved(saved)?))
}

async fn delete_dashboard(
    State(store): State<Arc<Store>>,
    Path(dashboard_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let requested_id = dashboard_id.clone();
    let outcome = tokio::task::spawn_blocking(move || store.delete_dashboard(&dashboard_id))
        .await
        .map_err(|error| ApiError::internal(format!("delete dashboard task failed: {error}")))?
        .map_err(|error| ApiError::internal(error.to_string()))?;
    match outcome {
        DashboardDeleteOutcome::Deleted => Ok(StatusCode::NO_CONTENT),
        DashboardDeleteOutcome::NotFound => Err(ApiError::not_found(&requested_id)),
    }
}

/// Build authenticated dashboard routes backed by the Finelog catalog.
pub fn router(store: Arc<Store>, auth: Arc<AuthPolicy>) -> Router {
    Router::new()
        .route("/v1/dashboards", get(list_dashboards))
        .route(
            "/v1/dashboards/:dashboard_id",
            get(get_dashboard)
                .put(put_dashboard)
                .delete(delete_dashboard),
        )
        .with_state(store)
        .layer(from_fn_with_state(auth, auth_gate))
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use axum::body::Body;
    use axum::extract::ConnectInfo;
    use axum::http::{Method, Request};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;

    fn test_router() -> Router {
        router(
            Arc::new(Store::new(None, String::new()).unwrap()),
            Arc::new(AuthPolicy::allow_localhost()),
        )
    }

    fn definition(id: &str) -> Value {
        json!({
            "version": 1,
            "id": id,
            "title": "Hardware health",
            "description": "Recent device faults.",
            "variables": [{"name": "job_id", "label": "Job", "default": ""}],
            "panels": [{
                "id": "faults",
                "title": "Faults",
                "kind": "table",
                "description": "Latest fault values.",
                "sql": "SELECT 1 AS value",
                "width": "full"
            }]
        })
    }

    fn request(method: Method, uri: &str, body: Body) -> Request<Body> {
        let mut request = Request::builder()
            .method(method)
            .uri(uri)
            .header(CONTENT_TYPE, "application/json")
            .body(body)
            .unwrap();
        request.extensions_mut().insert(ConnectInfo(
            "127.0.0.1:12345".parse::<SocketAddr>().unwrap(),
        ));
        request
    }

    async fn json_body(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn dashboard_crud_round_trips_through_http() {
        let app = test_router();
        let response = app
            .clone()
            .oneshot(request(
                Method::PUT,
                "/v1/dashboards/hardware-health",
                Body::from(definition("hardware-health").to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let saved = json_body(response).await;
        assert_eq!(saved["definition"]["title"], "Hardware health");

        let response = app
            .clone()
            .oneshot(request(Method::GET, "/v1/dashboards", Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let listed = json_body(response).await;
        assert_eq!(listed["dashboards"].as_array().unwrap().len(), 1);
        assert_eq!(
            listed["dashboards"][0]["definition"]["id"],
            "hardware-health"
        );

        let response = app
            .clone()
            .oneshot(request(
                Method::DELETE,
                "/v1/dashboards/hardware-health",
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);

        let response = app
            .oneshot(request(
                Method::GET,
                "/v1/dashboards/hardware-health",
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rejects_mismatched_ids_and_oversized_bodies() {
        let app = test_router();
        let response = app
            .clone()
            .oneshot(request(
                Method::PUT,
                "/v1/dashboards/expected",
                Body::from(definition("different").to_string()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let response = app
            .oneshot(request(
                Method::PUT,
                "/v1/dashboards/too-large",
                Body::from(vec![b'x'; MAX_BODY_BYTES + 1]),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
