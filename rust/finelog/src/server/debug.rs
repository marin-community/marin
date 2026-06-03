//! Flag-gated, NON-proto test-only admin surface (`--debug-admin`).
//!
//! The frozen RPC contract (LogService / StatsService) cannot force a
//! flush/compact/sync/evict cycle nor read per-segment level+location, but every
//! Phase-4 gating test does exactly that. This module exposes two plain axum
//! routes — `POST /debug/maintain` and `GET /debug/segments` — that drive the
//! SAME `Store::maintain_namespace` / `Store::list_segments` code the background
//! maintenance task uses (not a parallel implementation). It is mounted only
//! when `--debug-admin`/`FINELOG_DEBUG_ADMIN` is set and is OFF in production.
//!
//! Sanctioned by roadmap decision 9 (the test-forcing seam). Routes are mounted
//! BEFORE the connect fallback so RPC POSTs still reach the connect service.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::store::types::SegmentRow;
use crate::store::Store;

/// `POST /debug/maintain` body: which namespace, and whether to force an L0->L1
/// compaction regardless of planner policy.
#[derive(Debug, Deserialize)]
struct MaintainRequest {
    namespace: String,
    #[serde(default)]
    force_compact_l0: bool,
}

/// `GET /debug/segments?namespace=NS` query.
#[derive(Debug, Deserialize)]
struct SegmentsQuery {
    namespace: String,
}

/// One row in the `/debug/segments` JSON response. `path` is the basename only
/// (the absolute path differs across backends / log dirs, so the harness keys on
/// the filename + structured fields).
#[derive(Debug, Serialize)]
struct DebugSegment {
    path: String,
    level: i32,
    min_seq: i64,
    max_seq: i64,
    row_count: i64,
    byte_size: i64,
    location: String,
    created_at_ms: i64,
}

fn basename(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(path)
        .to_string()
}

fn to_debug_segment(row: SegmentRow) -> DebugSegment {
    DebugSegment {
        path: basename(&row.path),
        level: row.level,
        min_seq: row.min_seq,
        max_seq: row.max_seq,
        row_count: row.row_count,
        byte_size: row.byte_size,
        location: row.location.as_str().to_string(),
        created_at_ms: row.created_at_ms,
    }
}

async fn post_maintain(
    State(store): State<Arc<Store>>,
    Json(req): Json<MaintainRequest>,
) -> impl IntoResponse {
    // `maintain_namespace` is async and takes the query-visibility WRITE lock
    // INSIDE the engine (commit_swap / evict_segment via blocking_write), drained
    // against in-flight queries holding the READ side — so the handler must NOT
    // hold the write lock here (that would deadlock the blocking acquire).
    match store
        .maintain_namespace(&req.namespace, req.force_compact_l0)
        .await
    {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    }
}

/// `POST /debug/backdate` body: set a segment's `created_at_ms` so age-eviction
/// tests run without a wall-clock sleep.
#[derive(Debug, Deserialize)]
struct BackdateRequest {
    namespace: String,
    /// Segment filename (basename) to backdate.
    path: String,
    created_at_ms: i64,
}

async fn post_backdate(
    State(store): State<Arc<Store>>,
    Json(req): Json<BackdateRequest>,
) -> impl IntoResponse {
    let store2 = Arc::clone(&store);
    match tokio::task::spawn_blocking(move || {
        store2.backdate_segment(&req.namespace, &req.path, req.created_at_ms)
    })
    .await
    {
        Ok(Ok(())) => StatusCode::OK.into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        Err(join) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("backdate task panicked: {join}"),
        )
            .into_response(),
    }
}

async fn get_segments(
    State(store): State<Arc<Store>>,
    Query(q): Query<SegmentsQuery>,
) -> impl IntoResponse {
    let store2 = Arc::clone(&store);
    let ns = q.namespace.clone();
    match tokio::task::spawn_blocking(move || store2.list_segments(&ns)).await {
        Ok(Ok(rows)) => {
            let segments: Vec<DebugSegment> = rows.into_iter().map(to_debug_segment).collect();
            Json(segments).into_response()
        }
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        Err(join) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("segments task panicked: {join}"),
        )
            .into_response(),
    }
}

/// The `/debug/*` routes with `store` as state, for merging into the app router.
pub fn debug_router(store: Arc<Store>) -> Router {
    Router::new()
        .route("/debug/maintain", post(post_maintain))
        .route("/debug/segments", get(get_segments))
        .route("/debug/backdate", post(post_backdate))
        .with_state(store)
}
