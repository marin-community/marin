//! Build + storage introspection routes (`/api/server`, `/api/segments`).
//!
//! These answer the two questions an operator asks of a running finelog that
//! neither the RPC contract nor a SQL query can: *which source revision is this
//! binary*, and *what does this namespace's storage physically look like right
//! now*. The dashboard's System page and per-namespace Segments panel read them.
//!
//! Plain axum JSON rather than proto: the payloads describe this process and its
//! files, so they track the implementation rather than the wire contract, and
//! adding a field should not move a versioned RPC. They sit behind the same
//! default-deny [`auth_gate`](super::auth) as the `/debug/*` routes, so the
//! surface is never more open than the RPCs.

use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::query::metadata_cache_stats;
use crate::server::diagnostics::read_proc_self_status_kb;
use crate::store::index_bundle::SectionKind;
use crate::store::segment::{
    segment_id_and_row_group_rows, segment_physical, LAYOUT_VERSION, MAX_ROW_GROUP_ROWS,
    TARGET_ROW_GROUP_BYTES,
};
use crate::store::segment_index::parse_trigram_coverage;
use crate::store::segment_index::{parse_projection_reference, projection_path};
use crate::store::trigram::SIDECAR_SPAN_ROWS;
use crate::store::types::{basename, SegmentRow};
use crate::store::Store;

/// When this process started, stamped at router-build time so uptime counts
/// from the server coming up rather than from the first request.
static PROCESS_STARTED: OnceLock<SystemTime> = OnceLock::new();

fn process_started() -> SystemTime {
    *PROCESS_STARTED.get_or_init(SystemTime::now)
}

/// The source revision this binary was compiled from, baked in by `build.rs`.
///
/// Every field is empty when the build had no git checkout to read — a wheel
/// built from an sdist, for instance.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildInfo {
    pub version: &'static str,
    pub commit: &'static str,
    /// Tree hash of `commit`. Names the source content, so a rebased or
    /// cherry-picked rebuild of the same sources is visibly identical.
    pub tree: &'static str,
    /// Whether the checkout had uncommitted changes when it was built.
    pub dirty: bool,
    pub built_at_unix: i64,
    pub rustc: &'static str,
    pub profile: &'static str,
}

pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        commit: env!("FINELOG_BUILD_COMMIT"),
        tree: env!("FINELOG_BUILD_TREE"),
        dirty: matches!(env!("FINELOG_BUILD_DIRTY"), "true"),
        built_at_unix: env!("FINELOG_BUILD_AT_UNIX").parse().unwrap_or(0),
        rustc: env!("FINELOG_BUILD_RUSTC"),
        profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    }
}

/// This process: how long it has been up, where it is, and what it is holding.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProcessInfo {
    pid: u32,
    hostname: String,
    started_at_unix: i64,
    uptime_seconds: i64,
    rss_bytes: i64,
    vm_size_bytes: i64,
}

/// Where the store keeps its data, and how much of it is resident.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct StoreInfo {
    data_dir: String,
    remote_log_dir: String,
    namespaces: usize,
    ram_buffer_bytes: i64,
    ram_chunks: usize,
}

/// The parquet metadata cache, which is what stands between a query and
/// re-parsing every segment footer it touches.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct MetadataCacheInfo {
    limit_bytes: i64,
    size_bytes: i64,
    entries: i64,
    hits: i64,
}

/// Invalid derived index artifacts observed by this process. Both conditions
/// fall back to source Parquet scans, but a non-zero value merits rebuilding
/// the affected local bundle.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexCacheInfo {
    corrupt_bundles: i64,
    corrupt_sections: i64,
    exact_aggregate_full: i64,
    exact_aggregate_partial: i64,
    exact_aggregate_declined: i64,
    exact_aggregate_fallbacks: i64,
}

/// The on-disk format policy this binary writes. A segment whose
/// `layoutVersion` differs was written by an older policy and is queued for an
/// in-place re-encode by maintenance.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FormatInfo {
    layout_version: u32,
    target_row_group_bytes: i64,
    max_row_group_rows: i64,
    trigram_span_rows: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ServerInfoResponse {
    build: BuildInfo,
    process: ProcessInfo,
    store: StoreInfo,
    metadata_cache: MetadataCacheInfo,
    index_cache: IndexCacheInfo,
    format: FormatInfo,
}

/// `GET /api/segments?namespace=NS` query. `physical=true` additionally reads
/// each local segment's Parquet footer and index directory, which costs a tail
/// read per segment — cheap for one page load, not for a refresh loop.
#[derive(Debug, Deserialize)]
struct SegmentsQuery {
    namespace: String,
    #[serde(default)]
    physical: bool,
}

/// One segment: its catalog row, plus its physical shape when the caller asked
/// for it and the file is local.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentInfo {
    /// Basename. The directory is the store's, and repeating it per row buys
    /// nothing.
    path: String,
    level: i32,
    min_seq: i64,
    max_seq: i64,
    row_count: i64,
    byte_size: i64,
    created_at_ms: i64,
    /// `LOCAL`, `REMOTE`, or `BOTH`.
    location: String,
    min_key_value: Option<String>,
    max_key_value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    physical: Option<PhysicalInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PhysicalInfo {
    segment_identity: String,
    /// `None` for a segment written before the layout stamp existed.
    layout_version: Option<u32>,
    /// Whether that stamp matches what this binary writes today.
    layout_current: bool,
    row_groups: i64,
    /// Footer bytes on disk — the per-segment cost paid before any column is
    /// read.
    footer_bytes: i64,
    uncompressed_bytes: i64,
    created_by: Option<String>,
    /// Typed index bundle, when one exists and is bound to this segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    index_bundle: Option<IndexBundleInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexBundleInfo {
    /// Bytes in the checksummed `.fidx` bundle itself.
    bytes: i64,
    /// Bytes in covering-projection Parquets referenced by the bundle.
    external_bytes: i64,
    checksum: &'static str,
    sections: Vec<IndexSectionInfo>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexSectionInfo {
    id: String,
    kind: &'static str,
    exactness: &'static str,
    method_version: u8,
    checksum: &'static str,
    payload_bytes: i64,
    external_bytes: i64,
    /// Source or projected columns described by the section's typed coverage.
    columns: Vec<String>,
    /// False when a covering projection reference no longer resolves to the
    /// bound Parquet artifact. In-bundle sections are available with the
    /// readable directory; payload corruption is reported by `indexCache` once
    /// a query verifies its checksum.
    available: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SegmentsResponse {
    namespace: String,
    segments: Vec<SegmentInfo>,
}

/// Host name from `/proc`, or an empty string off Linux.
fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn unix_seconds(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn get_server(State(store): State<Arc<Store>>) -> impl IntoResponse {
    let started = process_started();
    let memory = store.memory_summary();
    let cache = metadata_cache_stats();
    let corruption = store.index_cache().corruption_counts();
    let aggregate = crate::query::exact_aggregate::stats();
    Json(ServerInfoResponse {
        build: build_info(),
        process: ProcessInfo {
            pid: std::process::id(),
            hostname: hostname(),
            started_at_unix: unix_seconds(started),
            uptime_seconds: started.elapsed().map(|d| d.as_secs() as i64).unwrap_or(0),
            rss_bytes: read_proc_self_status_kb("VmRSS") as i64 * 1024,
            vm_size_bytes: read_proc_self_status_kb("VmSize") as i64 * 1024,
        },
        store: StoreInfo {
            data_dir: store
                .data_dir()
                .map(|d| d.display().to_string())
                .unwrap_or_default(),
            remote_log_dir: store.remote_log_dir().to_string(),
            namespaces: memory.namespaces,
            ram_buffer_bytes: memory.ram_bytes,
            ram_chunks: memory.chunks,
        },
        metadata_cache: MetadataCacheInfo {
            limit_bytes: cache.limit_bytes as i64,
            size_bytes: cache.size_bytes as i64,
            entries: cache.entries as i64,
            hits: cache.hits as i64,
        },
        index_cache: IndexCacheInfo {
            corrupt_bundles: corruption.bundles as i64,
            corrupt_sections: corruption.sections as i64,
            exact_aggregate_full: aggregate.full as i64,
            exact_aggregate_partial: aggregate.partial as i64,
            exact_aggregate_declined: aggregate.declined as i64,
            exact_aggregate_fallbacks: aggregate.fallbacks as i64,
        },
        format: FormatInfo {
            layout_version: LAYOUT_VERSION,
            target_row_group_bytes: TARGET_ROW_GROUP_BYTES as i64,
            max_row_group_rows: MAX_ROW_GROUP_ROWS as i64,
            trigram_span_rows: SIDECAR_SPAN_ROWS as i64,
        },
    })
}

/// Read `path`'s footer and index bundle. `None` when the file is not readable here,
/// which is the normal state of a `REMOTE` segment after eviction.
fn physical_info(
    path: &str,
    index_cache: &crate::query::index_cache::IndexCache,
) -> Option<PhysicalInfo> {
    let path = Path::new(path);
    let physical = segment_physical(path).ok()?;
    let (source_id, rows) = segment_id_and_row_group_rows(path)?;
    let index_bundle = index_cache
        .get_header(path, source_id, rows.iter().sum::<usize>() as u64)
        .map(|header| {
            let sections = header
                .sections
                .iter()
                .map(|section| {
                    let reference = (section.kind == SectionKind::CoveringProjection)
                        .then(|| parse_projection_reference(&section.coverage))
                        .flatten();
                    let columns = match section.kind {
                        SectionKind::TrigramBloom => parse_trigram_coverage(&section.coverage)
                            .map(|coverage| vec![coverage.column])
                            .unwrap_or_default(),
                        SectionKind::ExactPostings | SectionKind::ValueCounts => {
                            std::str::from_utf8(&section.coverage)
                                .ok()
                                .map(|columns| {
                                    columns
                                        .split('\0')
                                        .filter(|column| !column.is_empty())
                                        .map(str::to_string)
                                        .collect()
                                })
                                .unwrap_or_default()
                        }
                        SectionKind::CoveringProjection => reference
                            .as_ref()
                            .map(|reference| reference.descriptor.columns.clone())
                            .unwrap_or_default(),
                    };
                    let available = if section.kind == SectionKind::CoveringProjection {
                        reference.as_ref().is_some_and(|reference| {
                            let projection = projection_path(path, reference);
                            std::fs::metadata(&projection)
                                .ok()
                                .is_some_and(|metadata| metadata.len() == reference.file_bytes)
                        })
                    } else {
                        true
                    };
                    let external_bytes = reference
                        .filter(|_| available)
                        .map(|reference| reference.file_bytes as i64)
                        .unwrap_or(0);
                    IndexSectionInfo {
                        id: section.id.clone(),
                        kind: section.kind.as_str(),
                        exactness: section.exactness.as_str(),
                        method_version: section.method_version,
                        checksum: section.checksum_algorithm.as_str(),
                        payload_bytes: section.len as i64,
                        external_bytes,
                        columns,
                        available,
                    }
                })
                .collect::<Vec<_>>();
            IndexBundleInfo {
                bytes: header.bundle_len as i64,
                external_bytes: sections.iter().map(|section| section.external_bytes).sum(),
                checksum: header.checksum_algorithm.as_str(),
                sections,
            }
        });
    Some(PhysicalInfo {
        segment_identity: source_id.to_string(),
        layout_version: physical.layout_version,
        layout_current: physical.layout_version == Some(LAYOUT_VERSION),
        row_groups: physical.row_groups as i64,
        footer_bytes: physical.footer_bytes,
        uncompressed_bytes: physical.uncompressed_bytes,
        created_by: physical.created_by,
        index_bundle,
    })
}

fn to_segment_info(
    row: SegmentRow,
    physical: bool,
    index_cache: &crate::query::index_cache::IndexCache,
) -> SegmentInfo {
    SegmentInfo {
        physical: physical
            .then(|| physical_info(&row.path, index_cache))
            .flatten(),
        path: basename(&row.path),
        level: row.level,
        min_seq: row.min_seq,
        max_seq: row.max_seq,
        row_count: row.row_count,
        byte_size: row.byte_size,
        created_at_ms: row.created_at_ms,
        location: row.location.as_str().to_string(),
        min_key_value: row.min_key_value,
        max_key_value: row.max_key_value,
    }
}

async fn get_segments(
    State(store): State<Arc<Store>>,
    Query(q): Query<SegmentsQuery>,
) -> impl IntoResponse {
    let namespace = q.namespace.clone();
    let index_cache = Arc::clone(store.index_cache());
    // Footer reads are blocking file I/O, and a large namespace has hundreds of
    // them, so the whole listing runs off the async runtime.
    let listed = tokio::task::spawn_blocking(move || {
        store.list_segments(&q.namespace).map(|rows| {
            rows.into_iter()
                .map(|row| to_segment_info(row, q.physical, &index_cache))
                .collect::<Vec<_>>()
        })
    })
    .await;
    match listed {
        Ok(Ok(segments)) => Json(SegmentsResponse {
            namespace,
            segments,
        })
        .into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        Err(join) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("segments task panicked: {join}"),
        )
            .into_response(),
    }
}

/// The `/api/*` introspection routes, for merging into the app router.
pub fn introspection_router(store: Arc<Store>) -> Router {
    process_started();
    Router::new()
        .route("/api/server", get(get_server))
        .route("/api/segments", get(get_segments))
        .with_state(store)
}
