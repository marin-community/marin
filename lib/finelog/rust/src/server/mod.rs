//! Native finelog HTTP server wiring.
//!
//! [`app::build_app`] registers BOTH services on one connect `Router` (so
//! `ctx.spec()`/`ctx.path()` are populated for the interceptors), wraps it in a
//! `ConnectRpcService` with raised 64 MB limits + zstd/gzip + the SlowRpc /
//! Concurrency interceptors, mounts `/health`, the SPA, and (optionally) the
//! `--debug-admin` routes, layers the legacy-path and forwarded-prefix transport
//! rewrites, and sets the connect service as the fallback. The connect service stays the FALLBACK
//! so RPC POSTs reach it while `/health`, `/debug/*`, `/static`, and the SPA
//! GET routes take precedence.

pub mod app;
pub mod debug;
pub mod diagnostics;
pub mod forwarded_prefix;
pub mod interceptors;
pub mod legacy_path;
pub mod log_service;
pub mod spa;
pub mod stats_service;

use std::sync::Arc;

use axum::Router;

use crate::store::Store;

pub use app::build_app as build_app_with_config;
pub use app::ServerConfig;

/// 64 MiB request/message limits (default is 4 MB — too small for WriteRows /
/// large query IPC). The Query handler reuses it as the result-size bound ->
/// `resource_exhausted`.
pub(crate) const MAX_MESSAGE_BYTES: usize = 64 << 20;

/// Build the axum app with production defaults and the given `--debug-admin`
/// flag. Thin convenience over [`app::build_app`] for the binary entry point.
pub fn build_app(store: Arc<Store>, debug_admin: bool) -> Router {
    app::build_app(store, ServerConfig::with_debug_admin(debug_admin))
}
