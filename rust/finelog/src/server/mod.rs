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
pub mod auth;
pub mod debug;
pub mod diagnostics;
pub mod forwarded_prefix;
pub mod forwarding;
pub mod interceptors;
pub mod legacy_path;
pub mod log_service;
pub mod spa;
pub mod stats_service;
#[cfg(test)]
pub mod test_support;

pub use app::build_app as build_app_with_config;
pub use app::ServerConfig;
pub use auth::AuthPolicy;
pub use forwarding::{spawn as spawn_forwarder, Forwarder, ForwardingConfig};

/// 64 MiB request/message limits (default is 4 MB — too small for WriteRows /
/// large query IPC). The Query handler reuses it as the result-size bound ->
/// `resource_exhausted`.
pub(crate) const MAX_MESSAGE_BYTES: usize = 64 << 20;
