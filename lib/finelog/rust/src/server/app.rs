//! axum app construction.
//!
//! [`build_app`] registers BOTH services on ONE connect `Router` via the
//! generated `register()` (REQUIRED so `ctx.spec()`/`ctx.path()` are populated
//! for the interceptors — `into_axum_service` would leave them empty and the
//! interceptor chain would silently no-op), wraps it in a `ConnectRpcService`
//! with raised 64 MiB limits (the 4 MB default would reject 16 MiB WriteRows and
//! large query IPC) and the default zstd/gzip `CompressionRegistry`, attaches the
//! SlowRpc (outermost) then Concurrency interceptors, and assembles the axum
//! `Router`:
//!
//! ```text
//! [strip-forwarded-prefix middleware]  (outermost; normalizes the URI path)
//! [legacy-path middleware]             (transport layer; rewrites the URI)
//!   /health
//!   /debug/*            (only with --debug-admin)
//!   /static, /favicon.ico, /, /{*rest}   (SPA, before the fallback)
//!   .fallback_service(connect)            (RPC POSTs land here)
//! ```
//!
//! Route precedence matters: the SPA/health/debug routes are matched FIRST and
//! the connect service is the FALLBACK, so an RPC POST to `/<pkg.Service>/<Method>`
//! still reaches connect while unmatched GETs serve the SPA.

use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use connectrpc::{ConnectRpcService, Limits, Router as ConnectRouter};

use crate::proto::finelog::logging::LogServiceExt;
use crate::proto::finelog::stats::StatsServiceExt;
use crate::server::auth::{auth_gate, AuthInterceptor, AuthPolicy};
use crate::server::interceptors::{
    ConcurrencyInterceptor, SlowRpcInterceptor, DEFAULT_SLOW_RPC_THRESHOLD_MS,
    MAX_CONCURRENT_FETCH_LOGS, MAX_CONCURRENT_QUERY,
};
use crate::server::{debug, forwarded_prefix, legacy_path, spa};
use crate::store::Store;

use super::log_service::LogServiceImpl;
use super::stats_service::StatsServiceImpl;
use super::MAX_MESSAGE_BYTES;

/// Server-shell tuning. The interceptor caps + slow threshold are fields so a
/// cargo/parity test can lower them; production uses [`ServerConfig::default`].
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Mount the non-proto `--debug-admin` `/debug/*` routes.
    pub debug_admin: bool,
    /// Concurrent FetchLogs cap.
    pub max_concurrent_fetch_logs: usize,
    /// Concurrent Query cap.
    pub max_concurrent_query: usize,
    /// Default per-method slow-RPC threshold (ms); `<= 0` disables.
    pub slow_rpc_threshold_ms: i64,
    /// The authenticated-ingress policy. Always enforced — the
    /// interceptor gates every RPC and the policy is default-deny. The default
    /// ([`AuthPolicy::allow_localhost`]) admits loopback only, so a bare finelog
    /// is reachable for local debugging but never open to its network; a shared
    /// global finelog sets a `cidr`+`jwt` stack.
    pub auth: Arc<AuthPolicy>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            debug_admin: false,
            max_concurrent_fetch_logs: MAX_CONCURRENT_FETCH_LOGS,
            max_concurrent_query: MAX_CONCURRENT_QUERY,
            slow_rpc_threshold_ms: DEFAULT_SLOW_RPC_THRESHOLD_MS,
            auth: Arc::new(AuthPolicy::allow_localhost()),
        }
    }
}

impl ServerConfig {
    /// Production defaults with only `debug_admin` set (the common CLI path).
    pub fn with_debug_admin(debug_admin: bool) -> Self {
        Self {
            debug_admin,
            ..Self::default()
        }
    }

    /// Set the authenticated-ingress policy (chainable). Replaces the
    /// allow-localhost default; the policy is always default-deny.
    pub fn with_auth(mut self, policy: AuthPolicy) -> Self {
        self.auth = Arc::new(policy);
        self
    }
}

/// Build the connect `ConnectRpcService`: both services on one Router (via
/// `register`), raised limits, default zstd/gzip, SlowRpc (outermost) then
/// Concurrency interceptors.
fn build_connect_service(
    store: Arc<Store>,
    config: &ServerConfig,
) -> ConnectRpcService<ConnectRouter> {
    let connect_router = {
        let r = ConnectRouter::new();
        let r = Arc::new(StatsServiceImpl::new(Arc::clone(&store))).register(r);
        Arc::new(LogServiceImpl::new(Arc::clone(&store))).register(r)
    };

    ConnectRpcService::new(connect_router)
        .with_limits(
            Limits::default()
                .max_message_size(MAX_MESSAGE_BYTES)
                .max_request_body_size(MAX_MESSAGE_BYTES),
        )
        // SlowRpc registered FIRST -> outermost -> times the whole chain
        // including the concurrency wait.
        .with_interceptor(SlowRpcInterceptor::new(config.slow_rpc_threshold_ms))
        // Auth sits just inside SlowRpc (so a rejected request is still timed) and
        // outside Concurrency (so it is rejected before consuming a permit). Always
        // installed; the default policy admits loopback only.
        .with_interceptor(AuthInterceptor::new(Arc::clone(&config.auth)))
        .with_interceptor(ConcurrencyInterceptor::new(
            config.max_concurrent_fetch_logs,
            config.max_concurrent_query,
        ))
    // gzip + zstd are already active via the default CompressionRegistry the
    // crate's `gzip`/`zstd` features install in ConnectRpcService::new.
}

/// Build the full axum app for `store` with `config`. See the module doc for
/// the route precedence. Re-exported as `server::build_app_with_config`.
pub fn build_app(store: Arc<Store>, config: ServerConfig) -> Router {
    let connect_service = build_connect_service(Arc::clone(&store), &config);

    let mut app = Router::new().route("/health", get(|| async { "ok" }));
    if config.debug_admin {
        // Mounted BEFORE the connect fallback so /debug/* is not shadowed. These
        // admin routes bypass the Connect interceptor chain, so they are gated by
        // the same auth policy here via `auth_gate` (default-deny, loopback-only
        // by default) — the admin surface is never more open than the RPCs.
        let debug = debug::debug_router(Arc::clone(&store)).layer(
            axum::middleware::from_fn_with_state(Arc::clone(&config.auth), auth_gate),
        );
        app = app.merge(debug);
    }
    // SPA routes BEFORE the connect fallback so unmatched GETs serve index.html.
    // The SPA catch-all forwards NON-GET (RPC POSTs) to a clone of the connect
    // service so an RPC POST to a SPA-matched path is dispatched, not 405'd; the
    // connect service is also the app's `.fallback_service` for paths the SPA
    // routes don't match.
    app = spa::spa_routes(app, spa::vue_dist_dir(), connect_service.clone());

    app.fallback_service(connect_service)
        // Transport layer: rewrite legacy /iris.logging.LogService/* onto
        // /finelog.logging.LogService/* before routing.
        .layer(axum::middleware::from_fn(
            legacy_path::rewrite_legacy_logging_path,
        ))
        // Outermost transport layer: strip X-Forwarded-Prefix from the path so
        // a dashboard fronted by a sub-path proxy routes correctly whether or
        // not that proxy strips the prefix itself. Runs before the legacy
        // rewrite so a prefixed legacy path is normalized first.
        .layer(axum::middleware::from_fn(
            forwarded_prefix::strip_forwarded_prefix,
        ))
}
