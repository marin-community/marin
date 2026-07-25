//! `finelog-server` binary entry point.
//!
//! Parse the CLI flags, open the `Store`, and serve `/health` plus the
//! StatsService RPCs.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use finelog::server::diagnostics::spawn_pool_diagnostics;
use finelog::server::{
    build_app_with_config, spawn_forwarder, AuthPolicy, Forwarder, ForwardingConfig, ServerConfig,
};
use finelog::store::Store;
use tokio::sync::Notify;

/// Bound process RSS. DataFusion frees its query buffers promptly (the pool
/// returns to ~0 between queries), but the default glibc allocator retains the
/// freed pages in its per-CPU arenas rather than returning them to the OS, so
/// RSS pins at the high-water mark of the largest query until restart (measured
/// ~3.5x higher and slowly drifting vs jemalloc over repeated heavy scans).
/// jemalloc's background thread purges dirty pages on a decay schedule, so RSS
/// follows real usage. The `background_threads` feature turns that thread on by
/// default. Unix-only; the binary ships on Linux.
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Parser, Debug)]
#[command(name = "finelog-server")]
struct Args {
    /// Port to bind.
    #[arg(long, env = "FINELOG_PORT", default_value_t = 8080)]
    port: u16,

    /// Local directory for parquet segments + catalog.
    #[arg(long, env = "FINELOG_LOG_DIR")]
    log_dir: Option<String>,

    /// Remote (gs:// or s3://) directory for offloaded segments. Empty disables
    /// sync. Read from `FINELOG_REMOTE_DIR`, set by the deploy environment.
    #[arg(long, env = "FINELOG_REMOTE_DIR", default_value = "")]
    remote_log_dir: String,

    /// Log level for the server's own tracing output.
    #[arg(long, env = "FINELOG_LOG_LEVEL", default_value = "info")]
    log_level: String,

    /// Mount the NON-proto test-only `/debug/*` admin routes (maintain/segments).
    /// Off the frozen contract; used only by the parity harness. Never set in
    /// production.
    #[arg(long, env = "FINELOG_DEBUG_ADMIN", default_value_t = false)]
    debug_admin: bool,

    /// Authenticated-ingress policy as a JSON list of layers (env
    /// `FINELOG_AUTH_POLICY`), e.g.
    /// `[{"type":"cidr","cidrs":["10.0.0.0/8","127.0.0.0/8"]},{"type":"jwt","keys":[{"cluster":"marin","public_keys":["<ed25519-pem>"]}]}]`.
    /// List order is evaluation order (first Allow admits, first Reject denies,
    /// none → deny). Empty (the default) installs the private allow-localhost
    /// policy: reachable from loopback for local debugging, never open to the
    /// network. A shared global finelog sets a `cidr`+`jwt` stack.
    #[arg(long = "auth-policy", env = "FINELOG_AUTH_POLICY", default_value = "")]
    auth_policy: String,

    /// Relay this store's logs to a shared hub finelog, as JSON (env
    /// `FINELOG_FORWARDING`): `{"target":"https://<hub>","cluster":"<this-cluster>"}`.
    /// Empty (the default) forwards nothing. Requires `--signing-key`.
    #[arg(long = "forwarding", env = "FINELOG_FORWARDING", default_value = "")]
    forwarding: String,

    /// This server's Ed25519 private key (PKCS#8 PEM, env `FINELOG_SIGNING_KEY`),
    /// which signs the `aud="finelog"` bearer the hub verifies against the matching
    /// public key in its `jwt` auth layer. Only the forwarder uses it. Deliver it
    /// through a secret; never inline it into a deploy manifest.
    #[arg(long = "signing-key", env = "FINELOG_SIGNING_KEY", default_value = "")]
    signing_key: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| args.log_level.clone().into()),
        )
        .init();

    let store = Arc::new(
        Store::new(
            args.log_dir.clone().map(PathBuf::from),
            args.remote_log_dir.clone(),
        )
        .map_err(|e| format!("failed to open store: {e}"))?,
    );
    // Start each namespace's maintenance task. Each task runs its boot remote
    // reconcile (adopt unknown remote parquet, redundancy-drop covered segments)
    // in the BACKGROUND as its first step, so a large first-time reconcile (e.g.
    // a namespace just self-healed into the catalog, whose thousands of archived
    // segments have never been footer-scanned) never blocks the listener bind
    // below. The server serves — and /health is green — while archived rows are
    // still being reconciled into the catalog.
    store.bootstrap_maintenance();
    // Auth is ALWAYS enforced (default-deny). No policy → the private
    // allow-localhost default (loopback only); a shared global finelog sets a
    // cidr+jwt stack. A malformed policy fails startup rather than mis-admitting.
    // The /debug/* admin routes are gated by this same policy (see `build_app`).
    let auth = if args.auth_policy.trim().is_empty() {
        AuthPolicy::allow_localhost()
    } else {
        AuthPolicy::parse(&args.auth_policy)
            .map_err(|e| format!("invalid FINELOG_AUTH_POLICY: {e}"))?
    };
    tracing::info!(policy = %auth.describe(), "finelog-server: auth policy active");
    let config = ServerConfig::with_debug_admin(args.debug_admin).with_auth(auth);
    let app = build_app_with_config(Arc::clone(&store), config);

    // Cross-cluster forwarding, when configured. Spawned before the listener binds
    // so a store with a backlog starts draining immediately, and latched off in the
    // shutdown block below.
    let (forward_stop, forward_task) = match build_forwarder(&args, Arc::clone(&store))? {
        Some(forwarder) => {
            let (tx, rx) = tokio::sync::watch::channel(false);
            (Some(tx), Some(spawn_forwarder(forwarder, rx)))
        }
        None => (None, None),
    };

    // Periodic pool/RSS diagnostics task; cancelled on shutdown via a latched
    // stop flag (set before the Notify, so a notify that races the task's emit
    // cannot be lost) plus the Notify for a prompt wakeup.
    let diag_stop = Arc::new(AtomicBool::new(false));
    let diag_shutdown = Arc::new(Notify::new());
    let diag = spawn_pool_diagnostics(
        Arc::clone(&store),
        Arc::clone(&diag_stop),
        Arc::clone(&diag_shutdown),
    );

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!(%addr, log_dir = ?args.log_dir, "finelog-server listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // Graceful shutdown: stop accepting and drain in-flight requests on the
    // first SIGTERM/SIGINT, then shut the store's background tasks down.
    // `into_make_service_with_connect_info` records each connection's peer
    // address so the auth CIDR rule can read it.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    tracing::info!("finelog-server draining background tasks");

    // Stop the forwarder first: it reads the store, so it must be off the segments
    // before the namespaces drain. It latches on the watch and interrupts its own
    // retry backoff, so the join is prompt; the bound is defense in depth.
    if let (Some(stop), Some(task)) = (forward_stop, forward_task) {
        let _ = stop.send(true);
        let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
    }

    // Stop the diagnostics task, then cooperatively cancel + join the
    // per-namespace flush/maintenance tasks. The per-namespace join is bounded;
    // an OUTER timeout here guarantees the process still exits promptly even if
    // a namespace shutdown is somehow slow (defense in depth; durability is
    // already preserved because writes ack only after L0 persist).
    diag_stop.store(true, Ordering::SeqCst);
    diag_shutdown.notify_waiters();
    // Bound the diagnostics join too: even with the latch the task does no
    // durable work, so it must never delay the store drain.
    let _ = tokio::time::timeout(Duration::from_secs(2), diag).await;
    let _ = tokio::time::timeout(
        Duration::from_secs(10),
        store.shutdown(Duration::from_secs(5)),
    )
    .await;
    tracing::info!("finelog-server stopped");
    Ok(())
}

/// Build the cross-cluster forwarder, or `None` when forwarding is unconfigured.
///
/// Every way the configuration can be wrong is an error here rather than a silent
/// no-op at the first push: a forwarder that never ships is invisible, and a store
/// whose logs stop reaching the hub looks exactly like a quiet cluster.
///
/// Forwarding needs a disk store. A memory-mode namespace publishes a durability
/// watermark but exposes no segments to a scan, so the forwarder would tail rows it
/// can never read.
fn build_forwarder(args: &Args, store: Arc<Store>) -> Result<Option<Forwarder>, String> {
    if args.forwarding.trim().is_empty() {
        return Ok(None);
    }
    if args.log_dir.is_none() {
        return Err(
            "forwarding needs a --log-dir: a memory-mode store has no segments to read".into(),
        );
    }
    if args.signing_key.trim().is_empty() {
        return Err(
            "forwarding needs --signing-key (env FINELOG_SIGNING_KEY) to mint its bearer".into(),
        );
    }
    let config = ForwardingConfig::parse(&args.forwarding)
        .map_err(|e| format!("invalid FINELOG_FORWARDING: {e}"))?;
    tracing::info!(
        target = %config.target,
        cluster = %config.cluster,
        "finelog-server: forwarding configured"
    );
    Ok(Some(Forwarder::new(store, config, &args.signing_key)?))
}

/// Resolve when the first SIGTERM or SIGINT (Ctrl-C) arrives.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => tracing::info!("received SIGTERM; shutting down"),
        _ = sigint.recv() => tracing::info!("received SIGINT; shutting down"),
    }
}
