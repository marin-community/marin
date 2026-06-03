//! Static Vue SPA serving.
//!
//! Resolves the dashboard `dist` directory at startup ([`vue_dist_dir`]), then
//! mounts:
//!
//! - `/static/*` via `tower_http::services::ServeDir` over `dist/static`,
//! - `/favicon.ico` (when present),
//! - `/` and `/{*rest}` -> the SPA index handler, which reads `dist/index.html`
//!   at request time and applies [`index_html_with_base`] (the byte-exact
//!   base-href rewrite keyed off `X-Forwarded-Prefix`).
//!
//! When `dist` is absent (CI has no built dashboard) the `/` route serves the
//! [`NOT_BUILT_HTML`] placeholder. All SPA routes are registered BEFORE the
//! connect fallback so RPC POSTs (POST `/<pkg.Service>/<Method>`) still reach the
//! connect service — only unmatched GETs fall through to the SPA index.

use std::convert::Infallible;
use std::path::PathBuf;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, get_service};
use axum::Router;
use tower::Service;
use tower_http::services::ServeDir;

/// Env override for the dashboard `dist` directory (used by the SPA parity test
/// to point at a tmp dist without a built dashboard).
const DIST_DIR_ENV: &str = "FINELOG_DASHBOARD_DIST";

/// Docker image location for the built dashboard.
const DOCKER_VUE_DIST_DIR: &str = "/app/dashboard/dist";

/// The `<base href="/"` placeholder the Vue build emits. The rewrite replaces
/// the FIRST occurrence so asset/router URLs resolve under a reverse-proxy
/// sub-path.
const BASE_HREF_PLACEHOLDER: &[u8] = b"<base href=\"/\"";

/// Placeholder served at `/` when no dashboard `dist` is present. It is a
/// human-facing diagnostic page, not a machine-parsed wire contract, so
/// exact-byte parity is not required (the load-bearing wire contract is the
/// base-href rewrite, which IS byte-checked).
pub const NOT_BUILT_HTML: &str = "<!doctype html><html><body>\
<h1>Dashboard not built</h1>\
<p>Run <code>npm run build</code> in <code>lib/finelog/dashboard</code>.</p>\
</body></html>";

/// Rewrite the first `<base href="/"` to `<base href="{prefix}/"` so a SPA
/// fronted by a reverse proxy at a sub-path resolves its asset/router URLs.
///
/// - empty prefix or `"/"` is a no-op (returns `raw` unchanged),
/// - the prefix is normalized to a leading AND trailing slash,
/// - only the FIRST placeholder occurrence is replaced.
pub fn index_html_with_base(raw: &[u8], prefix: &str) -> Vec<u8> {
    if prefix.is_empty() || prefix == "/" {
        return raw.to_vec();
    }
    let mut normalized = String::with_capacity(prefix.len() + 2);
    if !prefix.starts_with('/') {
        normalized.push('/');
    }
    normalized.push_str(prefix);
    if !normalized.ends_with('/') {
        normalized.push('/');
    }
    let replacement = format!("<base href=\"{normalized}\"").into_bytes();
    replace_first(raw, BASE_HREF_PLACEHOLDER, &replacement)
}

/// Replace the first occurrence of `needle` in `haystack` with `replacement`.
fn replace_first(haystack: &[u8], needle: &[u8], replacement: &[u8]) -> Vec<u8> {
    let Some(pos) = haystack.windows(needle.len()).position(|w| w == needle) else {
        return haystack.to_vec();
    };
    let mut out = Vec::with_capacity(haystack.len() - needle.len() + replacement.len());
    out.extend_from_slice(&haystack[..pos]);
    out.extend_from_slice(replacement);
    out.extend_from_slice(&haystack[pos + needle.len()..]);
    out
}

/// Resolve the dashboard `dist` directory, requiring `dist/index.html`.
///
/// Order: the `FINELOG_DASHBOARD_DIST` env override, then the in-repo
/// `lib/finelog/dashboard/dist` (relative to the crate manifest), then the
/// Docker image path `/app/dashboard/dist`. Returns `None` when none exists with
/// an `index.html`.
pub fn vue_dist_dir() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(over) = std::env::var(DIST_DIR_ENV) {
        if !over.is_empty() {
            candidates.push(PathBuf::from(over));
        }
    }
    // The crate lives at rust/finelog; the dashboard dist is at
    // lib/finelog/dashboard/dist from the repo root. CARGO_MANIFEST_DIR is an
    // absolute path to rust/finelog at build time; walk up to the repo root.
    if let Some(repo_root) = repo_root_from_manifest() {
        candidates.push(repo_root.join("lib/finelog/dashboard/dist"));
    }
    candidates.push(PathBuf::from(DOCKER_VUE_DIST_DIR));

    candidates
        .into_iter()
        .find(|c| c.is_dir() && c.join("index.html").is_file())
}

/// The repo root inferred from `CARGO_MANIFEST_DIR` (`<root>/rust/finelog`).
fn repo_root_from_manifest() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // rust/finelog -> rust -> <repo root>
    manifest.parent()?.parent().map(PathBuf::from)
}

/// SPA index handler: read `dist/index.html` at request time and apply the
/// base-href rewrite keyed off `X-Forwarded-Prefix`. `dist` is captured per
/// route (no axum State) so the SPA sub-router stays `Router<()>` and merges
/// cleanly with the rest of the app.
async fn spa_index(dist: PathBuf, headers: HeaderMap) -> Response {
    let prefix = headers
        .get("x-forwarded-prefix")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    match tokio::fs::read(dist.join("index.html")).await {
        Ok(raw) => {
            let html = index_html_with_base(&raw, prefix);
            ([(header::CONTENT_TYPE, "text/html")], html).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read index.html: {e}"),
        )
            .into_response(),
    }
}

/// The placeholder handler served at `/` when no `dist` is present.
async fn not_built() -> Response {
    ([(header::CONTENT_TYPE, "text/html")], NOT_BUILT_HTML).into_response()
}

/// Layer the SPA routes onto `app`, BEFORE the connect fallback is set.
///
/// With a `dist`: `/static` (ServeDir), `/favicon.ico` (when present), `/` and
/// `/*rest` (SPA index). Without one: just `/` -> the not-built placeholder.
///
/// Route-precedence note: the catch-all `/*rest` (and `/`) serve the SPA for
/// `GET` but `fallback_service` non-GET methods to `connect`. An RPC POST to
/// `/<pkg.Service>/<Method>` matches `/*rest` by PATH; without the method
/// fallback axum would 405 it (the path matched, the GET-only method did not)
/// instead of forwarding to the connect service.
///
/// A *GET* to an RPC path serves the SPA index here (the catch-all matches it).
/// Real connect clients only POST to RPC paths (which reach connect via the
/// method fallback above), so this is not externally observable — it is the
/// standard SPA semantics where any unknown GET serves index.html for
/// client-side routing.
///
/// `connect` is the same `ConnectRpcService` used as the app's `.fallback_service`;
/// it is `Clone`, so cloning it into the catch-all is cheap (Arc-shared state).
pub fn spa_routes<S>(app: Router, dist: Option<PathBuf>, connect: S) -> Router
where
    S: Service<Request, Error = Infallible> + Clone + Send + 'static,
    S::Response: IntoResponse + 'static,
    S::Future: Send + 'static,
{
    let Some(dist) = dist else {
        tracing::info!("dashboard dist not found; serving placeholder at /");
        // The placeholder is GET-only; non-GET to `/` falls back to connect so
        // an RPC POST to `/` (never happens in practice) is not 405'd.
        return app.route("/", get(not_built).fallback_service(connect));
    };

    let mut app = app.nest_service("/static", get_service(ServeDir::new(dist.join("static"))));

    let favicon = dist.join("favicon.ico");
    if favicon.is_file() {
        app = app.route("/favicon.ico", get(move || serve_file(favicon.clone())));
    }

    // Capture `dist` per route so the handlers need no axum State and the SPA
    // routes stay `Router<()>`.
    let dist_root = dist.clone();
    let dist_rest = dist;
    app.route(
        "/",
        get(move |headers: HeaderMap| spa_index(dist_root.clone(), headers))
            .fallback_service(connect.clone()),
    )
    // axum 0.7 catch-all syntax is `/*rest` (the `/{*rest}` form is axum 0.8).
    .route(
        "/*rest",
        get(move |headers: HeaderMap| spa_index(dist_rest.clone(), headers))
            .fallback_service(connect),
    )
}

/// Serve a single file's bytes (favicon).
async fn serve_file(path: PathBuf) -> Response {
    match tokio::fs::read(&path).await {
        Ok(bytes) => Response::builder()
            .status(StatusCode::OK)
            .body(Body::from(bytes))
            .unwrap(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_href_rewrite() {
        let raw = br#"<html><head><base href="/" /></head></html>"#;
        // Empty / "/" prefix is a no-op.
        assert_eq!(index_html_with_base(raw, ""), raw.to_vec());
        assert_eq!(index_html_with_base(raw, "/"), raw.to_vec());
        // A leading-slash prefix gets a trailing slash.
        assert_eq!(
            index_html_with_base(raw, "/proxy/log-server"),
            br#"<html><head><base href="/proxy/log-server/" /></head></html>"#.to_vec(),
        );
        // A no-leading-slash, trailing-slash prefix normalizes the same way.
        assert_eq!(
            index_html_with_base(raw, "proxy/log-server/"),
            br#"<html><head><base href="/proxy/log-server/" /></head></html>"#.to_vec(),
        );
        // Replacement happens only once even when the placeholder appears twice.
        let raw_dup = br#"<base href="/" /><base href="/" />"#;
        assert_eq!(
            index_html_with_base(raw_dup, "/p"),
            br#"<base href="/p/" /><base href="/" />"#.to_vec(),
        );
    }

    #[test]
    fn not_built_html_has_expected_markers() {
        // The placeholder is a human-facing diagnostic page (not a wire
        // contract), so assert its identifying markers rather than exact bytes.
        assert!(NOT_BUILT_HTML.starts_with("<!doctype html>"));
        assert!(NOT_BUILT_HTML.contains("Dashboard not built"));
        assert!(NOT_BUILT_HTML.contains("lib/finelog/dashboard"));
    }
}
