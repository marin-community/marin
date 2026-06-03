//! Legacy `/iris.logging.LogService/*` path rewrite (Phase 5e).
//!
//! Pre-#5212 (b212f0015) the LogService proto package was `iris.logging`, so old
//! worker images push to `/iris.logging.LogService/*`. The wire format is
//! identical across packages (same field numbers), so we rewrite the request
//! path to `/finelog.logging.LogService/*` BEFORE routing. Port of
//! `asgi.py::_LegacyIrisLoggingPathMiddleware`.
//!
//! This is a TRANSPORT-layer rewrite (an axum `middleware::from_fn` that mutates
//! `req.uri()`), NOT a connect `Interceptor` — interceptors run after path
//! routing, so by then the wrong route (the connect fallback's 404) has already
//! been chosen. Applied as the outermost app layer so the rewritten path is what
//! the router sees.
//!
//! VERIFICATION ITEM (do not silently drop): the CRON removal marker
//! (2026-05-12) has passed, BUT this is still live — the iris dashboard alias and
//! the worker push path depend on it. Port it; flag for removal only after
//! confirming no worker image / dashboard alias still emits the legacy prefix.

use axum::extract::Request;
use axum::http::uri::{PathAndQuery, Uri};
use axum::middleware::Next;
use axum::response::Response;

const LEGACY_PATH_PREFIX: &str = "/iris.logging.LogService/";
const CURRENT_PATH_PREFIX: &str = "/finelog.logging.LogService/";

/// Rewrite a `/iris.logging.LogService/<tail>` path to
/// `/finelog.logging.LogService/<tail>`, preserving the query string. Returns
/// `None` for any path that does not start with the legacy prefix.
fn rewritten_uri(uri: &Uri) -> Option<Uri> {
    let tail = uri.path().strip_prefix(LEGACY_PATH_PREFIX)?;
    let new_path = format!("{CURRENT_PATH_PREFIX}{tail}");
    let new_path_and_query = match uri.query() {
        Some(q) => format!("{new_path}?{q}"),
        None => new_path,
    };
    let pq = new_path_and_query.parse::<PathAndQuery>().ok()?;
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(pq);
    Uri::from_parts(parts).ok()
}

/// axum middleware that rewrites the legacy logging path before routing.
pub async fn rewrite_legacy_logging_path(mut req: Request, next: Next) -> Response {
    if let Some(new_uri) = rewritten_uri(req.uri()) {
        *req.uri_mut() = new_uri;
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_only_legacy_prefix() {
        let legacy: Uri = "/iris.logging.LogService/FetchLogs".parse().unwrap();
        let got = rewritten_uri(&legacy).unwrap();
        assert_eq!(got.path(), "/finelog.logging.LogService/FetchLogs");
    }

    #[test]
    fn preserves_query_string() {
        let legacy: Uri = "/iris.logging.LogService/PushLogs?x=1&y=2".parse().unwrap();
        let got = rewritten_uri(&legacy).unwrap();
        assert_eq!(got.path(), "/finelog.logging.LogService/PushLogs");
        assert_eq!(got.query(), Some("x=1&y=2"));
    }

    #[test]
    fn leaves_current_path_untouched() {
        let current: Uri = "/finelog.logging.LogService/FetchLogs".parse().unwrap();
        assert!(rewritten_uri(&current).is_none());
    }

    #[test]
    fn leaves_unrelated_path_untouched() {
        let health: Uri = "/health".parse().unwrap();
        assert!(rewritten_uri(&health).is_none());
        let stats: Uri = "/finelog.stats.StatsService/Query".parse().unwrap();
        assert!(rewritten_uri(&stats).is_none());
    }
}
