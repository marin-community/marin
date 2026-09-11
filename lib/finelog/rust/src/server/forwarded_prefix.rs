//! Strip `X-Forwarded-Prefix` from the request path before routing.
//!
//! finelog rewrites the SPA `<base href>` to `X-Forwarded-Prefix` (see
//! [`crate::server::spa::index_html_with_base`]), so a dashboard fronted by a
//! reverse proxy at a sub-path tells the browser that its assets and routes
//! live under `<prefix>/...`. A well-behaved proxy then strips that prefix
//! before forwarding, and finelog's routes (`/static`, the SPA index, and the
//! connect `/<pkg.Service>/<Method>` paths) all match at the root.
//!
//! When the proxy instead forwards the prefixed path unchanged, every asset
//! request (`<prefix>/static/...`) and RPC POST (`<prefix>/<pkg.Service>/<Method>`)
//! misses the root-mounted routes and falls through to the SPA catch-all
//! (returning `index.html` with the wrong content-type) or the connect 404.
//!
//! This transport-layer rewrite normalizes the inbound path to what the router
//! expects: if `X-Forwarded-Prefix` is set and the path starts with it on a
//! segment boundary, strip it. It is a no-op when the proxy already stripped
//! (the path does not start with the prefix), so finelog serves the dashboard
//! correctly behind either kind of proxy. The header itself is left intact so
//! the SPA index handler still rewrites `<base href>` from it.
//!
//! Like [`crate::server::legacy_path`] this is an axum `middleware::from_fn`
//! that mutates `req.uri()`, applied as the outermost app layer (ahead of the
//! legacy-path rewrite) so the normalized path is what the router sees.

use axum::extract::Request;
use axum::http::uri::{PathAndQuery, Uri};
use axum::middleware::Next;
use axum::response::Response;

const FORWARDED_PREFIX_HEADER: &str = "x-forwarded-prefix";

/// Strip a leading `prefix` from `uri`'s path, preserving the query string.
///
/// The prefix is matched on a path-segment boundary: `/p` strips from `/p` and
/// `/p/x` but not `/proxy`. A trailing slash on the prefix is ignored. Returns
/// `None` (no rewrite) when the prefix is empty or `/`, or when the path does
/// not start with it.
fn strip_prefix_from_uri(uri: &Uri, prefix: &str) -> Option<Uri> {
    let prefix = prefix.trim_end_matches('/');
    if prefix.is_empty() {
        return None;
    }
    let tail = uri.path().strip_prefix(prefix)?;
    // Segment boundary: after stripping, the remainder must be empty or start
    // with `/` — otherwise `/proxy` would match the prefix `/p`.
    if !tail.is_empty() && !tail.starts_with('/') {
        return None;
    }
    let new_path = if tail.is_empty() { "/" } else { tail };
    let new_path_and_query = match uri.query() {
        Some(q) => format!("{new_path}?{q}"),
        None => new_path.to_string(),
    };
    let pq = new_path_and_query.parse::<PathAndQuery>().ok()?;
    let mut parts = uri.clone().into_parts();
    parts.path_and_query = Some(pq);
    Uri::from_parts(parts).ok()
}

/// axum middleware that strips `X-Forwarded-Prefix` from the path before routing.
pub async fn strip_forwarded_prefix(mut req: Request, next: Next) -> Response {
    let prefix = req
        .headers()
        .get(FORWARDED_PREFIX_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    if let Some(prefix) = prefix {
        if let Some(new_uri) = strip_prefix_from_uri(req.uri(), &prefix) {
            *req.uri_mut() = new_uri;
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(path_and_query: &str, prefix: &str) -> Option<String> {
        let uri: Uri = path_and_query.parse().unwrap();
        strip_prefix_from_uri(&uri, prefix).map(|u| {
            u.path_and_query()
                .map(|pq| pq.as_str().to_string())
                .unwrap_or_default()
        })
    }

    #[test]
    fn strips_prefix_from_asset_path() {
        assert_eq!(
            strip(
                "/system.log-server/static/js/index.abc.js",
                "/system.log-server"
            ),
            Some("/static/js/index.abc.js".to_string()),
        );
    }

    #[test]
    fn collapses_bare_prefix_to_root() {
        // The SPA index request lands on the prefix itself, with or without a
        // trailing slash; both normalize to `/`.
        assert_eq!(
            strip("/system.log-server", "/system.log-server"),
            Some("/".to_string())
        );
        assert_eq!(
            strip("/system.log-server/", "/system.log-server"),
            Some("/".to_string())
        );
    }

    #[test]
    fn strips_prefix_from_rpc_path_preserving_query() {
        assert_eq!(
            strip("/p/finelog.stats.StatsService/Query?x=1", "/p"),
            Some("/finelog.stats.StatsService/Query?x=1".to_string()),
        );
    }

    #[test]
    fn trailing_slash_on_prefix_is_ignored() {
        assert_eq!(
            strip("/p/static/x.js", "/p/"),
            Some("/static/x.js".to_string())
        );
    }

    #[test]
    fn only_matches_on_segment_boundary() {
        // `/proxy` must not be treated as carrying the prefix `/p`.
        assert_eq!(strip("/proxy/static/x.js", "/p"), None);
    }

    #[test]
    fn empty_or_root_prefix_is_noop() {
        assert_eq!(strip("/static/x.js", ""), None);
        assert_eq!(strip("/static/x.js", "/"), None);
    }

    #[test]
    fn unprefixed_path_is_noop() {
        // A proxy that already stripped forwards the root path; leave it alone.
        assert_eq!(strip("/static/js/index.abc.js", "/system.log-server"), None);
    }
}
