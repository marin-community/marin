# Spec — iris_endpoint_proxy

Concrete contracts for the `EndpointProxy` design. Read alongside `design.md`.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `lib/iris/src/iris/cluster/controller/endpoint_proxy.py` | new | `EndpointProxy` class, constants, route handler |
| `lib/iris/src/iris/cluster/controller/dashboard.py` | modified | Add route + handler + lifespan chain |
| `lib/iris/tests/cluster/controller/test_endpoint_proxy.py` | new | Integration tests |

No proto changes. No schema changes. No `register_endpoint` changes. No new dependencies (httpx, starlette, BackgroundTask all already imported elsewhere).

## Module-level constants

In `endpoint_proxy.py`:

```python
PROXY_ROUTE = "/proxy/{endpoint_name:str}/{sub_path:path}"
PROXY_TIMEOUT_SECONDS: float = 30.0

# Headers stripped in both directions. `cookie` / `set-cookie` / `authorization`
# strip is a security choice (see design.md "Cookie stripping"); the rest are
# standard hop-by-hop.
_HOP_BY_HOP: frozenset[str] = frozenset({
    "host", "transfer-encoding", "connection", "keep-alive", "upgrade",
    "te", "trailer", "proxy-authorization", "proxy-authenticate",
    "cookie", "set-cookie", "authorization",
})

ALLOWED_METHODS: tuple[str, ...] = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")

# httpx connection pool. Defaults (100/100) are fine for now; making it explicit
# so the limits don't drift silently when httpx changes defaults.
_HTTPX_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
```

No body-size cap. The 30s timeout is the only backstop. Adding a cap is a follow-up if a real misbehaving upstream forces it.

## Public API: `EndpointProxy`

```python
class EndpointProxy:
    """Forwards arbitrary HTTP requests to a registered endpoint.

    The proxy resolves the endpoint name (with `.` → `/` substitution) against
    `ControllerStore.endpoints`, then forwards request method, path suffix,
    query string, and filtered headers to the upstream's `address`. Bodies are
    streamed in both directions with no size cap. Hop-by-hop headers,
    `Cookie` / `Set-Cookie`, and `Authorization` are stripped (see `_HOP_BY_HOP`).

    Lifecycle: construct once on dashboard startup; await `close()` on
    shutdown to drain the underlying httpx connection pool. The proxy is safe
    for concurrent use across requests.
    """

    def __init__(self, store: ControllerStore) -> None: ...

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient. Idempotent."""

    async def handle(self, request: starlette.requests.Request) -> starlette.responses.Response:
        """Handle one proxied request.

        Path params expected: `endpoint_name` (single segment, may contain `.`),
        `sub_path` (anything after, possibly empty).

        Returns a streaming response on success, or a JSONResponse with the
        error contract below on failure. Never raises.
        """
```

### Error contract (status codes returned by `handle`)

| Status | Condition | Body shape |
|--------|-----------|------------|
| `<upstream>` | Upstream responded — pass through verbatim status, headers (filtered), body (streamed) | upstream body |
| `404 Not Found` | `EndpointStore.resolve(name.replace(".", "/"))` returns `None` | `{"error": "No endpoint '<url_name>'"}` |
| `405 Method Not Allowed` | Method not in `ALLOWED_METHODS` (Starlette enforces via `methods=` arg before reaching `handle`; specified for completeness) | Starlette default |
| `502 Bad Gateway` | `httpx.ConnectError`, `httpx.RemoteProtocolError`, or any non-timeout `httpx.HTTPError` while connecting / sending | `{"error": "Upstream error: <repr>"}` |
| `504 Gateway Timeout` | `httpx.ConnectTimeout` or `httpx.ReadTimeout` (covers PROXY_TIMEOUT_SECONDS expiry on connect or read) | `{"error": "Upstream timeout after 30s"}` |

If the upstream connection drops mid-stream after the proxy has already emitted status + headers, the response is truncated at the TCP layer. There is no clean error path — status is committed. Document this; do not attempt to handle it in code.

## Wiring change in `dashboard.py`

Three edits inside `ControllerDashboard`:

**1. Construct the proxy in `__init__`** (next to the existing `self._actor_proxy = ActorProxy(...)`):

```python
self._endpoint_proxy = EndpointProxy(service.store)
```

**2. Add the route + handler** in the `routes = [...]` list at `dashboard.py:322`. The handler is a small wrapper that delegates to `self._endpoint_proxy.handle` so the auth decorator binds to the dashboard's method, not the proxy class:

```python
@requires_auth
async def _proxy_endpoint(self, request: Request) -> Response:
    return await self._endpoint_proxy.handle(request)

# in routes list, alongside the existing actor PROXY_ROUTE entry:
Route(endpoint_proxy.PROXY_ROUTE, self._proxy_endpoint, methods=list(endpoint_proxy.ALLOWED_METHODS)),
```

**3. Chain shutdown** — replace `lifespan=on_shutdown(self._actor_proxy.close)` (`dashboard.py:340`) with a chained version. `on_shutdown` accepts multiple callables; pass both:

```python
lifespan=on_shutdown(self._actor_proxy.close, self._endpoint_proxy.close)
```

(If `on_shutdown` doesn't already accept varargs, this is the moment to make it do so — it's a private helper in `iris.cluster.dashboard_common`.)

## Test contract: `tests/cluster/controller/test_endpoint_proxy.py`

Top-level pytest fixtures, parameterized where useful. No mocks at the proxy boundary — spin a real upstream Starlette app on `127.0.0.1:0` and a real `EndpointStore`-backed proxy. Mirror `test_actor_proxy.py` patterns.

```python
@pytest.fixture
def upstream() -> Iterator[UpstreamHandle]:
    """Spin up a fixture Starlette app exposing /echo, /large, /slow, /500."""

@pytest.fixture
def proxy(upstream) -> Iterator[ProxyHandle]:
    """Construct EndpointProxy + minimal Starlette host with the proxy route,
    register the upstream under name '/user/jobX/dash'."""

def test_round_trip_get(proxy):
    """GET /proxy/user.jobX.dash/echo?q=1 forwards method, path, query, body."""

def test_round_trip_post_body(proxy):
    """POST forwards a 1 MiB JSON body intact."""

def test_streams_large_response(proxy):
    """GET /proxy/user.jobX.dash/large returns 9 MiB body without buffering
    (assert via memory profile or by reading in chunks before upstream finishes)."""

def test_unknown_endpoint_returns_404(proxy):
    """Endpoint name not in store → 404 with JSON error body."""

def test_upstream_5xx_passes_through(proxy):
    """Upstream returns 500 → proxy returns 500, not 502."""

def test_upstream_connection_refused_returns_502(proxy):
    """Endpoint registered with unreachable address → 502."""

def test_upstream_timeout_returns_504(proxy):
    """Upstream takes > PROXY_TIMEOUT_SECONDS → 504."""

def test_cookies_stripped_both_directions(proxy):
    """Client sends Cookie: session=x → upstream sees no Cookie header.
    Upstream sends Set-Cookie: foo=bar → client sees no Set-Cookie header."""

def test_authorization_stripped(proxy):
    """Client sends Authorization: Bearer abc → upstream sees no Authorization."""

def test_dot_to_slash_transform(proxy):
    """name '/user/jobX/dash' resolves at /proxy/user.jobX.dash/...; a
    name registered as 'literal.dot' is unreachable through the proxy
    (404 — documented limitation)."""

def test_method_not_allowed_returns_405(proxy):
    """CONNECT / TRACE return 405 (Starlette route filter)."""

def test_disallowed_methods_not_listed():
    """Unit test: ALLOWED_METHODS contains GET/POST/PUT/PATCH/DELETE/HEAD/
    OPTIONS only; no CONNECT, no TRACE."""
```

A short auth integration test lives separately (one test in the existing dashboard auth test file): unauthenticated request to `/proxy/...` returns 401, mirroring how other `@requires_auth` routes are tested.

## Out of scope (explicit)

These are not part of this PR. Reviewers should not request them as blockers.

- WebSocket / SSE proxying.
- Link / URL rewriting in proxied HTML or JS responses.
- Removal of `ActorProxy` and migration of `iris.actor.resolver.ProxyResolver` to the path-based proxy. Tracked as a follow-up; design.md "Relationship to `ActorProxy`".
- An opt-in `metadata["proxy_pass_cookies"]` / `proxy_pass_auth` flag for endpoints that need credentials. Add only if a real consumer needs it.
- Per-endpoint timeout overrides via `metadata`. Constant is global for v1.
- Body-size cap (request or response). Add later if a real misbehaving upstream forces it.
- Rate limiting or concurrency caps per endpoint.
- HTTP/3, HTTP/2 server-push.
