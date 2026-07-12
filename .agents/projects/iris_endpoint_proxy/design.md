# Iris Endpoint Proxy

_Why are we doing this? What's the benefit?_

Per-task dashboards (Ray, JAX, TensorBoard, custom UIs) currently require a user to either know a worker's IP and reach it directly, or build their own ad-hoc tunnel. We want a single, authed entry point: hit `iris-controller/proxy/<endpoint-name>/<sub-path>` and the controller forwards normal HTTP/2 to the registered endpoint. This piggybacks on infrastructure that already exists — the endpoint registry, an in-memory store, and a working `ActorProxy` — and turns "expose a dashboard from a job" into a one-line `register_endpoint` call. Explicit non-goals: WebSockets, link-rewriting, server-sent-event semantics. This is for dashboards, not gateway features.

## Background

Iris already has a task-scoped endpoint registry (`EndpointRow` in `lib/iris/src/iris/cluster/controller/schema.py:1566`, write-through cache `EndpointStore` at `lib/iris/src/iris/cluster/controller/stores.py:97`, RPCs `register_endpoint` / `list_endpoints` / `unregister_endpoint` at `lib/iris/src/iris/cluster/controller/service.py:1702-1793`) and one consumer of it: `ActorProxy` at `lib/iris/src/iris/cluster/controller/actor_proxy.py:48`, which forwards `POST /iris.actor.ActorService/{method}` to the endpoint resolved from an `X-Iris-Actor-Endpoint` header. The new proxy is structurally the same — `httpx.AsyncClient`, hop-by-hop header filtering, `502` on upstream failure — but routes by URL path instead of header, supports any HTTP method, streams response bodies, and preserves query strings. See `research.md` for the full digest.

## Challenges

The only real design point is how to encode the endpoint name in the URL. Iris names are wire-format paths (`/user/<job_id>/<actor>`); embedding a `/`-containing string in `/proxy/<name>/<sub-path>` is ambiguous unless the name is escaped or the registry is queried for a longest prefix. Everything else — streaming, auth, header forwarding, lifecycle — is a copy of the `ActorProxy` shape with small extensions.

## Costs / Risks

- Every registered endpoint becomes implicitly proxyable for any authenticated controller user. Acceptable under the current threat model — the controller is the trust boundary and dashboards are not generally secret.
- Names containing `.` won't round-trip through the URL transform. Worse than unreachable: if `/user/job/foo.bar` and `/user/job/foo/bar` are both registered, the URL `/proxy/user.job.foo.bar/...` silently routes to the second one. Current callsites use slash-only wire-format names so this collision is theoretical, but it's a footgun if a future registrar gets clever. We accept it: the controller is single-tenant in trust terms, the registrar is the same user making the request, and the downside is "your dashboard URL doesn't go where you expected" rather than a privilege boundary.
- New surface area on the controller HTTP server: a Starlette catch-all route. If an endpoint is unhealthy or slow, traffic to its dashboard sits on the controller's connection pool. A 30s timeout is the only backstop; a body-size cap is left to a follow-up if it becomes necessary.
- Cookie and `Authorization` stripping (both directions for cookies; client→upstream for Authorization) means dashboards with their own cookie-based state or bearer-token auth lose access to those credentials. Acceptable for the use case — controller auth has already gated the request, and forwarding the controller's session JWT to an arbitrary dashboard upstream is a credential leak. If a specific dashboard genuinely needs either, that's a follow-up opt-in (`metadata["proxy_pass_cookies"]`, `metadata["proxy_pass_auth"]`).

## Design

Add `EndpointProxy` next to `ActorProxy`, mounted on the controller's `Starlette` app at `/proxy/{endpoint_name:str}/{sub_path:path}`. Endpoint name uses **domain-style encoding**: a registered name `/user/job/coordinator` is reachable at `/proxy/user.job.coordinator/...`. The transform is one-way at the proxy boundary — `.` → `/` on lookup. Names containing `.` are not reachable via the proxy URL (they resolve to a `/`-substituted form that won't match); this is documented, not enforced. No `register_endpoint` change.

```python
# lib/iris/src/iris/cluster/controller/endpoint_proxy.py (new)

PROXY_ROUTE = "/proxy/{endpoint_name:str}/{sub_path:path}"
PROXY_TIMEOUT_SECONDS = 30.0
_HOP_BY_HOP = frozenset({"host", "transfer-encoding", "connection",
                         "keep-alive", "upgrade", "te", "trailer", "proxy-authorization",
                         "cookie", "set-cookie", "authorization"})

class EndpointProxy:
    def __init__(self, store: ControllerStore):
        self._store = store
        self._client = httpx.AsyncClient(timeout=PROXY_TIMEOUT_SECONDS, follow_redirects=False)

    async def close(self) -> None:
        await self._client.aclose()

    async def handle(self, request: Request) -> Response:
        url_name = request.path_params["endpoint_name"]
        sub_path = request.path_params["sub_path"]
        name = url_name.replace(".", "/")  # bijective; "." disallowed at registration
        row = self._store.endpoints.resolve(name)
        if row is None:
            return JSONResponse({"error": f"No endpoint '{url_name}'"}, status_code=404)

        base = row.address if "://" in row.address else f"http://{row.address}"
        upstream_url = f"{base}/{sub_path}"
        if request.url.query:
            upstream_url += f"?{request.url.query}"
        forward_headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in _HOP_BY_HOP}

        upstream_req = self._client.build_request(
            request.method, upstream_url, headers=forward_headers, content=request.stream(),
        )
        try:
            upstream_resp = await self._client.send(upstream_req, stream=True)
        except httpx.HTTPError as exc:
            return JSONResponse({"error": f"Upstream error: {exc}"}, status_code=502)

        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers={k: v for k, v in upstream_resp.headers.items()
                     if k.lower() not in _HOP_BY_HOP},
            background=BackgroundTask(upstream_resp.aclose),
        )
```

Wiring (`dashboard.py:322-340`): add the route alongside `PROXY_ROUTE` for the actor proxy, decorate the handler with `@requires_auth`, and chain `endpoint_proxy.close` into the existing `on_shutdown(...)` lifespan.

The proxy uses `EndpointStore.resolve()` exclusively — no DB hit on the request path, no new index, no schema or proto change.

### Relationship to `ActorProxy`

`ActorProxy` (`actor_proxy.py:48`) is functionally subsumed by this design: an actor RPC at `controller/iris.actor.ActorService/Call` with header `X-Iris-Actor-Endpoint: /user/job/actor` is the same upstream call as `controller/proxy/user.job.actor/iris.actor.ActorService/Call`. The actor `ProxyResolver` (`lib/iris/src/iris/actor/resolver.py`, source of `ACTOR_ENDPOINT_HEADER`) currently builds a single Connect transport pointed at the controller and stamps the endpoint name into a header; it can instead build a per-actor transport with base URL `controller/proxy/<name-with-dots>` and drop the header.

**Subsumption applies to unary RPC only.** Connect streaming RPCs use chunked envelopes with in-band trailers and rely on long-lived connections, both of which collide with the response-body cap and 30s timeout. The current `ActorService` is unary, so this is fine for the migration. If streaming actors are ever introduced, they need their own proxy path with separate caps — or skip the proxy and address the actor server directly.

**Out of scope for the initial PR.** This design lands `EndpointProxy` alongside `ActorProxy` without touching the resolver. A follow-up PR migrates the resolver (path-based base URL, drop `ACTOR_ENDPOINT_HEADER`) and deletes `actor_proxy.py`. Splitting it keeps the first PR small and lets the actor migration ship behind a real-cluster validation pass.

## Testing

Two integration tests in `lib/iris/tests/actor/test_endpoint_proxy.py` (mirroring `test_actor_proxy.py`):

1. **Round-trip with a real upstream.** Spawn a tiny Starlette app on `127.0.0.1:0` exposing `/foo`, `/bar?x=1`, and a 1 MiB binary asset. Register it under `/user/jobX/dash` via the controller's endpoint store. Assert that `GET /proxy/user.jobX.dash/foo`, query string preservation, header forwarding, and the large asset (streamed) all return correct status / body / content-type.
2. **Failure modes.** Unknown endpoint → `404`. Upstream returns `500` → proxy returns `500` (passthrough, not `502`). Upstream connection refused → `502`. Endpoint registered with a `.` in its name → reachable at the slash-substituted URL is `404` (documented limitation).

Plus a unit test on the `.` → `/` transform. We do not re-test `EndpointStore` semantics — `test_endpoint_store.py` already covers them.

For live-cluster validation, expose Levanter's stats dashboard on a worker via `register_endpoint`, hit it through `iris-controller/proxy/...`, and confirm the page renders.

## Open Questions

- **Per-job rate limiting / concurrency cap?** A misbehaving dashboard polled by an open browser tab can keep a connection slot occupied indefinitely (within the 30s timeout). Probably not worth solving in v1, but worth flagging.
- **Address normalization at registration.** Today the proxy does a substring `://` test on `row.address` and prepends `http://` if missing. Cleaner to validate / normalize once at `register_endpoint` time and store a canonical form. Out of scope here, but a small follow-up.
- **CORS preflight.** Browser-driven dashboards may issue `OPTIONS` preflights; the controller's existing CORS posture (if any) needs to permit `/proxy/...`. If we discover a real consumer needs it, add it; not gating v1.
