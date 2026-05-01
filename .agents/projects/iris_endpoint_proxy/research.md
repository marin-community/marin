# Research — iris_endpoint_proxy

## Framing

A generic HTTP proxy on the Iris controller: `iris-controller/proxy/<name>/<sub-path>` forwards normal HTTP/2 requests to a registered endpoint's `host:port`. Target use case is exposing per-task dashboards (Ray, JAX, TensorBoard, custom UIs) through the controller's web origin without each user reaching the worker IP directly. Explicit non-goals: WebSockets, server-sent-event streaming with backpressure, link rewriting.

## In-repo findings

### Endpoint registry — already exists

The endpoint concept is fully wired up; the proxy just consumes it.

- `EndpointRow` dataclass: `lib/iris/src/iris/cluster/controller/schema.py:1566` — `endpoint_id`, `name`, `address`, `task_id`, `metadata: dict`, `registered_at`.
- SQLite schema: `schema.py:1049` (`ENDPOINTS = Table(...)`) — task-scoped, auto-cleaned on retry.
- In-memory write-through cache `EndpointStore`: `lib/iris/src/iris/cluster/controller/stores.py:97`. Reads never touch the DB.
  - `EndpointStore.resolve(name)` at `stores.py:203` — exact-name lookup, returns one row.
  - `EndpointStore.query(EndpointQuery(name_prefix=...))` at `stores.py:172` — prefix scan.
  - `EndpointStore.all()` at `stores.py:216` — full snapshot, used to build prefix index.
- RPC: `register_endpoint`, `unregister_endpoint`, `list_endpoints` in `lib/iris/src/iris/cluster/controller/service.py:1702-1793`.
- Proto: `lib/iris/src/iris/rpc/controller.proto:281-312`.
- Client API: `lib/iris/src/iris/cluster/client/remote_client.py:330` (`register_endpoint`).

Names are typically wire-formatted (`/user/<job_id>/<actor>`); `metadata` is a free-form `map<string,string>` already used to stash dashboard URLs and ports in some callsites.

### Existing proxy — `ActorProxy`

The shape we want to copy. `lib/iris/src/iris/cluster/controller/actor_proxy.py:48-105`.

Key patterns to reuse verbatim:
- `httpx.AsyncClient(timeout=PROXY_TIMEOUT_SECONDS)` constructed once, stored on the proxy instance.
- `_HOP_BY_HOP_HEADERS` filter (`actor_proxy.py:36`): `host`, `transfer-encoding`, `connection`, `keep-alive`, `upgrade`. We add `x-iris-actor-endpoint`-style headers for the new proxy too.
- Address normalization: accept `host:port` or `scheme://host:port` (`actor_proxy.py:75`).
- Error codes: `400` (bad request), `404` (endpoint missing), `502` (upstream error). We adopt the same.
- Lifespan: `Starlette(routes=routes, lifespan=on_shutdown(self._actor_proxy.close))` at `dashboard.py:340`. Same pattern for the new proxy's `aclose()`.

What ActorProxy does *not* do that we must add:
- Path forwarding — ActorProxy hard-codes `POST /iris.actor.ActorService/{method}`. We need any method, any sub-path.
- Streaming response bodies — ActorProxy buffers `upstream_resp.content` (line 94). Dashboards serve large static assets; we use `httpx.AsyncClient.stream()` and Starlette `StreamingResponse`.
- Query string forwarding — ActorProxy doesn't preserve `request.url.query`. We do.

### Controller HTTP server

- Framework: **Starlette + uvicorn**, all async. Not FastAPI.
- `ControllerDashboard`: `lib/iris/src/iris/cluster/controller/dashboard.py:214`.
- Route table: `dashboard.py:322-340`. New route slots in here.
- Auth middleware: `_RouteAuthMiddleware` at `dashboard.py:78`. Default-deny; routes must be decorated `@public` or `@requires_auth` (imported from `iris.cluster.dashboard_common`). The new proxy route uses `@requires_auth`.
- `lifespan=on_shutdown(...)` chain (line 340) — append the new proxy's `close` here.

### Auth

- JWT cookie / Bearer token, validated by `resolve_auth(token, verifier, optional)`.
- Existing browser sessions (controller dashboard SPA) carry the cookie and will satisfy `@requires_auth` automatically — dashboards opened via the proxy URL Just Work for logged-in users.
- CSRF check (`_check_csrf` at `dashboard.py:150`) only fires on auth state-changing endpoints; we don't need it on the proxy.

### Network

Controller has direct TCP to all workers (Iris assumes flat L3 within a VPC / k8s). No tunneling. Endpoint `address` is reachable from the controller process. This is the same assumption ActorProxy already relies on.

### HTTP client

`httpx>=0.28.1` (`lib/iris/pyproject.toml`) — already used by `ActorProxy` and `ProxyControllerDashboard`. No new dependency needed.

### Tests for similar shape

- `lib/iris/tests/actor/test_actor_proxy.py` — uses a `StandaloneActorProxy` helper with a dict-backed registry; spawns a real `ActorServer` on localhost; asserts round-trips, missing-header (`400`), unknown-endpoint (`404`). Same structure works for the new proxy: spin a tiny Starlette app with one `/foo` route, register it under a name, hit `/proxy/<name>/foo` through a controller test fixture.
- `lib/iris/tests/cluster/controller/test_endpoint_store.py` — covers the registry semantics; we don't re-test those.

## Prior-art pass

Skipped intentionally. This is a reverse proxy of a well-understood shape (httpx → Starlette `StreamingResponse`); the in-repo `ActorProxy` is the relevant reference implementation. Re-deriving "what does an HTTP reverse proxy look like" from OSS examples would not improve the design.

## Q&A summary (decisions surfaced before drafting)

**Q1 — name disambiguation in URL.** Iris names contain `/` (e.g. `/user/<job_id>/<actor>`). Options considered:
- (a) Longest-prefix match against `EndpointStore`.
- (b) UUID in URL.
- (c) Opt-in proxy alias stored in `metadata["proxy_alias"]`.
- (d) **Domain-style transform**: `/user/job/task` ↔ `user.job.task`. URL is `/proxy/user.job.task/<sub-path>`; parse single `:str` segment, transform `.` → `/`, look up via `EndpointStore.resolve()`.

**Decision: (d).** Bijective and trivially reversible, URLs are short and human-readable, every endpoint is automatically proxyable with no registry changes, no alias-collision problem, single O(1) `resolve()`. Bijectivity requires that registered names not contain `.`; we enforce this at `register_endpoint` time (raise `INVALID_ARGUMENT` if name contains `.`) — current callsites all use slash-only wire-format names so no migration is needed.

**Q2 — auth.** `@requires_auth`. The controller is the existing trust boundary; "logged-in dashboard user can see proxied dashboards" is the natural policy and matches the cookie that the SPA already carries.

**Q3 — streaming.** Use `httpx.AsyncClient.stream()` + Starlette `StreamingResponse`. Dashboards serve large assets (JS bundles, plot tiles); buffering them all in memory is wasteful and would block the event loop.

**Q4 — methods.** `GET POST PUT DELETE PATCH HEAD OPTIONS`. No `CONNECT`, no WebSocket upgrade, no SSE-specific handling (per user: "not for websockets or anything fancy").

**Q5 — backwards compat.** N/A — this is a new feature. No migration.

## Open follow-ups (surfaced in design.md)

- The `.` → `/` transform requires names not contain `.`. Survey current callsites of `register_endpoint` to confirm no `.`-containing names exist; add a `register_endpoint` boundary validation. Confirm with cluster-aware reviewer that no real callsite is harmed.
- Every registered endpoint becomes globally proxyable for any authed controller user. Acceptable for the current threat model (controller is the trust boundary), but worth flagging.
