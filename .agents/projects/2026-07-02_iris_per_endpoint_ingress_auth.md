# Per-endpoint auth-gated public ingress on the Iris controller

> Status: landed via PR #6857 (GitHub issue
> [#6847](https://github.com/marin-community/marin/issues/6847); follow-up to
> #6545 and #6556). This is the plan as written before implementation; the code
> is the source of truth. The largest drift from the plan: review moved the
> route-scoped auth layer into `rigging.server_auth` (`PolicyAuthInterceptor`,
> `RouteAuthMiddleware`, permissive/enforcing chains), replacing the
> `_DashboardAuthInterceptor`/`_enforce_http_auth` seams named below, and the
> Python code uses the proto `EndpointAccess` enum directly.
> Remaining operational step: run the `iap_gclb.py public-proxy` stage on real
> GCP (needs an operator with gcloud auth) and restart controllers.

## Problem

`marin-serve` (`quick_serve.py`) boots a vLLM slice, fronts it with an
OpenAI-compatible reverse proxy, and registers the dashboard as an Iris endpoint
reachable through the controller's `EndpointProxy` at `/proxy/serve.<ep>/`. That
route is gated by the dashboard-wide `@requires_auth`, so it is reachable only to
cluster-account holders behind the controller's IAP ingress.

Our agentic-RL / eval / datagen harnesses run inside **Daytona/Modal cloud
sandboxes** — a *different* cloud — that must call the on-cluster vLLM. Today we
bridge that with **one paid pinggy reverse tunnel per job**: a paid dependency, a
fragile SSH loop, and a **fully unauthenticated** public surface. We want to drop
pinggy and reach the endpoint directly through the controller.

We can't just remove `@requires_auth` from the proxy route: the *same* token that
passes it also passes the RPC interceptor, so any token good enough to reach a
served model would over-grant the cluster-control RPC surface. And on GCP the
controller sits behind IAP, which gates the whole origin — there is no route
today that a token-only (non-IAP) caller can reach.

## Goal

1. A per-endpoint **access mode** declared at registration:
   `{ PRIVATE | PUBLIC | BEARER }`.
2. For `BEARER`, a **scoped bearer token** minted by the controller (from its
   JWT signing key, with a deadline) that authorizes **only that endpoint's
   `/proxy/<name>/…` inference path** — never the dashboard's control RPCs.
3. A documented **public-ingress path per provider** that opens *only* the
   `/proxy` route off-cluster, without granting the controller any
   firewall/IAP-admin authority.

Non-goals: replacing IAP for the dashboard/RPC surface; a general API-gateway;
per-request quota/billing (rate-limiting stays at the Cloudflare WAF layer that
already fronts the ingress).

## What exists today (grounding)

- **Two name forms.** An endpoint's canonical **wire name** is the registered
  row `name`, `/`-separated (quick-serve's CLI default is `/serve/<job>`,
  `quick_serve_cli.py:191`). URLs carry an **encoded** `.`-separated form:
  `proxy_path()` maps `/serve/foo` → `/proxy/serve.foo`
  (`lib/rigging/src/rigging/connect.py:132-134`), and the subdomain form is
  `serve.foo.proxy.<base>`. The proxy decodes back: `dispatch` computes
  `slashed = encoded_name.replace(".", "/")` and tries `/{slashed}` then bare
  (`endpoint_proxy.py:256-257`) against `EndpointServiceImpl.resolve_endpoint`
  (`endpoint_service.py:180-188`), which does an exact-name row lookup and falls
  back to the in-memory `system_endpoints` map.
- **Proxy**: `controller/endpoint_proxy.py` — `EndpointProxy.dispatch(request, *,
  encoded_name, sub_path, proxy_prefix)`. Streams both ways and **strips
  `Authorization`/`Cookie` on the client→upstream hop** (credential-leak guard,
  `_HOP_BY_HOP`, `endpoint_proxy.py:82-97`).
- **Route wiring**: `controller/dashboard.py` mounts `PROXY_ROUTE =
  "/proxy/{endpoint_name}/{sub_path:path}"` behind `@requires_auth`
  (`_proxy_endpoint` at `dashboard.py:467`, `_proxy_endpoint_redirect` at
  `:477`, plus `_legacy_log_service` at `:490`), and a subdomain variant
  (`<name>.proxy.<base>`) handled by `_SubdomainProxyMiddleware.__call__`
  (`dashboard.py:350-371`) which enforces the *same* whole-dashboard policy via
  `_enforce_http_auth` (`dashboard.py:91-118`). Those three handlers are the
  **only** `@requires_auth` routes; everything else is `@public` or an RPC
  `Mount`.
- **RPC auth**: `_DashboardAuthInterceptor` runs `policy.resolve` then
  `authorize_method` on every RPC (`dashboard.py:267,279`); role RBAC in
  `rpc/auth.py` (`authorize_method` at `:89`, `DASHBOARD_READABLE_RPCS`). The
  same interceptor fronts the ControllerService, EndpointService, and
  StatsService mounts (`dashboard.py:437-462`).
- **Token minting**: `controller/auth.py` — `JwtTokenManager.create_token(user_id,
  role, key_id, ttl_seconds)` signs HS256 JWTs `{sub, role, jti, iat, exp}` with a
  persistent HMAC key in `controller_secrets`; `verify()` (`auth.py:212-234`)
  does a **single unqualified** `jwt.decode(token, key, algorithms=["HS256"])`
  (`auth.py:219`) and returns `VerifiedIdentity(user_id, role)`, checking an
  in-memory `jti` revocation set. API-key rows (`auth_api_keys_table`) exist for
  audit/revocation; `ControllerServiceImpl` already holds the manager
  (`self._auth.jwt_manager`, `service.py:1040`) and mints via it in
  `CreateApiKey`/`Login` (`service.py:2590,2629`).
- **Identity**: `rigging.server_auth.VerifiedIdentity(user_id, role)`
  (`server_auth.py:42-48`) — no scope field today.
- **Endpoints**: leased registry (`endpoint_service.py`,
  `projections/endpoints.py`) — `RegisterEndpointRequest{name, address, task_id,
  metadata, lease_duration, …}` fields 1–7 (`controller.proto:374-385`);
  frozen `EndpointRow` (`projections/endpoints.py:43-57`) persisted to
  `endpoints_table` (`schema.py:506-525`) with nullable `lease_deadline_ms`
  (`schema.py:516-521`). Latest migration: `0035_federation_unify.py`.
- **Ingress asymmetry**:
  - **GCP-TPU**: external HTTPS **GCLB → IAP → controller VM:10000**
    (`lib/iris/scripts/iap_gclb.py`). One shared frontend; **one IAP-gated
    backend service per cluster**, keyed by `Host`. A firewall allow-rule admits
    only Google LB ranges (`130.211.0.0/22,35.191.0.0/16`); an optional
    deny-rule blocks direct VM hits. **IAP is a per-backend-service setting.**
  - **CoreWeave (k8s)**: controller Service is **ClusterIP** — no public ingress;
    reached via `kubectl` tunnel.

The endpoint **lease** is the only per-endpoint deadline today, and it governs
*registration lifetime*, not *proxy access*. Auth is dashboard-wide. Those two
gaps are exactly what this design fills.

## Design

### 1. Endpoint access mode

Add an enum to the endpoint proto and persist it on the endpoint row.

```proto
// controller.proto, inside message Controller
enum EndpointAccess {
  ENDPOINT_ACCESS_PRIVATE = 0;  // cluster identity required (the default)
  ENDPOINT_ACCESS_PUBLIC  = 1;  // no auth on /proxy/<name>/*
  ENDPOINT_ACCESS_BEARER  = 2;  // scoped endpoint token (or full cluster identity)
}

message RegisterEndpointRequest {
  // … existing fields 1-7 …
  EndpointAccess access = 8;    // unset = PRIVATE (proto3 zero value)
}
message Endpoint { /* … fields 1-5 … */ EndpointAccess access = 6; }
```

Concrete touch points:

- **Migration `0036_endpoint_access.py`** (next free slot after
  `0035_federation_unify.py`): `ALTER TABLE endpoints ADD COLUMN access
  INTEGER`, nullable, no backfill — exactly the `lease_deadline_ms` pattern
  (`0031_endpoint_lease.py`, `schema.py:516-521`). **NULL decodes as
  `PRIVATE`**, so every pre-migration row keeps today's semantics.
- `endpoints_table` (`schema.py:506`) gains the nullable `access` column;
  frozen `EndpointRow` (`projections/endpoints.py:43`) gains
  `access: EndpointAccess = EndpointAccess.PRIVATE`; the two column mappings —
  `EndpointsProjection.rehydrate` (`projections/endpoints.py:130-138`) and
  `EndpointsProjection.add` (`:261-273`) — read/write it (NULL→PRIVATE on
  read).
- `EndpointServiceImpl.register_endpoint` builds the row from the request
  (`endpoint_service.py:101-109`) — carry `access` through; `list_endpoints`
  (`:165-176`) returns it so the dashboard endpoints tab can show a 🔓/🔑/🔒
  badge.
- Client plumbing: `RegisterEndpointRequest.access` flows through
  `EndpointClient.register` (`cluster/client/endpoint_client.py:98`), the
  `EndpointRegistry` protocol (`client/client.py:315`), and
  `NamespacedEndpointRegistry.register` (`client/client.py:355`), so
  `quick_serve.py` can register `BEARER`.

`PRIVATE` is the safe default and preserves the current behavior exactly.

### 2. Scoped endpoint tokens

Extend the JWT with a scope + audience so one token type can be *either* a full
cluster identity *or* an endpoint-scoped grant. Keep the generic mechanism in
rigging and the policy in iris.

**rigging** (`server_auth.py`) — add one optional, generic field; no iris
knowledge:

```python
@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    user_id: str
    role: str
    # Non-empty ⇒ this token is scoped to a single proxy audience and MUST NOT
    # authorize any RPC. None ⇒ full identity (today's behavior).
    audience: str | None = None
```

This is the **only** rigging change. The authenticator stack and
`RequestAuthPolicy.resolve` (`server_auth.py:512-530`) pass `VerifiedIdentity`
through opaquely and need no edits. Every other identity constructor —
`StaticTokenVerifier` (`server_auth.py:150`), `GcpAccessTokenVerifier` (`:184`),
`IapIdTokenVerifier` (`:218`), `IapAssertionVerifier` (`:296`), and
`LOOPBACK_IDENTITY` (`:54`) — leaves the new field at its `None` default, so a
non-JWT identity (IAP assertion, loopback, static token) can never present as
scoped.

**iris** (`controller/auth.py`) — mint and interpret:

```python
ENDPOINT_TOKEN_ROLE = "endpoint"          # a role with zero RPC authority
DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS = 3600 # 1h; caller may request less/more

def create_endpoint_token(self, endpoint_name: str, key_id: str,
                          ttl_seconds: int = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS) -> str:
    now = time.time()
    payload = {
        "sub": f"endpoint:{endpoint_name}",
        "role": ENDPOINT_TOKEN_ROLE,
        "aud": endpoint_name,             # canonical WIRE name, e.g. "/serve/foo" — see below
        "scope": "proxy",                 # explicit: proxy-only, no RPC
        "jti": key_id,
        "iat": int(now), "exp": int(now + ttl_seconds),
    }
    return jwt.encode(payload, self._signing_key, algorithm="HS256")
```

**`verify()` must disable PyJWT's implicit `aud` validation.** The current
decode (`auth.py:219`) is `jwt.decode(token, self._signing_key,
algorithms=["HS256"])` with no `audience=`. PyJWT 2.x defaults
`verify_aud=True` and its `_validate_aud` raises
`InvalidAudienceError("Invalid audience")` whenever the token carries a
non-empty `aud` claim but the caller passed no `audience=` (verified against
the pinned PyJWT 2.12.1; `InvalidAudienceError` ⊂ `InvalidTokenError`, which
`auth.py:222-223` maps to `ValueError("Invalid token: …")`). Since ONE shared
`verify()` must accept both full-identity tokens (no `aud`) and scoped tokens
(with `aud`), every scoped token would fail verification **before** the
`scope == "proxy"` check. The fix: decode with `options={"verify_aud": False}`
and read `aud`/`scope` manually:

```python
def verify(self, token: str) -> VerifiedIdentity:
    try:
        # verify_aud=False: this one verify() accepts both full-identity
        # tokens (no aud) and endpoint-scoped tokens (aud set). PyJWT 2.x
        # otherwise rejects any aud-bearing token when decode() gets no
        # audience= (InvalidAudienceError). The audience check happens at
        # the proxy, against the endpoint the request names.
        payload = jwt.decode(token, self._signing_key, algorithms=["HS256"],
                             options={"verify_aud": False})
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc
    # … existing jti revocation + touch …
    audience = payload.get("aud") if payload.get("scope") == "proxy" else None
    return VerifiedIdentity(user_id=payload["sub"],
                            role=payload.get("role", "user"),
                            audience=audience)
```

Trade-off, stated for the record: we could instead use a **private claim name**
(e.g. `endpoint_aud`) and leave the registered `aud` claim — and PyJWT's
default validation — untouched. Using registered `aud` with `verify_aud=False`
is fine (the audience check is *ours*, enforced at the proxy against a
per-request value PyJWT can't know), and keeps the claim recognizable to JWT
tooling; we just accept that this `verify()` never delegates audience
enforcement to the library.

**`aud` namespace — decided: the canonical wire name.** Mint and compare must
use ONE namespace. Two exist (see grounding): the `/`-separated wire name the
registry stores (`/serve/foo`) and the `.`-separated encoded form URLs carry
(`serve.foo`). Decision: **`aud` binds to the endpoint row's registered
`name`** — the wire name — and the encoded form is normalized to it **once at
the boundary**, by the same resolver that already owns the decode (§3). Mint
side: `MintEndpointToken` resolves the endpoint row and sets
`aud = row.name`. Compare side: `_authorize_proxy` resolves the row from the
request's `encoded_name` (path segment or subdomain label, §3/§5) and compares
`identity.audience == row.name`. The dotted encoded form never appears inside
a token. This matches what the code already treats as primary: `dispatch`
decodes `.`→`/` and tries the slash-prefixed wire name first
(`endpoint_proxy.py:256-257`), and quick-serve's default name contains slashes
(`/serve/<job>`), so binding to the encoded form would also break the moment
two encodings of one name disagree.

Reuse the existing `jti` revocation set + an `auth_api_keys` row (as the
`CreateApiKey` handler does, `service.py:2596-2629`) so a leaked endpoint token
can be revoked like any API key; the `exp` deadline is the primary bound
(rjpower's "time bound").

**Minting RPC** on `ControllerService`:

```proto
message MintEndpointTokenRequest {
  string endpoint_name = 1;     // wire name, e.g. "/serve/foo" (or endpoint_id)
  iris.time.Duration ttl = 2;   // clamped to a max (e.g. 24h)
}
message MintEndpointTokenResponse { string token = 1; iris.time.Timestamp expires_at = 2; }
rpc MintEndpointToken(...) returns (...);
```

Placement: `ControllerService`, not `EndpointService` — the handler needs
`self._auth.jwt_manager` and `create_api_key`, which live on
`ControllerServiceImpl` next to the other key-minting RPCs
(`service.py:1040,2590-2629`); `EndpointService` is a pure registry surface.
(Both mounts share the same auth interceptor, `dashboard.py:437-454`, so the
security posture is identical either way.)

Authz: the endpoint's owner is derived from the row it resolves to —
`EndpointRow.task_id` is a `JobName` (`/user/job/0`) whose first segment is the
submitting user (`JobName.user`, `cluster/types.py:224`). The handler resolves
the row (same resolver as the proxy, so name-form handling can't diverge), then
`authorize_resource_owner(row.task_id.user)` (`rpc/auth.py:120`), mirroring
`_authorize_job_owner` (`service.py:1067-1073`): only the owning user or an
admin may mint. Note one name can map to several rows
(`projections/endpoints.py:106-107`); authz runs against the row the resolver
picks — all rows for one quick-serve name share a task, so this is stable.

### 3. Proxy auth — split the arms out of `@requires_auth`

This is the crux: the over-grant comes from the proxy sharing the whole-dashboard
policy. Replace the blanket `@requires_auth` on the proxy handlers with a
dedicated resolver keyed on the endpoint's access mode.

**One lookup for access + address.** Add
`EndpointServiceImpl.resolve_endpoint_row(encoded_name) -> ResolvedEndpoint |
None` that owns the decode `dispatch` performs today (`.`→`/`, try
slash-prefixed then bare — `endpoint_proxy.py:256-257`) and returns the full
row view (`name`, `address`, `access`). Without this decode, a naive
`access_for(encoded_name)` string lookup would miss every slash-containing
endpoint (quick-serve's default `/serve/<job>` arrives as `serve.foo` via
`proxy_path`, `rigging/connect.py:132-134`) — treating it as unknown/PRIVATE
and wrongly rejecting BEARER/PUBLIC off-cluster requests. Returning the whole
row also means authorization and forwarding read the *same* resolution: hoist
the decode out of `dispatch` and pass the resolved address in, so "which
endpoint does this request name" is answered exactly once. `/system/`
endpoints come from the in-memory map (`endpoint_service.py:58`) which has no
access column — they resolve with `access=PRIVATE`, always.

```python
# endpoint_service.py
def resolve_endpoint_row(self, encoded_name: str) -> ResolvedEndpoint | None:
    slashed = encoded_name.replace(".", "/")
    for name in (f"/{slashed}", slashed):
        row = self._endpoints.resolve(name)
        if row is not None:
            return ResolvedEndpoint(name=row.name, address=row.address, access=row.access)
        address = self._system_endpoints.get(name)
        if address is not None:
            return ResolvedEndpoint(name=name, address=address, access=EndpointAccess.PRIVATE)
    return None

# dashboard.py — replaces @requires_auth on _proxy_endpoint /
# _proxy_endpoint_redirect and the _enforce_http_auth call inside
# _SubdomainProxyMiddleware.__call__ (dashboard.py:360-362).
async def _authorize_proxy(scope, receive, send, *, encoded_name, policy,
                           endpoint_service) -> ResolvedEndpoint | None:
    row = endpoint_service.resolve_endpoint_row(encoded_name)
    access = row.access if row is not None else EndpointAccess.PRIVATE  # unknown → PRIVATE
    if access is EndpointAccess.PUBLIC:
        return row
    token = extract_bearer_token(_scope_headers(scope), cookie_name=SESSION_COOKIE)
    identity = policy.resolve(token, client_address=..., headers=...)  # 401 on fail
    if access is EndpointAccess.BEARER:
        # A scoped token must match THIS endpoint's wire name; a full
        # cluster identity also passes.
        if identity.audience is not None and identity.audience != row.name:
            deny(403); return None
        return row
    # PRIVATE: must be a full cluster identity, never a scoped token.
    if identity.audience is not None:
        deny(403); return None
    return row  # None → dispatch 404s, as today
```

Both `@requires_auth` proxy sites move to this helper: the path-style handlers
`_proxy_endpoint` / `_proxy_endpoint_redirect` (`dashboard.py:467,477`) **and**
`_SubdomainProxyMiddleware.__call__`, which today calls `_enforce_http_auth`
(`dashboard.py:360-362`) — the subdomain arm must apply the identical logic or
`<name>.proxy.<base>` becomes a bypass.

**Close the over-grant on the RPC side** — a scoped identity is barred from
*every* RPC, in one place. `authorize_method` (`rpc/auth.py:89`) is the shared
choke point: `_DashboardAuthInterceptor` calls it on every authenticated RPC
(`dashboard.py:267,279`), and that interceptor fronts the ControllerService,
EndpointService, and StatsService mounts alike (`dashboard.py:437-462`).

```python
# rpc/auth.py::authorize_method
def authorize_method(identity, method_name):
    if identity.audience is not None:
        raise ConnectError(Code.PERMISSION_DENIED,
                           "endpoint-scoped token cannot call control RPCs")
    if identity.role == DASHBOARD_ROLE and method_name not in DASHBOARD_READABLE_RPCS:
        raise ConnectError(Code.PERMISSION_DENIED, ...)
```

**And on the remaining HTTP arm.** `_enforce_http_auth` (`dashboard.py:91-118`)
only *authenticates* — with `audience` populated, a scoped token would pass it
and reach any `@requires_auth` route it still guards. Concretely
`_legacy_log_service` (`dashboard.py:490`) stays on `@requires_auth` and
proxies to the log server: without a check, a model-inference token could read
cluster logs. Add a scoped-identity deny to `_enforce_http_auth` (identity
resolved with `audience is not None` → 403), making `_authorize_proxy` the
*only* place a scoped token is ever accepted — which also future-proofs any
`@requires_auth` route added later.

Net effect: an endpoint token reaches **only** `/proxy/<its-endpoint>/…` and
nothing else — not another endpoint, not the RPC surface, not the SPA, not the
legacy log route. The existing upstream `Authorization`-stripping stays, so the
controller token is consumed at the controller and never forwarded to vLLM.

**OpenAI-client ergonomics.** OpenAI-compatible clients already send
`Authorization: Bearer <api_key>`. So the scoped endpoint token *is* the
`api_key`: point the SDK at `https://<host>/proxy/serve.<ep>/v1` with
`api_key=<token>` and it works with no client changes. This is why `BEARER` is
the natural default for the datagen use case.

### 4. Public ingress per provider — "open just /proxy"

> rjpower: *"Can we open up just the /proxy route as un-authenticated via the IAP
> LB route?"* — **Yes, and the controller needs no firewall/IAP authority.**

**GCP-TPU.** IAP is configured **per backend service**. Add a *second* backend
service pointing at the **same NEG / controller VM**, with **IAP disabled**, and a
**URL-map path matcher** that routes `/proxy/*` (and `*.proxy.<host>` if we keep
the subdomain form) to it. The default backend (IAP-gated) keeps `/`, `/auth/*`,
the RPC mounts, everything else. So:

```
                       ┌─ path /proxy/*  → be-proxy   (IAP OFF) ─┐
client → GCLB (:443) → URL map                                   ├→ NEG → controller VM:10000
                       └─ default        → be-main    (IAP ON) ──┘
```

- Implemented as a new **operator-run** `iap_gclb.py` stage (e.g. `public-proxy`,
  alongside the existing `address`/`cert`/`backend`/`iap`/`frontend`/`route`/
  `grant`/`firewall` stages): create `be-proxy` on the existing NEG with
  `--no-iap`, add a path-matcher rule to the shared URL map. Idempotent, like
  the other stages.
- The controller code is unchanged; it simply now receives unauthenticated
  `/proxy/*` requests and applies §3. `PRIVATE`/`BEARER` endpoints stay protected
  by the controller's own check; `PUBLIC` are intentionally open.
- The existing firewall allow-rule (Google LB ranges only) still blocks direct
  VM hits, so nothing bypasses the LB. **The controller never touches firewall or
  IAP rules** — the admin runs the script once. This directly satisfies "expose
  the proxy route only… don't give the controller firewall access."
- Caveat to document: removing IAP on `/proxy/*` means the controller's
  `_authorize_proxy` is now the *sole* gate for that path. That is the whole
  point, and it is auditable in one function. The legacy
  `/finelog.logging.LogService/*` route is **not** under `/proxy/*` and stays
  IAP-gated.

**CoreWeave (k8s).** No IAP layer. Two documented options; recommend the first:
- **Path-restricted Ingress** exposing only `/proxy` (and `/health`) on the
  controller Service, e.g. an ingress rule with `path: /proxy`. RPC/SPA stay
  ClusterIP-internal. The controller's `_authorize_proxy` gates it.
- **`type: LoadBalancer` Service** on the controller port — simpler but exposes
  the whole origin; only acceptable when `auth.provider` is set (never
  null-auth), since the JWT check is then the only gate for RPCs too.

Ship a manifest/helm snippet + doc in `docs/coreweave.md`; the controller code is
identical across providers — only the ingress object differs.

### 5. Fallback — token-in-URL (rjpower's `/proxy/private/{hmac}`)

For transports that can't set an `Authorization` header (a raw browser link, a
webhook, a client that hard-codes headers), support the token as a **path
segment**:

```
/proxy/t/<token>/<encoded_name>/<sub_path>
```

The route handler lifts `<token>` from the path, validates it exactly like the
header case, then dispatches with the token stripped from the forwarded path.
The `<encoded_name>` path segment is dotted (`serve.foo`); it goes through the
same `resolve_endpoint_row` normalization as §3, and the `aud` compare runs
against the resolved row's wire name (`/serve/foo`) — consistent with the §2
namespace decision, so the two carriers cannot drift. Same JWT, different
carrier — no second credential system. Because OpenAI clients *do* send
`Authorization`, the header form (`BEARER`) is primary and this is a documented
fallback only.

## Implementation plan (spiral — each stage independently testable)

1. **proto** — `EndpointAccess` enum + `RegisterEndpointRequest.access = 8` +
   `Endpoint.access = 6` + `MintEndpointToken` RPC/messages; regenerate
   (`scripts/generate_protos.py`).
2. **schema** — migration `0036_endpoint_access.py` (nullable `access` column,
   NULL→PRIVATE, per the `lease_deadline_ms` pattern); `EndpointRow.access`;
   `EndpointsProjection.rehydrate`/`add` column mapping;
   `resolve_endpoint_row` on `EndpointServiceImpl` (hoisting the decode out of
   `EndpointProxy.dispatch`).
3. **rigging** — `VerifiedIdentity.audience` (default `None`); unit test that a
   scoped identity is distinguishable and that non-JWT constructors leave it
   `None`.
4. **auth** — `create_endpoint_token`; `verify()` decodes with
   `options={"verify_aud": False}` and populates `audience` when
   `scope == "proxy"`; regression test: a full-identity token AND an
   aud-bearing scoped token both round-trip through the one `verify()`;
   revocation via existing `jti` path.
5. **controller service** — `mint_endpoint_token` handler on
   `ControllerServiceImpl` (resolve row → `authorize_resource_owner(
   row.task_id.user)` → `create_api_key` audit row → mint with
   `aud = row.name`); persist `access` in `register_endpoint`.
6. **dashboard** — `_authorize_proxy` replacing `@requires_auth` on the two
   proxy handlers + inside `_SubdomainProxyMiddleware.__call__`;
   scoped-identity deny in `_enforce_http_auth` (guards
   `_legacy_log_service` and future `@requires_auth` routes); RPC deny for
   scoped identities in `authorize_method`; token-in-URL route.
7. **serve** — plumb `access` through `EndpointRegistry` /
   `NamespacedEndpointRegistry` / `EndpointClient.register`;
   `quick_serve.py` registers with `access=BEARER`
   (`quick_serve.py:287`); the CLI mints the token client-side after
   `_wait_for_endpoint` (`quick_serve_cli.py:100`) with the *user's* JWT and
   prints the off-cluster `base_url` + `api_key`
   (`quick_serve_cli.py:244`); drop the pinggy wrapper downstream. (In-job
   minting won't work as-is: in-job clients authenticate with the injected
   worker token — `composer.py:238`, `WORKER_USER = "system:worker"`,
   `auth.py:38` — which fails `authorize_resource_owner` against the endpoint's
   owning user; see open question 2.)
8. **infra** — `iap_gclb.py` `public-proxy` stage (IAP-free backend + `/proxy/*`
   URL-map route) + doc in `docs/iap-gclb.md`; CoreWeave path-restricted Ingress
   manifest + doc in `docs/coreweave.md`.
9. **docs + tests** — see below.

## Security considerations

- **No RPC over-grant**: a scoped identity (`audience != None`) is denied every
  RPC in `authorize_method` — the choke point every RPC mount's interceptor
  routes through — and denied on `@requires_auth` HTTP routes in
  `_enforce_http_auth`. A scoped token is accepted *only* by
  `_authorize_proxy`. New RPCs and new auth-gated routes are denied to scoped
  tokens by default.
- **Endpoint binding**: the token's `aud` binds to the canonical wire name (the
  row's registered `name`, e.g. `/serve/foo`), not the address, so a
  re-registered endpoint (address change / retry) keeps working while the token
  still can't reach a *different* endpoint. Both mint and compare sides read
  the same `row.name`, with the dotted URL form normalized at the boundary.
- **Deadline + revocation**: `exp` is the primary bound; `jti` revocation reuses
  the API-key revocation set so a leaked token can be killed immediately. The
  token deadline and the endpoint lease are **independent** bounds: a token can
  outlive the lease (the proxy then fails to resolve the row and treats the
  name as unknown/PRIVATE) and a lease can outlive the token (verify rejects on
  `exp`) — whichever expires first cuts access.
- **Upstream isolation**: the proxy still strips `Authorization` client→upstream,
  so the controller token never reaches vLLM; vLLM stays keyless behind the proxy.
- **PUBLIC is opt-in and per-endpoint**: default is `PRIVATE` (including NULL
  rows from before the migration and all `/system/` endpoints); nothing becomes
  public without an explicit `access=PUBLIC` at registration *and* the operator
  standing up the IAP-free `/proxy` route.
- **Ingress blast radius**: opening `/proxy/*` past IAP exposes only that path;
  the URL map keeps `/`, `/auth/*`, RPC mounts, and the legacy LogService route
  IAP-gated. Rate-limiting stays at the Cloudflare WAF layer.

## Testing

- `test_endpoint_proxy.py` — PUBLIC (no token) allowed; BEARER requires a
  matching scoped token; scoped token for endpoint A rejected on endpoint B;
  PRIVATE rejects a scoped token but accepts a full identity; a
  slash-containing endpoint name (`/serve/foo` ⇄ `serve.foo`) authorizes and
  resolves through both the path-style and subdomain arms (regression for the
  encode/decode namespace bug).
- `test_auth.py` — `create_endpoint_token`/`verify` round-trip sets `audience`;
  a full-identity token (no `aud`) still verifies through the same `verify()`
  (regression for PyJWT `verify_aud`); scoped identity denied on a
  representative RPC via `authorize_method`; revocation by `jti` rejects.
- `test_dashboard.py` — path-style + subdomain-style proxy both honor access
  mode; scoped token rejected on `_legacy_log_service` (and any
  `@requires_auth` route); token-in-URL fallback validates, normalizes the
  encoded name, and strips the token from the forwarded path.
- `test_api_keys.py` — `MintEndpointToken` owner/admin authz (owner =
  `row.task_id.user`; non-owner denied; worker-token caller denied); TTL clamp.
- migration test — pre-0036 rows load with `access=PRIVATE`.

## Decisions made during review

- **`aud` namespace**: canonical wire name (`row.name`, `/`-separated) on both
  mint and compare sides; encoded (`.`-separated) forms normalized once by
  `resolve_endpoint_row` (§2, §3, §5).
- **`verify()` and `aud`**: keep the registered `aud` claim, decode with
  `verify_aud=False`, enforce audience at the proxy (§2).
- **Access + address resolve in one lookup** returning the row, so
  authorization and forwarding cannot disagree (§3).
- **The public `/proxy` opening is the default on both provider arms** (user
  decision). GCP: `iap_gclb.py deploy` runs the `public-proxy` stage by default
  (idempotent; `--no-public-proxy` opts out). CoreWeave: the `/proxy` Ingress is
  part of `start_controller` — config-driven via `controller.coreweave`
  (`public_proxy_host`, `ingress_class`, `tls_secret`, `cluster_issuer`); an empty
  host leaves it ClusterIP-only. It publishes only `/proxy`; the controller's
  per-endpoint auth is the sole gate (no IAP layer on CoreWeave).
- **CoreWeave uses Traefik + cert-manager, verified against CKS docs** (not
  nginx — that was a wrong first assumption, since fixed). CKS ships no ingress
  controller/issuer, so `scripts/install_traefik_proxy.py` (operator-run, `--apply`-gated,
  mirrors `install_kueue.py`) installs `coreweave/traefik` + `coreweave/cert-manager`
  + HTTP-01 Let's Encrypt ClusterIssuers. Traefik's LB gets a stable
  `*.<ORG-ID>-<CLUSTER>.coreweave.app` FQDN (external-hostname); a custom `oa.dev`
  host CNAMEs to it. CoreWeave's bundled issuers are DNS-01 for `*.coreweave.app`
  only, so custom hosts need the HTTP-01 issuers `install_traefik_proxy.py` creates.
- **Minting stays a separate RPC, not folded into `RegisterEndpoint`** (user
  decision). `MintEndpointToken` keeps its owner-*user* authorization
  (`authorize_resource_owner(row.task_id.user)`); it is not merged into the
  leased, universal `RegisterEndpoint` path. Rationale: registration is a
  renewing, general service-discovery call made in-job under the worker token
  (`system:worker`), so folding minting in would either return a secret on every
  renewal or require first-registration idempotency, and would couple credential
  issuance to discovery. Keeping mint standalone preserves rotation, re-mint
  after TTL, one-token-per-consumer, and PRIVATE→BEARER upgrades. Consequence:
  the in-job process does **not** self-mint; a human/CLI holding the launching
  user's JWT mints and injects the token (see the serve flow in §Implementation
  step 7). Broadening mint authz to accept the owning task's worker token was
  considered and declined for now.

## Open questions for review

1. **Scope carrier**: `audience` claim as proposed, or a richer `scopes:
   frozenset[str]` on `VerifiedIdentity`? Single `audience` is enough for
   one-endpoint-per-token; a set generalizes to "this token for these N
   endpoints" if we ever want it.
2. **Subdomain form under the public route**: keep `*.proxy.<host>` as a public
   arm too (needs a wildcard cert + URL-map host rule), or restrict the public
   ingress to path-style `/proxy/*` only initially? I lean path-only first
   (shipped path-only).
