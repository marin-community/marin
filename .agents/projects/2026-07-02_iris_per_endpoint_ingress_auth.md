# Per-endpoint auth-gated public ingress on the Iris controller

> Status: **design — for review** (GitHub issue
> [#6847](https://github.com/marin-community/marin/issues/6847); weaver #4).
> Follow-up to #6545 (public-endpoint discussion) and #6556 (marin-serve).

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

- **Proxy**: `controller/endpoint_proxy.py` — `EndpointProxy.dispatch(request, *,
  encoded_name, sub_path, proxy_prefix)`. Resolves the wire name (`.`→`/`) via a
  `resolve: (name) -> address | None` callable, streams both ways, and **strips
  `Authorization`/`Cookie` on the client→upstream hop** (credential-leak guard).
- **Route wiring**: `controller/dashboard.py` mounts `PROXY_ROUTE =
  "/proxy/{endpoint_name}/{sub_path:path}"` behind `@requires_auth`, plus a
  subdomain variant (`<name>.proxy.<base>`) handled by `_SubdomainProxyMiddleware`
  which enforces the *same* whole-dashboard `policy.resolve`.
- **RPC auth**: `_DashboardAuthInterceptor` runs `policy.resolve` on every RPC;
  role RBAC in `rpc/auth.py` (`authorize_method`, `DASHBOARD_READABLE_RPCS`).
- **Token minting**: `controller/auth.py` — `JwtTokenManager.create_token(user_id,
  role, key_id, ttl_seconds)` signs HS256 JWTs `{sub, role, jti, iat, exp}` with a
  persistent HMAC key in `controller_secrets`; `verify()` returns
  `VerifiedIdentity(user_id, role)` and checks an in-memory `jti` revocation set.
  API-key rows (`auth_api_keys_table`) exist for audit/revocation.
- **Identity**: `rigging.server_auth.VerifiedIdentity(user_id, role)` — no scope
  field today.
- **Endpoints**: leased registry (`endpoint_service.py`,
  `projections/endpoints.py`) — `RegisterEndpointRequest{name, address, task_id,
  metadata, lease_duration, …}`; `EndpointRow` persisted with `lease_deadline`.
- **Ingress asymmetry**:
  - **GCP-TPU**: external HTTPS **GCLB → IAP → controller VM:10000**
    (`scripts/iap_gclb.py`). One shared frontend; **one IAP-gated backend service
    per cluster**, keyed by `Host`. A firewall allow-rule admits only Google LB
    ranges (`130.211.0.0/22,35.191.0.0/16`); an optional deny-rule blocks direct
    VM hits. **IAP is a per-backend-service setting.**
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
  ENDPOINT_ACCESS_UNSPECIFIED = 0;  // treated as PRIVATE
  ENDPOINT_ACCESS_PRIVATE = 1;      // cluster identity required (today's behavior)
  ENDPOINT_ACCESS_PUBLIC  = 2;      // no auth on /proxy/<name>/*
  ENDPOINT_ACCESS_BEARER  = 3;      // scoped endpoint token (or full cluster identity)
}

message RegisterEndpointRequest {
  // … existing fields 1-7 …
  EndpointAccess access = 8;        // default UNSPECIFIED → PRIVATE
}
message Endpoint { /* … */ EndpointAccess access = 6; }
```

- Persist as an `access` column on the endpoints table (**new migration**,
  `projections/endpoints.py::EndpointRow.access: EndpointAccess`), defaulting to
  `PRIVATE` so every existing/legacy registration keeps today's semantics.
- Plumb through `EndpointRegistry.register(..., access=...)` and the leased
  `EndpointClient`, so `quick_serve.py` can register `BEARER`.
- Return `access` in `ListEndpoints` / dashboard so the endpoints tab can show a
  🔓/🔑/🔒 badge.

`PRIVATE` is the safe default and preserves the current behavior exactly.

### 2. Scoped endpoint tokens

Extend the JWT with a scope + audience so one token type can be *either* a full
cluster identity *or* an endpoint-scoped grant. Keep the generic mechanism in
rigging and the policy in iris.

**rigging** (`server_auth.py`) — add optional, generic fields; no iris knowledge:

```python
@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    user_id: str
    role: str
    # Non-empty ⇒ this token is scoped to a single proxy audience and MUST NOT
    # authorize any RPC. Empty ⇒ full identity (today's behavior).
    audience: str | None = None
```

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
        "aud": endpoint_name,             # binds the token to one logical endpoint
        "scope": "proxy",                 # explicit: proxy-only, no RPC
        "jti": key_id,
        "iat": int(now), "exp": int(now + ttl_seconds),
    }
    return jwt.encode(payload, self._signing_key, algorithm="HS256")
```

`verify()` populates `VerifiedIdentity.audience` from the `aud` claim when
`scope == "proxy"`. Reuse the existing `jti` revocation set + `auth_api_keys`
row so a leaked endpoint token can be revoked like any API key; the `exp`
deadline is the primary bound (rjpower's "time bound").

**Minting RPC** on `ControllerService` (owner/admin only):

```proto
message MintEndpointTokenRequest {
  string endpoint_name = 1;     // or endpoint_id
  iris.time.Duration ttl = 2;   // clamped to a max (e.g. 24h)
}
message MintEndpointTokenResponse { string token = 1; iris.time.Timestamp expires_at = 2; }
rpc MintEndpointToken(...) returns (...);
```

Authz: only the endpoint's **owning task's user** or an admin may mint (reuse
`authorize_resource_owner`). The controller "mints it from its JWT token with a
time bound" exactly as rjpower described — same signing key, new claim set.

### 3. Proxy auth — split the arms out of `@requires_auth`

This is the crux: the over-grant comes from the proxy sharing the whole-dashboard
policy. Replace the blanket `@requires_auth` on the proxy handlers with a
dedicated resolver keyed on the endpoint's access mode.

```python
# dashboard.py — new helper, replaces @requires_auth on _proxy_endpoint /
# _proxy_endpoint_redirect and used inside _SubdomainProxyMiddleware.
async def _authorize_proxy(scope, receive, send, *, encoded_name, policy,
                          endpoint_service) -> bool:
    access = endpoint_service.access_for(encoded_name)   # PRIVATE if unknown
    if access is EndpointAccess.PUBLIC:
        return True
    token = extract_bearer_token(_scope_headers(scope), cookie_name=SESSION_COOKIE)
    identity = policy.resolve(token, client_address=..., headers=...)  # 401 on fail
    if access is EndpointAccess.BEARER:
        # A scoped token must match THIS endpoint; a full identity also passes.
        if identity.audience is not None and identity.audience != endpoint_wire_name:
            deny(403); return False
        return True
    # PRIVATE: must be a full cluster identity, never a scoped token.
    if identity.audience is not None:
        deny(403); return False
    return True
```

And **close the over-grant on the RPC side** — a scoped identity is barred from
*every* RPC, in one place:

```python
# rpc/auth.py::authorize_method (called by _DashboardAuthInterceptor)
def authorize_method(identity, method_name):
    if identity.audience is not None:
        raise ConnectError(Code.PERMISSION_DENIED,
                           "endpoint-scoped token cannot call control RPCs")
    if identity.role == DASHBOARD_ROLE and method_name not in DASHBOARD_READABLE_RPCS:
        raise ConnectError(Code.PERMISSION_DENIED, ...)
```

Net effect: an endpoint token reaches **only** `/proxy/<its-endpoint>/…` and
nothing else — not another endpoint, not the RPC surface, not the SPA. The
existing upstream `Authorization`-stripping stays, so the controller token is
consumed at the controller and never forwarded to vLLM.

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

- Implemented as a new **operator-run** `iap_gclb.py` stage (e.g. `public-proxy`
  / extend `route`): create `be-proxy` on the existing NEG with `--no-iap`, add a
  path-matcher rule to the shared URL map. Idempotent, like the other stages.
- The controller code is unchanged; it simply now receives unauthenticated
  `/proxy/*` requests and applies §3. `PRIVATE`/`BEARER` endpoints stay protected
  by the controller's own check; `PUBLIC` are intentionally open.
- The existing firewall allow-rule (Google LB ranges only) still blocks direct
  VM hits, so nothing bypasses the LB. **The controller never touches firewall or
  IAP rules** — the admin runs the script once. This directly satisfies "expose
  the proxy route only… don't give the controller firewall access."
- Caveat to document: removing IAP on `/proxy/*` means the controller's
  `_authorize_proxy` is now the *sole* gate for that path. That is the whole
  point, and it is auditable in one function.

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
header case (`scope=proxy`, `aud == endpoint`), then dispatches with the token
stripped from the forwarded path. Same JWT, different carrier — no second
credential system. Because OpenAI clients *do* send `Authorization`, the header
form (`BEARER`) is primary and this is a documented fallback only.

## Implementation plan (spiral — each stage independently testable)

1. **proto** — `EndpointAccess` enum + `RegisterEndpointRequest.access` +
   `Endpoint.access` + `MintEndpointToken` RPC/messages; regenerate
   (`scripts/generate_protos.py`).
2. **schema** — migration adding `access` to the endpoints table;
   `EndpointRow.access`; `EndpointsProjection` read/write; `access_for(name)`
   resolver alongside `resolve_endpoint`.
3. **rigging** — `VerifiedIdentity.audience`; unit test that a scoped identity is
   distinguishable.
4. **auth** — `create_endpoint_token`; `verify()` populates `audience`;
   revocation via existing `jti` path.
5. **controller service** — `mint_endpoint_token` handler (owner/admin authz);
   persist `access` in `register_endpoint`.
6. **dashboard** — `_authorize_proxy` replacing `@requires_auth` on the two proxy
   handlers + inside `_SubdomainProxyMiddleware`; RPC deny for scoped identities
   in `authorize_method`; token-in-URL route.
7. **serve** — `quick_serve.py` registers with `access=BEARER`, mints a token,
   prints the off-cluster `base_url` + `api_key`; drop the pinggy wrapper
   downstream.
8. **infra** — `iap_gclb.py` `public-proxy` stage (IAP-free backend + `/proxy/*`
   URL-map route) + doc in `docs/iap-gclb.md`; CoreWeave path-restricted Ingress
   manifest + doc in `docs/coreweave.md`.
9. **docs + tests** — see below.

## Security considerations

- **No RPC over-grant**: a scoped identity (`audience != None`) is denied every
  RPC in `authorize_method` — the single choke point both HTTP and RPC layers
  already share. New RPCs are denied by default to scoped tokens.
- **Endpoint binding**: the token's `aud` binds to the *logical name*, not the
  address, so a re-registered endpoint (address change / retry) keeps working
  while the token still can't reach a *different* endpoint.
- **Deadline + revocation**: `exp` is the primary bound; `jti` revocation reuses
  the API-key revocation set so a leaked token can be killed immediately.
- **Upstream isolation**: the proxy still strips `Authorization` client→upstream,
  so the controller token never reaches vLLM; vLLM stays keyless behind the proxy.
- **PUBLIC is opt-in and per-endpoint**: default is `PRIVATE`; nothing becomes
  public without an explicit `access=PUBLIC` at registration *and* the operator
  standing up the IAP-free `/proxy` route.
- **Ingress blast radius**: opening `/proxy/*` past IAP exposes only that path;
  the URL map keeps `/`, `/auth/*`, and RPC mounts IAP-gated. Rate-limiting stays
  at the Cloudflare WAF layer.

## Testing

- `test_endpoint_proxy.py` — PUBLIC (no token) allowed; BEARER requires a
  matching scoped token; scoped token for endpoint A rejected on endpoint B;
  PRIVATE rejects a scoped token but accepts a full identity.
- `test_auth.py` — `create_endpoint_token`/`verify` round-trip sets `audience`;
  scoped identity denied on a representative RPC via `authorize_method`;
  revocation by `jti` rejects.
- `test_dashboard.py` — path-style + subdomain-style proxy both honor access
  mode; token-in-URL fallback validates and strips.
- `test_api_keys.py` — `MintEndpointToken` owner/admin authz; TTL clamp.

## Open questions for review

1. **Scope carrier**: `audience` claim as proposed, or a richer `scopes:
   frozenset[str]` on `VerifiedIdentity`? Single `audience` is enough for
   one-endpoint-per-token; a set generalizes to "this token for these N
   endpoints" if we ever want it.
2. **Who mints in the serve flow**: `quick_serve.py` (task identity) mint at
   registration and print, vs. a `iris serve token <ep>` CLI the user runs. I'd
   do both — auto-mint on `--access=bearer`, plus the CLI for rotation.
3. **Subdomain form under the public route**: keep `*.proxy.<host>` as a public
   arm too (needs a wildcard cert + URL-map host rule), or restrict the public
   ingress to path-style `/proxy/*` only initially? I lean path-only first.
4. **CoreWeave default**: path-restricted Ingress (recommended) vs. plain
   `LoadBalancer` — do we want a managed cert / DNS story there now or later?
