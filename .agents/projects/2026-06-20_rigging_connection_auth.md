# `rigging.connect` — a composable connection & auth library

> Status: design proposal for review (weaver #236). Revised after a `codex`
> critical pass — see [§Codex review](#codex-review-and-how-the-design-changed)
> for every finding and how the design changed. The headline change: the
> controller already ships a generic reverse proxy and an endpoint resolver, so
> this is **much smaller** than a first draft assumed — a client-side
> transport+auth resolver, not a new proxy or a full "hop algebra".

## Problem

Reaching a Marin backend service from a client (laptop CLI, notebook, ops
script) is ad-hoc and per-service. Each service re-invents the same three moving
parts — *transport* (how bytes reach the box), *auth* (how the request proves
who it is), and *routing* (straight there, or through a proxy) — and bakes them
into a bespoke client constructor.

Concretely, today:

- **Iris** has its *own* tunnel implementations: `_gcp_tunnel` (gcloud SSH `-L`)
  in `backends/gcp/controller.py`, `kubectl.port_forward` in
  `backends/k8s/controller.py`, and `nullcontext` for manual/local — exposed
  through `ControllerProvider.tunnel(address) -> AbstractContextManager[str]`.
  The CLI enters that context lazily in `require_controller_url`
  (`cli/connect.py`) and threads a `TokenProvider` into `client_interceptors`
  to attach the JWT.
- **finelog**'s CLI has a *different* path: it builds a `rigging.tunnel`
  `TunnelTarget` from its own config (`deploy/cli.py:_tunnel_target`), calls
  `open_tunnel`, and points `LogClient.connect(url)` at the local port.
- **rigging** already owns `open_tunnel` (`GcpSshForwardTarget` /
  `K8sPortForwardTarget` → `http://127.0.0.1:<port>`), the `connect-python`
  interceptor utilities (`rpc.py`), and name→config resolution
  (`config_discovery.resolve_cluster_config`). Its own docstring already says
  *"iris's k8s port-forward should migrate here too."*

So there are **three parallel tunnel implementations** and **two unrelated ways
to attach auth**, and neither composes. The sharpest symptom the user named:

> To set the status text for a job we call `report_task_status_text` from the
> **Iris** client, even though that data only goes to the **finelog** log
> server.

That indirection exists because a user client has **no first-class way to obtain
an authenticated connection to a service behind a cluster** without spinning up
the whole `IrisClient`. `RemoteClusterClient` builds its `LogClient` with
`use_controller_proxy=True`, so log writes ride `client → SSH tunnel → Iris
controller → LogServiceProxy → finelog`. The hops are real and correct — but
they are welded inside `RemoteClusterClient`. You cannot write a 20-line `marin
status` CLI that just opens a finelog connection and writes a row.

PR #6466 (open) adds a *fourth* transport — a public controller behind Google
IAP — with its own client credential plumbing (`IapUserIdTokenProvider`,
`ProxyAuthTokenInjector`, `ClientCredentials` threaded through
`IrisClient.remote`/`rpc_client`/`call_rpc`). The bug that PR itself calls out
(*"`cluster vm status` rejected by IAP because a call site attached the JWT and
forgot the IAP token"*) is exactly the failure mode of threading credentials by
hand.

### What is *already solved* (and therefore not this design's job)

A first draft of this doc proposed adding a generic controller reverse proxy.
**That already exists.** The controller has:

- `EndpointProxy` (`controller/endpoint_proxy.py`) — a generic HTTP reverse
  proxy mounted at `/proxy/{endpoint_name}/{sub_path}` (and a subdomain variant),
  resolving the name (with `.`→`/` substitution) via a `resolve: (name) ->
  address | None` callable that consults both the SQL endpoint store and the
  in-memory `system_endpoints` map — i.e. it mirrors `ListEndpoints`. It streams
  bodies both ways, rewrites `Location`, and **strips `Authorization` and
  `Cookie` on the client→upstream hop** as a deliberate credential-leak guard.
- A typed mount of finelog's `LogService`/`StatsService` at their canonical RPC
  paths (`controller/dashboard.py`), backed by `LogServiceProxy`, behind the
  controller's own `auth_interceptor`.

So "the extra hop to finelog" is a controller capability that exists today. What
does **not** exist is a clean *client-side* way to say *"give me an
authenticated connection to `<service>` behind cluster `<name>`, and hand it to
my client factory"* — independent of `IrisClient`. That gap is this design.

## Goal

A small library in **`lib/rigging`** that makes a connection a **value you can
name, pass around, and resolve once**, regardless of transport/auth:

1. Express a connection as a **compact, consistent string** so a CLI flag,
   config field, or env var can say *"reach finelog through cluster marin"* in
   one token.
2. Resolve that string to a **transport** (tunnel or direct) plus **auth**
   (interceptors), establish the transport once, and hand the caller's own
   `factory(endpoint) -> client` the resolved endpoint. The user owns what a
   "client" is (a Connect/RPC client, almost always).
3. Keep tunnels alive for the connection's lifetime via a **context manager**,
   with a GC finalizer only as a *subprocess-leak backstop* (not a correctness
   mechanism — see [§Lifetime](#lifetime-no-magic)).
4. Reuse the controller's existing `EndpointProxy` + endpoint registry for the
   "extra hop" — do **not** invent a new proxy.
5. Live at the **bottom of the dependency graph** (`rigging` is a leaf), so
   `iris`, `finelog`, `zephyr`, and `marin` can all adopt it without cycles.

### Non-goals

- Not a service mesh, pooling, or load balancing. One client, one journey, torn
  down when done.
- Not a new auth *model*. The JWT/role/IAP design in
  `20260312_iris_auth_design.md` and PR #6466 is unchanged on the **server**.
  This library owns only the **client** side: acquiring tokens and attaching
  them as the transport requires.
- Not a new controller proxy, and not a re-litigation of the existing proxy's
  authorization model (that is separate controller security work — see
  [§Security](#security-the-proxy-is-a-boundary-already)).
- Not a uniform "everything is a hop" algebra. Transport and auth compose
  linearly; **logical endpoint resolution does not** (it needs an RPC), so it
  stays a resolver, not a hop (codex CRITICAL-1).

## The model

Three concepts, plus a deliberately separate resolver.

### `Endpoint` — what transport+auth produce

```python
@dataclass(frozen=True)
class Endpoint:
    """A reachable URL plus the auth a caller must attach to use it."""
    url: str                                   # "http://127.0.0.1:54321" | "https://iris-marin.oa.dev"
    interceptors: tuple[Interceptor, ...] = () # connect-python client interceptors (auth headers, etc.)

    def socket_address(self) -> tuple[str, int]:
        """(host, port) for socket-level callers. Valid only for a bare
        origin URL (no path); raises otherwise so a proxy-prefixed URL is
        never silently truncated to host:port."""
        parts = urlsplit(self.url)
        if parts.path not in ("", "/") or parts.hostname is None:
            raise ValueError(f"endpoint {self.url!r} is not a bare host:port")
        return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
```

`url` is the canonical address — it carries scheme, host, port, and any proxy
path prefix. `socket_address()` is a guarded convenience for the user's stated
`(ip, port) -> client` factory shape, but most Connect clients want the URL +
interceptors (codex MINOR-10).

> **Reconciling with "a function that takes IP:port".** The user asked for a
> factory `(ip, port) -> client`. Connect attaches interceptors at construction,
> so auth must reach the client; the factory therefore takes the richer
> `Endpoint`, which exposes `.url`, `.interceptors`, and `.socket_address()`.
> The two real adapters are one line each:
> `lambda e: LogClient.connect(e.url, interceptors=e.interceptors)` and
> `lambda e: ControllerServiceClientSync(e.url, interceptors=e.interceptors, ...)`.

### `Transport` — establishes the byte path

```python
class Transport(Protocol):
    """Establishes the network path to a service and yields its base Endpoint.

    `open` registers any background resource (a tunnel subprocess) on `stack`,
    so the connection's lifetime owns it, and returns the base Endpoint the
    client should target.
    """
    def open(self, stack: ExitStack) -> Endpoint: ...
```

| Transport | `open` does |
|-----------|-------------|
| `DirectTransport(url)` | returns `Endpoint(url)` — no tunnel (public HTTPS, loopback, in-cluster DNS). |
| `SshTunnel(GcpSshForwardTarget)` | `stack.enter_context(open_tunnel(target))` → `Endpoint("http://127.0.0.1:<port>")`. Wraps the **existing** `rigging.tunnel`. |
| `K8sPortForward(K8sPortForwardTarget)` | same, via `kubectl port-forward`. |

This is the user's "SSH is a tunnel that returns a new localhost connection",
unchanged — it just *is* `rigging.tunnel` with an `Endpoint` wrapper.

### `Auth` — contributes interceptors

```python
class Auth(Protocol):
    """Returns the client interceptors needed to authenticate to a target."""
    def interceptors(self) -> tuple[Interceptor, ...]: ...
```

| Auth | interceptors |
|------|--------------|
| `NoAuth()` | `()` (loopback/SSH-tunnel-trust clusters) |
| `JwtAuth(token_provider)` | `(AuthTokenInjector(token_provider),)` — Iris JWT in `Authorization` |
| `IapAuth(id_token_provider)` | `(ProxyAuthTokenInjector(id_token_provider),)` — IAP OIDC token in `Proxy-Authorization` (lands with PR #6466) |
| `ChainedAuth(*auths)` | concatenation — IAP + JWT together, attached **as a unit** so a call site can't attach one and forget the other (the PR #6466 bug, fixed structurally) |

Auth and transport are orthogonal: an IAP cluster is `DirectTransport("https://…")`
+ `ChainedAuth(IapAuth(...), JwtAuth(...))`; an SSH cluster is `SshTunnel(...)` +
`NoAuth()` (loopback-trusted) or `JwtAuth(...)`.

### `EndpointResolver` — the "extra hop", kept as a resolver (not a hop)

Reaching a service *behind* a cluster needs a name→address lookup that, for
in-cluster direct connections, is a live `ListEndpoints` RPC against the
controller. That is I/O with policy — it cannot be a pure pre-baked endpoint
rewrite (codex CRITICAL-1). The controller already exposes exactly this shape:
`EndpointProxy`'s `resolve: (name) -> address | None`, and `LogClient.connect`
already accepts a `resolver=` for re-resolution on retry. So we keep it as a
resolver and feed it to clients that understand re-resolution:

```python
@dataclass(frozen=True)
class Routing:
    """How to reach a named service relative to a connection's base endpoint."""
    PROXY:  ...   # service is served at the base endpoint (typed mount) or via /proxy/<name>
    DIRECT: ...   # resolve <name> via the controller's ListEndpoints, connect directly
```

- **PROXY (external clients)** — the controller already serves finelog's
  `LogService`/`StatsService` at canonical RPC paths, so the base controller
  `Endpoint` *is* the finelog endpoint; the factory just targets it. For other
  services, the URL becomes `f"{base}/proxy/{name.replace('/', '.')}"` using the
  existing `EndpointProxy` route (codex CRITICAL-2: reuse `/proxy/<dot-name>`,
  don't invent `/system/proxy`). **Caveat, documented:** the generic `/proxy/`
  hop strips `Authorization` upstream, so it suits services that don't require
  the Iris JWT; services that do (like finelog over the typed mount) use the
  canonical mount, which preserves auth via the controller's own interceptor.
- **DIRECT (in-cluster clients)** — resolve `name` via `ListEndpoints` and
  return that finelog address; no tunnel, no proxy. This is exactly what
  `RemoteClusterClient.resolve_endpoint` does today when
  `use_controller_proxy=False`, lifted out of the Iris client.

### `ConnectionSpec` + the compact string

```python
@dataclass(frozen=True)
class ConnectionSpec:
    transport: Transport
    auth: Auth
    routing: Routing | None = None       # set when targeting a service behind a cluster
    options: ConnectionOptions = ...      # connect timeout, rpc deadline (codex MAJOR-9)

    @classmethod
    def parse(cls, s: str, *, registry: "ClusterRegistry") -> "ConnectionSpec": ...
    def explain(self) -> str: ...         # codex MAJOR-7
```

`ConnectionOptions` carries the connect/tunnel timeout and the per-RPC deadline;
**retry and re-resolution stay inside the client** (LogClient already owns them
via its resolver) and **transport respawn stays inside `open_tunnel`'s
watchdog** — we do not invent a cross-hop retry layer (codex MAJOR-9).

`explain()` prints the resolved transport (SSH target vs IAP host), auth
providers by name (never token values), the config path the cluster resolved
from, and the routing mode — so *"why did this open SSH / use IAP / send no
JWT?"* is answerable without reading code (codex MAJOR-7).

#### String grammar — no chaining

The first draft proposed `::`-chained multi-hop URIs. **Dropped** (codex
MAJOR-8): once endpoint resolution is a resolver rather than a hop, there is no
user-visible chain to express. Two forms only:

**Clustered (the 95% case).** `iris://<cluster>[/<endpoint-wire-name>]`:

| String | Means |
|--------|-------|
| `iris://marin` | the controller of cluster `marin`; transport (SSH/IAP) + auth resolved from its config + token store |
| `iris://marin/system/log-server` | finelog's log server behind `marin`. The path after the authority is the **literal endpoint wire name** (`/system/log-server`) — the same string `ListEndpoints` and the typed mount use, so there is no dot-encoding ambiguity at this layer (encoding to `/proxy/system.log-server` happens only if the generic-proxy routing branch is taken). |

Transport and auth are intentionally **not** in the string — they resolve from
the cluster config, so `iris://marin` stays valid as `marin` migrates SSH→IAP
(PR #6466's gradual cutover) and the "forgot the IAP token" bug class is gone
(the registry expands the cluster into `ChainedAuth(...)` atomically). The cost
— hidden transport/auth — is bought back by `explain()` (codex MAJOR-7).

**Standalone (a service not behind any controller).** A single transport URL,
reusing the scheme vocabulary already in `iris/cluster/endpoints.py`:

| URI | Transport |
|-----|-----------|
| `ssh+gcp://[SA@]<project>/<zone>/<instance>:<port>` | `SshTunnel` (`?iap=true` → `--tunnel-through-iap`) |
| `k8s://[<context>/]<namespace>/<service>:<port>` | `K8sPortForward` |
| `https://<host>` / `http://<host:port>` | `DirectTransport` |

A standalone finelog VM is then
`ssh+gcp://logging@marin-prod/us-central1-a/finelog-1:7000` — exactly the hop
the finelog CLI builds today, now a string.

### `connect()` and `Connection` — resolve once, deterministic teardown

```python
ClientFactory = Callable[[Endpoint], ClientT]

def connect(spec: ConnectionSpec | str, factory: ClientFactory[ClientT],
            *, registry: "ClusterRegistry | None" = None) -> "Connection[ClientT]": ...

class Connection(Generic[ClientT]):
    """Owns the live tunnel(s) for one connection. Use as a context manager.

    `.client` is valid for the lifetime of the Connection. `close()` closes the
    client first (if it exposes close()/__exit__), then the tunnels — so a
    client's background flush threads are drained before their transport
    disappears.
    """
    client: ClientT
    endpoint: Endpoint
    def __enter__(self) -> ClientT: return self.client
    def __exit__(self, *exc) -> None: self.close()
    def close(self) -> None: ...          # idempotent: client teardown, then ExitStack
```

`connect` resolves the spec, `open`s the transport on an `ExitStack`, builds the
final `Endpoint` (base URL/proxy-rewritten URL + transport-contributed +
auth-contributed interceptors), calls `factory`, and returns a `Connection`.

#### Lifetime — no magic

The first draft made `Connection` a `__getattr__` transparent proxy with a
weakref finalizer "so holding the client keeps the tunnel alive". **Removed**
(codex MAJOR-4/5):

- `__getattr__` does **not** forward dunders (`__enter__`, `__exit__`, `__iter__`,
  truthiness) — they resolve on the class — so the proxy could not honestly
  stand in for a context-managed client.
- More dangerously, `LogClient.get_table()` returns `Table` objects with daemon
  flush threads that call back into the client. A caller holding a `Table` but
  dropping the `Connection` would have the finalizer kill the tunnel mid-write.

So: **the context manager is the contract.** `close()` tears down the client
before the tunnel. A `weakref.finalize(self, stack.close)` remains **only** as a
last-ditch guard against orphaned tunnel *subprocesses* at interpreter exit — it
is explicitly *not* a correctness mechanism, and the docstring says so. This
honors the user's "kept alive via context managers, GC as backstop, leakiness
accepted" — but reframes GC as preventing zombie `ssh` processes, not as
substituting for deterministic ownership.

```python
# Deterministic — the supported shape:
with connect("iris://marin/system/log-server", log_factory) as log:
    log.get_table(...).write([row])      # flushed and drained before the tunnel closes

# Standalone service:
with connect("ssh+gcp://logging@marin-prod/us-central1-a/finelog-1:7000", log_factory) as log:
    print(log.query("select * from \"iris.task_status\" limit 10"))
```

## Security — the proxy is a boundary, already

The existing `EndpointProxy` is an internal-network pivot surface: it forwards
arbitrary methods (minus CONNECT/TRACE), streams unbounded bodies, and can reach
any registry-resolved address. Today it is gated by `@requires_auth` and strips
`Authorization`/`Cookie` upstream. This design **does not widen** that surface —
it only *consumes* the proxy from the client side. Tightening proxy
authorization (per-endpoint allowlists for `/system/*`, owner checks for
task-registered endpoints, request/response size caps, audit logging) is real
and worth doing, but it is **controller security work, tracked separately**, not
part of this client library (codex CRITICAL-3). This doc flags it so it isn't
lost.

## Re-architecture

### `lib/rigging` — new `rigging/connect.py` + a conservative `rigging/auth.py`

- `rigging/connect.py`: `Endpoint`, `Transport` (+ `DirectTransport`,
  `SshTunnel`, `K8sPortForward` wrapping `rigging.tunnel`), `Auth`,
  `Routing`/`EndpointResolver`, `ConnectionSpec` (parse/explain), `connect`,
  `Connection`.
- `rigging/auth.py`: **only transport-generic client pieces** — the
  `TokenProvider` protocol, `AuthTokenInjector`, `GcpAccessTokenProvider`, and
  (post-#6466) `IapUserIdTokenProvider` / `ProxyAuthTokenInjector`. These depend
  only on `google-auth` + `connect-python`. **Staying in `iris`:** JWT minting
  and verification, role semantics, loopback trust, the SQLite token-store
  schema, and the desktop-OAuth login UX — none are service-neutral (codex
  MAJOR-6). This is a smaller move than the first draft proposed.

### `lib/iris`

- `ClusterRegistry` is an **iris-side** adapter (it reads iris's config proto and
  token store) implementing a tiny rigging protocol: `cluster_name → (Transport,
  Auth)`. This keeps the Marin-specific config-shape reader in iris and rigging a
  clean leaf (resolves the first draft's open question Q2 conservatively).
- `cli/connect.py::require_controller_url` → `connect(f"iris://{cluster}",
  rpc_client_factory)`, stashing the `Connection` on the click context (replaces
  the manual `tunnel_cm.__enter__()` / `ctx.call_on_close` dance).
- `ControllerProvider.tunnel()` and the three `_gcp_tunnel` /
  `kubectl.port_forward` / `nullcontext` impls **collapse** into rigging
  transports: GCP → `SshTunnel`, k8s → `K8sPortForward`, manual/local →
  `DirectTransport`. Kills two of the three duplicate tunnels.
- `RemoteClusterClient`'s `use_controller_proxy` flag + `resolve_endpoint`
  become a `Routing` value; its inner `LogClient` is built from
  `connect("iris://<cluster>/system/log-server", LogClient...)`.

### `lib/finelog`

- `deploy/cli.py`: replace `_tunnel_target` + `open_tunnel` + `LogClient.connect`
  with `connect(spec, lambda e: LogClient.connect(e.url, interceptors=e.interceptors))`.
  Standalone deployments produce a `ssh+gcp://…`/`k8s://…` spec from the finelog
  config; finelog-behind-a-cluster uses `iris://<cluster>/system/log-server`.
- **The payoff — a direct status-text path.** A standalone writer becomes:

  ```python
  with connect("iris://marin/system/log-server",
               lambda e: LogClient.connect(e.url, interceptors=e.interceptors)) as log:
      log.get_table(TASK_STATUS_NAMESPACE, TaskStatusRow,
                    storage_policy=TASK_STATUS_STORAGE_POLICY).write([TaskStatusRow(...)])
  ```

  No `IrisClient`. `report_task_status_text` becomes a thin finelog write over a
  `rigging` connection; the Iris client keeps a convenience wrapper delegating to
  the same path.

## Migration plan (validate the hard path first)

Reordered per codex MINOR-11 — prove controller auth + routing before extracting
the abstraction, since standalone-finelog is already handled by `open_tunnel`:

1. **Spike the concrete win, concretely.** Build the direct status-text path as
   a thin helper (`iris://<cluster>` → controller `Endpoint` + auth via the
   iris-side `ClusterRegistry`, point `LogClient` at it). This exercises the
   *hard* parts — cluster config resolution, transport choice, auth attach,
   routing to the typed log-server mount — against a live cluster, before any
   abstraction is frozen.
2. **Extract `rigging/connect.py`** from the spike: `Transport`, `Auth`,
   `Endpoint`, `Connection`, `ConnectionSpec.parse/explain` — only the seams the
   spike proved necessary. Unit-test with a fake `spawn` (the pattern
   `tunnel.py` already uses) and a fake factory.
3. **Relocate transport-generic auth** (`AuthTokenInjector`, `TokenProvider`,
   `GcpAccessTokenProvider`) into `rigging/auth.py`; iris imports from there. Pure
   move + import rewrite, no compat shims (repo policy). Sequence **after**
   PR #6466 merges so `IapAuth`/`IapUserIdTokenProvider` land on top.
4. **Adopt in iris** (`require_controller_url`, the `tunnel()` collapse, the
   `Routing` replacement for `use_controller_proxy`) and **finelog CLI**.
5. **Ship the direct status-text CLI** as the first capability the library
   unlocks for users.

Steps 1 + 5 deliver the user's concrete need even if the abstraction (2–4)
slips; that ordering is itself the hedge codex asked for.

## Testing

Follows `lib/rigging`'s conventions (`test_rpc.py`, `test_tunnel`) and root
`TESTING.md` (behavior-focused, injectable seams, no slop):

- `ConnectionSpec.parse`/`explain` round-trips for every scheme + the
  `iris://cluster/endpoint` form (table-driven).
- `connect` with a fake `spawn` + fake factory: assert transport opens, auth
  interceptors accumulate (and `ChainedAuth` attaches IAP+JWT together), and the
  factory receives the final localhost `Endpoint`.
- `Connection` teardown: `__exit__` closes the client before the ExitStack
  (order asserted via a recording fake client); `close()` is idempotent; the
  finalizer reaps an orphaned fake subprocess on `gc.collect()`.
- `Routing`: PROXY returns the base controller URL for the typed log-server
  mount and the `/proxy/<dot-name>` URL for a generic endpoint; DIRECT resolves
  via a fake registry; a missing endpoint name raises (not a silent no-op).

## Codex review and how the design changed

Full `codex` pass on the first draft; verdict *"revise, bordering on rethink —
the concrete need fits a much smaller design."* Findings and responses:

| # | Sev | Finding | Response |
|---|-----|---------|----------|
| 1 | CRIT | `ControllerProxyHop` can't be a pure hop — in-cluster resolution needs a `ListEndpoints` RPC (I/O + policy). | **Accepted.** Endpoint resolution is now an `EndpointResolver`/`Routing`, not a hop. Hops cover only transport+auth, which *do* compose linearly. |
| 2 | CRIT | A generic proxy already exists (`/proxy/<dot-name>`); don't invent `/system/proxy`; typed mounts ≠ generic proxy. | **Accepted.** Design now consumes the existing `EndpointProxy` and the typed `LogService` mount; invents no route. |
| 3 | CRIT | The generic proxy is a security boundary (network pivot, no size caps); needs an authz policy. | **Accepted, scoped out.** Documented as separate controller security work ([§Security](#security-the-proxy-is-a-boundary-already)); this client library doesn't widen it. |
| 4 | MAJ | Transparent-proxy `Connection` + GC finalizer is unsafe — `LogClient` `Table`s have daemon flush threads that outlive a dropped `Connection`. | **Accepted.** Dropped the transparent proxy; `close()` drains the client before the tunnel; GC is only a subprocess-leak backstop, not correctness. |
| 5 | MAJ | `__getattr__` doesn't forward dunders, so it can't honestly stand in for a context-managed client. | **Accepted.** No transparent substitution; `.client` is explicit, valid inside `with`. |
| 6 | MAJ | Auth relocation over-reached (moving login UX, token store, roles into a leaf lib). | **Accepted.** Only transport-generic injectors/providers move to `rigging/auth.py`; minting/verification/roles/loopback-trust/token-store/login stay in iris. |
| 7 | MAJ | Compact name hides operationally critical facts (SSH vs IAP, audience, token source). | **Accepted.** Added `ConnectionSpec.explain()`; kept hidden-transport for migration-stability but made it inspectable. |
| 8 | MAJ | `::` chaining is ambiguous vs existing endpoint/proxy encoding and reverses fsspec. | **Accepted.** Removed chaining entirely — resolver-not-hop means there's no chain to write. Two forms: `iris://cluster[/endpoint]` and a single standalone transport URL. |
| 9 | MAJ | Timeouts/retries aren't compositional across hops. | **Accepted.** `ConnectionOptions` for connect-timeout/deadline; retry/re-resolution stay in the client, transport respawn stays in `open_tunnel`'s watchdog. No cross-hop retry layer. |
| 10 | MIN | `Endpoint.address` snippet drops scheme/path and mishandles `None`. | **Accepted.** `url` is canonical; `socket_address()` is guarded and raises on path-prefixed URLs. |
| 11 | MIN | Migration starts with the algebra, not a real call site; standalone-finelog is already solved. | **Accepted.** Reordered: spike the status-text path first, extract the abstraction from what it proves. |

**Where the design deliberately keeps more than codex's minimal suggestion.**
Codex proposed *"just a `finelog_connection_from_iris_cluster(...)` helper + shared
interceptors."* The user explicitly asked for a *reusable, composable* connection
library with a compact string and a user-supplied factory, because the same
shape recurs (iris controller, finelog, future services) and the three duplicate
tunnels are real debt to consolidate. The synthesis: keep a **minimal**
composable core (`Transport` + `Auth` + `connect`), but (a) validate it through
the concrete status-text spike *first* (codex's step ordering), and (b) refuse
the parts that didn't survive scrutiny (hop-based resolution, transparent proxy,
new controller route, broad auth relocation). The result is closer to codex's
"small" than to the first draft, while still being the library the user asked
for.

## Open questions for the user

- **OQ1 — abstraction size.** This keeps a minimal `Transport`/`Auth`/`connect`
  core. Codex argued for even less (one helper function). Do you want the
  composable library now, or the helper first with the library extracted later
  once a second consumer appears?
- **OQ2 — `iris://cluster/endpoint` path semantics.** The path is the literal
  endpoint wire name (`/system/log-server`). Acceptable, or do you prefer an
  explicit separator (`iris:marin:system.log-server`) to make the
  cluster/endpoint split unmistakable?
- **OQ3 — generic-proxy auth gap.** finelog over the typed mount keeps the JWT;
  arbitrary services over `/proxy/<name>` lose `Authorization` (stripped
  upstream). Is "JWT-carrying services must use a typed mount" an acceptable
  rule, or should the proxy gain an opt-in auth-forwarding mode (controller
  work)?
