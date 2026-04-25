# Marin Log-Store Extraction Plan

**Status:** Draft · 2026-04-25
**Owner:** Russell
**Related design:** `docs/design/marin-service-architecture.md`

## Goal

Lift logging out of `lib/iris` into a new `lib/finelog` package, and in doing
so introduce the service primitives the architecture doc calls for:

1. A pluggable URL resolver in `lib/rigging` that converts logical service
   references (`gcp://<vm>[:port]`, `iris://<cluster>?endpoint=<name>`, bare
   `host:port`) into `(host, port)`. Schemes register handlers; `iris://`
   lives in `iris.client.resolver_plugin` so `rigging` stays proto-free and
   iris-free.
2. A domain-level `LogClient` / `LogServer` pair in `lib/finelog`, independent
   of the Iris controller process.
3. A Dockerfile and declarative `system-services` cluster YAML entry so the
   log-server runs as an independently-deployed GCP VM. Workers and the
   controller find it through the resolver, **not** through any registration
   RPC.

We deliberately do **not** introduce a new `EndpointsService` RPC, a
self-registration heartbeat, or any controller-mediated lookup for the log
server. System services like `finelog` live at a stable provider-managed
identity (e.g. a GCP VM named `finelog-server`); the resolver translates that
identity to a current `(host, port)` on demand. When the VM is recreated, the
resolver returns the new IP on the next call. See D3, D7.

(`finelog` because it is fine. The name was chosen for tone, not technical
content; do not read meaning into it.)

The migration must finish with no caller (worker, controller, CLI, test) still
importing `iris.log_server` or `iris.cluster.log_store`. Controller restarts
must not restart the log server, and log-server restarts must not restart the
controller.

## Current-state inventory

The explore pass produced the following facts. Anything below is a claim about
what exists today — verify before acting on it.

### What lives in `lib/iris` today

| Concern | Path | Notes |
| --- | --- | --- |
| Proto | `lib/iris/src/iris/rpc/logging.proto` | package `iris.logging`; imports `iris.time.Timestamp` |
| Generated stubs | `lib/iris/src/iris/rpc/logging_{pb2,pb2.pyi,connect}.py` | buf remote plugins (connect-python v0.9.0) |
| Server impl | `lib/iris/src/iris/log_server/server.py` | `LogServiceImpl` |
| Server entrypoint | `lib/iris/src/iris/log_server/main.py` | `python -m iris.log_server.main`; uvicorn + Starlette ASGI |
| Client | `lib/iris/src/iris/log_server/client.py` | `LogPusher`, `LogServiceProxy`, `RemoteLogHandler` |
| Storage | `lib/iris/src/iris/cluster/log_store/{duckdb_store,mem_store,_types}.py` | DuckDB in prod, MemStore under pytest/CI |
| Level util | `lib/iris/src/iris/logging.py` | `str_to_log_level` |
| In-proc handler | `lib/iris/src/iris/cluster/log_store/__init__.py` | `LogStoreHandler`, `LogCursor` |
| Subprocess supervisor (path A) | `lib/iris/src/iris/cluster/controller/main.py` `_start_log_server` at line 47, `subprocess.Popen` at line 230 | spawns `python -m iris.log_server.main`, waits on port |
| In-process supervisor (path B) | `lib/iris/src/iris/cluster/controller/controller.py` `_start_local_log_server` at line 1162, thread spawn at 1195 | spins up uvicorn on a dynamic port inside the controller process (thread, not subprocess) |
| Controller glue | `lib/iris/src/iris/cluster/controller/controller.py` lines 110–112, 1045–1071, 1256–1257 | builds `LogPusher` / `LogServiceProxy`; registers `/system/log-server`. Line 1060 picks path B when `config.log_service_address` is unset. |
| Worker resolver | `lib/iris/src/iris/cluster/worker/worker.py` lines 194–202, 266–275 | ad-hoc `_resolve_log_service` that calls `ControllerService.ListEndpoints` |
| Cross-proto refs to `iris.logging.LogEntry` | `lib/iris/src/iris/rpc/controller.proto:359`, `:393`; `lib/iris/src/iris/rpc/job.proto:179` | Three other protos import `iris.logging.LogEntry` by fully-qualified name — any package rename ripples through them. |
| Task-side push | `lib/iris/src/iris/cluster/worker/task_attempt.py` lines 42, 205–249 | `log_pusher.push(task_log_key(...), entries)` |
| Tests | `lib/iris/tests/{test_logging.py,test_remote_log_handler.py,e2e/test_attempt_logs.py,cluster/test_attempt_logs.py,cluster/controller/test_logs.py}` | mix of unit and e2e |
| Optional deps | `lib/iris/pyproject.toml` `[controller]` | `duckdb`, `pyarrow`, `kubernetes` |
| Dockerfile | `lib/iris/Dockerfile` | unified image; no log-server stage — it's a controller subprocess |

### Endpoint registration today

- RPCs live on `ControllerService` itself (`lib/iris/src/iris/rpc/controller.proto` lines 279–313): `RegisterEndpoint`, `UnregisterEndpoint`, `ListEndpoints`.
- Backed by `EndpointStore` in `lib/iris/src/iris/cluster/controller/stores.py` (write-through cache over `endpoints` SQL table).
- Used by coordinators / actors for rendezvous; unrelated to system services.
- **System endpoints** are a separate in-memory dict: `ControllerService._system_endpoints: dict[str, str]` with exactly one entry today (`/system/log-server`). Resolved via `ListEndpoints` (prefix match) because system endpoints are merged into the actor-endpoint response.

### What `lib/rigging` has today

Phase 0 + Phase 1 of this plan have already shipped on `rjpower/log-store`
(commits `244b57135`, `468dbfcaf`, `91027b13d`):

- `lib/rigging/src/rigging/auth.py` — `JwtVerifier` (Phase 0). The verifier
  is currently *not* fully stateless: it carries a `_revoked_jtis` set
  (`auth.py:27`) plus `revoke()` / `set_revocations()` plus a check at
  `auth.py:44`. We are jettisoning all four (see D4).
- `lib/rigging/src/rigging/resolver.py` — `ServiceURL`, `register_scheme`,
  `is_registered`, `resolve`, `gcp_vm_address`, plus the eagerly-registered
  `gcp://` handler (`resolver.py:116`). Single flat file (the originally-
  planned `resolver/` subpackage was collapsed in `468dbfcaf`; the
  speculative `provider=` arg was dropped from the helper in `9931883ea`).
- `lib/rigging/tests/test_resolver.py`, `test_jwt_verifier.py` cover them.

Other utilities unchanged: `filesystem.py`, `timing.py`, `log_setup.py`,
`distributed_lock.py`, `config_discovery.py`.

Dependencies: `fsspec`, `gcsfs`, `s3fs`, `google-auth`, `httpx`, `PyJWT`.
No protobuf. No connect-python.

### What's landed in `lib/iris` for this project

- `lib/iris/src/iris/client/resolver_plugin.py` — registers `iris://` with
  `rigging.resolver`. Delegates controller discovery to
  `ControllerProvider.discover_controller(...)`, so the scheme works on
  GCP (labeled VM), K8s/CoreWeave (Service DNS), and Manual/Local (static
  address from YAML). Used **only** for actor-endpoint lookup (job-scoped
  rendezvous). System services like `finelog` are addressed via `gcp://`
  (or whatever the cluster YAML carries) directly; the iris:// hop is
  unnecessary for them.
- `lib/iris/src/iris/client/__init__.py:29` — side-effect import that
  activates the plugin.
- `iris.cluster.config.IRIS_CLUSTER_CONFIG_DIRS` and
  `iris.cluster.config.load_cluster_config(name)` — the search path and
  helper the plugin uses to find a cluster YAML by name. Moved from
  `iris.cli.main` so the plugin doesn't depend on the CLI module.
- `iris.cluster.controller.auth.JwtTokenManager` (`auth.py:172`) composes a
  `JwtVerifier`, owns the revocation set today via `_verifier.revoke(...)`
  (callers at `service.py:2190,2264`) and `_verifier.set_revocations(...)`
  (callers at `auth.py:293,315`). The plan moves that revocation set into
  `JwtTokenManager` itself so the rigging verifier becomes pure stateless.

### Authn wrinkle (Phase 0, landed)

`iris.log_server.main._build_auth_interceptors` (line 112) used a
**function-scope import** of `iris.cluster.controller.auth.JwtTokenManager` —
the log server verifies tokens signed by the controller. The function-scope
import was a deliberate workaround acknowledged in the existing code
comments, because a top-level import would violate the iris rule "all
imports at top of file." Once the log server moved out of `lib/iris`, that
workaround was no longer viable: any import shape (`finelog →
iris.controller.auth`) is a reverse dependency.

The split landed in `244b57135`: `JwtTokenManager` now composes a
`rigging.auth.JwtVerifier` (signature/expiry only, intended-stateless) and
keeps the issuer half locally. The verifier is shared between the log
server and the controller.

**Followup in this plan (D4):** the verifier still carries a revocation set
that doesn't survive a process boundary cleanly. We drop revocation from
the verifier entirely; the controller-side `JwtTokenManager` owns its own
revocation set locally. The log server doesn't need revocation at all —
JWTs are short-lived (15 min default), so revocation is a nice-to-have on
the issuer side, not a security boundary on the verifier side.

### `time.proto` wrinkle

`logging.proto` imports `iris.time.Timestamp`. If logging moves to
`lib/finelog` we need a shared proto without a cyclic dep. Options in §3.

---

## Design decisions (made up-front so later steps are mechanical)

### D1. Proto package renames to `finelog`

When we move `logging.proto` into `lib/finelog`, we rename the proto package
from `iris.logging` to `finelog`. This is a wire-breaking change (the
Connect URL becomes `/finelog.LogService/...`), but since the client library
hides this and we redeploy everything in lock-step, it is fine. The
architecture doc explicitly treats wire-format churn as acceptable so long as
the public client interface is stable.

**Rename cascade.** Three other iris protos reference `iris.logging.LogEntry`
by fully-qualified name:

- `lib/iris/src/iris/rpc/controller.proto:359` — `repeated iris.logging.LogEntry logs = 2;`
- `lib/iris/src/iris/rpc/controller.proto:393` — `repeated iris.logging.LogEntry worker_log_entries = 9;`
- `lib/iris/src/iris/rpc/job.proto:179` — `repeated iris.logging.LogEntry log_entries = 2;`

Those are **also** wire-breaking once the package is renamed: field 2 on
`iris.controller.*` now references `finelog.LogEntry`. The plan does not
try to preserve wire compatibility on those RPCs. We flip them at the same
moment we flip the log-server package (Phase 5), documented as part of that
phase. Phase 3 keeps them pointing at the old `iris.logging.LogEntry` so that
`lib/iris` builds on its own; Phase 5 rewrites them to import
`finelog/logging.proto` and regenerates.

This means Phase 3's "lib/iris unchanged" claim is literally true (iris's
`.proto` files don't move), but Phase 3 does introduce a **second,
independent** `LogEntry` definition in `finelog/logging.proto` with the same
schema. Both coexist until Phase 5's proto-import rewrite. This is
intentional — it keeps each phase independently mergeable.

### D2. Shared `Timestamp` moves to `lib/rigging`

`lib/iris` already imports `lib/rigging`, and `lib/finelog` will too. We
create `lib/rigging/src/rigging/proto/time.proto` with a minimal
`rigging.time.Timestamp` (just `int64 epoch_ms = 1`), generate stubs, and
both iris and finelog depend on it. `iris.time.Timestamp` goes away after a
one-shot rewrite of `.proto` imports.

### D3. No new endpoints RPC; system services bypass the controller entirely

(Revised 2026-04-25: was originally "add `rigging.endpoints.EndpointsService`
as a second service alongside `ControllerService`." Then briefly "piggy-back
on `ControllerService.ListEndpoints` for system endpoints." Settled on the
shape below: the controller is not in the system-service lookup path at all.)

Two independent things carry the word "endpoint":

- **Actor endpoints** — job-scoped rendezvous written by coordinators and
  read by workers. Stored in the `EndpointStore` SQL table; reachable via
  `ControllerService.{Register,Unregister,List}Endpoint`. Continues
  unchanged. The `iris://<cluster>?endpoint=<name>` resolver scheme exists
  for these.
- **System endpoints** — cluster-scoped service addresses like the
  log-server.

For system services we **bypass** the controller's endpoint lookup. The log
server is a GCP-managed VM with a stable name (e.g. `finelog-server`).
Cluster YAML names it; the cluster YAML value is itself a resolver URL
(`gcp://finelog-server` in prod, `127.0.0.1:<port>` in dev). Workers and the
controller pass that URL straight to `LogClient.connect(...)`, which calls
`rigging.resolver.resolve(...)`.

In this project we:

- **Do NOT** add a new `EndpointsService` proto/server/client.
- **Do NOT** seed `/system/log-server` into the controller's
  `_system_endpoints` dict in prod. (The dev/in-process path keeps doing
  it for backwards-compat with any code that still queries
  `ListEndpoints(prefix="/system/...")`. New code does not.)
- **Do NOT** add a self-registration RPC or heartbeat. The log server
  doesn't tell anyone where it is. Its identity is its GCP VM name; the
  resolver is the only mediator.
- Move VM-recovery logic into the resolver retry path (D11): when a `gcp://`
  resolution becomes stale (VM recreated, IP changed), the next
  `RemoteLogClient` call evicts its cached transport and re-resolves,
  which finds the new address.

This is materially simpler than going through the controller for system
services: no new RPC, no controller-side state, no controller-restart
recovery story, no race between seeding and clients connecting, no
controller hop on every push.

### D4. JWT verifier in `lib/rigging` is pure stateless

The Phase-0 extraction (landed) put the verifier in `rigging.auth`. We now
strip the revocation set out of it as well:

- `rigging.auth.JwtVerifier` becomes signature + expiry only. Drop
  `_revoked_jtis`, `revoke()`, `set_revocations()`, and the in-`verify_full`
  revocation check (`auth.py:27,44,53,56`).
- `iris.cluster.controller.auth.JwtTokenManager` keeps a revocation set
  *of its own*, checked locally inside its `verify()` after delegating to
  `_verifier.verify_full`. Existing callers
  (`service.py:2190`, `:2264`, `auth.py:293`, `:315`) continue to work
  unchanged.
- The log-server's verifier doesn't get a revocation set at all. JWTs are
  15-minute-TTL by default; issuer-side revocation is a soft-delete that
  takes effect within one TTL when the controller stops reissuing.

Why this matters for the log-server extraction: the verifier is the only
auth code that crosses the new process boundary. Keeping it stateless means
no revocation-list sync between controller and log-server, no shared
mutable state, no "what if the controller's revocation set is newer than
the log-server's" race. The cost is at most one TTL of acceptance lag for a
revoked credential — acceptable for our threat model and operationally
identical to the cost we already pay when a token is signed but
not-yet-revoked.

### D5. URL-scheme resolver is a small registry (landed)

The resolver lives at `lib/rigging/src/rigging/resolver.py` (single flat
file; the originally-planned `resolver/` subpackage was collapsed in
`468dbfcaf`).

```python
def register_scheme(scheme: str, handler: Callable[[ServiceURL], tuple[str, int]]) -> None: ...
def resolve(ref: str) -> tuple[str, int]: ...
```

`resolve` short-circuits bare `host:port`, parses URLs into `ServiceURL`,
and dispatches to a registered handler.

Built-in handlers (one, today):

- `gcp://<vm-name>[:port]` — registered eagerly at import
  (`resolver.py:116`). Calls `gcp_vm_address(name, port=url.port or 10002)`.
  Port is taken from the URL when present (commit `91027b13d`), otherwise
  defaults to 10002. This is one form a system-service URL can take.

`iris://<cluster>?endpoint=<name>` is **not** built into rigging. It lives
in `iris.client.resolver_plugin` and is side-effect imported by
`iris/client/__init__.py:29`. The handler is **platform-agnostic**: it
loads the cluster's YAML config, asks the platform's `ControllerProvider`
to discover the controller, and queries `ListEndpoints` over that
address. Concretely:

```python
cluster_config = load_cluster_config(cluster)            # from YAML
bundle = cluster_config.provider_bundle()                # ControllerProvider + WorkerProvider
controller_addr = bundle.controller.discover_controller(  # platform-specific
    cluster_config.proto.controller
)
# GCP   → labeled VM's internal_address:port
# K8s   → "iris-controller-svc.iris.svc.cluster.local:10000"
# Manual/Local → static address from config
```

The handler then opens a `ControllerServiceClientSync` to that address
and runs `ListEndpoints(prefix=name, exact=True)`. No hard-coded
`gcp_vm_address(f"iris-controller-{cluster}")` call — that was the
GCP-only shape and it shipped briefly in `244b57135`/`9931883ea` but is
gone now.

`iris://` is for **actor endpoints** (coordinator → worker rendezvous
within a job). Per D3, system services do not pass through this path —
they're addressed by `gcp://...` (or whatever YAML carries) directly.

**Off-cluster access.** Loading the YAML and calling `discover_controller`
work fine from a laptop. **Connecting to the resolved address** does not,
because internal IPs / cluster-local DNS aren't reachable from outside
the cluster network. The existing pattern (used by the CLI at
`iris/cli/main.py:117–127`) wraps the resolved address in
`bundle.controller.tunnel(addr)` — a context manager that does SSH
port-forward on GCP, `kubectl port-forward` on K8s, and nullcontext
locally. The architecture doc's `maybe_proxy()` is the planned wrapper
that fuses tunnel + resolver behind a context-managed
`LogClient.connect(...)`. Until then, off-cluster code uses the explicit
tunnel pattern; on-cluster code uses the resolver directly.

**Why this matters for K8s clusters.** Earlier drafts of the plugin called
`gcp_vm_address(f"iris-controller-{cluster}")` directly, which raises
`LookupError` on CoreWeave/K8s clusters because there's no GCP VM with
that name. Going through `ControllerProvider.discover_controller` makes
`iris://` work on every cluster type iris already supports without per-
scheme code.

The architecture doc calls out `coreweave://` and `k8s://` as **known
followups** — we do not stub them. A fresh provider-style URL scheme
(e.g. for a system service that lives on a K8s Service) is a one-line
`register_scheme` call from whichever package owns the cluster type.

### D6. Resolver is synchronous and returns `(host, port)`

Matches the signature already used by `LogPusher`'s existing `resolver`
callback, so callers can swap implementations without touching call sites.

### D7. No auto on/off-cluster detection — and no `maybe_proxy` stub in this project

The architecture doc's open question — recommendation is "prefer explicit."
We defer `maybe_proxy()` and the whole remote-access appendix to a followup
project. The CLI invocation shape does not need to leak tunnel details
today: the CLI runs on a cluster VM via the existing SSH path, same as
today.

### D8. Cluster YAML gains `system_services:` as a name → resolver-URL map

`IrisClusterConfig` is a **proto** message (`lib/iris/src/iris/rpc/config.proto:476`),
not a Python dataclass. Adding system-services is a proto-schema change.

```protobuf
// lib/iris/src/iris/rpc/config.proto
message IrisClusterConfig {
  // ... existing fields ...

  // System services live as cluster peers (typically GCP VMs). Each value
  // is a resolver URL understood by rigging.resolver.resolve(). Today
  // that means gcp://<vm>[:port] for VMs and bare host:port for pinned
  // dev endpoints.
  map<string, string> system_services = 90;
}
```

YAML form:

```yaml
system_services:
  log-server: gcp://finelog-server
```

That's the entire schema. We deliberately do not introduce
`SystemServiceConfig` with `provider/vm_type/image/health/env` fields in
this project — those are for the reconciler followup, and adding dead
config now would pre-commit to a schema we haven't validated. The
controller and the worker only need a string they can hand to
`LogClient.connect(...)`.

Reconciliation (auto-spinning the VM) is **not** part of this project; we
accept that the GCP VM exists, named as the YAML promises. Phase 7 documents
the operator runbook.

### D9. Local-dev fallback: `finelog.client.StderrLogClient`

A zero-dep client that implements the same interface and writes log entries
to stderr instead of the network. This is what unit tests inject and what
`dev_tpu` uses by default. No running log-server required for most
development work.

### D10. `str_to_log_level` lives in `finelog`

The helper depends on `finelog.LogLevel` enum values. Putting it in
`rigging.log_setup` would force `rigging` to import `finelog.proto.*`
(reverse dependency). Place it at `finelog/store/_types.py` next to the
other enum/keyname helpers. Close Open Question #2.

### D11. `RemoteLogClient` caches, re-resolves on failure (load-bearing for VM moves)

`RemoteLogClient.connect(url)` stores the URL and lazily resolves it on
first send. The resolved `(host, port)` is cached for the lifetime of the
client. On any transport-layer error that indicates the target has gone
away — connect refused, TLS handshake failure, HTTP 5xx from a stale VM —
the client evicts its cached transport and re-runs the resolver on the
next call.

This is **the only mechanism** by which we recover from a log-server VM
moving (deletion + recreation, IP reassignment, image roll). Specifically:

1. Operator recreates the `finelog-server` GCP VM. The new VM gets a new
   internal IP.
2. Workers are still pushing to the old IP. The next push fails with
   connect-refused.
3. `RemoteLogClient` evicts the cached transport and calls
   `rigging.resolver.resolve("gcp://finelog-server")` again.
4. The resolver hits the GCP API (`resolver.py:_fetch_vm_aggregated`) and
   returns the new IP.
5. Push succeeds against the new VM.

There is no controller involvement in this loop. The resolver going
directly to GCP is what makes it work without a heartbeat.

Backoff/retry semantics on `LogPusher` are unchanged from today — push
failures already retry with backoff. The new behavior is purely "evict
cached `(host, port)` after a transport error" inside `RemoteLogClient`.

### D12. Restart-independence is a prod-config guarantee, not a dev-config one

The goal "controller restarts do not restart the log server" holds **only**
when `system_services.log-server` is a `gcp://...` URL pointing at a
real, externally-managed VM. In the dev / single-box path
(`system_services: {}` or unset), the controller spins up an in-process
log server on a dynamic port and uses `127.0.0.1:<port>` as the URL;
controller restart does restart it — same as today. Document this in the
cluster YAML reference.

---

## Phasing

Phases 0–2 have already shipped on `rjpower/log-store` (commits
`244b57135`, `468dbfcaf`, `91027b13d`). Phase 3 introduces `lib/finelog`.
Phase 4 wires cluster YAML. Phase 5 cuts Iris callers over to
`finelog.client`. Phase 6 deletes the old paths. Phase 7 is operational
polish.

**Core migration:** Phases 0–6 minus the "optional cut-overs" noted inside
Phase 5 (CLI direct-connect and dashboard forwarder deletion).

**Operational followup:** Phase 7 and the optional Phase 5 cut-overs can
land as a separate PR stack and do not block deletion in Phase 6.

### Phase 0 — Split `JwtTokenManager` into verifier + issuer **(landed; followup pending)**

Status: shipped in `244b57135`. Followup work for **this** plan: drop the
revocation set from `rigging.auth.JwtVerifier` (D4). After the followup:

- `JwtVerifier` is pure stateless: signature + expiry checks only.
- `iris.cluster.controller.auth.JwtTokenManager` keeps its own
  `_revoked_jtis` set and applies the check inside its `verify()` after
  delegating to `_verifier.verify_full(...)`.
- Existing controller callers (`service.py:2190`, `:2264`, `auth.py:293`,
  `:315`) keep working unchanged because `JwtTokenManager.revoke(...)` and
  `JwtTokenManager.load_revocations(db)` survive — only their internal
  delegation to the verifier changes.
- The log-server's verifier (in finelog) takes a stock `JwtVerifier` with
  no revocation list. JWTs are short-lived (15 min default); revoke-on-
  controller eventually-consistent on log-server is acceptable.

**Done when:** `uv run pyrefly` passes; all existing auth tests pass; the
verifier exposes only `verify(token)` and `verify_full(token)`.

### Phase 1 — Resolver in `lib/rigging` **(landed)**

Status: shipped in `244b57135`, refined in `468dbfcaf`, `91027b13d`. The
landed file is **flat** (`lib/rigging/src/rigging/resolver.py`), not a
subpackage.

What it provides today (verified against `resolver.py`):

- `ServiceURL.parse(ref)` (`resolver.py:32–43`) — wraps `urllib.parse.urlsplit`.
  Surfaces `scheme`, `host`, `port`, and a `query: dict[str, str]`
  (first value per key). Rejects URLs with userinfo or a missing host.
- `register_scheme(scheme, handler)` (`resolver.py:51`) — adds a handler
  to a module-level `_HANDLERS` dict.
- `resolve(ref)` (`resolver.py:55`) — short-circuits bare `host:port`,
  otherwise parses and dispatches via `_HANDLERS`. Unknown scheme →
  `ValueError`.
- `gcp_vm_address(name, *, port=10002)` (`resolver.py:66`) — hits Compute
  Engine `aggregated/instances` via `google.auth` + `httpx` (lazy-imported),
  returns `(internal_ip, port)`. CoreWeave / k8s would each get their own
  helper if added.
- `is_registered(scheme)` (`resolver.py:54`) — predicate so callers and
  tests don't have to peek at `_HANDLERS`.
- `gcp://` is registered eagerly (`resolver.py:116`):
  `register_scheme("gcp", lambda url: gcp_vm_address(url.host, port=url.port or 10002))`.
  This lets callers say `gcp://my-vm:9000` to override the default port.

`iris://` lives in `iris.client.resolver_plugin` (Phase 2 below). Per D3
and D5, `iris://` is **only** for actor endpoints; system services use
`gcp://...` directly.

Tests: `lib/rigging/tests/test_resolver.py`,
`lib/rigging/tests/test_jwt_verifier.py`.

### Phase 2 — Iris resolver plugin **(landed; cross-platform refactor follow-up)**

Status: initial GCP-only version shipped in `244b57135`. The plugin now
delegates to `ControllerProvider.discover_controller(...)` so `iris://`
works on every iris-supported platform. Concretely:

```python
# lib/iris/src/iris/client/resolver_plugin.py
def _resolve_iris(url):
    cluster = url.host
    name = url.query["endpoint"]
    cfg = load_cluster_config(cluster)
    bundle = cfg.provider_bundle()
    controller_addr = bundle.controller.discover_controller(cfg.proto.controller)
    with ControllerServiceClientSync(f"http://{controller_addr}") as client:
        response = client.list_endpoints(... prefix=name, exact=True ...)
    ...
```

This unblocks K8s/CoreWeave clusters (where `discover_controller` returns
the Service DNS name `iris-controller-svc.iris.svc.cluster.local:10000`)
without per-scheme code in the plugin.

Off-cluster callers (laptop, CI runner) need to wrap usage in
`bundle.controller.tunnel(addr)` because the resolved address is a
cluster-internal IP / DNS name not routable from outside the VPC. The CLI
already does this at `iris/cli/main.py:117–127`. `maybe_proxy()` (architecture-doc appendix) is the planned ergonomic wrapper.

Tests: `lib/iris/tests/client/test_resolver_plugin.py` (six cases,
including a K8s-Service-DNS controller address path and a GCP-IP path).

### Phase 3 — Create `lib/finelog`

**Package skeleton.**

```
lib/finelog/
  pyproject.toml
  buf.yaml
  buf.gen.yaml
  Dockerfile
  README.md
  src/finelog/
    __init__.py
    proto/
      logging.proto
    client/
      __init__.py    # LogClient, StderrLogClient, LogQuery, LogRecord
      pusher.py      # LogPusher (formerly iris.log_server.client.LogPusher)
      remote_handler.py  # RemoteLogHandler
      proxy.py       # LogServiceProxy
    server/
      __init__.py
      service.py     # LogServiceImpl
      app.py         # build_asgi(...) (formerly build_log_server_asgi)
      main.py        # `python -m finelog.server.main` entrypoint
    store/
      __init__.py    # LogStore (env-conditional), LogCursor, LogStoreHandler
      _types.py      # keys, enums, protocols
      duckdb_store.py
      mem_store.py
  scripts/
    cli.py           # finelog cli (query/tail/push-test)
  tests/
    ...
```

**Dependencies.** `lib/finelog/pyproject.toml`:

```toml
[project]
name = "marin-finelog"
dependencies = [
    "marin-rigging",
    "connect-python>=0.9.0",
    "grpcio>=1.76.0",
    "httpx>=0.28.1",
    "protobuf",
    "starlette>=0.50.0",
    "uvicorn[standard]>=0.23.0",
    "PyJWT>=2.12.0",  # for auth interceptor (verifier half in rigging)
    "fsspec>=2024.0.0",
    "gcsfs>=2024.0.0",
    "s3fs>=2024.0.0",
]
[project.optional-dependencies]
server = [
    "duckdb>=1.0.0",
    "pyarrow>=19.0.0",
]
[project.scripts]
finelog = "finelog.scripts.cli:main"
finelog-server = "finelog.server.main:main"
```

`duckdb` / `pyarrow` move out of `lib/iris` `[controller]` extra and land in
`lib/finelog` `[server]` extra. This is a concrete win of the migration: the
Iris controller wheel shrinks.

**Proto.** Create a fresh `lib/finelog/src/finelog/proto/logging.proto`:

- `package finelog;` (fresh — `lib/iris/src/iris/rpc/logging.proto` is left
  in place and still compiled for iris; see D1 for why this duplication is
  intentional and temporary).
- `import "rigging/time.proto";` (from Phase 1).
- Same message shapes as the iris version so the stores/handlers port over
  without schema changes.
- Run `buf generate`, commit generated stubs under
  `src/finelog/proto/logging_{pb2,pb2.pyi,connect}.py`.

**Watch for generator-output collisions.** connect-python v0.9.0 writes
`<proto>_connect.py` next to the proto. If anyone later adds a human-written
`endpoints_connect.py`, `logging_connect.py`, etc., `buf generate` will
silently overwrite it. Document this in `lib/finelog/README.md`.

**Code moves (copy then delete later).** Rather than cut over in place, do a
**copy + adjust imports** into `lib/finelog/src/finelog/...`, leaving the
`lib/iris/src/iris/log_server/...` code untouched. Deletion happens in Phase 6
after all callers cut over. This keeps `main` green at every step.

Specifically:

- `iris.log_server.server.LogServiceImpl` → `finelog.server.service.LogServiceImpl`.
- `iris.log_server.main.run_log_server` / `build_log_server_asgi` →
  `finelog.server.app.build_asgi` / `finelog.server.main.run`.
- `iris.cluster.log_store.*` → `finelog.store.*`.
- `iris.log_server.client.LogPusher` → `finelog.client.pusher.LogPusher`.
- `iris.log_server.client.LogServiceProxy` → `finelog.client.proxy.LogServiceProxy`.
- `iris.log_server.client.RemoteLogHandler` → `finelog.client.remote_handler.RemoteLogHandler`.
- `iris.cluster.log_store.LogStoreHandler` → `finelog.store.LogStoreHandler`.
- `iris.logging.str_to_log_level` → `finelog.store._types.str_to_log_level` (per D10).

**New public client surface.** The architecture doc mandates a domain client,
not generated proto exposure. Define:

```python
# finelog/client/__init__.py
@dataclass
class LogMessage:
    key: str
    data: str
    source: str = "app"
    timestamp_ms: int | None = None
    level: LogLevel = LogLevel.UNKNOWN
    attempt_id: int = 0

@dataclass
class LogQuery:
    source: str
    substring: str | None = None
    since_ms: int = 0
    cursor: int = 0
    max_lines: int = 1000
    tail: bool = False
    min_level: LogLevel | None = None

@dataclass
class LogRecord:
    timestamp_ms: int
    source: str
    data: str
    level: LogLevel
    key: str
    attempt_id: int

class LogClient(Protocol):
    @staticmethod
    def connect(endpoint: str | tuple[str, int]) -> "LogClient": ...

    def write_batch(self, messages: Sequence[LogMessage]) -> None: ...
    def query(self, query: LogQuery) -> Sequence[LogRecord]: ...
    def close(self) -> None: ...

class RemoteLogClient(LogClient): ...
class StderrLogClient(LogClient): ...
```

`RemoteLogClient.connect(endpoint)` uses `rigging.resolver.resolve(endpoint)`
to obtain `(host, port)` and constructs a `LogServiceClientSync`
internally. Callers never see the generated proto types.

**`LogPusher` keeps its semantics** (buffering, batching, backoff) but its
constructor now takes a `LogClient` rather than a raw `resolver`:

```python
class LogPusher:
    def __init__(self, client: LogClient, max_queue: int = 10_000): ...
    def push(self, key: str, entries: Sequence[LogMessage]) -> None: ...
    def flush(self, timeout_s: float | None = None) -> None: ...
```

The resolver is now a concern of `LogClient.connect`, not of `LogPusher`.
This is the interface-stability payoff the architecture doc describes.

**Tests.** Port existing tests over:

- `lib/iris/tests/test_logging.py` → `lib/finelog/tests/test_push_pull.py`.
- `lib/iris/tests/test_remote_log_handler.py` →
  `lib/finelog/tests/test_remote_log_handler.py`.
- Keep duckdb_store and mem_store tests co-located with the code.
- Add `tests/test_stderr_client.py` for the fallback client.
- Add `tests/test_client_resolver.py` confirming `LogClient.connect` honors
  URL references.

**Done when:** `cd lib/finelog && uv run pytest` is green. `lib/iris` is
unchanged at this point — we only **added** a package.

### Phase 4 — Cluster YAML carries the system-service URL

No new RPC. No controller-side seeding logic for the prod path. The
controller's only job in this phase is to read a string from cluster YAML
and pass it through to whoever connects.

**Proto change.** Add a field to `IrisClusterConfig`
(`lib/iris/src/iris/rpc/config.proto:476`):

```protobuf
// Cluster-scoped service URLs (resolver references). Each value is
// understood by rigging.resolver.resolve(). Today: gcp://<vm>[:port] for
// GCP-hosted services, bare host:port for pinned dev endpoints.
map<string, string> system_services = 90;
```

This is wire-additive (new field number; existing clients ignore it).

**Plumbing.** Three places touch this:

1. `iris.cluster.config.IrisConfig` (the Python proto wrapper at
   `config.py:1082`) — gains an accessor like `system_service(name) -> str
   | None`. No new dataclass; we read the proto map directly.
2. `iris.cluster.controller.controller.ControllerConfig.log_service_address`
   (`controller.py:959`) is **removed**. Replaced by reading
   `cluster_config.system_services["log-server"]` at the same boot site.
3. The controller hands the string to `LogClient.connect(...)` — see
   Phase 5.3.

**The dev / in-process path** (`system_services` does not contain
`log-server`): the controller spins up an in-process `LogServiceImpl` on a
dynamic port and uses `f"127.0.0.1:{port}"` (a bare host:port URL) as its
log-server URL. Same shape, different value — every caller still goes
through `LogClient.connect`. The `_system_endpoints` dict is no longer
written to in this phase; if any legacy reader still needs
`ListEndpoints(prefix="/system/log-server")` to return something, that
reader needs a Phase-5 cutover instead of new seeding.

**What we don't add:** `SystemServiceConfig` with provider/vm-type/image/
health/env fields. Those are reconciler-scope. Adding them now as dead
config preselects a schema we haven't validated.

**Tests.**

- `lib/iris/tests/cluster/test_system_services_config.py` — load a YAML
  fragment with `system_services: {log-server: gcp://finelog-server}`;
  assert `IrisConfig` exposes the URL string verbatim.
- A fake-handler resolver test: register a fake `gcp` handler in
  `rigging.resolver`, set `system_services.log-server: gcp://finelog-test`,
  call `resolve(...)` against the loaded URL, assert the fake handler
  fires with `host="finelog-test"`.
- Existing actor-endpoint tests must not regress.

**Done when:** `IrisClusterConfig` carries the new field; the controller
reads it; the dev/in-process fallback is unchanged in user-visible
behavior. No new RPC surface.

### Phase 5 — Cut Iris over to `finelog.client`

**This is the only risky phase.** We are rewriting imports in the running
controller and worker. Do it in one sequenced commit per caller, not all at
once.

**5.1 — Worker.**

`lib/iris/src/iris/cluster/worker/worker.py`:

- Delete `_resolve_log_service` and the `ControllerService.ListEndpoints`
  call at `worker.py:194–202,266–275`.
- Replace `LogPusher(server_url=..., resolver=self._resolve_log_service)` with:

  ```python
  from finelog.client import RemoteLogClient, LogPusher

  # Set by the controller at job-launch time. In prod this is the value
  # of system_services["log-server"] from cluster YAML (e.g.
  # "gcp://finelog-server"). In dev it is the controller's
  # 127.0.0.1:<port>. Bare host:port also works.
  log_endpoint = self._worker_config.log_endpoint
  log_client = RemoteLogClient.connect(log_endpoint)
  self._log_pusher = LogPusher(log_client)
  ```

- `WorkerConfig` gains `log_endpoint: str` (required, not optional). The
  controller populates it from `cluster_config.system_services["log-server"]`
  for prod and from `127.0.0.1:<dev-port>` in the in-process dev path.

- **No stderr fallback in the worker.** If the log endpoint is unreachable,
  let `LogPusher`'s existing retry/backoff handle it; if it stays down,
  let it fail loudly. Stderr-on-a-headless-VM is silent loss in disguise,
  and the cluster is misconfigured if the log endpoint is missing — fail
  fast.

**5.2 — LogEntry → LogMessage call-site rewrite.** The `LogPusher` queue
element type changes from `iris.rpc.logging_pb2.LogEntry` to
`finelog.client.LogMessage`. Every site that constructs a `LogEntry()` and
hands it to `LogPusher.push()` must be rewritten. Enumerated from a
`grep -rn "logging_pb2.LogEntry("` pass (re-run before touching any of
these):

- `lib/iris/src/iris/log_server/client.py` lines 133, 163, 257, 315, 327,
  341 — internal `LogPusher` machinery (moves to finelog wholesale).
- `lib/iris/src/iris/log_server/client.py:452` — `RemoteLogHandler.emit()`
  (moves to finelog).
- `lib/iris/src/iris/cluster/log_store/__init__.py:77` —
  `LogStoreHandler.emit()` (moves to finelog).
- `lib/iris/src/iris/cluster/worker/task_attempt.py` lines 42, 205–249 —
  stdout/stderr polling loop. Stays in iris; flips to `LogMessage`.
- Any test construction of `logging_pb2.LogEntry(...)` — move to
  `LogMessage(...)` or `finelog.proto.logging_pb2.LogEntry(...)` depending
  on test layer.

Phase 3 already defines `LogMessage` as a dataclass with the same fields.
`LogPusher.push()` converts `LogMessage` to wire `finelog.LogEntry`
internally — callers never see the proto type.

**5.2.1 — Cross-proto import rewrite.** Simultaneously, flip the three
iris protos that reference `iris.logging.LogEntry`:

- `lib/iris/src/iris/rpc/controller.proto:359` → `repeated finelog.LogEntry logs = 2;`
- `lib/iris/src/iris/rpc/controller.proto:393` → `repeated finelog.LogEntry worker_log_entries = 9;`
- `lib/iris/src/iris/rpc/job.proto:179` → `repeated finelog.LogEntry log_entries = 2;`

Add the buf dependency in `lib/iris/buf.yaml` so the iris module can import
from `lib/finelog/src/finelog/proto/logging.proto`. Regenerate iris stubs.
All fields 2 / 9 on those RPCs are now wire-typed against `finelog.LogEntry`
— accept the break; it rolls with the rest of Phase 5.

**5.3 — Controller.**

`lib/iris/src/iris/cluster/controller/controller.py`:

- Delete the import `from iris.log_server.main import build_log_server_asgi`
  (`controller.py:111`).
- Delete `self._log_server: uvicorn.Server | None = None`
  (`controller.py:1055`).
- Delete the in-process `LogPusher` / `LogServiceProxy` wiring at
  `controller.py:1045–1071`, including the conditional at
  `controller.py:1057–1060` (`if config.log_service_address: … else: …
  self._start_local_log_server()`).
- Delete the `_start_local_log_server` method (`controller.py:1162–1201`).
- Delete the `self._service._system_endpoints["/system/log-server"] = …`
  write (`controller.py:1256`). Per D3 we no longer write
  `/system/log-server` into the controller's endpoint dict.
- Delete the dashboard's legacy `PushLogs` forwarder at
  `controller.py:281–282` (or defer it to 5.5 if Phase 5 is too big).
- Delete `ControllerConfig.log_service_address` (`controller.py:959`) and
  the `main.py:241,253` plumbing that constructs it.

Replace with:

```python
from finelog.client import RemoteLogClient, LogPusher
from finelog.server.app import build_asgi as build_finelog_asgi
from finelog.server.service import LogServiceImpl

log_url = cluster_config.system_services.get("log-server")
if log_url:
    # Prod path: log-server is a separate VM (typically gcp://finelog-server).
    pass  # log_url goes straight to LogClient.connect below.
else:
    # Dev / single-box mode: bind finelog locally, get back a host:port URL.
    log_url = self._start_in_process_finelog()

log_client = RemoteLogClient.connect(log_url)
self._controller_log_pusher = LogPusher(log_client)

# Workers receive the same URL via WorkerConfig at launch time.
self._worker_log_endpoint = log_url
```

`_start_in_process_finelog` is the Phase-5.3 replacement for
`_start_local_log_server`. It:

1. Builds a `LogServiceImpl` against a MemStore (dev) or `DuckDBLogStore`
   (if configured).
2. Spawns a uvicorn thread on a dynamic port (existing
   `self._threads.spawn_server` pattern).
3. **Returns** the bare URL `f"127.0.0.1:{port}"`. It does **not** write
   to `self._service._system_endpoints`. Callers reach it through the
   returned URL like any other.

The subprocess supervisor `iris.cluster.controller.main._start_log_server`
(`main.py:47`, Popen at `:230`) is removed in Phase 5.3. Operators running
that flow migrate to the Phase-7 Docker image.

**5.4 — CLI (optional — can ship after 6).**

`lib/iris/src/iris/cli/*` uses `RemoteClient.fetch_logs()` which today hits
the controller. After 5.1–5.3 the controller still forwards `FetchLogs` to
the log server (extra hop). Migrating the CLI to
`finelog.client.RemoteLogClient.query()` directly is a strict improvement
but does **not** block Phase 6 deletion — the forwarder can live a little
longer. Ship as a follow-up PR if Phase 5 is already carrying too much.

**5.5 — Dashboard (optional — can ship after 6).**

Dashboard's "PushLogs forwarder" (`controller.py:281–282`) is legacy
backwards-compat for a previous extraction. Deleting it is the same pattern
as 5.4: right thing to do, not blocking.

**5.6 — Tests.** Update all test imports from `iris.log_server.*` and
`iris.cluster.log_store.*` to their `finelog.*` equivalents. The fixtures
mostly construct `MemStore` + `LogServiceImpl` in-process; only the import
paths change. Double-check that e2e tests pointing at a real cluster still
find the log-server endpoint (they will, because the controller's
EndpointsService has it).

**Validation.**

- `cd lib/iris && uv run pytest` green.
- `cd lib/iris && uv run pytest -m e2e` green.
- `./infra/pre-commit.py --all-files --fix` clean.
- `uv run pyrefly` clean.
- Manual: `iris cluster up` on a dev cluster, submit a tiny job, `iris logs
  <job>` returns task output.

**Done when:** no caller imports `iris.log_server` or
`iris.cluster.log_store`. Iris controller starts without spawning the log
server subprocess when `system_services.log-server.address` is set.

### Phase 6 — Delete old code

Mechanical. With nothing importing them:

- `rm -rf lib/iris/src/iris/log_server/`.
- `rm -rf lib/iris/src/iris/cluster/log_store/`.
- `rm lib/iris/src/iris/logging.py`.
- `rm lib/iris/src/iris/rpc/logging.proto` and its generated `logging_{pb2,pb2.pyi,connect}.py`.
- Remove `duckdb` and `pyarrow` from `lib/iris` `[controller]` extra.
- Remove `LogServiceProxy` proxy handler from controller.
- Remove `_start_log_server` subprocess supervisor from
  `iris.cluster.controller.main` including the `--legacy-inprocess-log-server`
  flag added in 5.3 (keep if user wants a soak period; delete otherwise).
- `iris` `Dockerfile` controller stage loses the log-server-subprocess-related
  baggage (duckdb, pyarrow move to finelog's image).
- Grep for any stale references (`PushLogs` in proto, `iris.logging` package
  string, `log_server` directory name) — fix or delete.

### Phase 7 — Docker + ops polish

**`lib/finelog/Dockerfile`.** Minimal image:

```dockerfile
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

FROM base AS app
WORKDIR /app
COPY lib/rigging /app/rigging
COPY lib/finelog /app/finelog
RUN pip install --no-cache-dir /app/rigging /app/finelog[server]

ENV FINELOG_PORT=10002
ENV FINELOG_LOG_DIR=/var/cache/finelog/logs
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -sf http://localhost:${FINELOG_PORT}/healthz || exit 1

ENTRYPOINT ["finelog-server"]
CMD ["--port", "10002", "--log-dir", "/var/cache/finelog/logs"]
```

**No registration, no heartbeat.** Per D3, the `finelog-server` process
does not register itself with the controller. The cluster YAML names the
GCP VM (`gcp://finelog-server`); workers and the controller resolve that
URL through the GCP Compute API on demand. Recovery from a VM
restart/recreate is D11's cached-transport eviction on the next push.

**Operator runbook (VM moves).** When an operator recreates the
`finelog-server` VM:

1. The new VM gets a new internal IP. The VM name is unchanged.
2. In-flight workers' next `RemoteLogClient` push fails with
   connect-refused.
3. `RemoteLogClient` evicts its cached `(host, port)` and re-resolves
   `gcp://finelog-server`. The GCP API returns the new IP.
4. Push succeeds. No worker restart, no controller restart, no operator
   touchpoint beyond replacing the VM.

If the VM **name** changes, operators must update cluster YAML and bounce
the controller (workers pick up the new URL via `WorkerConfig` on next
launch). This is the documented operator step; it is rare and worth a
controller restart.

**Cluster CLI.** `iris cluster log-server restart` is the same shape as any
other system-service lifecycle command. The CLI contract is part of the
reconciler followup, not this project.

---

## Testing strategy (across phases)

### Unit tests

- `lib/rigging/tests/test_resolver.py` (landed) — URL parsing, resolver
  dispatch, gcp:// handler, register_scheme override path.
- `lib/rigging/tests/test_jwt_verifier.py` (landed) — verifier signature
  + expiry. After D4 lands, no revocation tests here (revocation moves
  to iris-side `JwtTokenManager` tests).
- `lib/finelog/tests/` — client behaviour (Pusher batching/backoff,
  connect()), server (MemStore + DuckDB stores), proto round-trips.

### Integration tests

- `lib/finelog/tests/test_end_to_end.py` — real finelog-server on a
  dynamic port; `LogClient.connect("127.0.0.1:<port>").write_batch(...)
  .query(...)` returns the written entries.
- `lib/iris/tests/cluster/test_system_services_config.py` (new) — load a
  YAML with `system_services: {log-server: gcp://finelog-test}`; a fake
  `gcp` handler installed via `register_scheme` for the test resolves
  the URL; assert the controller's `_worker_log_endpoint` matches what
  was configured and `WorkerConfig.log_endpoint` carries it through.
- `lib/iris/tests/e2e/test_attempt_logs.py` (existing, ported) — end-to-
  end task log flow with the split log-server binary. Keystone test.

**Lifecycle tests focused on D11 (the only recovery mechanism).**

- `test_logserver_moves_reresolve`: simulate two GCP IPs returned by a
  fake `gcp_vm_address` — first call returns IP A, second returns IP B
  (mimicking a VM recreation). Spin two `LogServiceImpl` instances on
  those ports. Have a `RemoteLogClient` push successfully against IP A;
  shut down the IP-A server; assert the next push, after a transport
  error, re-resolves and lands on IP B without restarting the client.
- `test_client_cache_eviction_on_rpc_failure`: unit-level check for D11
  — after a transport error, the next call on the same `RemoteLogClient`
  re-invokes `rigging.resolver.resolve(...)`.
- `test_auth_verifier_pure_stateless`: D4 sanity — a token issued by
  `JwtTokenManager` verifies in a stock `JwtVerifier(signing_key)` with
  no revocation list configured.

### Manual / acceptance tests

Before merging Phase 5:

1. `iris cluster up` on a dev cluster with `system_services: {}`.
   Controller starts an in-process finelog (Phase 5.3
   `_start_in_process_finelog`). Submit a tiny job; confirm logs appear
   in `iris logs`.
2. Stand up a finelog-server VM on GCP named `finelog-server`; add
   `system_services: {log-server: gcp://finelog-server}` to cluster YAML;
   restart controller. Submit a job; confirm logs appear.
3. Kill the controller VM mid-job. Restart it. The log-server keeps
   running; workers' in-flight pushes succeed against their cached
   `(host, port)`. Workers launched after restart get the same URL via
   the new controller's `WorkerConfig`.
4. Recreate the `finelog-server` VM mid-job (new IP, same name). The
   first push after the change fails with connect-refused; the cached
   transport is evicted; the next push re-resolves through the GCP API
   and lands on the new IP. No worker restart.

### Pyrefly / lint

Run after each phase, not at the end. Easier to isolate the failure.

### Regression guards

- `lib/iris/tests/cluster/controller/test_logs.py` must keep passing
  verbatim (just with import rewrites).
- Log-scrape SQL queries in `lib/iris/OPS.md` must still work — the DuckDB
  schema is unchanged. This is a documentation-level check.

---

## Risk table

| Risk | Probability | Severity | Mitigation |
| --- | --- | --- | --- |
| `package iris.logging` rename breaks unknown external callers | Medium | Medium | Grep the whole tree for `iris.logging.` literals; communicate in PR description. |
| Worker without log endpoint configured fails at startup | Medium | Low | `WorkerConfig.log_endpoint` is required, populated by the controller from cluster YAML or the dev in-process URL. A misconfigured cluster fails fast at job launch rather than silently dropping logs. |
| GCP API rate-limit / outage during VM-recreate recovery | Low | Medium | Resolver caches `(host, port)`; only a transport error triggers a re-resolve. A burst of failures after a VM move means at most N re-resolves where N = active workers. Acceptable for our scale (~100s of workers). Worth measuring under traffic. |
| Migration straddles multiple PRs; `main` breaks between them | Medium | High | Phases 0–3 are additive-only. Phase 5 per-caller commits. No phase requires a squash-merge across boundaries. |
| DuckDB `.duckdb` files on controller VM get orphaned after split | Low | Low | Document a one-time `gsutil cp` migration in the Phase 7 runbook. Log data in GCS is unaffected. |
| `fetch_logs` latency regresses when CLI talks directly to log-server | Low | Low | Direction is controller→log-server proxy today (extra hop). Going direct is an improvement, not a regression. |
| Dependency direction inversion (finelog importing iris) | Low | High | Enforce via import-lint check in pre-commit. `AGENTS.md` codifies only the iris-downstream edges (`{iris, haliax} → {levanter, zephyr} → marin`); rigging's upstream position is de-facto (verified: `iris/pyproject.toml` declares `marin-rigging`, `rigging/pyproject.toml` has no iris dep). This project should amend `AGENTS.md` to state the rule explicitly: `rigging` upstream of `{iris, finelog, levanter, zephyr, marin}`. |

---

## What this plan deliberately does not do

- Introduce a new `EndpointsService` RPC, or any other service-discovery RPC.
  Per D3, system services live at provider-managed identities; the resolver
  is the only mediator.
- Migrate **actor** endpoints (coordinators registering via
  `ControllerService.RegisterEndpoint`) anywhere. They keep their existing
  shape. The `iris://` resolver scheme already covers the read path for
  out-of-process readers.
- Implement CoreWeave / k8s VM-address helpers. GCP only today
  (`gcp_vm_address`).
- Implement `maybe_proxy` tunnel logic. Off-cluster access lives in the
  architecture-doc appendix; not in scope here.
- Reconcile system-services from cluster YAML (i.e., auto-spin-up VMs).
  We accept that the named VM exists.
- Move `connect-python` / proto infrastructure into a shared location.
  Each package has its own buf.yaml.

These are all followups the architecture doc already calls out.

---

## Open questions resolved

1. **No new `EndpointsService` RPC** — D3. System services bypass the
   controller entirely; the resolver goes straight to GCP.
2. **No heartbeat / self-registration** — D3, D11. The resolver re-resolves
   on transport failure, which is sufficient to follow a VM through a
   recreate.
3. **JWT verifier is pure stateless** — D4. Revocation set lives only on
   the iris-side `JwtTokenManager`; the verifier in rigging just checks
   signature + expiry.
4. **`rigging` does NOT depend on `connect-python`** — confirmed; with
   no `EndpointsService`, no proto infrastructure is needed in rigging.
5. **`str_to_log_level` location** — `finelog.store._types` (D10).
6. **Proto package name: `finelog` vs `finelog.v1`** — `finelog`. Matches
   existing iris convention (unversioned).
7. **`LogMessage` shape** — dataclass. No runtime validation needed for
   internal types.
8. **Controller `FetchLogs` proxy** — keep it until Phase 5.4/5.5 ships as
   a follow-up. Not blocking Phase 6 deletion.
9. **Cluster YAML schema** — `system_services: map<string, string>` (URLs).
   Provider/vm-type/image/health/env stay out until the reconciler
   followup.

## Still-open questions (not blocking — called out for the implementer)

1. **DuckDB page cache behaviour** — the existing log server caps
   `FetchLogs` concurrency at 4 to avoid thrashing. When log-server moves
   to a dedicated VM, the cap may relax. Leave the default at 4; tune on
   real traffic.
2. **GCP API quota during recovery** — see risk table. After a VM move,
   the entire fleet of workers re-resolves at roughly the same time.
   Compute Engine API quotas are generous, but worth confirming under
   our worst-case fleet size.
3. **Off-cluster CLI access** — `iris://` resolution itself works from a
   laptop (load YAML, call `discover_controller`), but the resolved
   address is cluster-internal. Today's CLI handles this via
   `bundle.controller.tunnel(addr)` (`iris/cli/main.py:117`). For
   `gcp://` system-service URLs there is no equivalent, so a laptop
   running `LogClient.connect("gcp://finelog-server")` would fail at
   transport time. The architecture doc's `maybe_proxy()` is the right
   place to solve this; until it ships, off-cluster `iris logs` should
   route through the controller's `FetchLogs` proxy or open an explicit
   tunnel.

---

## Rollout checklist

- [x] Phase 0 landed (`244b57135`); `JwtVerifier` in `rigging.auth`,
      `JwtTokenManager` composes it in `iris.cluster.controller.auth`.
- [x] Phase 1 landed (`244b57135`, `468dbfcaf`, `91027b13d`); resolver
      tests green; `gcp://` scheme works including port override.
- [x] Phase 2 landed (`244b57135`); `iris://` plugin registered for
      actor endpoints. Followup test `test_resolver_plugin.py` pending.
- [ ] Phase 0 followup (D4): drop revocation from `rigging.auth.JwtVerifier`;
      move the set into iris-side `JwtTokenManager`.
- [ ] Phase 3 landed; `lib/finelog` tests green; `lib/iris` unchanged.
- [ ] Phase 4 landed; `IrisClusterConfig.system_services` proto field
      added; controller reads it; no behavior change yet for the dev
      path.
- [ ] Phase 5.1–5.3 landed; worker + task + controller cut over; D11
      re-resolve test green; e2e log flow green.
- [ ] Phase 6 landed; `iris/log_server/`, `iris/cluster/log_store/`,
      `iris/logging.py`, `iris.logging` proto all deleted; iris protos
      reference `finelog.LogEntry` directly.
- [ ] Phase 7 landed; log-server Docker image published; cluster YAML
      reference documented; operator runbook for VM-recreate documented.
- [ ] Optional: Phase 5.4 (CLI goes direct to log-server); Phase 5.5
      (dashboard forwarder removed).
- [ ] Docs updated: `AGENTS.md` (rigging dep-direction rule),
      `lib/iris/AGENTS.md`, `lib/iris/OPS.md`, cluster-config reference.
- [ ] Followup issue filed: CoreWeave/k8s VM-address helpers
      (`coreweave_vm_address`, `k8s_vm_address`).
- [ ] Followup issue filed: `maybe_proxy` off-cluster tunnel integration.
- [ ] Followup issue filed: declarative reconciler for `system_services`
      (auto-provisioning of the named VMs).
