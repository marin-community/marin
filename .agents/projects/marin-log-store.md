# Marin Log-Store Extraction Plan

**Status:** Draft · 2026-04-24
**Owner:** Russell
**Related design:** `docs/design/marin-service-architecture.md`

## Goal

Lift logging out of `lib/iris` into a new `lib/finelog` package, and in doing so
introduce the generic service primitives the architecture doc calls for:

1. A reusable `EndpointsService` in `lib/rigging` (Register / Lookup / List /
   Unregister) that Iris hosts but does not own.
2. A URL-scheme resolver in `lib/rigging` supporting `iris://`, `gcp://`, and
   bare `host:port`.
3. A domain-level `LogClient` / `LogServer` pair in `lib/finelog`, independent
   of the Iris controller process.
4. A Dockerfile and declarative `system-services` entry so the log-server can
   run on its own VM and be restarted independently of the Iris controller.

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

- Utilities only: `filesystem.py`, `timing.py`, `log_setup.py`, `distributed_lock.py`, `config_discovery.py`.
- No proto infra, no buf.yaml, no service-layer code.
- Dependencies are limited to `fsspec`, `gcsfs`, `s3fs`. No protobuf.

### Authn wrinkle

`iris.log_server.main._build_auth_interceptors` (line 112) uses a
**function-scope import** of `iris.cluster.controller.auth.JwtTokenManager` —
the log server verifies tokens signed by the controller. The function-scope
import is a deliberate workaround acknowledged in the existing code comments,
because a top-level import would violate the iris rule
"all imports at top of file." When the log server moves out of `lib/iris`,
that workaround is no longer viable: any import shape
(`finelog → iris.controller.auth`) is a reverse dependency.

This forces a concrete extraction: `JwtTokenManager` splits into
`rigging.auth.JwtVerifier` (stateless verify/decode) and a thin iris-side
issuer (DB-backed token creation, revocation). Both the log server and the
controller import the verifier from rigging. See D4 and Phase 0.

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

### D3. We do not add a new endpoints RPC at all

(Revised 2026-04-25: was originally "add `rigging.endpoints.EndpointsService` as
a second service alongside `ControllerService`." That added too much surface for
the actual benefit at our scale.)

Two independent things carry the word "endpoint":

- **Actor endpoints** (job-scoped rendezvous written by coordinators and read
  by workers — the `EndpointStore` SQL table).
- **System endpoints** (cluster-scoped service addresses like
  `/system/log-server`).

The architecture doc's `EndpointsService` was a generic primitive for the
latter. We can defer that abstraction: today there's exactly one cluster type
(`marin`), and iris's existing `ControllerService.ListEndpoints` already
returns both flavors merged (`_system_endpoints` dict + `EndpointStore`). We
piggy-back on it.

In this project we:

- **Do NOT** add a new `EndpointsService` proto/server/client.
- For system-service lookup, callers route through the URL resolver (D5),
  whose `iris://` handler is a plugin that calls
  `ControllerServiceClientSync.list_endpoints(prefix=..., exact=True)`.
- The plugin lives in `iris.client`, not `rigging`. `rigging.resolver` stays
  proto-free and has no iris dependency.
- System-service addresses are still seeded into the controller's existing
  `_system_endpoints` dict from cluster YAML at boot, exactly as today.
- Self-registration RPCs are out of scope; cluster orchestration places
  services and writes their addresses into config.

This keeps the blast radius small and is genuinely simpler than the original
two-service plan: no new wire surface, no new bootstrap order, no
double-source-of-truth between actor and system endpoints.

### D4. JWT verifier moves to `lib/rigging`

`JwtTokenManager` has two halves: issuance (needs DB for revocation) and
verification (stateless). The verifier half moves to
`rigging.auth.JwtVerifier`. Iris keeps the issuer (a subclass / composition)
in `lib/iris`. This is strictly additive; no wire changes. The concrete
extraction lives in **Phase 0** (see §Phasing) — it ships before anything
else because both the new log-server process and the existing in-iris
verification path depend on it.

### D5. URL-scheme resolver is a small registry

(Revised 2026-04-25: was originally a single `match` statement; we found a
registry buys nontrivial decoupling at almost no cost — `rigging` doesn't have
to know about iris-specific RPCs.)

The resolver in `lib/rigging/src/rigging/resolver/` exposes:

```python
def register_scheme(scheme: str, handler: Callable[[ServiceURL], tuple[str, int]]) -> None: ...
def resolve(ref: str) -> tuple[str, int]: ...
```

`resolve` short-circuits bare `host:port`, parses URLs into `ServiceURL`, and
dispatches to a registered handler. Built-in handlers:

- `gcp://<vm-name>` — registered eagerly in `rigging.resolver`. Uses
  `vm_address(name, provider="gcp")`.

`iris://` is **not** built into rigging. `iris.client.resolver_plugin` calls
`register_scheme("iris", ...)` at import time; the handler uses iris's
existing `ControllerServiceClientSync.list_endpoints`. Importing `iris.client`
activates the plugin. Off-cluster contexts that don't import iris simply
can't resolve `iris://`, which is the right failure mode.

The architecture doc calls out `coreweave://` and `k8s://` as **known
followups** — we do not stub them in this project. A fresh scheme is a
one-line `register_scheme` call from whichever package owns the cluster type.

### D6. Resolver is synchronous and returns `(host, port)`

Matches the signature already used by `LogPusher`'s existing `resolver`
callback, so callers can swap implementations without touching call sites.

### D7. No auto on/off-cluster detection — and no `maybe_proxy` stub in this project

The architecture doc's open question — recommendation is "prefer explicit."
We defer `maybe_proxy()` and the whole remote-access appendix to a followup
project. The CLI invocation shape does not need to leak tunnel details
today: the CLI runs on a cluster VM via the existing SSH path, same as
today.

### D8. Cluster YAML gains `system-services:`

```yaml
system-services:
  log-server:
    provider: gcp
    vm-type: n2-standard-4
    image: marin/finelog:${TAG}
    health: /healthz
    env:
      FINELOG_REMOTE_DIR: gs://marin/logs
```

Reconciliation is the cluster CLI's job, not the controller's. For this
migration we wire the declaration into config-loading only; the actual
reconciler is added in a sibling phase (§4).

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

### D11. `RemoteLogClient.connect()` caches, re-resolves on failure

The resolved `(host, port)` is cached for the lifetime of the client. On
any RPC-layer error that indicates the target has gone away (connect
refused, TLS handshake failure, HTTP 5xx from a stale VM), the client
evicts its cached transport and re-runs the resolver on the next call.
This matches today's `LogPusher` retry/backoff semantics and lets us move
the log server to a new VM without restarting workers.

### D12. Restart-independence is a prod-config guarantee, not a dev-config one

The goal "controller restarts do not restart the log server" holds **only**
when `system_services.log-server.address` is set in cluster config. In the
dev / single-box path (`system_services: {}`), the log server runs
in-process as a sibling thread; controller restart does restart it — same
as today. Document this in the cluster YAML reference.

---

## Phasing

Phases are ordered so each one can land, be tested, and be reverted on its own.
Phase 0 is an auth prerequisite. Phases 1–3 are strictly additive (no Iris
behavior change). Phase 4 is the bootstrap glue. Phase 5 cuts over Iris
internals. Phase 6 deletes the old paths. Phase 7 is operational polish.

**Core migration:** Phases 0–6 minus the "optional cut-overs" noted inside
Phase 5 (CLI direct-connect and dashboard forwarder deletion).

**Operational followup:** Phase 7 and the optional Phase 5 cut-overs can land
as a separate PR stack and do not block deletion in Phase 6.

### Phase 0 — Split `JwtTokenManager` into verifier + issuer

Smallest possible step, runs before proto work. Pure refactor, no wire
changes.

- Extract the stateless verify/decode half of
  `iris.cluster.controller.auth.JwtTokenManager` into
  `rigging.auth.JwtVerifier`.
- Leave the issuer half (DB-backed token creation, revocation list
  hydration) in `iris.cluster.controller.auth`; have it compose a
  `JwtVerifier` rather than subclassing.
- Update `iris.log_server.main._build_auth_interceptors` to import
  `JwtVerifier` from rigging at the top of the file, removing the
  function-scope workaround.
- Port existing JWT tests to cover the verifier in its new home.

**Done when:** `uv run pyrefly` passes, all existing auth tests pass,
`JwtTokenManager` no longer exposes its verify half directly.

### Phase 1 — Resolver registry in `lib/rigging`

(Revised 2026-04-25: this phase originally added an `EndpointsService` proto +
in-memory server + Connect client. That was struck out per D3. The phase now
delivers only the resolver, which keeps `rigging` proto-free.)

Phase 1 has only the resolver infrastructure (formerly Phase 2). No new RPCs,
no buf.yaml, no proto generation in rigging.

The resolver and its layout move here from the old Phase 2; see §Phase 2
below for the full spec.

### Phase 2 — Resolver in `lib/rigging`

**Add the URL-scheme resolver.**

```
lib/rigging/src/rigging/resolver/
  __init__.py     # exports resolve, ServiceURL, vm_address
  url.py          # ServiceURL parse/format
  providers.py    # vm_address(name, provider) with gcp/coreweave/k8s match
  resolver.py    # resolve(ref: str) -> tuple[str, int]
```

`ServiceURL`:

- Parses `scheme://authority?endpoint=<path>`.
- Uses stdlib `urllib.parse`. No URL validation beyond scheme / authority /
  query-param extraction.

`resolve`:

```python
_HANDLERS: dict[str, Callable[[ServiceURL], tuple[str, int]]] = {}

def register_scheme(scheme: str, handler) -> None:
    _HANDLERS[scheme] = handler

def resolve(ref: str) -> tuple[str, int]:
    if "://" not in ref:
        host, port = ref.rsplit(":", 1)
        return host, int(port)
    url = ServiceURL.parse(ref)
    handler = _HANDLERS.get(url.scheme)
    if handler is None:
        raise ValueError(f"unsupported scheme: {url.scheme!r}")
    return handler(url)
```

`gcp://` is registered eagerly at module-load time inside
`rigging.resolver`:

```python
def _resolve_gcp(url: ServiceURL) -> tuple[str, int]:
    return vm_address(url.host, provider="gcp")

register_scheme("gcp", _resolve_gcp)
```

`iris://` is **not** registered by rigging — that handler lives in
`iris.client.resolver_plugin` (see Phase 1.5). Off-cluster code that doesn't
import iris cannot resolve `iris://` by design.

`vm_address`:

- Initial implementation: **GCP only**. Non-gcp `provider` values raise
  `ValueError(f"unsupported provider: {provider}")`. CoreWeave and k8s are
  known followups per the architecture doc.
- Accepts `name` and `provider`; returns `(host, port)` by metadata server
  lookup (GCP instance attributes). Tests mock the GCP client.

**Where does the GCP lookup code live today?** Provisioning has bits of this
scattered; we'll consolidate them here in a followup, not in this phase. For
now, `vm_address(gcp)` uses the same `google.cloud.compute_v1` client that
iris already uses (already a transitive dep via `google-cloud-tpu`).

**Tests.** Under `lib/rigging/tests/resolver/`:

- `test_url.py` — parse / format round-trips, bare `host:port`, missing scheme,
  missing authority, query param extraction.
- `test_resolve.py` — resolve(bare) short-circuits; resolve(gcp://) uses
  mocked `vm_address`; unknown scheme raises; `register_scheme` allows a
  test-local handler to be installed and called.
- `test_providers_gcp.py` — `vm_address` with a mock GCP client.

**Done when:** `cd lib/rigging && uv run pytest` is green.

### Phase 1.5 — Iris resolver plugin

A small new module `lib/iris/src/iris/client/resolver_plugin.py`. On import
it registers an `iris://` handler:

```python
from rigging.resolver import register_scheme, ServiceURL
from rigging.resolver.providers import vm_address
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.controller_pb2 import Controller as _Controller

def _resolve_iris(url: ServiceURL) -> tuple[str, int]:
    cluster = url.host
    name = url.query["endpoint"]
    controller_host, controller_port = vm_address(
        f"iris-controller-{cluster}", provider="gcp"
    )
    client = ControllerServiceClientSync(
        address=f"http://{controller_host}:{controller_port}"
    )
    response = client.list_endpoints(
        _Controller.ListEndpointsRequest(prefix=name, exact=True)
    )
    if not response.endpoints:
        raise KeyError(f"endpoint not found: {name}")
    address = response.endpoints[0].address
    host, port = address.rsplit(":", 1)
    return host, int(port)

register_scheme("iris", _resolve_iris)
```

Importing `iris.client` activates the plugin (via a side-effect import in
`iris/client/__init__.py`). Off-cluster code that doesn't import iris has no
`iris://` handler and gets `unsupported scheme` — that's the right failure.

**Tests.** Under `lib/iris/tests/client/`:

- `test_resolver_plugin.py` — registers a fake `vm_address` and a fake
  in-process `ControllerService` to verify the iris:// path. Use the
  existing iris testing patterns for spinning a controller stub.

**Done when:** `iris.client` import installs the handler and a workspace
import of `rigging.resolver.resolve("iris://marin?endpoint=/x")` succeeds
in an iris-imported context.

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

### Phase 4 — Embed `EndpointsService` in the Iris controller

**Single integration point.** In `iris.cluster.controller.controller.py`
where the controller's ASGI stack is assembled, add one more mount:

```python
from rigging.endpoints import EndpointsServiceImpl
from rigging.endpoints_connect import EndpointsServiceWSGIApplication

endpoints_svc = EndpointsServiceImpl()
# Pre-seed ONLY the explicitly-configured external services. Self-registered
# system services (Phase 5.3's in-process log server, Phase 7's out-of-
# process log server) register themselves via the service's own
# EndpointsClient — Phase 4 does not seed those to avoid race conditions
# with the self-registering writer.
for name, svc in cluster_config.system_services.items():
    if svc.address:
        endpoints_svc.register(f"/system/{name}", svc.address)

routes.append(Mount(
    EndpointsServiceWSGIApplication.path,
    app=WSGIMiddleware(EndpointsServiceWSGIApplication(service=endpoints_svc))
))
```

The existing `ControllerService.ListEndpoints` RPC is unchanged. We now have
**two** ways to look up `/system/log-server`:

- Legacy: `ControllerService.ListEndpoints(prefix="/system/log-server")` —
  still works; merges actor and system endpoints.
- New: `EndpointsService.Lookup(name="/system/log-server")` — only system
  endpoints.

New callers (phase 5) use the new path. Legacy callers keep working.

**Cluster config.** Add a `system_services` section to cluster YAML parsing
(`lib/iris/src/iris/cluster/config.py` or wherever `IrisClusterConfig` is
defined):

```python
@dataclass
class SystemServiceConfig:
    provider: str               # "gcp" | "external"
    address: str | None = None  # for external or already-provisioned services
    vm_type: str | None = None
    image: str | None = None
    health: str = "/healthz"
    env: dict[str, str] = field(default_factory=dict)

@dataclass
class IrisClusterConfig:
    ...
    system_services: dict[str, SystemServiceConfig] = field(default_factory=dict)
```

The controller only reads `address` from this config and seeds the
`EndpointsService`. **Provisioning** (spinning up the actual VM) is a sibling
concern; for this phase we accept a pre-provisioned address (`address:
10.0.0.5:10002` or `iris-controller-marin.internal:10002`). Full
declarative provisioning is a followup.

**Tests.**

- `lib/iris/tests/cluster/controller/test_endpoints_service.py` — controller
  starts, exposes EndpointsService over its port, pre-seeded entries are
  present, Register/Unregister work.
- Existing actor-endpoint tests must not regress.

**Done when:** both endpoint services run side-by-side in the controller and
a client can round-trip through `rigging.resolver.resolve("iris://marin?endpoint=/system/log-server")`.

### Phase 5 — Cut Iris over to `finelog.client`

**This is the only risky phase.** We are rewriting imports in the running
controller and worker. Do it in one sequenced commit per caller, not all at
once.

**5.1 — Worker.**

`lib/iris/src/iris/cluster/worker/worker.py`:

- Delete `_resolve_log_service`.
- Replace `LogPusher(server_url=..., resolver=self._resolve_log_service)` with:

  ```python
  from finelog.client import RemoteLogClient, LogPusher, StderrLogClient

  log_endpoint = self._worker_config.log_endpoint
      or f"iris://{self._cluster_name}?endpoint=/system/log-server"

  try:
      log_client = RemoteLogClient.connect(log_endpoint)
  except Exception:
      # Off-cluster or unavailable — fall back to stderr so task logs
      # aren't silently dropped.
      logger.warning("log server unreachable; using stderr fallback")
      log_client = StderrLogClient()

  self._log_pusher = LogPusher(log_client)
  ```

- `WorkerConfig` gains `log_endpoint: str | None = None`.

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

- Delete top-of-file imports from `iris.log_server.*` (lines 110–112).
- Delete the in-process LogPusher / LogServiceProxy wiring (lines
  ~1045–1071), including the `self._log_server: uvicorn.Server | None`
  field at 1055.
- Delete the `_start_local_log_server` method and its call site at 1060
  (method body runs ~1162–1201).
- Delete the `self._service._system_endpoints["/system/log-server"] = ...`
  insertion (line ~1256).
- Delete the dashboard's legacy `PushLogs` forwarder at
  `controller.py:281–282` (unless deferred to the optional follow-up
  stack).
- Replace with:

  ```python
  from finelog.client import RemoteLogClient, LogPusher
  from finelog.server.app import build_asgi as build_finelog_asgi
  from finelog.server.service import LogServiceImpl

  log_svc_cfg = config.system_services.get("log-server")
  if log_svc_cfg and log_svc_cfg.address:
      # Prod path: out-of-process log server declared in cluster YAML.
      log_client = RemoteLogClient.connect(log_svc_cfg.address)
  else:
      # Dev / single-box mode: finelog code bound in a local uvicorn
      # thread, registered in our own EndpointsService. Same behaviour as
      # today's _start_local_log_server, just reading finelog's modules.
      log_client = self._start_in_process_finelog(endpoints_svc)

  self._controller_log_pusher = LogPusher(log_client)
  ```

- `_start_in_process_finelog` is the Phase-5.3 replacement for
  `_start_local_log_server`. It:
  1. Builds a `LogServiceImpl` against a MemStore (dev) or
     `DuckDBLogStore` (if configured).
  2. Spawns a uvicorn thread on a dynamic port (existing
     `self._threads.spawn_server` pattern).
  3. Registers `/system/log-server` into the controller's
     `EndpointsServiceImpl` directly (same process, same object — no RPC
     hop to register).
  4. Returns a `RemoteLogClient` that targets that local port, so the
     controller and the workers use the same code path (no
     in-process-only client type).

- The subprocess supervisor `iris.cluster.controller.main._start_log_server`
  (main.py:47, Popen at :230) is removed outright in Phase 5.3. Anyone
  running the old bash flow should use the new Phase 7 Docker image. No
  `--legacy-inprocess-log-server` flag.

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

**Registration on boot.** The `finelog-server` entrypoint registers itself
with the cluster EndpointsService on startup:

```python
# finelog/server/main.py
if os.environ.get("FINELOG_REGISTER_URL"):
    # e.g. "iris://marin"
    client = EndpointsClient.connect(resolve_controller(os.environ["FINELOG_REGISTER_URL"]))
    client.register("/system/log-server", f"{my_host}:{port}")
```

Controller restarts drop registrations (because the EndpointsServiceImpl is
in-memory). The log server re-registers on a periodic heartbeat (every 30s)
to heal this. That heartbeat is part of Phase 7, not Phase 3.

**SQLite-backed endpoints (followup, not in this plan).** The architecture
doc calls this out as "if controller restarts dropping registrations becomes a
problem, we can back it with SQLite." Heartbeat is simpler; defer the SQLite
work.

**Cluster CLI.** `iris cluster log-server restart` invokes the same path as
any other system service. The CLI contract for this is part of the
provisioning followup, not this project.

---

## Testing strategy (across phases)

### Unit tests

Each phase lands its own pytest suite:

- `lib/rigging/tests/endpoints/` — Register/Unregister/Lookup/List semantics.
- `lib/rigging/tests/resolver/` — URL parsing, resolver dispatch, provider
  stubs.
- `lib/finelog/tests/` — client behaviour (Pusher batching/backoff, Stderr
  fallback, connect()), server (MemStore + DuckDB stores), proto round-trips.

### Integration tests

- `lib/rigging/tests/test_end_to_end.py` — real uvicorn process hosting
  EndpointsService; resolver round-trips.
- `lib/finelog/tests/test_end_to_end.py` — real finelog-server on a dynamic
  port; `LogClient.connect("host:port").write_batch(...).query(...)` returns
  the written entries.
- `lib/iris/tests/cluster/controller/test_endpoints_service.py` — Iris
  controller exposes EndpointsService, preseeds system-services from config,
  a client outside iris can resolve them.
- `lib/iris/tests/e2e/test_attempt_logs.py` (existing, ported) — end-to-end
  task log flow with the split log-server binary. This is the keystone test.

**Multi-process lifecycle tests (new — one of the review's blocker calls).**

- `test_bootstrap_order_controller_before_logserver`: start a controller
  with `system_services.log-server.address` pointing at a port that is
  not yet listening. Start a worker. Worker's first `push()` must fall
  back to `StderrLogClient` **or** retry-and-backoff without crashing.
  Then start the finelog-server at that address; the worker's next
  `push()` must succeed. This exercises the "controller exists, log
  server does not yet exist" window.
- `test_heartbeat_reregister_after_controller_restart`: start finelog
  server with `FINELOG_REGISTER_URL=iris://marin`; it registers. Kill and
  restart the controller process (discarding the in-memory
  EndpointsService state). Within one heartbeat interval, `Lookup` must
  return the log server again. Use a short interval in the test fixture
  (1s) so this runs in seconds.
- `test_logserver_moves_vm_reresolve`: simulate the log server process
  moving to a new address. Confirm `LogPusher` / `RemoteLogClient`
  re-resolves through `EndpointsService` rather than reusing the stale
  `(host, port)` (D11 behaviour).
- `test_client_cache_eviction_on_rpc_failure`: unit-level check for D11:
  after a transport error, the next call on the same `RemoteLogClient`
  re-invokes the resolver.
- `test_auth_verifier_shared_between_controller_and_logserver`: Phase 0
  sanity — a token issued by the controller's `JwtTokenManager` passes
  verification in `finelog-server`'s `JwtVerifier`.

### Manual / acceptance tests

Before merging Phase 5:

1. `iris cluster up` on a dev cluster with `system_services: {}` (no
   log-server configured). Controller should start an in-process log server
   via the Phase 5.3 `_start_local_inprocess_log_server` path. Submit a tiny
   job; confirm logs appear in `iris logs`.
2. Stand up a finelog-server container on a second VM; add `log-server:`
   under `system_services` in cluster config with that address; restart
   controller. Submit a job; confirm logs appear.
3. Kill the controller VM mid-job. Restart it. The log-server keeps running;
   the worker's next push re-resolves through the new controller's
   EndpointsService (populated from the heartbeat in Phase 7).
4. Kill the log-server VM mid-job. Workers fall back to stderr (visible in
   `docker logs`). Restart log-server. Workers resume pushing (cached client
   re-resolves after the first failed send — existing behaviour of
   `LogPusher`).

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
| `JwtTokenManager` split creates cycle | Medium | Low | Move verifier to `rigging.auth`. Iris imports rigging already. |
| In-memory EndpointsService drops registrations on controller restart | High | Medium | Log-server heartbeat every 30s (Phase 7). |
| Worker without log-server endpoint drops task logs silently | Medium | High | `StderrLogClient` fallback + explicit warn log. |
| Migration straddles multiple PRs; `main` breaks between them | Medium | High | Phases 1–3 are additive-only. Phase 5 per-caller commits. No phase requires a squash-merge across boundaries. |
| DuckDB `.duckdb` files on controller VM get orphaned after split | Low | Low | Document a one-time `gsutil cp` migration in the Phase 7 runbook. Log data in GCS is unaffected. |
| `fetch_logs` latency regresses when CLI talks directly to log-server | Low | Low | Direction is controller→log-server proxy today (extra hop). Going direct is an improvement, not a regression. |
| Dependency direction inversion (finelog importing iris) | Low | High | Enforce via import-lint check in pre-commit. `AGENTS.md` codifies only the iris-downstream edges (`{iris, haliax} → {levanter, zephyr} → marin`); rigging's upstream position is de-facto (verified: `iris/pyproject.toml` declares `marin-rigging`, `rigging/pyproject.toml` has no iris dep). This project should amend `AGENTS.md` to state the rule explicitly: `rigging` upstream of `{iris, finelog, levanter, zephyr, marin}`. |

---

## What this plan deliberately does not do

- Migrate **actor** endpoints (coordinators registering via `ControllerService.RegisterEndpoint`) to `EndpointsService`. That is the architecture doc's "Migrate Iris endpoint writers" followup; do it after this.
- Implement `vm_address` for CoreWeave or k8s. Stub only.
- Implement `maybe_proxy` tunnel logic. No-op shim only.
- Introduce SQLite-backed endpoint persistence. Heartbeats only.
- Reconcile system-services from cluster YAML (i.e., auto-spin-up VMs). We
  accept pre-provisioned addresses.
- Move `connect-python` / proto infrastructure into a shared location. Each
  package has its own buf.yaml.

These are all followups the architecture doc already calls out.

---

## Open questions resolved

1. **`rigging` depending on `connect-python`** — accepted. Unavoidable once
   it hosts a service. connect-python is small and iris already pulls it in.
2. **`str_to_log_level` location** — `finelog.store._types` (D10).
3. **Proto package name: `finelog` vs `finelog.v1`** — `finelog`. Matches
   existing iris convention (unversioned).
4. **`LogMessage` shape** — dataclass. Consistent with rest of
   iris/rigging; no runtime validation needed for internal types.
5. **Controller `FetchLogs` proxy** — keep it until Phase 5.4/5.5 ships as a
   follow-up. Not blocking Phase 6 deletion.

## Still-open questions (not blocking — called out for the implementer)

1. **Heartbeat interval and jitter** — Phase 7 says 30s. Under heavy load or
   JWT TTL constraints this may need to shift. Implementer picks a value
   and documents it in Phase 7's PR.
2. **DuckDB page cache behaviour** — the existing log server caps
   `FetchLogs` concurrency at 4 to avoid thrashing. When log-server moves
   to a dedicated VM, the cap may relax. Leave the default at 4; tune on
   real traffic.
3. **Cluster YAML schema finalization** — `SystemServiceConfig` shape in D8
   is a straw-man. The `provider`/`vm_type`/`image` fields are not used
   until the reconciler followup; should they live there now, or should
   Phase 4 accept only `address` and defer the full schema?

---

## Rollout checklist

- [ ] Phase 0 landed; `JwtVerifier` in `rigging.auth`, issuer composes it in `iris.cluster.controller.auth`.
- [ ] Phase 1 landed; `lib/rigging` tests green; `EndpointsService` round-trips.
- [ ] Phase 2 landed; resolver tests green; `iris://` and `gcp://` schemes work.
- [ ] Phase 3 landed; `lib/finelog` tests green; `lib/iris` unchanged.
- [ ] Phase 4 landed; controller exposes `EndpointsService`; behaviour change = nil (no log-server reconfigured yet).
- [ ] Phase 5.1–5.3 landed; worker + task + controller cut over; bootstrap-order and heartbeat tests green; e2e log flow green.
- [ ] Phase 6 landed; `iris/log_server/`, `iris/cluster/log_store/`, `iris/logging.py`, `iris.logging` proto all deleted; iris protos reference `finelog.LogEntry` directly.
- [ ] Phase 7 landed; log-server Docker image published; heartbeat wired; cluster YAML reference documented.
- [ ] Optional: Phase 5.4 landed (CLI goes direct to log-server); Phase 5.5 landed (dashboard forwarder removed).
- [ ] Docs updated: `AGENTS.md` (rigging dep-direction rule), `lib/iris/AGENTS.md`, `lib/iris/OPS.md`, cluster-config reference.
- [ ] Followup issue filed: migrate actor endpoints to `EndpointsService`.
- [ ] Followup issue filed: CoreWeave/k8s `vm_address` providers.
- [ ] Followup issue filed: `maybe_proxy` off-cluster tunnel integration.
- [ ] Followup issue filed: declarative reconciler for `system_services` (auto-provisioning).
- [ ] Followup issue filed: SQLite-backed `EndpointsService` (currently in-memory + heartbeat).
