# Marin Services

## Motivation

Logging and bundling were originally embedded in the Iris controller. That was convenient while the system was small: service ports were firewalled, and bundling let us reuse the controller’s existing proxy connection. Once those pieces became performance bottlenecks, extracting them was painful. We had to define a new service, proxy old connections to it, migrate clients one by one, and then tear down the proxy path.

Most of that migration pain came from coupling callers to the original placement of the implementation. A separate client interface from the start — for example, `LogClient` or `StorageClient` — would have let us move logging from in-process, to out-of-process, to an independently deployed VM without touching worker code.

This RFC defines a small service model for Marin/Iris: services are addressed by stable logical names, resolved lazily, and accessed through typed client libraries. The goal is not to introduce a large microservice platform. The goal is to make the ~10 standing services we are likely to need easy to add, move, debug, and replace.

We assume services usually live within a single provider network, such as GCP or CoreWeave, and that VMs inside that network can talk to each other directly. Clients outside the provider network — laptops, dashboards, ad hoc tools — need a proxy, VPN, or tunnel; that is covered in the appendix.

Non-goals:

* 100% uptime
* automatic failover
* multi-region replication
* a general-purpose service mesh
* Kubernetes as a hard dependency

A service going down for a short period is acceptable. The important property is that recovering, moving, or replacing a service does not require rewriting callers.

## Service Guidelines

### Avoid persistent services unless necessary

Prefer processes that run in the context of a user job and write outputs to object storage. Those processes do not need stable APIs, backwards compatibility, monitoring, or independent deployment.

A persistent service is appropriate when multiple jobs need shared state, low-latency coordination, or a common control plane. Once we add one, we should assume it requires maintenance, monitoring, rollout discipline, and a migration path.

### Stabilize the client interface, not the wire format

Callers should depend on a domain client, not generated transport types. This follows the same pattern used by stable SDKs: the public interface is a small library with domain-specific request and response types, while HTTP, Connect, gRPC, protobufs, retries, batching, and transport details remain implementation details.

For example, worker code should call something like:

```python
client = LogClient.connect(endpoint)
client.write_batch(messages)
```

It should not construct RPC stubs or pass generated proto messages directly.

This gives the client room to evolve. It can add buffering, batch small writes, convert polling to streaming, retry idempotent requests, or swap the backing service without changing callers. Duplicating a proto message as a dataclass is often worth it if it keeps the caller isolated from transport churn. Enum-like constants are usually fine to share when they represent real domain concepts.

### Use proto service descriptions

Each standing service should have a proto service definition. Proto gives us a consistent way to describe methods, request/response shapes, streaming behavior, and future Rust/Python interop.

The proto is not the public API. It is the wire contract. The public API is the client library.

This keeps the system easy for humans and agents to inspect while avoiding direct coupling between application code and transport-generated types.

## Design

### Service design

In Connect, the proto service name is part of the client-visible contract. Moving a method from one proto service to another is therefore a breaking change: callers generated against `ServiceA.Method` do not automatically follow the method to `ServiceB.Method`. Supporting that migration requires compatibility shims, duplicate registrations, or proxy methods that forward from the old service to the new one.

That means service boundaries should be chosen carefully. A single mega-service is easy to start with but becomes load-bearing: every method belongs to the same generated client, and splitting it later is disruptive. At the same time, creating a separate network process for every small concern adds operational overhead we do not want.

Prefer one server process that can host multiple smaller proto services.

For example:

```text
logger process
  finelog.LogService
  finelog.AdminService
  rigging.HealthService
```

This gives us smaller generated clients and clearer ownership boundaries without forcing each service definition to become its own VM, container, port, deployment unit, or health check. The deployment unit and the proto service boundary do not need to be the same thing.

Guidelines:

* Do not create one mega-service per process.
* Do not create one process per proto service unless operationally useful.
* Group methods by caller-facing domain, not by current implementation detail.
* Expose clients for each proto sub-service rather than one catch-all client.
* Treat moving a method across proto services as a compatibility event.

This lets us write code against `LogClient`, `EndpointClient`, or `KVClient` rather than `IrisSystemClient`. If logging later moves out of an Iris-adjacent process into a standalone VM, its client and proto service name can remain stable. Only resolution changes.

### Resolution

Services should be referenced by stable logical names, not hardcoded hosts and ports. This is the standard service-discovery pattern used by systems like Kubernetes Services, DNS-based discovery, and gRPC name resolution: callers depend on a logical endpoint, while infrastructure owns placement and addressing.

We do not need a full service mesh or Kubernetes control plane for this. Our scale is small, and most services live inside a single cluster or provider network. A lightweight resolver is enough: first locate the cluster’s endpoints service, then ask it where a named service is running.

Resolution is lazy. Callers pass a URL-like service reference and receive a ready-to-use client. The physical `(host, port)` is resolved at connection time, which gives us room to move a service, restart it on a different VM, or swap implementations without changing callers.

Resolution composes two lookups:

* **Infrastructure lookup:** ask the provider where a named VM or controller lives.
* **Endpoint lookup:** ask a running endpoints service where a named service is registered.

For example:

```python
def endpoint_lookup(server_addr, name) -> tuple[str, int]:
    """Connect to an EndpointsService and resolve `name`."""
    ...
```

Infrastructure lookup is intentionally small:

```python
def vm_address(name: str, provider: str) -> tuple[str, int]:
    match provider:
        case "gcp":
            ...
        case "coreweave":
            ...
        case "k8s":
            ...
```

We already do versions of this in provisioning and cluster startup. This RFC turns that into a shared helper rather than scattering provider-specific lookup logic across services.

### Endpoints service

Iris already has endpoint registration semantics: coordinators register themselves, workers look them up, and the controller acts as the rendezvous point. The problem is that those semantics currently live inside Iris. A logging service, KV service, or future storage service should not need to import Iris internals just to resolve names.

We move the endpoints service into `rigging` as a small reusable primitive. `rigging` is our home for transport-layer building blocks (server runtime, health, endpoints) and is intentionally not Iris-specific — any service, including Iris itself, consumes it as a library.

```text
lib/rigging/endpoints/
  service.proto     # Register, Lookup, List, Unregister
  client.py
  server.py         # in-memory, good enough initially
```

The service provides:

* `Register(name, addr)`
* `Lookup(name)`
* `List()`
* `Unregister(name)`

Iris embeds it like any other service:

```python
from rigging.endpoints import EndpointsService

server.register_service(EndpointsService())
```

Existing in-process endpoint writers become clients of `EndpointsService`. The semantics stay the same; the implementation moves behind a service boundary.

The initial implementation can be in-memory. If controller restarts dropping registrations becomes a problem, we can back it with SQLite or Iris state later. The interface does not need to change.

### URL scheme

For CLI args, config files, and client constructors, we need a compact serialization format for logical service references. URLs are a good fit because they already encode scheme, authority, path, and query parameters.

A service reference is either a bare `host:port` literal or a URL. Bare `host:port` is not a scheme — it is the degenerate case where no resolution is needed, and is handled directly without entering the scheme dispatch. This keeps the common "I know exactly where the service is" case trivial.

URL-form references use two schemes:

```text
iris://<cluster>?endpoint=<name>    # go via a named cluster's endpoints service
gcp://<vm-or-service-name>          # resolve a VM / managed instance by provider name
```

The `<cluster>` authority in `iris://` is a cluster key — the same identifier already used by the `--cluster` flag and cluster config files. Today this is effectively always `marin`, but the scheme is multi-cluster ready: a second deployment (e.g. `openathena`) is just another entry.

Examples:

```text
iris://marin?endpoint=/system/logger       # logger on the marin cluster
iris://openathena?endpoint=/system/logger  # logger on a separate cluster
gcp://log-server                            # skip the cluster hop, hit the VM directly
log-server.internal:8080                    # bare host:port, no resolution
```

`iris://marin?endpoint=/system/logger` resolves in four steps:

1. Resolve the `marin` cluster's Iris controller VM.
2. Connect to the endpoints service on that controller.
3. Look up `/system/logger`.
4. Connect the client to the returned `(host, port)`.

Dispatch is small:

```python
def resolve(ref: str) -> tuple[str, int]:
    # Bare host:port short-circuits — no URL parsing, no lookup.
    if "://" not in ref:
        host, port = ref.rsplit(":", 1)
        return host, int(port)

    url = ServiceURL.parse(ref)
    match url.scheme:
        case "iris":
            controller = vm_address(f"iris-controller-{url.host}", provider="gcp")
            return endpoint_lookup(controller, url.query["endpoint"])
        case "gcp":
            return vm_address(url.host, provider="gcp")
```

A registry of resolver plugins is unnecessary until we have enough schemes to justify it. A single `match` statement is easier to inspect and debug.

`gcp://log-server` and bare `host:port` are first-class: if the endpoints service itself moves off the Iris controller (see §System services), clients can still reach it without going through `iris://`. `iris://` is the convenience form that chases the cluster's endpoints service; the other two forms are escape hatches that skip that hop.

### System services

Some standing services are part of the cluster’s control plane rather than a user job. Current candidates are logging and shared KV. Both are used by the Iris controller itself, so running them as ordinary Iris-submitted jobs creates a bootstrapping problem.

The model is: **system services are peers of the Iris controller, not children of it.** They are deployed and updated independently, owned by the cluster configuration rather than by Iris. Iris consumes them like any other client — by logical URL through the endpoints service — and has no privileged in-process pathway. This is what lets us move logging from in-controller to a dedicated VM without rewriting Iris's callers, and (symmetrically) lets a logging restart not require restarting Iris.

Define system services in the cluster YAML:

```yaml
system-services:
  log-server:
    gcp: { vm-type: ..., tags: ..., image: ... }
    health: /healthz
  kv:
    external: { url: xyz, auth-token: AUTH_TOKEN_ENV }
    health: /weirdo/url
```

Properties:

* started or reconciled during cluster startup
* independently restartable, independent of the Iris controller lifecycle
* registered with the cluster endpoints service on boot
* addressable through the same URL scheme as any other service

Every caller — Iris included — reaches them the same way:

```text
iris://marin?endpoint=/system/logger
```

Operationally, the cluster CLI manages them like any other lifecycle target:

```bash
iris cluster log-server restart
```

Here `iris cluster` is the cluster-management CLI; `log-server` is the system-service name from the YAML. The CLI is not running logger in-process — it is reconciling the declared system service against its provider.

Endpoints update when a service moves to a new VM. Callers continue to use the logical name.

### Example: lifting out logging

Logging is a good first service because the current controller coupling has already caused migration pain, and the domain boundary is clear.

Extract it into its own package:

```text
lib/finelog/Dockerfile
src/finelog/
  client.py
  server.py
  duckdb_store.py
  mem_store.py
  logging.proto
scripts/cli.py
```

The public client is domain-level:

```python
@dataclass
class LogQuery:
    source: str | None = None
    body_like: str | None = None

class LogClient:
    @staticmethod
    def connect(endpoint: str | tuple[str, int]) -> "LogClient": ...

    def write_batch(self, messages: Sequence[LogMessage]) -> None: ...
    def query(self, query: LogQuery) -> Sequence[LogRecord]: ...
```

The CLI resolves and connects:

```python
client = LogClient.connect("iris://marin?endpoint=/system/logger")
```

From a cluster VM, this resolves directly to the logger’s internal address. From a laptop, it runs inside a proxy context, described in the appendix. The client API is the same in both cases.

There is no controller-specific proxy service and no `iris-proxy://` dispatch. The controller hosts a generic endpoints service; logging is just one registered service.

## Open Questions

### Detecting on-cluster vs off-cluster

Remote clients need a proxy; cluster-internal clients do not.

Options:

* explicit flag or context manager
* environment variable
* metadata server probe
* route probing

Recommendation: prefer explicit behavior. Hidden auto-detection can be surprising, especially on developer machines with partial cloud access.

### URL scheme extensibility

We currently need `iris://`, `gcp://`, and bare `host:port` support. Bare `host:port` is handled as a pre-match short-circuit; only URL-form references hit the scheme dispatch.

Options for additional schemes:

* single `match` statement
* resolver registry
* plugin mechanism

Recommendation: use a `match` statement until we have at least four or five schemes and real extension pressure.

## Known Followups

### Migrate Iris endpoint writers

Endpoints are currently coupled to Iris directly. We'll need to adjust these
users to operate against the generic Endpoints service with the Iris controller
as the target.

1. Embed `EndpointsService` in the controller.
2. Migrate existing writers to use `EndpointsClient`.
3. Migrate readers to use resolver URLs.
4. Remove the in-process endpoint path.

### Add provider coverage for `vm_address`

Implement provider lookup incrementally:

1. GCP
2. CoreWeave

The interface should stay stable:

```python
vm_address(name: str, provider: str) -> tuple[str, int]
```

## Appendix: Remote access

Inside a provider network, VMs talk to each other directly. Clients outside the provider network need a tunnel, proxy, or VPN. Today that usually means SSH forwarding through a bastion. We want the common case to be ergonomic without leaking tunnel details into every client constructor.

A small context helper handles the distinction:

```python
with maybe_proxy("marin"):
    client = LogClient.connect("iris://marin?endpoint=/system/logger")
```

On a cluster VM, this is a no-op. Off-cluster, it establishes the needed tunnel and makes service resolution work through it. If we later adopt Tailscale, Cloudflare Zero Trust, or another private networking layer, `maybe_proxy` can become a no-op everywhere.

### Option 1: SOCKS via `ssh -D`

SOCKS gives us one tunnel for many services. The client chooses the destination per request, which fits dynamic service discovery well.

Example environment:

```bash
ALL_PROXY=socks5h://localhost:<port>
```

This works with common HTTP clients:

* Python `httpx` with `httpx[socks]`
* Rust `reqwest` with the `socks` feature

The drawback is library support. `grpc-python` does not support SOCKS as a client-side forward proxy, and upstream support has historically been limited. If our Connect/RPC stack uses an HTTP transport built on `httpx`, SOCKS is a good default. If it depends on grpc-python, SOCKS is not sufficient.

### Option 2: per-service `ssh -L`

Local port forwarding allocates one local port per remote service:

```text
localhost:47829 -> log-server:8080
localhost:47830 -> kv:8080
```

This requires more bookkeeping, but it is universally compatible because client libraries only see `localhost:<port>`.

A proxy session can manage this lazily:

```python
with proxy_session("marin") as proxy:
    local = proxy.local_addr("log-server")
    client = LogClient.connect(local)
```

The session owns the `{service_name: local_port}` map and opens forwards on demand.

### Option 3: private networking overlay (Tailscale, Cloudflare Zero Trust)

A mesh VPN or zero-trust proxy removes the tunnel problem entirely. Laptops, CI runners, and cluster VMs all become addressable on the same private network; the tunnel layer disappears from application code, and `maybe_proxy` becomes a no-op everywhere. There is no client-library compatibility matrix to maintain (`grpc-python`, SOCKS, `ALL_PROXY`, etc.) because the transport below the client is just normal TCP over a routed interface.

Rough cost for our scale (~25 users, pricing as of 2026):

| Option | Plan | Cost/user | Annual @ 25 users | Notes |
| --- | --- | --- | --- | --- |
| Tailscale | Personal (free) | $0 | $0 | capped at 6 users — too small for us |
| Tailscale | Standard | $8/mo | ~$2.4k/yr | unlimited users and devices |
| Tailscale | Premium | $18/mo | ~$5.4k/yr | JIT access, network flow logs, priority support |
| Cloudflare Zero Trust | Free | $0 | $0 | up to 50 users — covers us today |
| Cloudflare Zero Trust | Pay-as-you-go | $7/mo | ~$2.1k/yr | billed annually, no user cap |

Cloudflare Zero Trust's free tier is interesting precisely because it covers our current user count. The tradeoffs vs. Tailscale are the usual ones: Tailscale is a simpler WireGuard-style mesh; Cloudflare is an identity-aware reverse proxy with more policy surface and more lock-in. Either removes the SOCKS/`ssh -L` apparatus.

### Recommendation

Near-term, standardize on HTTP-based Connect clients that support SOCKS, and keep the `ssh -L` helper for libraries (notably `grpc-python`) that ignore proxy env vars. This keeps us unblocked without a new vendor dependency.

Medium-term, seriously consider a private networking overlay — most likely Cloudflare Zero Trust's free tier while we are under 50 users, or Tailscale Standard once we exceed that threshold. Adopting one is non-breaking: `maybe_proxy` becomes a no-op, and service URLs and client APIs do not change.

Auth remains SSH-based for now. A future private-networking layer can replace the tunnel implementation without changing service URLs or client APIs.
