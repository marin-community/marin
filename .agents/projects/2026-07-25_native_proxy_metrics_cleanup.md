# Native-proxy RPC telemetry: uniform cleanup

Status: implemented (2026-07-25), revised after codex peer review. Weaver issue #160.
The plan below is the original design; **read the As-built section first** for how
the shipped branch diverges from it.

## As-built (2026-07-25)

- **WS2 was already merged; this PR adds only test coverage.** The finelog server
  already stamps a credential-bound origin `cluster` on forwarded telltale/stats
  rows (`write_origin_cluster` + `stamp_cluster_column`, the stats-plane twin of
  the log path's `authorized_cluster`; the #138 line of work). Local writes leave
  the column empty/NULL. So the P1 hub-mixing fix is **panel-side only** (WS3's
  cluster filter), not a finelog schema change. The PR adds Rust unit tests that
  pin the guarantee: a forwarding JWT stamps its key's cluster, a spoofed
  caller-supplied `cluster` is overwritten, a trusted-network local write stamps
  nothing, and a write with no identity is refused (fails closed). CI defers
  finelog `cargo test`, so these run locally (4/4 green) and stand as executable
  documentation of the invariant the panel filter relies on.
- **No native wheel is published in this PR.** CI builds `marin-iris-native` from
  source via the `setup:rust` tag (the diff touches `lib/iris/rust/`); the
  requester drives the deploy/wheel rollout. The Rollout section is the contract
  to follow then, not work done here.
- **WS1 bounding is cap + idle eviction + overflow, cumulative counters.**
  `scope=total` is an exact, never-evicted aggregate. Per-endpoint series live in
  a map bounded to `PROXY_ENDPOINT_CAP`, evicted after
  `PROXY_ENDPOINT_IDLE_EVICTION` idle, with overflow folded into `__other__`.
  Counters are cumulative while an endpoint is resident; eviction+reintroduction
  resets its per-endpoint counter, which rate queries already treat as a reset
  (the collector discards negative deltas). No separate aborted-stream count and
  no per-process discriminator were added — the exporter sums across a process's
  proxies, following the existing `iris_rpc_*` precedent. Bytes are booked on the
  body stream via a `Drop` flush, so a client abort still accounts the bytes
  transferred before it.
- **WS3 freshness is a trailing 1-hour window** (`epoch_ms(ts) >= now - 1h`) plus
  the local-cluster filter `(cluster IS NULL OR cluster = '')`, with `cluster`
  added to the `QUALIFY PARTITION BY`. The window is what retires the stale
  duplicate rows; dropping the shim alone would leave them.
- **WS4 keeps the ControllerService endpoint shims for now; only the clients
  migrate.** Removing the shims in this PR would 404 a pre-migration worker or
  task (notably an already-running `logship.py` log shipper) the moment the
  controller updates, since it still calls `ControllerService.ListEndpoints`. So
  the three RPCs stay as deprecated forwarding shims to `EndpointService`, every
  shipping client is moved to `EndpointService` (verified: resolver, logship,
  worker second client, CLI list, benchmark, e2e helpers, and the
  `RemoteClusterClient` facade — no shipping caller hits the ControllerService
  endpoint methods), and the panel freshness window is what removes the visible
  duplicate rows regardless. The dashboard route tests run against both services
  so the shim keeps working. Shim removal is a follow-up once old callers drain.

The Iris controller dashboard's **RPC Methods** panel
(`lib/iris/dashboard/src/components/controller/RpcStatsPanel.vue`) is built from
native-proxy counters that the process exposes through `rigging.telltale` and the
controller forwards into its finelog `telltale` table. Four issues motivate a
cleanup of how these metrics are produced, tagged, forwarded, and displayed.

## Current data path (evidence)

1. **Rust native proxy** counts requests in-process
   (`lib/iris/rust/src/lib.rs`). `RpcMetricKey{service, method, upstream}` +
   `RpcMetricState{requests, responses[status], in_flight, latency buckets,
   latency_sum}`. `rpc_metric_key` (lib.rs:945) derives the key by scanning path
   segments for the first starting with `iris.` (the Connect service) → service,
   next → method; returns `None` if there is no `iris.` segment or a third
   segment follows. `upstream = "endpoint"` when the path begins `/proxy/`, else
   `"controller"`. `ingress` (lib.rs:1396) wraps every request in an
   `RpcRequestTimer` that records nothing when `begin` returns `None`, and
   `finish(status)` fires at **response-header** time — before the body streams.
2. **native_proxy_metrics.py** reads `proxy.rpc_metrics_json`, aggregates by
   `(service, method, upstream)`, emits `iris_rpc_requests_total`,
   `iris_rpc_responses_total{status}`, `iris_rpc_in_flight`,
   `iris_rpc_duration_seconds_{bucket{le},sum,count}` into the process Telltale
   registry with one global label `source="iris"` (no cluster). The Rust-JSON ↔
   Python-collector field set is a compatibility contract between the wheel and
   the pure package.
3. **controller.py:765** `telltale.start_forwarding(FinelogMetricSink(...))`
   ships those samples into the **local** finelog `telltale` table
   (`TelltaleMetric`, `lib/rigging/src/rigging/telltale.py:207`) — append-only
   samples over time.
4. **RpcStatsPanel.vue** queries the local log server
   (`proxy/system.log-server/finelog.stats.StatsService`, useRpc.ts:149) with
   `WHERE source='iris'` and `QUALIFY row_number() OVER (PARTITION BY name,
   service, method, upstream, status, le ORDER BY ts DESC) = 1` — i.e. the
   greatest-`ts` row **per label tuple over all history** — then groups in JS by
   `${service}/${method} (${upstream})`.
5. **Cross-cluster forwarding**: the finelog server (Rust, `lib/finelog/rust`)
   with `ForwardingConfig` (deploy/config.py:168) ships every row to a hub
   finelog. `forwarding.rs:114` stamps this store's cluster on every forwarded
   row **that has an origin column**; the query layer already handles segments
   with and without a `cluster` column (`query/mod.rs:693,739`). Log rows have
   the origin column; `TelltaleMetric` does not, so telltale rows forward
   unstamped.

## Problems

- **P1 — Hub mixes clusters.** `TelltaleMetric` has no `cluster`/origin column,
  so forwarded child telltale rows land on the hub with the column *absent*
  (not literally NULL), indistinguishable from hub-local rows. The panel's
  `QUALIFY` picks the greatest-`ts` row independently per label tuple, so on
  iris.oa.dev a single displayed series can be a **cross-cluster mosaic** —
  request count from one cluster, a histogram bucket from another. (Same root
  cause as the known #138 unstamped-forwarding issue.)
- **P2 — Proxied load is uncounted by these RPC metrics.** `rpc_metric_key`
  records only paths carrying an `iris.<service>/<method>` pair: controller
  Connect RPCs (`upstream=controller`) and proxied `iris.actor.*`
  (`upstream=endpoint`). Plain proxied HTTP — vLLM/OpenAI endpoints, capability
  URLs, federation relays — has no `iris.` segment → `begin()` returns `None` →
  absent from `iris_rpc_*`. (Other logs may still observe it.) The segment scan
  is also fragile: any path with a later `iris.*/*` suffix is misclassified.
- **P3 — No byte accounting.** The proxy streams every body but records no
  bytes, so per-method/per-endpoint transfer weight is unknown.
- **P4 — Duplicate `ListEndpoints`/`UnregisterEndpoint` rows, and the panel
  never retires them.** Two causes: (a) `EndpointService` and `ControllerService`
  both expose `Register/Unregister/ListEndpoints` (EndpointService is the leased
  canonical one, #6728; ControllerService keeps compat shims, service.py
  2450-2483; `MintEndpointToken` is Controller-only); (b) even after a service or
  method stops receiving traffic, its last-written telltale rows persist and the
  greatest-`ts`-per-series panel keeps showing them **forever**. So removing the
  shims does not by itself clear the duplicate rows — the panel needs
  current-snapshot/freshness semantics.

## Decisions (locked with requester; refined by review)

- Proxied-load view: an **aggregate "all proxied" total** plus the **top endpoints
  by recent load** — "how much is proxying hurting us." Not one persistent series
  per endpoint.
- Cluster view: panel **defaults to this cluster**, resolved from a canonical
  cluster identifier the controller already knows.
- **Drop the ControllerService endpoint shims** (requester: clients are current).
  Independently, add panel freshness so the duplicate actually disappears.
- Add **per-series request/response bytes**.
- Ship as **one PR** (requester). Feasible via the native release contract below.

## Design

### WS1 — Proxy transport metrics + bytes (Rust `marin-iris-native`)

- **Separate `iris_proxy_*` family**, not an overload of `iris_rpc_*`. Keep
  `iris_rpc_*` for Connect service/method semantics; add `iris_proxy_*` for
  transport load with labels `{endpoint, method (normalized to a fixed verb set
  + OTHER), route_kind}` where `route_kind ∈ {endpoint, relay}`. A proxied actor
  Connect call contributes to both views; the panel/docs state the two views are
  **not summable**.
- **Attribution from the typed route** (`ProxyRoute`/decision result), never a
  raw `iris.` segment scan, so capability tokens, cluster tags, and malformed
  paths never become labels.
- **Bounded cardinality (genuinely).** Aggregate totals in fixed counters.
  Per-endpoint ranking uses bounded heavy-hitter state over a recent window
  (or prune against the active endpoint registry with a hard cap + idle
  eviction), ranked by **recent rate** (not lifetime — lifetime lets retired
  endpoints squat and hides newly-hot ones). Since the motive is load, rank by a
  bounded union of recent requests **and** recent bytes (request count alone
  hides low-QPS/high-bandwidth streams). Emit top-K as **window gauges**, not
  cumulative counters (so evict/reintroduce is not read as a reset). The proxy
  map is memory-bounded, and the emitted label set is bounded, so historical
  finelog cardinality is bounded too.
- **Bytes on the body stream.** Wrap request/response `Body` and increment as
  data frames are **polled** (not on completion — aborts/errors often skip
  completion). Count data-frame payload bytes only (not HTTP/1 chunk framing,
  headers, or trailers). Define directions explicitly:
  `iris_proxy_request_bytes` = bytes read from the client;
  `iris_proxy_response_bytes` = bytes delivered toward the client (name the
  upstream-side counters separately if added). Expose an **aborted / stream-error
  count** so a 200 header followed by a broken body is not shown as clean.
  Exclude the endpoint-decision/control sub-request from user-payload counters.
  Handle early rejection, unread request bodies, 1xx, `101` upgrades, and relay
  hops. Use per-request atomics flushed once, **no shared lock per chunk**. Note:
  byte/stream-duration accounting cannot hang off the existing
  `finish(response.status())` (header-time) — it needs the body lifecycle.
- **Process identity.** If a process can host multiple proxies/controllers,
  include a process/proxy discriminator so the panel's per-snapshot selection
  does not silently drop or mix process-local counters.

### WS2 — Cluster provenance (finelog server `marin-finelog-server`)

- Give the finelog telltale/stats table the **server-assigned origin (`cluster`)
  column**, reusing the existing forwarding-stamp + schema-evolution machinery
  (`forwarding.rs:114`, `query/mod.rs`). Do **not** add a cluster field to
  `TelltaleMetric` — producers must not assert their own provenance.
- **Credential-bound identity.** Stamp from a server-side binding of the
  authenticated writer to one canonical cluster (per-cluster credential or an
  authoritative credential→cluster map). A shared signing key that merely signs a
  caller-supplied cluster string proves key possession, not identity — reject or
  ignore caller-provided provenance.
- **Fail closed locally.** Only a trusted local credential/transport is stamped
  as the server's own cluster; an unauthenticated remote write must not become
  hub-local. Confirm `FinelogConfig.name` is the canonical Iris cluster id; if
  not, add an explicit required cluster setting.
- **Legacy + multi-hop.** Old rows without the column stay unknown (do not
  backfill as hub). Preserve the **original** origin across multi-hop forwarding
  (grandchild → child → hub) rather than restamping the immediate peer. Keep
  read compatibility while old/new finelog servers coexist.

### WS3 — Panel (dashboard; no wheel)

- **Current-snapshot / freshness semantics** so retired series (dropped shims,
  gone endpoints, dead clusters) disappear: select only the latest complete
  scrape per (cluster, series) — a shared snapshot/batch id if we add one, else a
  freshness cutoff. This is what actually fixes the P4 duplicate.
- **Cluster-correct.** Select `cluster`; **default-filter to the dashboard's own
  cluster** using the canonical id from an existing controller status/config
  payload (never inferred from hostname). Include `cluster` in the
  `QUALIFY PARTITION BY` now so a future all-cluster view cannot reintroduce
  mixing. Delay the stats query until the identity is available — no brief
  unfiltered query.
- **Proxied-load section** from `iris_proxy_*`: the aggregate total up top and
  the top-K endpoints; controller RPCs stay in the existing table. Add
  **bytes in/out** columns.

### WS4 — Drop ControllerService endpoint shims (iris proto + callers)

- Remove `RegisterEndpoint/UnregisterEndpoint/ListEndpoints` from
  `service ControllerService` in `controller.proto`; regenerate every checked-in
  target through the one canonical workflow (`scripts/generate_protos.py`,
  `npx @bufbuild/buf` — verified reachable). Remove the three shims
  (service.py:2456-2483). Keep `MintEndpointToken`.
- Migrate the read callers to `EndpointServiceClientSync`:
  `cli/endpoints.py:49` (list moves; **mint stays on the controller client**),
  `client/resolver.py:56/89` (client is list-only → switch type; update its
  public type + fakes), `cluster/worker/worker.py:517` (holds a controller
  client for register/heartbeat → **add a second endpoint client sharing the same
  address, interceptors, compression, and shutdown**),
  `cluster/backends/k8s/logship.py:324` (throwaway → switch type),
  `scripts/benchmark_controller.py:751/851`. Update docs and any literal Connect
  route strings.
- **Version-skew note.** New client vs old controller is safe (EndpointService
  already exists); **old worker vs new controller is the dangerous direction** —
  a still-running pre-migration worker calling the removed method 404s. Requester
  accepts (clients current); the panel-freshness change is what removes the
  visible rows regardless.

## Rollout (one PR, staged deploy)

Dev/CI build both native crates from source (`scripts/rust_mode.py dev`), so no
publish is needed to develop or test. Deploy installs pinned wheels from
`uv.lock`, so per `.agents/projects/2026-07-24_iris_native_release_contract.md`
the PR:

1. Publishes a **nightly of `marin-finelog-server`** and a **nightly of
   `marin-iris-native`** from the PR commit, raises both floors, and refreshes
   `uv.lock` in-PR (makes CI consume the exact native code and the merge
   deploy-safe).
2. Documents deploy order: **finelog (cluster column) → new clients/workers →
   iris wheel + exporter → dashboard → shim removal after old callers drain.**
3. After merge, stable tags + a small dependency-only lock bump follow the
   contract.

## Test plan

- WS1 (Rust): typed-route attribution; recent-window top-K churn + idle
  eviction; window-gauge semantics across evict/reintroduce; restart/reset;
  byte counting across a streamed body including **abort/stream-error/partial**,
  trailers, 1xx/101, retries, and exclusion of the decision sub-request;
  no-per-chunk-lock.
- WS2 (finelog): forwarded batch stored under the peer's bound cluster; local
  write under the server's own cluster; spoofed caller-provided cluster ignored;
  unauthenticated write not stamped hub-local; legacy rows stay unknown;
  multi-hop preserves original origin.
- WS3 (panel): freshness drops a retired series; default filter shows only the
  local cluster; two clusters with identical labels do not mosaic; proxied-load
  section renders aggregate + top-K + bytes.
- WS4 (`uv run pytest`): migrated callers; ControllerService no longer serves the
  three methods, EndpointService still does; worker second-client lifecycle;
  generated-artifact drift check.

## Open questions

1. Snapshot identity: add an explicit per-scrape batch id to telltale (clean
   current-snapshot selection) vs. a freshness-cutoff heuristic?
2. Do we count upstream-side bytes (bytes to/from the upstream) in addition to
   client-side, or client-side only for v1?
3. `FinelogConfig.name` canonical as the Iris cluster id, or add an explicit
   `cluster` setting to finelog + the controller status payload?
