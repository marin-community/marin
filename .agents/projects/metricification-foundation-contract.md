# Metricification foundation contract

Status: coordinator-approved through producer/agent contract revision 21; checkpoint A implementation corrections under review
Date: 2026-07-31
Parent work: #204; benchmark #205 is closed and accepted.

## First commit boundary

The first recoverable commit contains two seams and no Telltale call-site
migration:

1. `rigging.telemetry` descriptors, resource/config types, implicit no-op
   handles, same-config idempotence, conflicting-config isolation, bounded
   process-local aggregation, and failure-injection safety tests.
2. Finelog `WriteRows` idempotency keyed by `(namespace, batch_id)`. The server
   recomputes a domain-separated digest over authenticated origin plus exact
   IPC bytes and writes batch ID, digest, row count, and sequence range into the
   L0 footer and one checksummed immutable manifest per sealed L0. SQLite
   receipts are an index over that metadata, not the source of truth. Startup
   rebuilds missing receipts and verifies existing ones before accepting
   writes. A retry after a lost response or server restart returns the original
   acknowledgement without appending rows. Reusing a batch ID for different
   bytes or authoritative origin fails.

The second checkpoint adds approved v1 schemas, catalog-backed declarations,
durable receipt timestamps, historical-format repair, and lazy referenced-
namespace query binding. The REST/OTLP checkpoint adds bounded authenticated
ingestion and two-phase batch completion. Worker-agent registration, WAL, and
spill/replay follow independently. Benchmark #205 selects generation-keyed
manifest prefiltering and rejects a physical partition/clustering layout and
generic rollups for this foundation.

## Python API

`rigging.telemetry` is import-safe and process-global. Instrument declarations
return stable handles before configuration. Every emission returns `None`, does
no network or filesystem I/O, and catches validation/runtime failures. Before
`configure` and after shutdown, emission is a no-op except for bounded
process-local loss counters. A conflicting reconfiguration keeps the first
runtime active, so handles continue emitting through it.

Emission takes a lock-free snapshot of the configured runtime and acquires the
aggregation lock only with a nonblocking attempt. Contention drops the
observation. Loss accounting is also a best-effort nonblocking attempt, so a
contended loss counter may itself omit a loss. Configuration, declaration, and
status inspection may coordinate synchronously; emission and shutdown never
wait for those locks.

```python
from datetime import timedelta
from rigging import telemetry

OUTCOME = telemetry.AttributeSpec("outcome", ("success", "failure"))
METER = telemetry.meter(
    scope="skyrl.inference",
    owner="skyrl",
    default_cadence=timedelta(seconds=15),
)
REQUESTS = METER.counter(
    "requests",
    description="Completed inference requests",
    unit="{request}",
    attributes=(OUTCOME,),
    delivery_class=telemetry.DeliveryClass.BUFFERED,
)
LATENCY = METER.histogram(
    "request_duration",
    description="Inference request latency",
    unit="s",
    attributes=(OUTCOME,),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
    delivery_class=telemetry.DeliveryClass.BUFFERED,
)
QUEUE = METER.gauge(
    "queue_depth",
    description="Requests waiting for inference",
    unit="{request}",
    attributes=(),
    delivery_class=telemetry.DeliveryClass.COALESCING,
)

telemetry.configure(
    service_name="skyrl-inference",
    role="inference",
    root_run_uid=root_run_uid,
    service_version=git_revision,
    exporter=telemetry.HttpExporterConfig(
        endpoint=local_agent_url,
        token_provider=producer_token_provider,
        export_interval=timedelta(seconds=5),
        request_timeout=timedelta(seconds=2),
        shutdown_timeout=timedelta(seconds=1),
        max_queue_records=10_000,
        max_queue_bytes=16 * 1024 * 1024,
    ),
)

REQUESTS.add(1, outcome="success")
LATENCY.record(elapsed, outcome="success")
QUEUE.set(queue_depth)
telemetry.event(
    "skyrl.worker.ready",
    delivery_class=telemetry.DeliveryClass.DURABLE,
    severity=telemetry.Severity.INFO,
    outcome="success",
    engine_id=engine_id,
    model_revision=model_revision,
)
with telemetry.logging_context(root_run_uid=root_run_uid, engine_id=engine_id):
    logger.info("worker ready")
```

The public types are frozen dataclasses or `StrEnum`s:

- `AttributeSpec(name: str, allowed_values: tuple[str, ...])`.
- `MetricDescriptor(name, scope, description, unit, instrument_kind,
  temporality, attributes, buckets, owner, cadence, delivery_class,
  cardinality_limit, maturity)`.
- `meter(*, scope, owner, default_cadence) -> Meter` captures declaration
  defaults once per module. `Meter.counter`, `Meter.gauge`, and
  `Meter.histogram` accept the remaining descriptor fields. The top-level
  factories remain available as the lower-level explicit form.
- `Resource` holds nullable canonical identity fields. `service_name` and
  `service_instance_id` identify a unique process. High-cardinality identity is
  resource or record data, never a metric attribute.
- `counter(...) -> Counter`, `gauge(...) -> Gauge`, and
  `histogram(...) -> Histogram` have the keyword parameters shown above plus
  optional `cardinality_limit=100` and
  `maturity=Maturity.EXPERIMENTAL`. Counters and histograms are cumulative and
  carry `start_ts`/`reset_id`; gauges have no temporality.
- `Counter.add(value=1, **attributes)`, `Gauge.set(value, **attributes)`, and
  `Histogram.record(value, **attributes)` return `None` and never raise.
- `event(event_name, *, delivery_class=BUFFERED, severity=INFO, outcome=None,
  body=None, trace_id=None, span_id=None, **attributes) -> None` accepts only
  names in the checked-in event catalog.
- `logging_context(**fields)` is a `contextvars`-backed context manager. It
  enriches structured records but does not duplicate stdout or invoke a logging
  handler during emission.
- `configure(*, service_name, role=None, root_run_uid=None,
  service_instance_id=None, service_version=None, exporter) -> None` is the
  application convenience form. It auto-mints a process-unique
  `service_instance_id` when omitted. `configure(*, resource: Resource,
  exporter)` remains the lower-level form for Iris and tests.
- Configuration is idempotent for an equal configuration. A conflicting second
  call increments `telemetry.runtime.configuration_conflicts`, keeps the first
  runtime active, and every existing or future handle continues emitting
  through that first runtime. `shutdown()` is idempotent and obeys the
  configured fixed budget without waiting for the aggregation lock.

Descriptor conflicts, invalid names, unknown or invalid attributes, non-finite
values, cardinality overflow, queue overflow, fork state, and exporter failure
catch ordinary internal `Exception`s, increment bounded self-telemetry, and
otherwise disappear from the application path. Emission never catches
`BaseException` or process interrupts and never performs synchronous logging.
Checked-in catalog validation remains strict in CI.

`lib/finelog/rust/telemetry_catalog.v1.json` is authoritative because it is
inside the production Rust build context. Rigging ships a byte-identical package
mirror. After editing the canonical file, run
`uv run python scripts/sync_telemetry_catalog.py`; CI/tests use
`uv run python scripts/sync_telemetry_catalog.py --check`.

## Record schema

The dependency-light wire model is `TelemetryBatchV1`:

```json
{
  "schema_version": 1,
  "catalog_version": "telemetry-catalog.v1",
  "batch_id": "0198... (UUIDv7)",
  "records": [
    {
      "record_index": 0,
      "signal": "metric",
      "event_ts_unix_nano": "1785456000000000000",
      "observed_ts_unix_nano": "1785456001000000000",
      "resource": {
        "service_name": "skyrl-inference",
        "service_instance_id": "...",
        "entity_authority": "iris:us-central2",
        "entity_type": "worker",
        "entity_uid": "...",
        "root_run_uid": "...",
        "iris_job_id": null,
        "attempt_uid": null,
        "worker_id": null,
        "node_id": null,
        "cluster": null
      },
      "metric": {
        "scope": "skyrl.inference",
        "scope_version": null,
        "name": "requests",
        "description": "Completed inference requests",
        "unit": "{request}",
        "instrument_kind": "counter",
        "temporality": "cumulative",
        "start_ts_unix_nano": "1785455900000000000",
        "reset_id": "...",
        "series_id": "sha256:...",
        "value": 42,
        "device_uid": null,
        "device_type": null,
        "attributes": {"outcome": "success"}
      }
    }
  ]
}
```

`catalog_version` is required and identifies the exact descriptor/routing
catalog against which the SDK validated the batch. The server rejects an
unknown version instead of silently routing with another catalog. `signal`
selects exactly one of `metric`, `event`, `log`, or `artifact`.
Histogram metrics replace scalar `value` with `count`, `sum`,
`explicit_bounds`, and `bucket_counts`; `len(bucket_counts)` is
`len(explicit_bounds) + 1`. Events use `event_name`, severity number/text,
outcome, phase, normalized error type, bounded body, attributes, trace/span, and
evidence/result URIs. Probe events carry a typed `probe_status` of `success`,
`failed`, or `unsupported`; missing samples are never interpreted as healthy.
A record's immutable `point_id` is
`<batch_id>:<record_index>`.

`series_id` is the server-canonical hash of descriptor identity, authenticated
cluster, entity authority/type/UID, service instance, attempt, actor/engine,
rank/process identity, typed device UID/type, and normalized bounded attributes.
Custom senders must
supply that canonical value; a mismatch after authoritative stamping is a
schema error, rather than a producer-selected series. OTLP and custom records
with the same authoritative identity derive the same value. `reset_id` changes
when a cumulative stream restarts. Collector/Finelog credentials overwrite
infrastructure-owned cluster, Iris, Kubernetes, node, pod, container, and
placement fields. The server looks up `(signal, scope, name)` in the approved
catalog and chooses the namespace; callers cannot name or create Finelog tables.
At a JWT-authenticated Finelog boundary, `cluster` is stamped from the verified
key and unverified Iris job/task/attempt, worker/node/pod/container, and entity
authority/type/UID claims are cleared. Only a key configured by Finelog with
the `trusted_collector` role may preserve signed infrastructure enrichment;
network/CIDR authentication alone clears it.

Authority-scoped identities use typed `entity_authority`, `entity_type`, and
`entity_uid` resource columns. `entity_uid`, when present, is the canonical
global join key. Otherwise a local worker/node/actor/engine identifier joins by
`(entity_authority, cluster, entity_type, local_id)` until an
`telemetry.entity_link.v1` row maps it to a minted UID. Infrastructure
collectors own and overwrite authority and placement identity; producer-owned
aliases cannot replace them. Hardware metrics carry typed `device_uid` and
`device_type` columns rather than hiding accelerator identity in attributes.

## HTTP contract and acknowledgements

- `POST /v1/telemetry` accepts `TelemetryBatchV1` as `application/json` or
  `application/x-protobuf`. `Content-Encoding: gzip` and `zstd` are supported.
  `Idempotency-Key` is required and equals the body `batch_id`.
- `POST /v1/metrics` and `POST /v1/logs` accept OTLP/HTTP protobuf or JSON and
  normalize into the same records.
- The worker agent later exposes local-authenticated
  `POST /v1/scrape-targets` and
  `DELETE /v1/scrape-targets/{target_id}`.

REST admission is independently bounded at 10,000 normalized records, 32 MiB
estimated normalized memory, 64 detailed validation errors, four concurrent
requests, and four blocking decode/storage tasks. The wire and uncompressed
body limits remain 64 MiB; each normalized namespace must also fit the 16 MiB
WriteRows limit. Before generated-message construction, protobuf uses a
schema-aware constant-memory wire scan and JSON uses a streaming visitor. Each
repeated container is capped at 10,000 elements, nested structural items are
capped at 640,000 total (64 fields per maximum-size record), nesting is capped
at 32 levels, and strings/bytes are capped at 64 KiB. Every protobuf wire field
counts toward the global structural budget, including Buffa-preserved unknown
custom fields. Well-formed unknown protobuf groups are skipped compatibly with
Buffa/prost while their fields and nesting consume the same item, per-group,
and depth quotas; unbalanced groups are invalid protobuf. This admits a fully
populated 10,000-record custom batch while rejecting repeated empty message
graphs or compact unknown-field streams before allocating their
`Vec`/`String`/`Option` trees. Gzip/zstd decoding
reads at most the uncompressed limit plus one byte on a blocking worker. Zstd
also caps `window_log_max` at 23 (8 MiB) before the first decoded byte. Every
OTLP metric point scanner traverses its exemplars, including bounded
`filtered_attributes`, `span_id`, and `trace_id`. Resource scanners also
traverse `entity_refs`, bounding each reference's schema URL, type, identifying
keys, and description keys. Every request has one 30-second deadline shared by
admission, validation, every
durability wait, and completion. Admission
wraps the entire handler, so the permit and deadline are established before
Axum polls the body. Overloaded bodies are never polled; a stalled body is
canceled under the deadline and releases its permit. Request `Content-Type`
selects the success/schema-response representation. Custom endpoint errors use
stable JSON `{code,message}`. OTLP errors use `google.rpc.Status` in the
request's protobuf or JSON representation, including auth, malformed input,
body/admission limits, deadline, and storage failures. Only HTTP `429`, `502`,
`503`, and `504` are retryable and carry `Retry-After: 1`; `500` is a
nonretryable server invariant failure.

For Finelog, `201` means every namespace sub-batch and its durable
segment/manifest receipt metadata are persisted. SQLite receipt rows may lag
and are repaired from that metadata on startup. `200` with
`status="duplicate"` returns the original ack for the same key and payload. The
router derives stable namespace sub-batch IDs from
`(batch_id, namespace, delivery_class)`. A crash after one namespace commits
but before the parent response cannot duplicate it.

The parent batch uses a two-phase durable fence. After all namespace Arrow IPC
is built and exact WriteRows alignment/size validation succeeds,
`telemetry.batch_intent.v1` durably reserves the authenticated
`(batch_id,payload_digest)`. No child append starts before that receipt is
durable. Signal children then append and become durable under the request-wide
deadline. `telemetry.batch.v1` is appended last as the completion marker and is
the only batch namespace dashboards treat as accepted. A partial child failure
leaves the intent but no completion; a same-payload retry deduplicates completed
children and finishes with `201`, while a changed payload—including one routing
to disjoint namespaces—conflicts with the intent. Only a pre-existing
completion yields `200 status="duplicate"` and its child receipts reconstruct
the original acknowledgement.

A reused key with another payload digest returns `409`. Custom telemetry
batches are all-or-nothing on schema and catalog validation: any invalid record
returns `400` with stable indexed errors and commits no namespace sub-batch.
`401/403` are auth failures, `413` is the fixed body/record limit, and `429` is
retryable overload. Transient storage and deadline failures are `503`/`504`;
an internal invariant failure is `500` and must be dropped rather than retried
indefinitely.

OTLP requests with a valid explicit `Idempotency-Key` use it under the same
conflict contract. Headerless OTLP requests derive a stable internal batch ID
from a domain-separated hash of authenticated origin, signal endpoint, content
type, and uncompressed request bytes. `/v1/metrics` and `/v1/logs` return the
normal OTLP HTTP `200` response with `partial_success` when individual points
are rejected; they never use `207`.

Internal `WriteRows` rejects an empty RecordBatch; it never returns an
unreconstructible durable acknowledgement. Concurrent requests with the same
namespace, batch ID, and payload share one pending receipt and sequence range.
A same-ID conflict is detected while that receipt is still pending. New writes
and pending duplicates wait for their segment; catalog-durable duplicates
return their original acknowledgement without consulting `persisted_seq`.

```json
{
  "schema_version": 1,
  "batch_id": "0198...",
  "status": "accepted",
  "durability": "finelog_local",
  "accepted_records": 10,
  "rejected_records": [],
  "commits": [
    {
      "namespace": "telemetry.workload_metric.v1",
      "first_seq": 120,
      "last_seq": 129
    }
  ]
}
```

An agent ack reports `durability="agent_wal"` and means the batch can be
replayed locally. It does not mean hub Finelog has received it. Hub completeness
is a separate watermark contract.

## Delivery classes

- `coalescing`: keep one latest gauge value per series; counter deltas may add
  into one pending delta. Cardinality eviction increments a gap/drop counter.
  Per-observation history is not promised before collector acceptance.
- `buffered`: use a fixed-memory FIFO for ordinary counters, histogram deltas,
  and events. Process overflow follows one configured fixed drop policy. After
  agent acknowledgement, the batch is in the bounded WAL and survives agent
  restart until delivered or its recovery budget expires.
- `durable`: the process path remains nonblocking bounded memory. After agent
  acknowledgement, it also qualifies for immutable object spill/replay.
  Lifecycle, failure, alert, and accounting descriptors use this class.
  Exhausted WAL/spill retention emits an explicit gap and never silently
  downgrades to coalescing.

No class guarantees survival between an application emission and background
handoff to the agent. No class may block the application.

## Producer and worker/node agent contract

This gate adds one upstream path:

```text
rigging.telemetry background exporter
  -> injected local worker/node agent endpoint
  -> append-only framed agent WAL
  -> authenticated Finelog POST /v1/telemetry
```

The agent is a `finelog-telemetry-agent` host process built from Finelog's Rust
crate. Kubernetes runs the same binary as a node agent; VM/TPU Iris workers run
it as a managed host process. Task identity and scrape-target registration are
the next subgate and do not add another collector hop. This path does not put
payloads into a Finelog `Store`: its failed-flush buffers are not hard-bounded.
It also does not reuse forwarding's skip-on-lag or permanent-rejection cursor
advance, or the physical segment object-sync format. Low-level TLS, JWT,
`object_store`, and HTTP helpers may be extracted when their semantics match,
but the agent owns its envelope WAL, replay state machine, and spill format.

### Producer batches

`delivery_class` is a required top-level field on every
`TelemetryRecordV1`. It is the sole authoritative delivery lane; a nested
metric, event, log, or artifact value cannot override it. Rigging derives
metric and declared-event classes from the checked-in catalog. Generic OTLP
logs default to `buffered`; cataloged lifecycle events and artifacts use their
cataloged class, including `durable`. Validation rejects an absent or
catalog-inconsistent class.

Producer requests are delivery-homogeneous. A snapshot contains exactly one of
`coalescing`, `buffered`, or `durable`; stable selection order is durable,
buffered, then coalescing. REST normalization may split an OTLP or custom
parent into deterministic `(namespace, delivery_class)` child batches after
whole-request validation. The child ID is the domain-separated SHA-256 of the
parent batch ID, namespace, and delivery class. The existing point ID remains
bound to the parent ID and original record index, so splitting cannot change
point identity.

The process queue defaults to 10,000 records and 16 MiB across queued events,
the bounded batch-building reservation, and the one immutable batch currently
being retried. Gauges coalesce, counters add, and histograms update fixed
buckets outside that record queue. The exporter holds at most one
unacknowledged batch, assigns its nonempty UUID before the first request, and
retries identical content with that ID across every transport retry. Under a
short runtime lock, snapshot extraction copies bounded metric state, removes
the exact selected event objects, and publishes a building reservation that
emitters count at the configured queue caps. Record construction, series
hashing, and deterministic JSON encoding happen after releasing the lock. The
builder enforces the exact record and byte caps across metric and event
records; deterministic per-lane metric cursors continue any snapshot that
does not fit in one batch. When encoding completes, the reservation becomes
the exact immutable in-flight batch under another short swap.

An acknowledgement clears that exact object with a blocking state swap on the
daemon exporter thread; it never performs a later “pop N” against the live
deque and cannot remain in retry solely because the tiny swap was contended.
New overflow therefore evicts only unsnapshotted entries and cannot remove
newer survivors on behalf of an older in-flight request. Queue byte accounting
includes one exact deterministic envelope per nonempty delivery lane plus
record separators. The selected lane's building reservation or immutable
pending body and every other lane's queued events must fit the same global
record and byte limits. A same-lane arrival may replace only an eligible queued
event; it cannot evict another lane or an event already reserved or pending.
An individually unshippable event is immediately terminally accounted and
cannot pin later work. Cancellation restores the exact reserved objects before
newer survivors while preserving both caps.

When one lane contains both events and cumulative metrics, the exporter
alternates which signal gets first use of the bounded batch. A deterministic
metric cursor advances only for encoded metric records, so continuously
refilled event traffic cannot starve cumulative metric series. Runtime status
does not take the aggregation lock and reports the building reservation's
actual `reserved_records` and `reserved_bytes`, or the immutable pending
batch's exact size. Application emission remains nonblocking and catches
ordinary `Exception`, never `BaseException`.

`HttpExporterConfig` keeps the agent endpoint and renewable `TokenProvider`
explicit and defaults to the starting queue caps. Each request has a configured
deadline. Missing credentials, transport failure, an unverifiable `200/201`
acknowledgement, `401`, `429`, `502`, `503`, and `504` keep the same immutable
batch and ID. The provider is called again on every retained retry, allowing
credential renewal after `401` or temporary unavailability. `Retry-After`
integer or HTTP-date guidance is honored up to 60 seconds; otherwise the fixed
export interval bounds the retry schedule. `403`, every other `4xx`, and
exactly `500` are terminal for those bytes and record bounded process loss
state. Other `5xx` responses remain retryable. Only an acknowledgement with
the exact batch ID, `accepted` or `duplicate` status, and
`durability="agent_wal"` settles success. `shutdown()` sets the stopped flag
without taking the aggregation lock, asks the daemon exporter to make one
final attempt, and joins for at most the configured shutdown timeout.
Production configuration rejects a shutdown timeout over five seconds. A
blocked exporter may outlive that join only as a daemon and cannot delay
process exit.

### Agent canonical identity matrix

The agent applies this matrix to every record after credential verification and
before canonical encoding. “Clear” means encode the protobuf field as absent,
not as a producer-provided empty sentinel.

| Action | `TelemetryResourceV1` fields | Authority |
| --- | --- | --- |
| Stamp | `cluster`, `iris_job_id`, `iris_task_id`, `attempt_id`, `attempt_uid`, `worker_id`, `node_id` | Exact verified `aud="telemetry-agent"` claims; producer values are ignored |
| Clear | `task_index`, `pod_uid`, `container_id`, `entity_authority`, `entity_type`, `entity_uid` | No signed claim exists in v1, so the agent must not forward a value |
| Preserve | `service_name`, `service_instance_id`, `role`, `root_run_uid`, `service_version`, `run_id_alias`, `rank`, `process_index`, `actor_id`, `engine_id`, `repository`, `git_revision`, `image_digest`, `model_id`, `model_revision`, `policy_step`, `owner`, `experiment_issue` | Producer-owned application, process, model, and source identity |

This list is exhaustive for `TelemetryResourceV1`; adding a resource field
requires assigning it to one row before the canonical schema version can
advance. Metric `device_uid` and `device_type` remain catalog-validated signal
fields, not infrastructure resource claims, until the hardware collector gate.
After stamping and clearing, the agent recomputes every metric `series_id` from
the canonical descriptor, resource, attempt/process, device, and normalized
attribute identity. A supplied `series_id` is never forwarded unchanged.
Point IDs remain derived from the stable batch ID and record index.
Before a batch can reach the WAL, the agent runs the same complete custom
semantic validator as Finelog REST over the canonical batch: schema/catalog
versions, record indexes and timestamps, exactly one signal payload,
descriptor/event/artifact catalog membership, descriptor fields, bounded
attributes, histogram consistency, resource requirements, delivery class, and
post-stamp series identity must all pass. Delivery-lane validation alone is
never sufficient for an `agent_wal` acknowledgement.

### Agent acknowledgement and framed WAL

The local agent admits custom JSON/protobuf only after the request fits its
per-request limit and the global 128 MiB RAM admission budget. It verifies the
producer credential before normalization and computes two distinct digests:

- `producer_request_digest` is
  `SHA-256("marin.telemetry-agent.request.v1\0" || identity_tuple ||
  content_type || received_body)`. Each component is length-prefixed.
  `identity_tuple` contains the verified cluster/job/task/attempt/worker/node
  identity claims in fixed field order, excluding renewable `iat` and `exp`.
  `content_type` is the exact trimmed HTTP header value and `received_body` is
  the exact compressed-decoded request body. This digest enforces
  `(batch_id, producer_request_digest)` retry/conflict identity.
- `canonical_payload_digest` is
  `SHA-256("marin.telemetry-agent.canonical.v1\0" || canonical_body)`.
  `canonical_body` is deterministic `TelemetryBatchV1` protobuf after the
  agent overwrites every record's infrastructure resource from the verified
  claims. It preserves record order and point IDs, serializes attribute maps
  in key order, uses the canonical protobuf media type, and rejects unknown v1
  record fields rather than forwarding them across the trust boundary.

Only the canonical body is durable or replayable; raw producer bytes never
enter the WAL or spill store. A new homogeneous batch receives the next
agent-local `agent_seq`; sequence numbers never come from a producer or
Finelog. The durable receipt retains both digests so a retry can be classified
after the canonical payload has already been delivered or reclaimed.

The durability source is a directory of append-only WAL segment files. SQLite
or another sidecar may accelerate lookup, but is disposable and is rebuilt
solely from WAL frames. The agent takes a nonblocking exclusive advisory lock
on `<wal_dir>/LOCK` for its lifetime and refuses to start if another process
owns the directory.

Sequence commitment also has a redundant fence outside the active WAL suffix.
`<wal_dir>/sequence-a` and `sequence-b` each contain a checksummed fixed record
with a fence magic/version, monotonically increasing fence generation,
`reserved_through`, `committed_through`, and `receipt_floor`. An update writes
and file-`fsync`s a temporary copy, atomically replaces the older slot, and
directory-`fsync`s before mirroring the same generation through the other slot.
Both valid copies must reflect a reservation before its sequence is appended,
and both must reflect commitment before the agent acknowledges that batch.
Reservation may advance in bounded blocks; one serialized WAL writer commits
batch sequences without holes.

The commit order for sequence `N` is:

1. Durably advance both fence copies so `reserved_through >= N`.
2. Append and file-`fsync` the complete canonical batch frame.
3. Durably advance both copies so `committed_through = N`.
4. Return the `agent_wal` acknowledgement.

A crash after step 1 but before a valid frame leaves a never-committed/torn
reservation; `N` may be assigned by the retry. A valid frame after step 2 is
recovered and promotes the fence before admission resumes. A missing
representation for any sequence in
`(receipt_floor, committed_through]` was durably committed even when it is in
the final corrupt active suffix with no later generation: recovery appends and
`fsync`s an explicit maximal contiguous `wal_corruption` gap range and never
reuses those sequence numbers. Receipt GC advances `receipt_floor` only for a
contiguous settled prefix after the replacement receipt checkpoint is durable.
Recovery validates each fence checksum and requires
`reserved_through >= committed_through >= receipt_floor`. Two valid copies at
the same generation must have byte-identical content; divergence fails closed.
A checksum-valid copy that violates that field ordering is semantic corruption
and fails closed even if the other copy is usable.
For different generations, every high-water in the newer copy must be greater
than or equal to its value in the older copy; a backwards field or any
otherwise invalid monotonicity fails closed. The highest valid generation is
authoritative. Its atomic-write path repairs a lower-generation, torn, or
checksum-invalid slot and `fsync`s the directory before admission resumes. If
neither copy is valid, recovery fails closed. Fence files and their temporaries
count against `wal_max_bytes`.

Every frame has this versioned, little-endian envelope:

```text
magic[8] = "MTWAL001"
version:u16 = 1
kind:u16
lane:u8
reserved:u8
header_len:u32
payload_len:u64
agent_seq:u64
accepted_at_ms:u64
record_count:u32
batch_id_len:u16
content_type_len:u16
producer_request_digest[32]
canonical_payload_digest[32]
frame_checksum[32]
variable_header[header_len]
payload[payload_len]
```

The variable header carries the UTF-8 batch ID and canonical protobuf content
type plus kind-specific transition data. Batch frames contain the canonical
body only. Transition and receipt frames repeat the referenced sequence, batch
ID, both digests, time, and lane. `frame_checksum` is
`SHA-256("marin.telemetry-agent.wal.frame.v1\0" || fixed header without the
checksum || variable_header || payload)`. The canonical payload digest is
verified against the canonical body before delivery. Supported kinds are
batch, Finelog acknowledgement, spill acknowledgement, terminal gap, and
receipt checkpoint.

An active segment is created through a temporary file, file `fsync`, atomic
rename, and WAL-directory `fsync` before admission starts. A batch response is
sent only after its complete frame has been appended and the active file has
been `fsync`ed. Rotation `fsync`s the active file, renames it to an immutable
sealed generation, `fsync`s the directory, then durably creates the next active
file. A successful first response is `201`; a same-request-digest retry is
`200 status="duplicate"` with the original `agent_seq` and acceptance time;
changed authoritative claims, content type, or received bytes under the same ID
is `409`, even if canonical normalization would produce equal output. Both
success responses say `durability="agent_wal"` and claim neither Finelog nor
object-store durability.

Startup scans generations in order without materializing payloads. It validates
magic, version, lengths, checksum, canonical payload digest, and strictly
increasing new batch sequences; transitions reference an existing sequence and
may arrive out of order. It then rebuilds receipts and state transitions. EOF
inside an active frame is a torn tail: truncate to the last valid frame,
`fsync` the file, and `fsync` the directory. Any other corrupt suffix is copied
to a bounded quarantine generation, `fsync`ed, atomically renamed, and
directory-`fsync`ed. A bounded resynchronization scan accepts only complete
checksum-valid frame headers after the corrupt offset. Every recoverable batch
sequence found in that suffix, and every missing sequence inferred before the
next valid generation or the redundant committed fence, is covered by one of
the maximal contiguous `wal_corruption` gap ranges before the valid prefix
replaces the damaged segment. This includes missing committed sequences in the
last active generation. No quarantined recoverable or fenced-committed sequence
is skipped or reused. The replacement uses the same durability barriers. A
corrupt or stale sidecar is deleted and reconstructed. Recovery never treats
sidecar state as evidence of an acknowledged batch.

Quarantine policy is also explicit:
`quarantine_max_bytes >= segment_max_bytes`,
`quarantine_payload_retention`, and `quarantine_metadata_retention` are
required. A quarantine entry stores the complete corrupt suffix, its digest,
source generation/offset, discovered sequence ranges, reason, and transition
reference. Its payload may be reclaimed at the configured payload age or under
the quarantine byte cap only after every recoverable/fenced sequence has a
file-`fsync`ed terminal transition. If no transitioned entry is reclaimable,
recovery/admission stops instead of crossing the cap. Checksummed metadata
remains in receipt checkpoints until
`quarantine_metadata_retention` after settlement; that duration must be at
least `receipt_retention`. Metadata is removed only after a replacement
checkpoint is durable. Quarantine payload, metadata, and rewrite temporaries
all count against `wal_max_bytes`.

Receipt checkpoints compact old batch IDs, digests, original acknowledgement
fields, committed times, terminal state, and `settled_at_ms` without carrying
delivered payloads. Unsettled receipts, including every spilled receipt, are
never age-pruned. The `receipt_retention` clock begins only when a file-`fsync`ed
Finelog-ack or terminal-gap transition assigns `settled_at_ms`; acceptance,
WAL age, and spill creation do not start it. A replacement checkpoint or
recovered segment is file-`fsync`ed, renamed, and directory-`fsync`ed before
superseded files are unlinked; the directory is `fsync`ed again after unlink.
Thus a crash selects either a complete old generation or a complete new
generation. Batch IDs may be reused only after their settled receipt has
exceeded the configured receipt horizon and contiguous-prefix GC has durably
advanced `receipt_floor`.

Disk policy is explicit. `wal_max_bytes`, `wal_max_age`,
`receipt_retention`, `segment_max_bytes`, `maintenance_reserve_bytes`, and
`emergency_gap_bytes` are required positive configuration.
`receipt_retention` must cover the configured producer retry/backfill horizon;
it is intentionally independent of WAL/spill age because it starts at
settlement rather than acceptance.
`wal_max_bytes` is a hard total for active, sealed, checkpoint, quarantine,
temporary, lock, and sidecar files. The maintenance reserve is at least one
maximum segment plus one maximum frame, allowing bounded rewrite/quarantine.
Normal batches may consume only
`wal_max_bytes - maintenance_reserve_bytes - emergency_gap_bytes`; admission
returns `429` before an append could cross that boundary. Recovery fails closed
if external files already exceed the hard limit. Rebuildable sidecars are
discarded before durable data. The 128 MiB RAM limit covers request bodies,
indexes, one active export, and recovery buffers; payloads are streamed from
the WAL rather than loaded wholesale.

The emergency reserve is available only to compact terminal-gap frames and
their cataloged `telemetry.runtime.gap` batches. Normal admission stops before
using it. If the reserve cannot hold the maximum configured gap frame, startup
fails. If corruption or an external disk change exhausts it, admission remains
stopped and health is failed rather than silently losing the gap. Gap frames
aggregate contiguous sequence ranges by reason, so outage volume cannot create
one metadata record per dropped point. Exportable gap batches carry an
internal `gap_depth=1`. If one expires, is corrupt, or receives a permanent
rejection, the agent settles it into the reserved non-exported
`gap_delivery_failed` aggregate and never creates another gap batch from that
failure. Status retains its cumulative ranges and reason counts. This terminal
rule makes gap reporting non-recursive under a permanent outage.

Coalescing and buffered batches are settled with a durable gap transition when
their byte or age horizon is exhausted. Quota reclamation selects the oldest
coalescing, then buffered batch first. For an oldest durable batch it attempts
configured spill before loss. If spill is disabled or fails, and normal WAL
space cannot be restored by checkpointing, the durable batch is settled as a
`wal_quota_exhausted` gap before its canonical payload is reclaimed. An
unspilled durable batch reaching `wal_max_age` similarly becomes
`wal_retention_exhausted`. A spilled durable batch remains live through the
explicit spill horizon; reaching it without a Finelog acknowledgement appends
and `fsync`s `spill_retention_exhausted` before object deletion. No class is
silently downgraded. A permanent Finelog rejection is quarantined and settled
explicitly; it never advances a cursor as if delivery succeeded. Later
independent batches may be attempted, but all published watermarks remain
contiguous and stop at the hole until its gap transition is durable.

### Durable spill and replay

A durable batch may spill only when an explicit `SpillConfig` supplies a
`gs://`, `s3://`, or test-local root and positive retention. Spill is otherwise
disabled. This is a dedicated immutable telemetry envelope, not a Finelog
Parquet segment or `RemoteStore` sync. It may reuse configured
`object_store` clients, not their physical layout.

The object key is
`<agent_origin>/v1/<agent_seq:020d>/<batch_id>-<canonical_digest>.pb`.
The fixed-width decimal sequence makes key order equal agent sequence order.
The spill writer considers committed batches in increasing `agent_seq` order;
after each sequence is either ineligible, already settled, or durably spilled,
it advances a contiguous `spill_create_through` decision frontier. It never
creates an object at or below that frontier later. The frontier is
reconstructible from WAL lanes and transitions. A conditional create writes the
canonical protobuf media type, canonical body, both digests, and receipt
metadata; an existing key is accepted only after checksum verification. The
agent appends and `fsync`s a spill-ack transition before advancing the frontier
or allowing compaction to omit the local payload.

Spill discovery is a paginated or streaming walk and is never collected into
one in-memory listing. `spill_scan_page_entries` defaults to 1,000 and
`spill_scan_page_bytes` to 8 MiB; both count against the global 128 MiB budget.
Each reconciliation pass holds the spill-mutation lock only long enough to
snapshot `committed_through` and `spill_create_through`; the latter is the
inclusive scan high-water. It then releases the lock before any remote listing
or body validation. The pass scans the complete agent prefix from its first key
through that fixed-width high-water, validates and checkpoints one page before
requesting the next, and keeps at most one page and one object body resident.
Concurrent creates have greater keys and cannot appear behind its cursor.
Concurrent cleanup is permitted; a listed object that is deleted before
metadata/body fetch returns `404` and is treated as settled cleanup after its
durable deletion-authorizing transition is observed. Mutations above the
captured frontier are covered by the next complete pass. A crash discards the
cursor and restarts at the prefix beginning. Providers without a native page
API are consumed as a stream with the same entry/byte window.

Startup handles the conditional-put-before-spill-ack crash separately. Before
enabling any spill mutation, it performs the same bounded complete-prefix scan
through the recovered `committed_through`, adopts checksum-valid orphan objects,
and appends their spill-ack transitions. With no concurrent creator during that
bootstrap, an orphan cannot hide beyond the reconstructed frontier.

Replay sends the original batch ID with the canonical protobuf content type and
canonical body. Only after Finelog returns its durable `200/201`
acknowledgement does the agent append and `fsync` the Finelog-ack transition;
that transition is the only delivered outcome. A spill object may be deleted
only after either that durable Finelog-ack transition or a file-`fsync`ed
`spill_retention_exhausted` terminal transition. The latter is loss, advances
only `settled_through`, and never advances `finelog_acked_through`. Deletion is
retryable cleanup after the transition; a crash before deletion finds the
object on the next complete scan. Every retry and restart preserves the
original batch ID, producer-request digest, canonical digest, and canonical
body.

Retry starts at one second, doubles to at most 60 seconds, and uses bounded
jitter. Five consecutive retryable failures open the circuit until the current
backoff expires; one half-open probe is allowed. Success closes it. Backoff and
shutdown are interruptible. This state machine is dedicated to agent
envelopes; it does not inherit Finelog forwarding's lag skipping.

### Credential and identity boundary

The producer-to-agent credential is an Iris-minted Ed25519 JWT with
`aud="telemetry-agent"`, a positive lifetime no longer than 3,600 seconds, and signed
`cluster`, `iris_job_id`, `iris_task_id`, `attempt_id`, `attempt_uid`,
`worker_id`, and `node_id` claims. `sub` is the attempt UID. The agent verifies
signature, audience, `iat`/`exp`, `sub == attempt_uid`, nonnegative
`attempt_id`, and that cluster/worker/node claims match its Iris
registration. It rejects missing, mismatched, expired, or network-only
attribution. For every accepted record it overwrites the corresponding
resource identity fields from those claims; producer-supplied infrastructure
identity is never authoritative. Application service, rank, and descriptor
identity remain producer data.

Agent-to-Finelog uses a different Ed25519 identity with `aud="finelog"`.
Finelog configuration, not a self-asserted claim, assigns its verification key
the `trusted_collector` role and binds the allowed cluster. Only that role may
preserve the agent's signed job/task/attempt/worker/node enrichment; Finelog
still stamps the configured cluster. Ordinary cluster JWTs continue to have
producer infrastructure fields cleared. CIDR/network authentication also
clears them and is never sufficient attribution. A valid trusted-collector
bearer is therefore evaluated before an ordered CIDR/network fallback even
when the deployment lists CIDR first; bearerless and unverifiable callers
retain the normal first-match policy. The same Ed25519 public-key fingerprint
may appear more than once only with an identical `(cluster, role)` binding.
Any cross-layer cluster or trusted-collector ambiguity fails server
configuration before admission. Iris credential minting and deployment land
later, but this gate freezes the claim schema and tests it with real signed
fixtures.

The process runtime records its creating PID. An inherited post-fork runtime
does no emission or export, does not join the vanished parent exporter thread,
and records bounded fork loss. An at-fork child hook replaces module
coordination locks before recording that loss, so a lock held by a vanished
parent thread cannot deadlock status or configuration. Explicit configuration
in the child creates a new runtime and auto-mints a new service instance ID.
Runtime status uses lock-free snapshots of aggregation and
building/in-flight state, so inspection cannot wait behind batch extraction or
another aggregation holder.

### Status, watermarks, and shutdown

`GET /v1/status` reports bounded structured state: all quota partitions and
horizons; active/sealed/checkpoint/quarantine bytes; pending
batches/records/bytes and oldest age per lane; spill objects/bytes/oldest age;
gap and corruption totals; last success; retry attempt/next retry; circuit
state; directory-lock ownership; and shutdown/health state.

Watermarks use agent-local sequence space and are contiguous:

- `wal_durable_through` is the highest sequence for which every earlier
  acceptance remains represented by a durable batch or receipt frame. Payload
  reclamation after acknowledgement does not move this watermark backward.
- `finelog_acked_through` is the highest sequence for which every earlier batch
  has a durable Finelog acknowledgement; gaps do not count as delivery.
- `settled_through` is the highest sequence for which every earlier batch has a
  Finelog acknowledgement or durable terminal-gap transition.

Later successes never jump these values across a hole. Status separately
reports the oldest unsettled sequence and bounded counts by state, avoiding a
misleading max-sequence watermark.

SIGTERM stops admission, interrupts HTTP/backoff, and leaves unsettled frames
in the WAL. Shutdown waits only for the configured fixed budget. Restart
resumes at the lowest unsettled `agent_seq`. A Finelog acknowledgement is
required before a batch is counted as delivered. Spill deletion additionally
permits the separately reported, durably settled retention-exhaustion path
defined above.

Checkpoint A contains the authoritative record lane, signed identity fixtures,
Finelog trusted-collector boundary, and bounded Rigging exporter described
above. It is committed only after independent implementation review.
Checkpoint B then adds framed WAL ingest, recovery, receipt, quota, gap, and
status core. Checkpoint C adds exact hub delivery/retry and opt-in immutable
durable spill/replay with fakes. All three remain on the same Marin PR.
Iris deployment, scrape-target registration, Prometheus/Ray/vLLM/node/DCGM
adapters, Telltale migration, and physical Finelog layout remain later gates.

## Resolved and later choices

Coordinator review resolved commit 1:

- Declaration and emission catch ordinary `Exception`, return inert handles for
  invalid declarations, and preserve process interrupts.
- Emission and shutdown never wait for process-global, aggregation, or loss
  locks. Contention drops the observation and loss accounting remains
  best-effort.
- The server computes the payload digest. The wire sends only `batch_id`.
- `batch_id` is required for every internal `WriteRows`; controlled writers
  update together.
- Durable segment/manifest metadata is the receipt source of truth. SQLite is a
  repairable index. A crash after segment rename but before SQLite receipt
  commit is an explicit test point.
- Each L0 has at most one checksummed receipt manifest containing all batches
  sealed into that segment. The manifest remains after compaction or data
  retention until the replay/backfill horizon; there is no per-batch file.
  Startup is `O(manifests + L0 filenames + receipts)` and reads only an L0
  footer whose validated manifest is missing. Legacy L0s receive an empty
  version marker on first adoption. Startup diagnostics record
  `manifest_count`, `footer_repairs`, and recovered `receipts`.
- The in-memory receipt map contains pending writes only and is capped at
  10,000 batches per namespace. Durable lookup is indexed by SQLite after the
  manifest transaction commits.
- The idempotency digest covers the authenticated origin, including a distinct
  trusted/local sentinel, and exact Arrow IPC bytes.
- Empty `WriteRows` batches are rejected before receipt allocation. New and
  pending-duplicate requests await persistence; recovered/catalog-durable
  duplicates do not depend on the local sequence watermark.
- Receipt metadata lives for at least the maximum configured replay/backfill
  horizon, even if the corresponding data expires earlier.
- Receipt manifest generations and their SQLite index rows carry a durable
  `committed_at` timestamp so later receipt garbage collection can enforce that
  horizon independently of data-segment retention. The timestamp is assigned
  once when a buffer first seals and remains in its restorable retry state, so
  a post-manifest/pre-catalog retry writes identical footer and manifest
  metadata. Legacy v1 receipts omit a zero timestamp when checksummed, preserving
  their original serialized representation; the SQLite migration adds zero and
  startup repairs it from the durable manifest.
- Every telemetry batch carries `catalog_version`.

Lazy query binding preserves DataFusion reference shape. A quoted dotted name
such as `"telemetry.event.v1"` is a bare Finelog namespace; the SQL-qualified
`telemetry.event.v1` is a three-part catalog/schema/table reference and never
aliases the dotted namespace. CTEs are removed by the SQL resolver before
storage lookup, multiple bare namespaces bind independently, and `SELECT 1`
touches no namespace. Unknown names remain normal client errors.
`information_schema` is disabled because lazy registration would expose a
misleading partial catalog, and public table functions are not registered.

Coordinator review resolved the REST gate:

- Headerless OTLP derives its batch ID from authenticated origin, signal
  endpoint, content type, and uncompressed request bytes under a domain
  separator. A valid explicit key overrides the derived ID.
- Custom `/v1/telemetry` validation is atomic and returns `400` with indexed
  errors without committing valid siblings.
- OTLP uses its protocol-native `200 partial_success` response.
- REST decompression, parsing, normalization, and storage work are bounded and
  kept off Tokio executor threads; validation errors and normalized memory have
  independent caps. OTLP admission multiplies shared resource/scope/descriptor
  string memory by its descendant point count before row construction and caps
  each string/byte value at 64 KiB.
- Protobuf repeated-message structure and JSON arrays/maps are quota-scanned
  before generated-message deserialization. Repeated empty `0a 00` messages
  cannot allocate a decoded ownership graph, including when nested under an
  OTLP exemplar's filtered attributes or a resource's entity references.
  Every wire field, including Buffa-preserved unknown custom fields, consumes
  the global budget. Balanced unknown groups follow decoder skip semantics
  under item/depth quotas; mismatched or missing end groups are rejected.
  Exemplar trace/span byte fields and `EntityRef` strings are bounded, and
  zstd frames cannot request a decoder window above 8 MiB.
- OTLP failures are protocol-native `google.rpc.Status` protobuf/JSON. Only
  `429`, `502`, `503`, and `504` carry retry guidance; `500` is nonretryable.
- Official OTLP `LogRecord.event_name` takes precedence over the legacy
  `event.name` attribute fallback. Integer metric values outside the exact
  binary64 range are rejected rather than rounded.
- `telemetry.batch_intent.v1` is the global digest reservation and
  `telemetry.batch.v1` is the last durable completion marker.

Later budget choices do not block the current gate: application queue
bytes/records, agent WAL bytes/age, durable spill retention, export bandwidth,
and raw/rollup retention. Benchmark #205 leaves the current name-keyed physical
segments unchanged.

Nonblocking next-gate ledger:

- Once hardware descriptors exist, extract OTLP hardware device UID/type into
  the typed resource identity and add custom/OTLP series-identity parity tests.
- Pre-register telemetry schemas before the REST router serves requests. Until
  then, make schema-registration guard acquisition and registration
  cancellation-safe so a request deadline cannot leave a registration race.
