# Metricification foundation contract

Status: coordinator-approved foundation and REST/OTLP gate
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
authority/type/UID claims are cleared. A trusted-network collector may supply
those fields because it is the infrastructure enrichment boundary.

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
router derives stable namespace sub-batch IDs from `(batch_id, namespace)`. A
crash after one namespace commits but before the parent response cannot
duplicate it.

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
