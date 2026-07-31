# Metricification foundation contract

Status: coordinator-approved for commit 1
Date: 2026-07-30
Parent work: #204; the physical-layout benchmark remains #205.

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

The REST router and approved v1 schema routing are the next commit. Worker-agent
registration, WAL, spill/replay, and lazy query binding are later independent
gates. Clustering, partitioning, manifests, and rollups remain blocked on #205.

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
evidence/result URIs. A record's immutable `point_id` is
`<batch_id>:<record_index>`.

`series_id` hashes descriptor identity, authoritative service instance or
attempt identity, and normalized bounded attributes. `reset_id` changes when a
cumulative stream restarts. Collector/Finelog credentials overwrite
infrastructure-owned cluster, Iris, Kubernetes, node, pod, container, and
placement fields. The server looks up `(signal, scope, name)` in the approved
catalog and chooses the namespace; callers cannot name or create Finelog tables.

## HTTP contract and acknowledgements

- `POST /v1/telemetry` accepts `TelemetryBatchV1` as `application/json` or
  `application/x-protobuf`. `Content-Encoding: gzip` and `zstd` are supported.
  `Idempotency-Key` is required and equals the body `batch_id`.
- `POST /v1/metrics` and `POST /v1/logs` accept OTLP/HTTP protobuf or JSON and
  normalize into the same records.
- The worker agent later exposes local-authenticated
  `POST /v1/scrape-targets` and
  `DELETE /v1/scrape-targets/{target_id}`.

For Finelog, `201` means every accepted namespace sub-batch and its durable
segment/manifest receipt metadata are persisted. SQLite receipt rows may lag
and are repaired from that metadata on startup. `200` with
`status="duplicate"` returns the original ack for the same key and payload. The
router derives stable namespace sub-batch IDs from `(batch_id, namespace)`. A
crash after one namespace commits but before the parent response cannot
duplicate it.

A reused key with another payload digest returns `409`. Schema-invalid records
produce `207` with stable indexed rejections while valid records are durably
committed. A batch with no valid records returns `400`. `401/403` are auth
failures, `413` is the fixed body/record limit, `429` is retryable overload with
`Retry-After`, and `5xx` is retryable because no durable ack was observed.

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
- Every telemetry batch carries `catalog_version`.

The REST commit still chooses the headerless-OTLP idempotency rule and whether
partial validation uses `207` or whole-batch rejection. Later budget choices do
not block commit 1: application queue bytes/records, agent WAL bytes/age,
durable spill retention, export bandwidth, raw/rollup retention, and the
benchmark-selected physical layout.
