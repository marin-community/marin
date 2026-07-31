# Telemetry foundation contract

Status: superseded by repo-shared `metricification-foundation-contract` revision 35; retained as the implementation-size sketch and release-order record.

## Release split

The implementation is split across two PRs because Iris embeds the separately
published `marin-finelog-server` wheel:

1. Commit `88cc5f483` contains the bounded Rigging client and Finelog
   `POST /v1/telemetry` endpoint. Its native Finelog wheel must be published
   before the cutover deploys.
2. The release-gated cutover migrates producers and consumers, then deletes
   Telltale. It must require the published wheel that contains the endpoint.

There is no compatibility route or dual write between the two deployments.
Until the native wheel and Iris dependency floor advance, the cutover remains
undeployed.

## Scope

The client sends telemetry directly to Finelog. Finelog validates and stores it.
The cutover replaces Marin's `rigging.telltale` call sites with the client.

## Client API

Instrumentation may be declared before configuration:

```python
from rigging import telemetry

requests = telemetry.counter("inference.requests")
latency = telemetry.histogram("inference.request_duration", unit="ms")

def main() -> None:
    telemetry.configure(
        endpoint="https://finelog.example/v1/telemetry",
        service="skyrl-inference",
        attributes={"root_run_uid": "...", "role": "inference"},
    )
```

`counter.add`, `gauge.set`, `histogram.record`, and `telemetry.event` are no-ops
before `configure`. Telemetry calls never raise into application code and never
perform network I/O on the caller thread.

`configure` is idempotent for the same configuration. Invalid configuration
disables export and emits one rate-limited warning. It does not terminate the
application.

## Export behavior

- One daemon thread drains a bounded in-memory queue.
- The default queue holds at most 10,000 records or 16 MiB.
- Queue overflow drops the new record and increments an in-process loss counter.
- The exporter batches records by count and encoded byte size.
- Each batch has one stable UUID reused for every retry.
- Connect and request timeouts are finite and configurable.
- Transport errors, HTTP 429, and HTTP 5xx retry with capped exponential backoff.
- Other HTTP 4xx responses drop the batch and emit one rate-limited warning.
- `shutdown(timeout=...)` attempts one bounded flush, then abandons remaining data.
- A process crash may lose queued telemetry. This is acceptable for v1.

There is no local agent, WAL, SQLite outbox, sequence fence, checkpoint,
quarantine protocol, spill-to-object-storage path, or exactly-once guarantee.
SQLite or an OpenTelemetry Collector persistent queue may be added later if a
measured workload needs crash persistence.

## REST API

`POST /v1/telemetry` accepts one versioned JSON batch. Requests are bounded by
body size, record count, attribute count, and string length before storage.

Required headers:

- `Content-Type: application/json`
- `Idempotency-Key: <batch UUID>`

The body contains `version`, `batch_id`, resource attributes, and records. A
record is one counter delta, gauge value, histogram observation, or structured
event. Records carry a timestamp, name, value/body, unit, and bounded attributes.

Responses use a stable JSON envelope:

- `200`: accepted, including a repeated batch already known by this Finelog process.
- `400`: invalid JSON or record schema.
- `401` or `403`: authentication failure.
- `409`: the same batch ID was reused with different content.
- `413`: request exceeds a configured bound.
- `429`: server admission limit reached.
- `500` or `503`: transient server failure.

Finelog keeps a bounded batch-ID cache to suppress ordinary retry duplicates.
The cache does not promise deduplication across every server crash. Every stored
record includes `batch_id` and `record_index`, so later compaction or queries can
deduplicate if needed.

## Storage

Finelog writes normalized telemetry rows through its existing namespace and
`WriteRows` machinery. The REST handler does not implement a second storage or
recovery layer.

Metrics and events share the `telemetry_v1` namespace. Physical partitioning
and rollups remain separate work.

## Safety checks

The PR must demonstrate:

- an unconfigured call is a no-op;
- caller latency does not include network latency;
- queue count and byte bounds hold under concurrency;
- exporter failures do not escape the daemon thread;
- retry uses the same batch ID;
- shutdown respects its timeout;
- REST request bounds and error codes are stable;
- a successful request is queryable from Finelog;
- Telltale call sites covered by the PR use telemetry directly.

## Size guardrail

The handwritten client runtime should stay below 600 lines. The handwritten REST
handler and normalization code should stay below 1,000 lines. Exceeding either
limit requires a design review before more code is added.
