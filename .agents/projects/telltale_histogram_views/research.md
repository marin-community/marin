# Telltale Histogram Background Research

## Background Research Brief

- Effort: medium
- Stop rule: stop when internal history, primary format specifications, the
  adversarial Map-cost source, and a live Finelog measurement agree on the
  storage/display boundary.
- Date: 2026-07-25

### Question

Should Telltale promote the Prometheus `le` label into a Finelog column, adopt a
native histogram storage type, or keep generic sample rows and reconstruct
histograms in views?

### Current Marin Context

`rigging.telltale.samples()` walks the Prometheus registry after collectors have
flattened classic histograms into scalar `_bucket`, `_sum`, and `_count`
samples. `scrape_metrics()` writes each scalar as one `TelltaleMetric`. Typed
process/job identity is stored in top-level Finelog columns; remaining
Prometheus labels use `Map<Utf8,Utf8>`. The namespace key is `name`, so
compaction clusters one metric's rows for Parquet statistics and Bloom-filter
pruning.

Each 15-second forwarding snapshot produces these rows for one
`engine,model_name` histogram series:

| Prometheus sample | Finelog `name` | Finelog `value` | Finelog `labels` |
| --- | --- | ---: | --- |
| count | `vllm:request_prompt_tokens_count` | observations so far | `engine`, `model_name` |
| sum | `vllm:request_prompt_tokens_sum` | token sum so far | `engine`, `model_name` |
| bucket | `vllm:request_prompt_tokens_bucket` | cumulative observations at the bound | `engine`, `model_name`, `le` |

All three also carry one shared `ts`, `kind='histogram'`, `source='vllm'`, and
the typed Iris identity columns. Finelog does not flatten one histogram into a
single row during ingestion.

The process-local Telltale page renders the same flattened samples directly.
This is why one vLLM histogram occupies many nearly identical rows. The Iris
RPC dashboard already performs view-layer reconstruction: its SQL selects
`json_get(labels, 'le')`, and Vue sorts cumulative buckets, subtracts adjacent
values, estimates p50/p95/p99 bounds, and draws compact bars.

### Internal Prior Work

- [PR #7319](https://github.com/marin-community/marin/pull/7319) added
  `json_get` and related UDFs specifically so semi-structured label columns can
  be filtered and grouped in Finelog SQL. Native maps use a direct per-row
  lookup; JSON strings require parsing.
- [PR #7324](https://github.com/marin-community/marin/pull/7324) changed
  semi-structured labels from JSON text to native `Map<Utf8,Utf8>`. The change
  intentionally retained arbitrary label sets and duck-typed query
  compatibility.
- [PR #7336](https://github.com/marin-community/marin/pull/7336) defined the
  durable Telltale row. It explicitly placed `le` and ad hoc Prometheus labels
  in the Map while lifting stable process/job identity into typed columns.
- [PR #7349](https://github.com/marin-community/marin/pull/7349) mirrors native
  vLLM `/metrics` families into Telltale while preserving Prometheus
  counter/histogram semantics for query-time rate and latency calculations.
- [PR #7362](https://github.com/marin-community/marin/pull/7362) verified Map
  labels survive a 40-segment compaction and remain queryable through
  `json_get`.
- [PR #7598](https://github.com/marin-community/marin/pull/7598) moved Iris RPC
  metrics into Telltale. `RpcStatsPanel.vue` reconstructs classic histogram
  buckets from the same Finelog representation and draws log-scaled compact
  histograms.

No internal artifact proposed a typed `le` column. Echo corpus search was
attempted with the queries `telltale histogram finelog labels le dashboard`,
`finelog telltale Map labels json_get performance`, and
`Prometheus native histogram telltale`; it could not authenticate to Cloud SQL
because this session lacks a resolvable ADC database identity. GitHub and local
git history supplied the durable internal record instead.

### External Prior Art

[Prometheus histogram guidance](https://prometheus.io/docs/practices/histograms/)
distinguishes classic histograms, where `_sum`, `_count`, and every configured
bucket are separate time series, from native histograms, where one composite
sample contains the histogram. Prometheus now recommends native histograms when
the producer and backend support them because they reduce series count and
retain a mergeable distribution.

The [Prometheus native histogram
specification](https://prometheus.io/docs/specs/native_histograms/) defines a
composite value with count, sum, schema, zero bucket, and sparse positive and
negative buckets. It does not model a native histogram as classic scalar rows
plus a promoted `le` column.

The [OpenTelemetry metrics data
model](https://opentelemetry.io/docs/specs/otel/metrics/data-model/) likewise
transports a histogram point as count, sum, explicit bounds, and bucket counts;
its exponential histogram uses a scale, offsets, and dense bucket counts.
OpenTelemetry's [Prometheus/OpenMetrics
compatibility](https://opentelemetry.io/docs/specs/otel/compatibility/prometheus_and_openmetrics/)
converts cumulative Prometheus buckets into per-bucket counts by subtracting
the next-lowest bucket and drops the explicit `+Inf` bound.

[Grafana's Prometheus query
documentation](https://grafana.com/docs/grafana/latest/datasources/prometheus/query-editor/)
does the same work at the view boundary for heatmaps: it converts cumulative
histograms to regular bucket populations and sorts by boundary.

The adversarial result is from ClickHouse's observability work. [ClickStack
materializes hot Map
attributes](https://clickhouse.com/blog/clickstack-faster-observability)
because scanning and decoding a generic Map costs I/O and CPU at large
observability volumes. This supports selective materialization after profiling,
not predeclaring every possible attribute.

### Live Finelog Measurement

Queries ran against the Marin Finelog hub on 2026-07-25. They selected the exact
key `name = 'vllm:request_prompt_tokens_bucket'`, `source = 'vllm'`, and rows
from 2026-07-24 onward. Wall times include CLI startup, OAuth, network, the Iris
proxy, and server execution.

| Query | Rows examined/result | End-to-end wall time |
| --- | ---: | ---: |
| `count(*)` using typed predicates | 1,574,629 | 2.49 s |
| `count(*)` plus `json_get(labels, 'model_name')` filter | 51,712 matched | 3.24 s |
| group all rows by `json_get(labels, 'le')` | 17 groups | 3.40 s |

These single runs include different result shapes, OAuth, network, proxy, and
cache state, so their differences do not isolate Map overhead. They establish
only that a bounded scan with Map lookup and grouping completes within a few
seconds at this volume. The grouped query is not histogram reconstruction: it
mixes snapshots and series. A correct dashboard query must first select a
point or compute reset-aware deltas per complete series identity.

For a current-snapshot view, the existing pattern is:

```sql
SELECT name, value, ts,
       json_get(labels, 'engine') AS engine,
       json_get(labels, 'model_name') AS model_name,
       json_get(labels, 'le') AS le
FROM telltale
WHERE source = 'vllm'
  AND name IN (
    'vllm:request_prompt_tokens_bucket',
    'vllm:request_prompt_tokens_sum',
    'vllm:request_prompt_tokens_count'
  )
  AND job_id = '<job>'
QUALIFY row_number() OVER (
  PARTITION BY name, cluster, run, job_id, task_index, attempt,
               worker, region, process_index,
               json_get(labels, 'engine'),
               json_get(labels, 'model_name'),
               json_get(labels, 'le')
  ORDER BY ts DESC
) = 1
```

The view groups by `engine,model_name`, sorts numeric `le` with `+Inf` last,
and computes each bucket population by subtracting the preceding cumulative
value. A range view must calculate reset-aware deltas before that subtraction.

A typed `le` column would not add useful selectivity because reconstruction
normally reads every boundary. Series identity would still require `engine`,
`model_name`, and any future producer dimensions from the Map. The feasibility
measurement therefore supports deferring a migration, while the ClickHouse
experience supports profiling again if a real dashboard misses its SLO.

### Negative / Failed Leads

- A typed `le` column alone does not create a native histogram. It reduces one
  lookup but keeps classic bucket-series multiplication and still requires Map
  extraction for other dimensions.
- Native histogram storage would be a larger ingest and wire-format project.
  The installed `prometheus_client` 0.24.1 `Histogram` exposes only classic
  explicit buckets, and the vLLM bridge parses the classic text exposition.
- A persistent SQL view is not available through Finelog's SELECT-only Query
  API. A product-specific query or dashboard component can reconstruct the
  current snapshot without changing storage.
- No source gave a universal volume threshold for materializing Map keys.
  Query SLOs and profiles are the appropriate trigger.

### Evidence Map

#### Claim: Current rows preserve classic Prometheus histogram semantics

- Support:
  - Prometheus histogram guidance: classic buckets, sum, and count are separate
    scalar series.
  - Marin PRs #7336 and #7349: Telltale deliberately stores the flattened
    samples and preserves vLLM semantics.
- Contradictions:
  - Prometheus recommends native histograms where supported.
- Directness to Marin: exact current producer and storage path.
- Confidence: stable.
- Action: preserve the durable schema in this change.

#### Claim: Map extraction is acceptable now but should remain measurable

- Support:
  - Live Finelog measurement: a bounded `le` grouping over 1.57 million rows
    completed in 3.40 seconds end to end.
  - Finelog clusters by metric name, bounding the relevant scan.
- Contradictions:
  - ClickStack materializes common attributes because Map lookup becomes
    expensive at larger observability scale.
- Directness to Marin: live production query on the reported metric family.
- Confidence: exploratory; the measurement is not server-only and does not
  isolate Map extraction.
- Action: keep Map storage and define a profile/SLO gate for materialization.

#### Claim: Histogram reconstruction belongs in the view

- Support:
  - Grafana converts cumulative Prometheus buckets for histogram/heatmap views.
  - Iris `RpcStatsPanel.vue` already reconstructs Telltale histograms this way.
  - OpenTelemetry's classic conversion subtracts adjacent cumulative buckets.
- Contradictions:
  - A backend with native histogram samples can perform distribution functions
    directly without client-side grouping.
- Directness to Marin: exact existing dashboard code and current classic
  producer format.
- Confidence: stable.
- Action: group the local HTML page while keeping `/metrics` raw.

### Recommended Next Experiments

#### 1. Compact the process-local histogram display

- Minimum experiment: render one labelled classic histogram through the
  Telltale TestClient and verify one compact row with exact bucket tooltips.
- Baseline/control: the current raw `_bucket`, `_sum`, and `_count` rows.
- Expected signal: about 20 rows collapse to one per vLLM
  `engine,model_name` series without changing `/metrics`.
- Falsifier: grouping loses a non-`le` label combination or miscomputes
  per-bucket populations, including malformed/incomplete families.
- Cost/risk: small, local HTML-only change.
- Sources: Prometheus histogram guidance, OpenTelemetry compatibility, Marin
  `RpcStatsPanel.vue`.

#### 2. Re-profile before materializing any label

- Minimum experiment: repeat typed-only and Map-extraction queries over the
  exact metric and time window when a dashboard misses its latency budget; use
  server-side profile/scan metrics.
- Baseline/control: current `json_get` query.
- Expected signal: materialize only if Map extraction dominates the missed
  budget.
- Falsifier: scan volume, sorting, auth/network, or another label dominates.
- Cost/risk: low read cost when bounded by metric and time.
- Sources: live measurement and ClickStack materialized-column experience.

### Hypothesis Queue Update

- Add: compact server-rendered histogram rows improve the reported debugging
  experience without storage migration.
- Revise: Map lookup is not free; it is acceptable at current scale, not
  categorically cheap.
- Falsify / stop: adding a nullable `le` column now.
- Promote: revisit a composite native histogram only with producer support or
  storage-volume evidence.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| Prometheus histogram guidance | official docs | https://prometheus.io/docs/practices/histograms/ | classic vs native series shape | high | Current guidance recommends native histograms where available |
| Prometheus native histogram spec | official spec | https://prometheus.io/docs/specs/native_histograms/ | native composite shape | high | Shows why `le` alone is not native |
| OpenTelemetry metrics model | official spec | https://opentelemetry.io/docs/specs/otel/metrics/data-model/ | structured histogram points | high | Explicit and exponential forms |
| OTel Prometheus compatibility | official spec | https://opentelemetry.io/docs/specs/otel/compatibility/prometheus_and_openmetrics/ | cumulative-to-regular conversion | high | Direct conversion rule |
| Grafana Prometheus query docs | official docs | https://grafana.com/docs/grafana/latest/datasources/prometheus/query-editor/ | view-layer histogram conversion | high | Heatmap behavior |
| ClickStack Map materialization | first-party engineering | https://clickhouse.com/blog/clickstack-faster-observability | adversarial Map-cost evidence | medium | Larger scale and different engine |
| Marin PR #7319 | PR | https://github.com/marin-community/marin/pull/7319 | label-query UDF intent | high | Exact Finelog code |
| Marin PR #7324 | PR | https://github.com/marin-community/marin/pull/7324 | native Map storage intent | high | Exact Finelog code |
| Marin PR #7336 | PR | https://github.com/marin-community/marin/pull/7336 | Telltale row/schema decision | high | Exact current contract |
| Marin PR #7349 | PR | https://github.com/marin-community/marin/pull/7349 | vLLM bridge semantics | high | Exact producer path |
| Marin PR #7362 | PR | https://github.com/marin-community/marin/pull/7362 | Map compaction verification | high | 40-segment check |
| Marin PR #7598 | PR | https://github.com/marin-community/marin/pull/7598 | existing view reconstruction | high | Exact current UI pattern |
| Marin hub query, 2026-07-25 | live measurement | `telltale` namespace | bounded-query feasibility and volume | medium | Single end-to-end runs, not server-only or a reconstruction benchmark |

### Handoff

- Suggested issue `Prior work` block: Telltale deliberately stores classic
  Prometheus histogram samples with `le` in a native Map. A bounded live query
  grouped 1.57 million bucket rows by `le` in 3.40 seconds end to end, which is
  enough to defer a schema migration but not enough to attribute Map overhead.
  Keep the schema and compact histograms in the local view.
- Suggested logbook entry: no typed `le`; add display grouping and profile
  before materializing labels.
- Open questions: when producers expose native histograms; what dashboard
  latency budget should trigger materialization.
- Stop reason: internal decisions, external models, adversarial Map evidence,
  and the production measurement all support the same near-term design.
