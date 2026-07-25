# Compact Telltale Histogram Views

The process-local Telltale page should show one compact row per Prometheus
histogram series instead of exposing every `_bucket`, `_sum`, and `_count`
sample as an unrelated row. Finelog should continue storing the original
Prometheus samples. Its existing `Map<Utf8,Utf8>` labels column is fast enough
for the current volume and preserves arbitrary producer labels without schema
churn.

For the reported `vllm:request_prompt_tokens` family, the local page will
replace about 20 rows per `engine,model_name` series with output shaped like:

```text
vllm:request_prompt_tokens  histogram  engine=4,model_name=...  n=3 avg=64  ····████████████
```

Each spark character represents the non-cumulative population of one bucket.
Its tooltip shows the bucket interval and count. The raw Prometheus exposition
remains available at `metrics`.

## Challenges

Prometheus classic histograms are flattened before Telltale sees them. A
histogram family arrives as scalar `_bucket{le=...}`, `_sum`, and `_count`
samples. Buckets are cumulative, so a useful display must group by the family
and every label except `le`, sort `+Inf` last, and subtract adjacent cumulative
counts.

The persisted rows are cumulative counter snapshots. Queries over a time range
must select a point or compute reset-aware deltas before calculating a
distribution. A display-only grouping must not imply that summing snapshots or
taking `max(value)` is generally correct.

## Costs / Risks

- The local HTML renderer gains histogram-specific grouping and formatting.
  `/metrics`, `scrape_metrics`, and the Finelog wire/storage schema do not
  change.
- A sparkline gives shape and exact tooltip values, but it cannot estimate
  within-bucket values. The displayed quantiles would only be bucket bounds, so
  this change does not add quantile labels.
- Native Prometheus histograms remain unsupported. vLLM and
  `prometheus_client` 0.24.1 currently expose classic histograms to this path.
  Supporting native histograms would require a composite sample type at the
  scrape and storage boundaries.

## Design

Keep `TelltaleMetric` unchanged:

```python
TelltaleMetric(
    name="vllm:request_prompt_tokens_bucket",
    value=3.0,
    kind="histogram",
    labels={"engine": "4", "model_name": "...", "le": "100.0"},
    # ts and typed process/job identity omitted
)
```

Finelog continues to cluster the `telltale` namespace by `name`. `le`,
`engine`, and `model_name` remain entries in the native Map column and are
read with `json_get(labels, 'le')`. This matches Prometheus classic histogram
storage: `_bucket{le}`, `_sum`, and `_count` are distinct scalar series. It
also keeps arbitrary label sets possible across vLLM, Levanter, Zephyr, Iris,
and Python/process collectors.

Adding only a nullable `le` column is not a native histogram representation. It
would optimize one label lookup on bucket rows while `_sum`, `_count`, other
histogram dimensions, and all non-histogram labels retain their current shape.
If Telltale later adopts a composite histogram sample, use an explicit
`bounds[]`, `bucket_counts[]`, `count`, and `sum` representation like
OpenTelemetry or a Prometheus native histogram. Do not evolve toward that
representation one producer-specific scalar column at a time.

Change `rigging.telltale._render_index` to group the current registry snapshot
for display:

1. Pass non-histogram samples through as today.
2. Group histogram samples by Prometheus family and the sorted labels excluding
   `le`.
3. Record `_sum`, `_count`, and numeric/`+Inf` bucket samples for each group.
4. Validate that bounds are unique, cumulative values are finite,
   non-negative, and non-decreasing, and `_count` agrees with the `+Inf`
   bucket. A malformed family renders one compact warning row without a
   distribution, so the page cannot present invented bucket populations.
5. If `_count` is present but `+Inf` is absent, synthesize the standard overflow
   bucket by subtracting the last finite cumulative value from `_count`.
6. Convert valid cumulative bucket values to non-cumulative populations and
   render one row containing count, average when defined, and a log-scaled
   Unicode sparkline with exact interval tooltips.
7. Use `_count` as the observation count, falling back only to the `+Inf`
   cumulative bucket. Omit the average unless both a finite sum and a positive,
   finite count are present.
8. Drop histogram `_created` samples from this compact view, matching the
   durable forwarding path.
9. Fall back to the data that is present for an incomplete custom histogram;
   the status page must not fail because a collector omitted `_sum` or
   `_count`.

The Finelog query path needs no code change. From 2026-07-24 through the
measurement time on 2026-07-25,
`vllm:request_prompt_tokens_bucket` had about 1.57 million rows and 17 distinct
bounds. End-to-end queries through OAuth and the Iris proxy took 2.49 seconds
for a typed-column `count(*)`, 3.24 seconds when filtering `model_name` through
`json_get`, and 3.40 seconds when grouping rows by `le`. These single runs show
that bounded Map lookup and grouping complete within a few seconds. They do not
isolate Map CPU, benchmark a correct histogram reconstruction query, or define
a dashboard latency SLO.

A typed `le` column would not make the common histogram query selective:
reconstruction normally reads every bound. It also cannot identify a series
without `engine`, `model_name`, and producer-specific dimensions from the Map.
It therefore adds little pruning value while leaving the expensive grouping
and classic-series multiplication unchanged.

Promote a label to a typed/materialized column only when a repeated production
query misses its latency or resource budget and a profile attributes the miss
to Map extraction. A candidate must also be meaningful across enough rows to
justify a permanent Telltale schema field. `le` does not meet that threshold.

## Testing

Extend `lib/rigging/tests/test_telltale.py` through the rendered HTTP page:

- a labelled histogram renders one row without raw `_bucket`, `_sum`, `_count`,
  or `le=` cells;
- two non-`le` label combinations remain separate histogram rows;
- cumulative buckets become the expected interval tooltip counts;
- bounds sort numerically (`2`, `10`, `+Inf`) and `+Inf` is last;
- an empty histogram renders without division by zero;
- `_count` falls back only to `+Inf`, and a missing `+Inf` bucket is synthesized
  from a valid `_count`;
- invalid/duplicate bounds, decreasing/non-finite cumulative values, duplicate
  count/sum samples, and count/`+Inf` mismatches render as malformed without a
  sparkline;
- histogram `_created` is omitted;
- an ordinary gauge remains a scalar row;
- metric and label text remains HTML-escaped.

Run the Rigging package tests, then the repository changed-file checks. Finelog
tests are unchanged because the persisted schema and query behavior are
unchanged.

## Open Questions

- The local view will use log-scaled bars so small tail buckets remain visible,
  matching the existing Iris RPC histogram panel. A later shared Vue component
  could unify the visual treatment, but the process-local page cannot depend on
  bundled assets because it is served behind arbitrary Iris proxy prefixes.
- Native histogram ingestion may become worthwhile if vLLM exposes it and
  Telltale volume grows enough that classic bucket-series multiplication, not
  query latency, becomes the dominant cost.
