# Compact Telltale Histogram View Contract

## Persisted and exposition formats

`TelltaleMetric` and the Finelog `telltale` namespace do not change. A classic
Prometheus histogram remains scalar rows:

```text
<family>_bucket  value=<cumulative count>  labels={..., le=<upper bound>}
<family>_sum     value=<sum>               labels={...}
<family>_count   value=<count>             labels={...}
```

`GET metrics` remains the unmodified Prometheus text exposition.

## Local HTML format

`GET /` renders one table row per histogram family and non-`le` label set.

- Sample cell: Prometheus family name without `_bucket`, `_sum`, or `_count`.
- Type cell: `histogram`.
- Labels cell: sorted labels excluding `le`.
- Value cell: observation count, average when `count > 0` and sum is present,
  and one spark character per sorted bucket.
- Bucket order: numeric ascending, then `+Inf`.
- Bucket population: `cumulative[i] - cumulative[i - 1]`.
- Spark characters: `▁▂▃▄▅▆▇█`. For positive population `p` and largest
  population `m`, use level
  `ceil(log1p(p) / log1p(m) * 8) - 1`, clamped to `[0, 7]`. Zero is a middle
  dot.
- Tooltip: exact `≤ <bound>: <population>` for the first bucket and
  `(<previous>, <bound>]: <population>` thereafter.
- Observation count: `_count`, falling back only to the `+Inf` cumulative
  value.
- Histogram `_created` samples: omitted, matching durable forwarding.

An absent `_count` falls back to a valid `+Inf` bucket. If `_count` is present
and `+Inf` is absent, the view appends an overflow bucket with population
`count - last_finite_cumulative`; this must be non-negative. With neither
`_count` nor `+Inf`, finite buckets render and the value cell states that the
distribution is truncated. An absent, zero, negative, or non-finite effective
count omits the average.

The whole histogram group is malformed and renders no sparkline when any of
these conditions hold:

- an invalid or duplicate numeric bound;
- duplicate `_sum` or `_count` samples;
- a negative or non-finite cumulative bucket value;
- cumulative values decrease after numeric sorting;
- `_count` is negative/non-finite or differs from the `+Inf` cumulative value;
- a synthesized overflow population would be negative.

A non-finite sum only suppresses the average; bucket counts may still be valid.
Malformed and truncated groups remain one display row and link implicitly to
the raw `metrics` exposition for diagnosis.

Non-histogram rows keep the existing sample/type/labels/value representation.
All metric names, label names/values, bounds, and tooltip text are HTML-escaped.
The heading reports both raw sample count and rendered display-row count so
histogram compaction does not relabel display rows as samples.
