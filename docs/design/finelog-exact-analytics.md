# Finelog Exact Analytics

Finelog should answer recurring dashboard filters and low-cardinality rollups in
well under its 10-second query deadline without becoming a general-purpose OLAP
database. On copied production telemetry, the motivating training-status query
took 3.92 seconds and a service-count rollup took 60 milliseconds. This design
targets at least a 10× reduction for both shapes while preserving the existing
DataFusion path for every other query.

The production slowdown combined segment-layout rewrites with repeated Grafana
queries. Concurrency and a 10-second default query deadline contained the
incident, but copied production shards showed that containment did not make the
recurring query family cheap. The incident record is
[Finelog production CPU saturation from Grafana training queries](https://echo.oa.dev/wiki/90).

## Challenges

Finelog queries a snapshot containing many independently written Parquet
segments. An auxiliary index is safe only when it describes the exact segment
generation being scanned, and a filtered data copy is safe only when it contains
every row that could satisfy the predicate. Historical namespaces also need to
gain the new artifacts without concentrating enough backfill work to recreate
the CPU saturation this work is intended to avoid.

The useful workloads have two different shapes. A training dashboard repeatedly
filters a wide table to three known metric names before performing windowed
aggregations. A query such as `SELECT service, count(service) ... GROUP BY
service` needs every row logically, but only the group counts physically. One
index structure does not efficiently serve both.

## Costs and risks

- Exact metadata and filtered projections added 20.4 MB beside 171.9 MB of
  copied production Parquet, an 11.9% local-storage overhead.
- Building one historical shard took roughly three seconds on the local
  benchmark machine. Backfill must remain bounded and lower priority than
  serving queries.
- Exact projections accelerate only values declared in the schema. Value-count
  summaries are omitted above 4,096 distinct values, where an in-sidecar
  histogram would stop being compact.
- Derived artifacts are local-only. Restored or newly indexed historical
  segments use the ordinary scan until local maintenance rebuilds them.

## Design

Add two independent policies to a string column's `ColumnIndex`:

```protobuf
message ColumnIndex {
  optional bool trigram = 1;
  repeated string exact_values = 2;
  optional bool value_counts = 3;
}
```

For `exact_values`, each segment gets an `.eqi` metadata sidecar containing the
matching global row runs and an `.eqp` Parquet file containing the union of all
matching rows. The projection keeps the original schema and uses small
byte-bounded row groups, so ordinary key and timestamp pruning still applies.
Equality, same-column `IN`, and equivalent same-column `OR` filters may
substitute projections only when every visible segment has a current artifact
covering every requested value. Otherwise Finelog uses exact row selections
where possible or the unchanged Parquet scan.

For `value_counts`, `.eqi` stores a complete per-value count, including null,
when the segment has at most 4,096 distinct values. Finelog recognizes an
unfiltered one-column group with one `COUNT(*)` or `COUNT(group_column)` and
combines the segment summaries directly. It uses this path only when every
visible segment has a complete summary whose row count matches the Parquet
footer. Filters, joins, multiple grouping columns, other aggregates, and
high-cardinality columns continue through DataFusion.

L0 flushes and compaction outputs write current artifacts immediately. A level
bump renames them with the segment. Deletion and eviction remove them with the
segment. Maintenance backfills at most one exact artifact per namespace per
tick, reading indexed columns first and reading complete projected rows only
when necessary. Missing, malformed, incomplete, or stale state is always a
performance miss rather than a correctness failure.

`telemetry_v1` enables counts for `service`, `kind`, and `name`, and exact
projection for the `phase`, `step`, and `progress_time_seconds` metric names.
This covers the observed dashboard and broad analytic queries while keeping
index policy explicit in the table schema.

### Schema contract

`ColumnIndex.exact_values` is a sorted, deduplicated set. Registration merges
both policies monotonically: an older client cannot disable an existing index or
remove an exact value. Both policies are valid only on string columns. The
Python and Rust schema representations and both checked-in proto copies
round-trip these fields identically.

### Segment-artifact contract

For a segment `<path>.parquet`:

- `<path>.parquet.eqi` contains an `FLEQ` magic value, format version, source row
  count, optional projection row count, and per-column payloads.
- Exact-value payloads map every configured value, including absent values, to
  sorted non-overlapping half-open row runs.
- A value-count payload is present only when it completely accounts for the
  segment and contains at most 4,096 distinct non-null values plus null.
- `<path>.parquet.eqp` preserves the source Arrow schema and contains exactly the
  union of rows selected by every configured exact value.

Writers publish each artifact by temporary-file rename. Artifacts are derived
local state and are not uploaded as source segments. A compaction merge writes
new artifacts; a level bump renames existing artifacts; unlink and eviction
remove them.

### Query contract

Finelog may extract equality, literal `IN`, and same-column equality `OR`
predicates. It may intersect same-column exact constraints under top-level
`AND`. It must not infer a constraint from a mixed-column disjunction or another
expression whose truth does not imply the extracted values.

Projection substitution is all-or-nothing for a snapshot. Every visible source
segment must have a parseable sidecar whose row count matches the source footer,
whose configured values cover the predicate, and whose projection exists with
the declared row count. Any failed check retains the source files. When a
projection is unavailable but exact row runs are valid, Finelog may attach
Parquet row selections without changing the residual filter.

The summary path accepts only an unfiltered query with one string grouping
column and one `COUNT(*)` or `COUNT(group_column)`. Every visible segment must
have a complete summary for that column. `COUNT(column)` excludes the null
bucket; `COUNT(*)` includes it. Any unsupported logical-plan shape or incomplete
segment returns control to DataFusion without a partial result.

## Alternatives considered

Parquet min/max statistics do not help the training predicate because metric
names are interleaved within time-oriented row groups. Parquet Bloom filters can
reject absent values, but all three motivating values occur throughout the
segments, so they cannot avoid reading the matching row groups.

Row-run metadata alone is exact and compact, but selecting sparse rows from wide
source row groups still performs scattered reads and decoding. It remains a
useful fallback during partial projection coverage.

A materialized aggregate table would make each chosen rollup fast but would
require query-specific ingestion paths and freshness semantics. Complete
low-cardinality per-segment counts cover the generic one-column count family
without introducing another table.

A general secondary-index engine or full OLAP store would broaden the supported
query space, but would add compaction, consistency, and operational systems that
duplicate Finelog's existing segment lifecycle. The observed query families do
not justify that expansion.

## Testing and benchmark

Unit tests pin sidecar parsing, null counts, the cardinality cap, row-run
selection, projection completeness, predicate extraction, all-or-nothing
snapshot fallback, aggregate recognition, schema evolution, L0 creation,
compaction output, level bumps, backfill, and artifact lifecycle.

The production-shaped benchmark uses four copied telemetry shards: 29.2 million
rows and 171.9 MB of source Parquet. The training query returns the same schema
and 88 rows before and after indexing; its median latency falls from 3.92
seconds to 237.7 milliseconds, a 16.5× improvement. The unfiltered service-count
query falls from 60 milliseconds to 4.8 milliseconds, a 12.5× improvement.
Equivalent counts grouped by `kind` and `name` complete in 5.7 and 6.0
milliseconds.

Rollout should first confirm artifact backfill rate, CPU, and query equivalence
on `marin-dev`. Production rollout should verify that ordinary queries retain
their current plans, indexed queries gain coverage gradually, and the 10-second
deadline remains a final containment boundary rather than a normal termination
path.

## Open questions

- Should future schema registrations expose a metric for exact-artifact coverage
  so dashboard owners can see when a newly declared value is fully accelerated?
- If filtered projections grow materially beyond the measured overhead, should
  policy accept a per-column storage budget or should operators remove cold
  values explicitly?
