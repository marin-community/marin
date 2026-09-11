# Finelog Segment Index Families

Finelog uses a small planner-facing index family to make recurring filters and
low-cardinality analytics fast without adding a second mutable storage engine.
Each immutable Parquet segment owns one checksummed `.fidx` metadata bundle.
Large covering projections remain ordinary narrow Parquet files referenced by
that bundle.

This design follows the production query-saturation incident recorded in
[Echo](https://echo.oa.dev/wiki/90). Repeated Grafana training-status scans
continued after layout rewrites had finished, held the query-visibility read
lock longer, and delayed otherwise cheap compaction commits. Concurrency limits
and a 10-second default deadline contain that failure mode. Segment indexes make
the recurring queries cheap enough not to reach the containment boundary.

The architecture was reviewed against PostgreSQL access methods, BRIN, Iceberg
Puffin, ClickHouse skip indexes and projections, and Parquet page indexes. The
accepted [Weaver design](https://loom.oa.dev/s/qx90ms6l/artifacts/design) and
[Claude review](https://loom.oa.dev/s/qx90ms6l/artifacts/claude-review) contain
the longer comparison.

## Decisions

### One bundle per segment

The files for a segment are:

```text
seg_L3_….parquet
seg_L3_….parquet.fidx
seg_L3_….parquet.fidx.training-status.parquet
```

`.fidx` has an internal prefix directory. Its fixed header binds the derived
state to the source segment identity, row count, schema fingerprint, and
complete index-policy fingerprint. New segments carry a UUID in Parquet
key-value metadata. Existing current-layout segments that predate that stamp use
a local generation identity derived from device, inode, length, and nanosecond
mtime, so adopting indexes does not require another fleet-wide layout rewrite.
Level-only renames preserve either identity; replacement or layout rewrite
invalidates the old bundle. A layout rewrite gives an unstamped segment a UUID.

Each directory entry records a stable section ID, method kind, method format
version, exactness class, concrete coverage, offset, length, checksum algorithm,
and SHA-256 checksum. The header also records the total bundle length and a
directory checksum. Readers load the bounded directory first and then issue
positioned reads only for methods required by the query.

A bad directory, invalid extent, unexpected length, or source-ID mismatch
disables the bundle. A bad payload checksum disables only that section. Both
paths increment separate corruption counters and continue with source Parquet.
Malformed derived data is never a query error.

A filesystem directory containing one file per method would still need a
last-published manifest to define completeness. It would add open, stat, rename,
cleanup, and cache-key states for every segment while recreating the same
directory in the filesystem. A namespace-wide index file would be worse: it
would introduce mutable shared state, its own compaction and recovery, and a
second locking domain. The segment bundle follows Finelog's existing immutable
lifecycle.

Covering projections stay outside `.fidx` because DataFusion can scan Parquet
directly and projection data can be much larger than metadata. A builder writes
and syncs projection files first and publishes the bundle last. The bundle
reference includes a stable projection-spec ID, projection segment ID, byte
length, row count, included columns, predicate, and source-row identity domain.

### A closed method family

User schemas remain concise: column flags declare trigram, exact-value, and
value-count policy, while `Schema.projections` declares independent named
covering projections. The server compiles those declarations into a closed Rust
`IndexSpec` enum:

```text
TrigramBloom { column }
ExactPostings { column, values }
ValueCounts { column }
CoveringProjection { projection }
```

This is the useful part of PostgreSQL's access-method and operator-class split:
storage methods are distinct from the SQL shapes they support. It deliberately
does not reproduce PostgreSQL's dynamic extension catalog. Adding a method
requires a typed enum variant, schema validation, a versioned section codec, a
planner rule, fallback tests, and a copied-shard benchmark. There is no
free-form method string or runtime plugin.

The initial methods do not overlap:

| Method | Query family | Planner result |
| --- | --- | --- |
| `TrigramBloom` | `contains` and literal runs in `LIKE` | Conservative source-row span mask; residual filter retained |
| `ExactPostings` | Configured `=`, `IN`, and same-column `OR` | Exact source-row selection; residual filter retained |
| `ValueCounts` | Unfiltered one-column `GROUP BY` with `COUNT(*)` or `COUNT(column)` | Exact aggregate subtree replacement |
| `CoveringProjection` | Configured predicate whose referenced columns are covered | Per-segment replacement Parquet file |

Parquet already supplies row-group min/max statistics, so Finelog does not add
a second zone-map or BRIN implementation. A generic equality Bloom is also
omitted: common telemetry values occur in nearly every time-oriented span and
would retain most of the data. New sketches, multi-column summaries, token
indexes, or automatic index selection require measured workloads before joining
the family. Current source and projection files already carry Parquet column and
offset page indexes, and DataFusion page-index pruning is enabled. This existing
layer remains the fine-grained range/equality fallback rather than becoming a
duplicate `.fidx` method.

### Planner substitution and cost guards

All methods fold into DataFusion planning. Missing, stale, malformed,
incomplete, or inapplicable state returns `NotApplicable` and leaves the
ordinary plan intact.

Trigram masks and postings apply per source segment. A covered projection also
substitutes per segment; uncovered segments continue through postings or the
source scan. This makes partial backfill monotonically useful. The residual
filter remains, so combining these paths cannot add wrong rows.

Every row-addressed decision states its row-identity domain. Source postings or
trigram spans are discarded after a projection changes the row space. A future
projection-specific method must name that projection's stable spec ID as its
domain.

Exact postings are used only when they retain at most 25% of a segment. Above
that threshold, applying a fragmented row selection is more expensive than a
contiguous Parquet scan. This is a planner cost guard, not a correctness rule.

Value-count substitution requires complete coverage for every visible segment.
Each segment omits a summary above 4,096 distinct non-null values, and the
optimizer aborts substitution if the combined result exceeds 16,384 values.
The logical optimizer emits `FinelogIndexAggregate`; `EXPLAIN`, outer
projections, `ORDER BY`, and `LIMIT` therefore see and compose with the rewrite.
`COUNT(column)` excludes the null bucket while `COUNT(*)` includes it.

### Named projections, not column or JSON shattering

Parquet already stores ordinary columns separately and DataFusion projects only
referenced columns. One file per source column would require stable row-position
joins, more file opens, and a late-materialization operator without reducing the
column bytes read by the current path.

Each covering projection instead declares one predicate and one explicit
included-column list. The planner substitutes it only when the query's predicate
values and every scan or filter column are covered. A second exact-value policy
cannot silently widen an existing projection.

The initial `training-status` projection contains rows whose `name` is `phase`,
`step`, or `progress_time_seconds` and only these columns:

```text
seq, timestamp_ms, service, name, value, resource_attributes_json, cluster
```

Arbitrary JSON shattering is out of scope. OpenTelemetry attribute bags are
sparse and evolve by producer. A JSON path should become a typed generated
column only when a recurring query uses it, copied-shard profiling shows JSON
extraction remains material after pruning, and the path has a stable type and
meaning. Once promoted, indexes and projections treat it like any ordinary
column.

## Build, cache, and lifecycle

L0 flushes and compaction already hold Arrow batches. One builder feeds every
configured method and projection from those batches. Historical backfill reads
the union of required columns once and is bounded to a few bundles per namespace
per maintenance tick. Index building happens outside the process-wide
query-visibility write lock.

The query cache keys directories and decoded sections by segment identity and
section ID. Trigram and exact payloads are charged by decoded heap size. A
backfill publish invalidates the old entries. Equality queries do not read
trigram bytes, and aggregate queries do not read postings.

DataFusion repartitions multi-file scans above 1 MiB rather than its 10 MiB
default. Narrow projections otherwise form one serial scan despite spanning
several files and hundreds of thousands of rows. Sub-megabyte scans retain the
lower-overhead serial path.

Level bumps rename the segment, bundle, and named projections together.
Compaction input deletion, eviction, and table deletion remove all derived
artifacts. Rebuilds remove projection files no longer referenced by policy.
Migration cleanup continues deleting legacy `.tgm`, `.eqi`, and `.eqp` files,
although the query path no longer reads them. Derived files remain local and are
never uploaded or adopted as source segments.

Faster queries shorten the visibility read-lock hold but do not remove the
query/compaction publication coupling. Changing that lock is a separate design.

## Observability

`GET /api/segments?namespace=NS&physical=true` reports the source segment ID and
the readable bundle directory, including method, exactness, version, checksum,
and payload size. `GET /api/server` reports separate corrupt-bundle and
corrupt-section counters.

Planner debug events report covered or pruned segments, retained rows, and
high-selectivity posting fallbacks. Exact aggregate substitution is visible in
`EXPLAIN` as `FinelogIndexAggregate`. These signals distinguish incomplete
backfill, an unsupported query shape, poor selectivity, corruption, and a
planner regression.

The deployment option `query_index_cache_mb` controls decoded bundle-section
cache size. `query_metadata_cache_mb` remains the independent DataFusion
Parquet-footer cache.

## Benchmark and acceptance criteria

The completed `.fidx` implementation was measured on four copied production
telemetry segments containing 29.2 million rows and 171.9 MB of source Parquet.
The training-status query fell from a 3.601-second median to 232.6 milliseconds
with the same schema, 88 rows, and result digest: a 15.5× improvement. The
unfiltered service-count query fell from 59.4 milliseconds to 1.5 milliseconds,
about 40×. Derived bundles and projections added 8.8 MB, or 5.1%.

With half the projection files deliberately unavailable, the training query
still completed in 447.4 milliseconds with the same digest, proving per-segment
fallback is correct and useful during backfill. A 50%-selectivity posting case
retains the contiguous scan. `EXPLAIN` shows 12 projection scan partitions and
the `FinelogIndexAggregate` rewrite. Benchmarking uses copied local shards; it
does not run expensive operations on the production server or bucket.

Parquet page indexes were inspected before accepting another physical method:
both source and projection columns already carry column and offset indexes, and
DataFusion enables their pruning by default. The copied-shard result meets the
target without another projection, smaller pages, JSON shattering, or a new
page-level `.fidx` method. Those remain evidence-driven follow-ups if a measured
query still spends material time below the current pruning layers.
