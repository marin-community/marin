# Finelog Telemetry Layout

Finelog should answer recurring metric-name and run-scoped telemetry queries in
well under the ten-second server deadline while retaining its current Parquet
and local-hot/GCS-archive model. Production replay found that exact-index
planning, physical row ordering, and JSON-only resource identity each impose
avoidable work. The selected changes address those mechanisms without a table
format migration.

The full measurements and rejected alternatives are recorded in
[research.md](research.md). Public and persisted contracts are pinned in
[spec.md](spec.md).

## Challenges

The hot `telemetry_v1` table contains about 1.36 billion rows across hundreds of
segments. Its `.fidx` exact-posting payloads are useful for a small metric
allowlist, but the planner previously decoded every segment payload before it
could learn that another value was absent. A fixed equality query therefore
spent seconds constructing a plan even when execution needed little data.

The retention key, physical sort order, and row-group sizing were one policy.
Sorting only by time preserves retention semantics but provides weak Parquet
statistics for the usual `(service, run, name, time)` filters. Stable resource
identity also lived in `resource_attributes_json`, preventing typed predicates
and encouraging repeated JSON extraction in dashboards and alerts.

The store is live and persisted. New binaries must read existing Parquets and
version-1 index bundles while a partial backfill is in progress. Existing
producers must continue to send the resource-attribute envelope until they opt
into explicit fields.

## Costs / Risks

- The selected 128K-row, service/run/name layout increased Parquet bytes by
  6.6% and increased recent service aggregation from 28.5 to 34.9 milliseconds
  in the 48.1-million-row production sample.
- Old rows expose null promoted dimensions. Training and Zephyr alerts retain a
  JSON fallback; other identity-filtered views show only post-deploy samples
  until the roughly 40-hour hot window drains.
- Version-1 exact bundles remain correct scan fallbacks. Rebuilding the 232
  production bundles at four per 30-second maintenance tick takes about 29
  minutes when the namespace is otherwise quiet.
- A server rollback can read the additive nullable string columns and old/new
  Parquets while producers continue to send identity through attributes. The
  previous HTTP parser rejects the new explicit resource fields. Producers must
  remain attribute-only through the rollback window. An old server also drops
  the layout policy and may rebuild version-2 index bundles as version 1;
  rolling forward reapplies both.

## Design

Exact-posting sections move to method version 2. Their `.fidx` header coverage
contains the values present for each indexed column. Equality and same-column
`IN`/`OR` planning checks this small header before opening or decompressing the
postings payload. Version-1 bundles keep their previous behavior and are queued
for normal background rebuild. A corrupt or unknown header remains a source
scan fallback.

`Schema` gains `sort_columns` and `max_row_group_rows`. The event/retention key
remains `timestamp_ms`; the compactor orders telemetry by
`(service, run_id, name, timestamp_ms, seq)` and limits row groups to 131,072
rows in addition to the encoded-byte target. Empty layout fields preserve the
historical key-plus-sequence behavior for other namespaces and older clients.
New L0 and merged files use the row limit. Compaction applies sorting when it
rewrites multiple inputs. The change does not bump the global layout revision,
so terminal production files are not rewritten solely for this policy; they
leave the hot cache through normal compaction and retention.

Telemetry promotes `run_id`, `job_id`, `execution_uid`, `region`, `node_name`,
`process_index`, and `alert_tag` to nullable string columns. The HTTP resource
object accepts each explicitly. Missing values fall back to same-named resource
attributes, with legacy `root_run_uid` also feeding `run_id`. Explicit values
win and canonicalize the same key in `resource_attributes_json`. First-party
Grafana and dashboard queries use the scalar columns. Safety-critical alert
queries retain a JSON fallback for pre-deploy and rollback rows. Covering
projections include the promoted fields required by those selectors.

GCS remains the durable archive, and local SSD remains the bounded hot query
surface. A direct bounded GCS query took 19.42 seconds versus about 240
milliseconds over local source files, so raw GCS Parquet reads are not placed on
the dashboard or alert path.

## Testing

Unit coverage checks header-only rejection without touching a deliberately
corrupt postings payload, version-1 fallback, schema/protobuf/catalog
round-trips, multi-column compaction ordering, row-group ceilings, and explicit
resource fields overriding inferred attributes. Grafana tests execute migrated
SQL against DuckDB fixtures containing structured identity columns.

The production replay uses 244 copied Parquets and 232 copied index bundles:
1,360,540,454 rows and 16.06 GB. The fixed exact-name query improved from
8,563.6 milliseconds to 189.3 milliseconds warm p50; `EXPLAIN` improved from
6,550.1 to 15.9 milliseconds. The separate physical-layout sample improved the
same query from 194.8 to 15.8 milliseconds.

The staged rollout starts at `marin-dev` with a captured image-digest rollback
target. Gates include health-body success, zero unexpected container restarts,
clean startup logs, the effective telemetry schema, fresh-row ingestion, exact
query latency, index-backfill progress, and a rollback drill before any producer
uses explicit resource fields. Physical-layout acceptance compares point-query
latency, service aggregation, bytes, compaction backlog, and the proportion of
query-visible segments produced by multi-input merges. The new sort order does
not guarantee that every file converges because single-input promotions are
rename-only.

## Open Questions

- Production can let the share of old physical layout decline with the hot
  window, or introduce a layout revision and pay an immediate local rewrite.
  The current proposal chooses gradual adoption unless the dev and production
  canaries show a reason to force migration.
- Catalog timestamp bounds can remove file-footer fanout before Parquet planning.
  This remains follow-up work: a one-hour exact query reads less data but checks
  more files when `LIMIT 100` must find nearly every recent match.
