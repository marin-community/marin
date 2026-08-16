# EvalDash catalog contract

## Persistence

`eval_record_sources` contains one row per object path. Its owning row in `eval_record_prefixes`
supplies the configured priority; `(prefix priority, path)` selects the winning valid source for a
`run_id`. A row may have a null `record` when a new object has never validated; its path-derived run
ID still prevents a migrated serving row from being pruned. A later invalid rewrite preserves the
previous valid body.

`eval_record_prefixes` records configured priority and active membership along with the latest probe,
latest successful complete listing, candidate count, and error. Listing failure does not change
source membership. Reordering or retiring prefixes rematerializes affected run IDs from retained
sources.

`eval_catalog_runs` and `eval_catalog_metrics` are seeded once from the legacy serving tables. New
revisions never use the legacy tables for serving or reconciliation.

`eval_catalog_state` contains one `singleton=true` row. `generation` increments in the transaction
that changes the selected catalog projection. `updated_at` is the commit time shown
by `/api/status`.

## Reconciliation

1. List `{prefix}/*/record.json` candidates every 600 seconds.
2. Read candidates absent from `eval_record_sources`.
3. For a source whose `next_verify_at` is due, fetch its version. Reread only after a version change or
   prior validation/stat error.
4. A successful first read schedules a deterministic first verification within 24 hours. Later
   successful checks schedule exactly 24 hours from the check.
5. A transient check or validation error schedules retry after 600 seconds and retains the last valid
   record.
6. A source is removed after two consecutive successful observations that its object is missing.
7. Recompute and replace only affected serving rows in the same transaction as source changes.
8. Do not prune serving rows that predate the inventory until all configured prefixes have one
   successful listing.

## Status API

`store` adds:

```text
record_count: int
catalog_generation: int | null
snapshot_updated_at: ISO-8601 string | null
```

`ingest` adds `revalidate_after_seconds`, null for the direct local scanner and `86400` for the
PostgreSQL reconciler.

## Configuration

- `EVALDASH_INGEST_INTERVAL`: discovery and error-retry cadence, default `600` seconds.
- `EVALDASH_REVALIDATE_AFTER`: successful known-object verification cadence, default `86400` seconds.
