# EvalDash PostgreSQL serving catalog

EvalDash will answer its first request from PostgreSQL in seconds instead of waiting for a complete
object scan and thousands of row rewrites. Object-store `record.json` files remain the durable
producer and recovery input. Reconciliation runs after startup and updates PostgreSQL only when the
set or version of source objects changes.

## Challenges

The four configured roots contain canonical and legacy duplicates, with prefix order defining the
winner. A selected-row-only cache loses the fallback record. Listing failure is also different from
confirmed deletion: an unavailable CoreWeave prefix must not remove its prior rows. Finally, known
paths are writable today, so discovery by path alone cannot bound staleness.

The dashboard has both SQL-backed endpoints and aggregate views over an in-memory list. Booting from
PostgreSQL therefore requires a committed full-record snapshot, not only flattened run columns.

## Costs and risks

- The source catalog stores a second copy of each record JSON. This is about 9 MB at the current
  3,000-run scale and buys deterministic duplicate fallback and last-valid retention.
- A rewritten object can remain stale for up to 24 hours plus one polling interval. New paths remain
  discoverable on the next 10-minute listing.
- Object deletion takes two successful observations. This deliberately favors serving a stale run
  for one extra interval over deleting it after a partial listing.
- The application database user applies migrations at startup. Migrations are additive and guarded
  by a PostgreSQL advisory lock; future destructive changes require expand/contract rollout.
- The old `eval_runs` tables remain during the rollout. This small temporary duplication prevents an
  overlapping old revision from writing the new commit-token projection.

## Design

`eval_catalog_runs` and `eval_catalog_metrics` become the canonical serving projection. `PgRecordStore` loads every
validated `eval_catalog_runs.record` plus the current catalog generation during construction, then serves
immediately. The existing in-memory aggregate code runs over that DB-loaded snapshot. Object work
starts as a Starlette lifespan task after startup.

`eval_record_sources` retains each object path, owning prefix, run ID, opaque backend version, full
validated record, next verification deadline, missing marker, and last error. Lower priority
duplicates remain present. `eval_record_prefixes` records configured priority, active membership,
health, and the last successful complete listing of each root. A failed listing updates only its
error state.

Every 10 minutes the reconciler lists immediate run directories. It reads all new candidates. Known
sources carry a staggered verification deadline: their first checks are spread across the first day,
then each receives one GCS metadata GET, S3 HEAD, or local version check per 24 hours. An unchanged
version advances its deadline. A changed version is read and validated with the exact version returned
by `ConditionalObject.read`. A record whose payload run ID does not match its parent directory is
invalid.

An invalid rewrite retains the source's last valid JSON and is retried on the next interval. A missing
object is marked on the first successful check and removed on the second. Prefix failure never advances
the missing marker.

Each successful prefix result applies in one DB transaction. Only changed run IDs are recomputed; the
winner is lowest prefix priority and then lexical path. Their serving rows and metrics are replaced in
bulk. The same transaction increments the singleton `eval_catalog_state.generation`. After commit,
the process loads and swaps the full snapshot only when that token changed. Normal cycles perform four
listings, the due metadata checks, and no serving-row rewrite.

Numbered modules in `infra/evaldash/src/migrations/` replace `metadata.create_all` as the evolution
mechanism. `0001_initial.py` creates or adopts the original unversioned tables. `0002_record_sources.py`
adds inventory and catalog state, creates isolated projection tables, and seeds them from `eval_runs`.
An old revision may keep writing the legacy tables during handoff without bypassing the catalog token.
The application applies pending migrations under an advisory lock before loading the first snapshot.
It fails fast when the database ledger contains a migration unknown to the binary.
On the initial catalog build, seeded records remain served;
existing rows are neither replaced nor pruned until every configured prefix has completed one
successful listing. New run IDs can still be added while one prefix is unavailable.

## Testing

SQLite-backed behavior tests run the real migrations and catalog transactions. They seed the old
schema to prove adoption and DB-only boot. A local object root plus fake clock exercises new discovery,
deadline-gated rewrites, canonical/legacy precedence, two-check deletion, invalid rewrites, and failed
prefixes. Rigging tests verify that metadata-only version checks use GCS generation and S3 ETag APIs.
The existing EvalDash API suite confirms local mode remains unchanged; the dashboard typecheck/build
checks the expanded status payload.

Production rollout should verify that container readiness is no longer gated on object scanning,
`/api/status` exposes the boot generation immediately, all expected runs are present, and steady-state
logs show only the daily due subset being checked.

The rollout uses a manual hard restart after the new revision is ready. The separate catalog
projection remains necessary because migration and revision overlap can occur before that restart;
it also makes rollback behavior explicit. Catalog writes take a row lock, prefix configuration is
durable, invalid first reads retain the directory run ID, snapshot swaps are atomic, and each process
polls the generation independently so concurrent revisions converge on the same committed view.

## Decisions

- PostgreSQL is canonical for serving; object records are canonical producer/recovery inputs.
- Use a persisted source catalog, not source metadata attached only to selected rows.
- Discover every 10 minutes and revalidate each object version daily on a staggered schedule.
- Require two successful missing observations before removing a source.
- Treat the catalog generation as the serving snapshot's commit token.
- Keep the new projection isolated from old-revision writes during rollout.
