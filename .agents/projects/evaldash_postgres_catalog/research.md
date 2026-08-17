# Background Research Brief

- Effort: low
- Stop rule: stopped when code, production measurements, and repository migration precedents converged
- Date: 2026-08-16

## Question

How can EvalDash start quickly from PostgreSQL while detecting new, rewritten, and removed object-store
records without making an object-store outage destructive?

## Current Marin context

The current process lists four roots and keeps a process-local path cache in
`lib/marin/src/marin/evaluation/records.py`. `PgRecordStore.refresh` then rewrites every selected run
and its metrics in a separate transaction. Aggregate views still depend on an in-memory record list,
so a restart waits for object discovery before those views are complete.

A read-only production measurement on 2026-08-16 found 2,970 `eval_runs`, 18,851 metric rows, and
9.35 MB of record JSON. Loading and validating every PostgreSQL record took 3.54 seconds. The warm
object listing took about five seconds, while the complete warm ingest took 3m34s; the per-record DB
rewrite is the dominant avoidable work.

## Internal prior work

- [PR #7752](https://github.com/marin-community/marin/pull/7752) introduced the PostgreSQL query
  index while retaining object records as the producer format.
- [PR #7975](https://github.com/marin-community/marin/pull/7975) added prefix invalidation and a
  process-local path cache. That cache cannot bootstrap a new process or detect a rewrite at a known
  path.
- `rigging.filesystem.conditional_object` already supplies provider-correct opaque versions and exact
  reads: GCS generations, S3 ETags, and local content hashes.
- Echo and Iris use numbered migration modules plus a durable migration ledger. EvalDash's existing
  `metadata.create_all` can create tables but cannot evolve an existing one.
- The Cloud Run component fixes EvalDash at one warm instance. Migration locking still matters during
  revision overlap and leaves scale-out safe.

## Prior-art shape

This is a materialized catalog, not a second object store. The source inventory retains every
candidate, including lower-priority duplicates; isolated `eval_catalog_runs` is the selected serving projection. A
single catalog generation plays the role of a commit sequence number. No new lakehouse, queue, or
sidecar catalog is needed at this scale.

## Negative leads

- Putting source version columns only on `eval_runs` loses the lower-priority copy needed when a
  canonical duplicate disappears.
- Rewriting all rows in one transaction would improve atomicity but retain the measured startup and
  steady-state cost.
- Treating one missing listing as deletion makes an eventually inconsistent or partial listing
  irreversible.
- Polling every known object every 10 minutes costs about 432,000 metadata requests/day. One daily
  check per object is about 3,000/day and bounds undetected rewrites without dominating discovery.

## Evidence map

### Claim: PostgreSQL can serve the initial dashboard snapshot

- Support: the production JSON payload is under 10 MB and loaded plus validated in 3.54 seconds.
- Caveat: all aggregate APIs must hydrate from that same committed snapshot.
- Confidence: high.
- Action: seed and load `eval_catalog_runs.record` at store construction, before the background task starts.

### Claim: a persisted source inventory is necessary for correct reconciliation

- Support: prefix precedence and lower-priority duplicates are current behavior; a selected-row-only
  schema cannot promote the fallback without another read.
- Caveat: it duplicates roughly 9 MB of JSON at current scale.
- Confidence: high.
- Action: retain path, owning prefix, version, validation state, and record JSON per source; keep
  precedence once per prefix.

### Claim: a catalog generation is a useful commit token

- Support: selected runs and metrics already form a transactional projection; incrementing one row in
  the same transaction identifies the complete result.
- Caveat: the token does not replace object-store versions or producer transactions.
- Confidence: high.
- Action: expose the generation and snapshot commit time on `/api/status`.

## Source ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| EvalDash server | Marin code | `infra/evaldash/src/server.py` | startup and rewrite behavior | high | current implementation |
| Eval record scanner | Marin code | `lib/marin/src/marin/evaluation/records.py` | listing and process cache | high | current implementation |
| Conditional objects | Marin code | `lib/rigging/src/rigging/filesystem/conditional_object.py` | portable object versions | high | already used by FineStore |
| Echo migrations | Marin code | `infra/echo/migrate.py` | numbered migration precedent | high | PostgreSQL |
| Production DB probe | live read-only measurement | 2026-08-16 session result | boot cost and scale | high | 2,970 rows at measurement time |

## Handoff

Implement the catalog and reconciliation contract in [spec.md](spec.md). Verify DB-only boot, new
discovery, due rewrite detection, duplicate fallback, two-check deletion, invalid rewrite retention,
and failed-prefix retention with a fake clock and a real local SQL database.
