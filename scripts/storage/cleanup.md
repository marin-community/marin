# Marin Storage Cleanup Plan

## Goal

Delete cold, unprotected objects from the `marin-*` GCS buckets to reduce storage costs.
We keep everything that's either actively used (STANDARD class) or explicitly protected.

## What gets deleted

Objects that meet **both** criteria:

1. **Storage class is non-STANDARD** (NEARLINE, COLDLINE, or ARCHIVE) — Autoclass
   moved them to a cold tier because they haven't been accessed recently.
2. **Not in the protect set** — the object's name doesn't match any rule in the
   `protect_rules` table (prefix matches or LIKE patterns from wildcard globs).

Objects in STANDARD class are never deleted, even if unprotected.  Autoclass promotes
recently-accessed objects back to STANDARD, so STANDARD is a reliable proxy for "in use."

## Safety net: soft-delete

Instead of copying protected data to backup buckets (the previous STS-based approach),
we use GCS's native **soft-delete** feature:

- Before deleting anything, enable soft-delete with a 3-day retention window.
- All deleted objects remain recoverable via `gcloud storage restore` for 3 days.
- After the safety window, disable soft-delete to finalize.

## Architecture: DuckDB state store

All state lives in a DuckDB database at `scripts/storage/purge/storage.duckdb`.
Objects are stored as sorted, ZSTD-compressed parquet segments under
`scripts/storage/purge/objects_parquet/` and exposed via a DuckDB `VIEW`.

### Tables

- **`storage_classes`** — pricing per GiB/month for US and EU regions
- **`protect_rules`** — protection rules: direct prefix matches (`pattern_type='prefix'`)
  and wildcard LIKE patterns (`pattern_type='like'`) converted from globs
- **`scanned_prefixes`** — tracks which top-level prefixes have been scanned (for resume)
- **`split_cache`** — cached sub-prefix listings for adaptive scan resume
- **`objects`** (VIEW) — every object in every bucket, backed by parquet segments

Protection checks and savings estimates are SQL queries against these tables.
Wildcard globs from the classified CSV are converted to SQL LIKE patterns at load time
(e.g. `checkpoints/dclm_1b*` → `checkpoints/dclm_1b%`), eliminating the need for
GCS listing calls to resolve them.

## Cost accounting

The estimate step queries the object catalog and groups by storage class.
Monthly savings are computed using per-class GCS list prices (stored in the
`storage_classes` table) with a 50% CUD discount:

| Class    | US ($/GiB/mo) | EU ($/GiB/mo) | After 50% discount |
|----------|---------------|---------------|---------------------|
| STANDARD | 0.020         | 0.023         | 0.010 / 0.0115      |
| NEARLINE | 0.010         | 0.013         | 0.005 / 0.0065      |
| COLDLINE | 0.004         | 0.006         | 0.002 / 0.003       |
| ARCHIVE  | 0.0012        | 0.0025        | 0.0006 / 0.00125    |

## Workflow steps

```
prep.load_protect_rules          Load protect globs + direct prefixes into DB
prep.scan_objects                Scan bucket objects into DuckDB (parquet-backed)
prep.estimate_savings            Estimate deletion savings via SQL queries
cleanup.enable_soft_delete       Enable 3-day soft-delete on each source bucket
cleanup.delete_cold_objects      Delete non-STANDARD unprotected objects (batch, DB-driven)
cleanup.wait_for_safety_window   Checkpoint: wait for soft-delete retention to pass
cleanup.disable_soft_delete      Turn off soft-delete (finalizes the purge) [optional]
```

### Running

```bash
# See current step status
uv run scripts/storage/purge.py plan

# Run all prep steps (read-only against buckets)
uv run scripts/storage/purge.py run --to prep.estimate_savings

# Scan objects with more workers
uv run scripts/storage/purge.py prep scan-objects --scan-workers 64

# Dry-run the deletion to see what would be removed
uv run scripts/storage/purge.py cleanup delete-cold-objects --dry-run

# Execute the full cleanup
uv run scripts/storage/purge.py run --from cleanup.enable_soft_delete

# Run a single step
uv run scripts/storage/purge.py run --only cleanup.enable_soft_delete

# Limit to one region
uv run scripts/storage/purge.py run --region us-central2
```

### Step details

**prep.load_protect_rules**
Reads both `protect_prefixes_classified.csv` and `protect_prefixes_direct.csv`.
Direct prefixes become `pattern_type='prefix'` rules. Wildcard globs (previously
requiring GCS listing to resolve) are converted to SQL LIKE patterns
(`pattern_type='like'`) — no GCS calls needed. Writes `cleanup_plan.csv` as the
manifest for downstream steps.

**prep.scan_objects**
Discovers top-level prefixes in each bucket, then fans out workers to list every
object. Objects are buffered, sorted by `(bucket, name)`, and flushed as
ZSTD-compressed parquet segments. Timestamps are stored as `TIMESTAMPTZ`.
Skips already-scanned prefixes (tracked in `scanned_prefixes`) unless `--force`
is given. Uses adaptive prefix splitting for large flat prefixes.

**prep.estimate_savings**
SQL queries against the `objects` view and `protect_rules` table. Joins with
`storage_classes` for pricing, uses `NOT EXISTS` to find unprotected cold objects,
and computes monthly savings with the 50% discount. Writes per-region JSON
estimates and a summary CSV.

**cleanup.enable_soft_delete**
Sets `--soft-delete-duration=259200s` (3 days) on each source bucket.
If soft-delete is already enabled with sufficient retention, this is a no-op.

**cleanup.delete_cold_objects**
Queries the DuckDB catalog for cold unprotected objects per top-level prefix,
then batch-deletes from GCS with workers for throughput. Writes a
`deletion_log_{region}.json` with counts and class breakdowns.

**cleanup.wait_for_safety_window**
Records a settle deadline (default 72 hours from deletion).  Refuses to proceed
until the deadline passes.  This is a checkpoint, not a sleep.

**cleanup.disable_soft_delete** *(optional)*
Clears soft-delete on each bucket, permanently removing soft-deleted objects.
Only run after confirming no important data was accidentally deleted.

## Rollback

During the soft-delete safety window (3 days by default):

```bash
# Restore a specific object
gcloud storage restore gs://marin-us-central2/path/to/object

# Restore an entire prefix
gcloud storage restore gs://marin-us-central2/prefix/** --all-versions
```

After `cleanup.disable_soft_delete` runs, deletions are permanent.
