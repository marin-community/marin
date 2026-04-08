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

## Safety net: STS backup

Before deleting anything, we copy the exact delete set to same-region temporary
backup buckets using Google Cloud Storage Transfer Service (STS):

- The `deletion_manifest.csv` defines the prefixes to delete.
- STS jobs copy those prefixes to per-region temp buckets (same region, no egress cost).
- Only after all STS jobs complete does the delete step proceed.
- Recovery: copy objects back from the temp bucket if needed.
- After confirming the purge is correct, delete the temp buckets manually.

### Temp backup buckets

| Source | Backup |
|--------|--------|
| `marin-eu-west4` | `marin-tmp-backup-eu-west4-purge-tmp-20260326` |
| `marin-us-central1` | `marin-tmp-backup-us-central1-purge-tmp-20260326` |
| `marin-us-central2` | `marin-tmp-backup-us-central2-purge-tmp-20260326` |
| `marin-us-east1` | `marin-tmp-backup-us-east1-purge-tmp-20260326` |
| `marin-us-east5` | `marin-tmp-backup-us-east5-purge-tmp-20260326` |
| `marin-us-west4` | `marin-tmp-backup-us-west4-purge-tmp-20260326` |

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

## Workflow steps

```
scan         — Scan bucket objects into parquet segments
summarize    — Materialize per-directory summaries from objects
compute      — Evaluate delete/protect rules, collapse, write CSV manifest
backup-prep  — Plan STS backup jobs and write to disk for inspection
backup-run   — Create STS jobs from the backup plan
delete       — Delete objects from manifest (requires backup completion)
```

### Running

```bash
# See current step status
uv run scripts/storage/cleanup.py plan

# Run all prep steps (read-only against buckets)
uv run scripts/storage/cleanup.py run --to compute

# Scan objects with more workers
uv run scripts/storage/cleanup.py scan --workers 64

# Plan backup jobs (writes plan to disk for inspection)
uv run scripts/storage/cleanup.py backup-prep

# Inspect the plan
cat scripts/storage/purge/backup/backup_plan.json

# Dry-run to see what STS jobs would be created
uv run scripts/storage/cleanup.py backup-run --dry-run

# Create the STS jobs
uv run scripts/storage/cleanup.py backup-run

# Check backup job progress
uv run scripts/storage/cleanup.py backup-status

# Dry-run the deletion to see what would be removed
uv run scripts/storage/cleanup.py delete --dry-run

# Execute the deletion (requires backup to be complete)
uv run scripts/storage/cleanup.py delete --workers 64

# Run the full workflow
uv run scripts/storage/cleanup.py run
```

### Step details

**scan**
Discovers top-level prefixes in each bucket, then fans out workers to list every
object. Objects are buffered, sorted by `(bucket, name)`, and flushed as
ZSTD-compressed parquet segments.

**summarize**
Aggregates objects into per-directory rows with per-storage-class columns,
collapsing small deep directories into their depth-2 ancestors.

**compute**
SQL queries against the `dir_summary` view, `protect_rules`, and `delete_rules`.
Produces `deletion_manifest.csv` with collapsed directory prefixes marked for
deletion, plus a SHA-256 sidecar for integrity.

**backup-prep**
Reads `deletion_manifest.csv` and plans STS jobs to copy each bucket's delete
prefixes to the corresponding temp backup bucket.  STS jobs are batched at 1000
prefixes per job (the STS API limit).  Writes the full plan (including per-job
prefix lists) to `scripts/storage/purge/backup/backup_plan.json` for inspection
before execution.

**backup-run**
Reads the plan from `backup_plan.json` and creates the actual STS transfer jobs.
State is tracked in `scripts/storage/purge/backup/backup_state.json`.

**delete**
Reads `deletion_manifest.csv`, resolves individual object names from DuckDB,
and batch-deletes from GCS.  Refuses to run unless all STS backup jobs have
completed successfully.

## Rollback

While the temp backup buckets exist:

```bash
# Restore a prefix from the backup
gsutil -m cp -r gs://marin-tmp-backup-us-central2-purge-tmp-20260326/path/prefix/ gs://marin-us-central2/path/prefix/
```

After you delete the temp buckets, recovery is not possible.
