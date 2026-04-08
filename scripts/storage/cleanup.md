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
scan      — Scan bucket objects into parquet segments
summarize — Materialize per-directory summaries from objects
compute   — Evaluate delete/protect rules, collapse, write CSV manifest
delete    — Delete objects from manifest
```

### Running

```bash
# See current step status
uv run scripts/storage/cleanup.py plan

# Run all prep steps (read-only against buckets)
uv run scripts/storage/cleanup.py run --to compute

# Scan objects with more workers
uv run scripts/storage/cleanup.py scan --workers 64

# Dry-run the deletion to see what would be removed
uv run scripts/storage/cleanup.py delete --dry-run

# Execute the deletion
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

**delete**
Reads `deletion_manifest.csv`, resolves individual object names from DuckDB,
and batch-deletes from GCS using a multi-threaded worker pool with live progress
display.
