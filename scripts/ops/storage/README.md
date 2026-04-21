# Storage Purge Workflow

Tools for auditing, triaging, and deleting cold objects from Marin's GCS
buckets to reduce storage costs.

## Overview

The workflow has four phases:

```
prep  ──>  server (human triage)  ──>  compute (manifest)  ──>  cleanup (delete)
```

1. **Prep** scans every object in the `marin-*` buckets into a local DuckDB
   database and computes per-directory cost summaries.
2. **Server** runs a dashboard ("delete-o-tron") where humans browse the
   directory tree, add protect rules (keep) and delete rules, and see cost
   impact in real time.
3. **Compute** evaluates the curated rules against the object catalog and
   outputs a collapsed CSV manifest of directory prefixes to delete. This is
   the human inspection point — review before proceeding.
4. **Cleanup** reads the manifest and deletes all objects under those
   prefixes, with soft-delete as a safety net.

## Architecture

All state lives in `scripts/ops/storage/purge/`:

```
purge/
  storage.duckdb              # DuckDB catalog (schema, markers, rules, scan state)
  objects_parquet/             # sorted ZSTD parquet segments of every scanned object
  dir_summary_parquet/         # per-directory aggregates by storage class
  protect_rules.json           # source of truth for protect rules
  delete_rules.json            # source of truth for delete rules
  deletion_manifest.csv        # computed deletion set (output of compute step)
  logs/                        # per-run log files
```

Objects are stored as parquet segments and exposed via DuckDB views. Rules
are persisted as JSON files and loaded into DuckDB tables on startup.

### Key tables

| Table | Purpose |
|-------|---------|
| `objects` (view) | Every object across all buckets, backed by parquet |
| `dir_summary` (view) | Per-directory aggregates with per-storage-class counts/bytes |
| `protect_rules` | Glob patterns marking directories to keep |
| `delete_rules` | Glob patterns marking directories to delete |
| `rule_costs` | Materialized cost of each protect rule |
| `delete_rule_costs` | Materialized cost of each delete rule (excluding protected dirs) |
| `scanned_prefixes` | Tracks which prefixes have been scanned (for resume) |
| `step_markers` | Tracks which workflow steps have completed |

## Phase 1: Prep

```bash
# See step status
uv run scripts/ops/storage/prep.py plan

# Run all prep steps
uv run scripts/ops/storage/prep.py run

# Run a single step
uv run scripts/ops/storage/prep.py prep scan-objects --scan-workers 64

# Force re-scan (ignore cached markers)
uv run scripts/ops/storage/prep.py run --force
```

### Steps

**`prep.scan_objects`** — Lists every object in each of the 6 `marin-*`
buckets using an adaptive two-level scan: tries a flat listing first, and if a
prefix exceeds 10k objects, splits via delimiter listing and recursively
enqueues children (up to depth 2). Results stream into an `ObjectBuffer` and
flush as sorted parquet segments. Skips already-scanned prefixes unless
`--force` is given.

**`prep.materialize_dir_summary`** — Groups all objects by parent directory
and computes per-storage-class aggregates (count + bytes for STANDARD,
NEARLINE, COLDLINE, ARCHIVE). Small deep directories (< 10 GB, deeper than
depth 2) are collapsed into their depth-2 ancestor to keep the row count
manageable.

**`prep.materialize_rule_costs`** — Joins protect rules against dir_summary
to compute the storage cost of each rule, accounting for region-specific GCS
pricing and a 30% CUD discount.

## Phase 2: Server (delete-o-tron)

```bash
# Local development
uv run scripts/ops/storage/server.py --dev

# Production (deployed via Docker)
uv run scripts/ops/storage/deploy.py deploy
```

The dashboard is a FastAPI app serving a Vue frontend from
`scripts/ops/storage/dashboard/`. It provides:

- **Overview** (`/api/overview`) — Per-region storage breakdown by class with
  monthly cost estimates.
- **Explore** (`/api/explore/unified`) — Hierarchical directory browser with
  inline keep/delete status computed at query time by joining against
  protect and delete rules. No materialized status table needed.
- **Protect rules** (`/api/rules`) — CRUD for protect rules (glob patterns
  that mark directories as "keep"). Rules can target a specific bucket or `*`
  for all buckets.
- **Delete rules** (`/api/delete-rules`) — CRUD for delete rules (glob
  patterns that mark directories for deletion). Can optionally target a
  specific storage class.
- **Savings estimate** (`/api/savings`) — Computes deletable bytes as
  total minus protected, with monthly cost savings.
- **Simulate** (`/api/rules/simulate?exclude=1,2`) — "What if we unprotect
  these rules?" Upper-bound estimate without re-scanning.
- **Pattern estimate** (`/api/delete-patterns/estimate`) — Cost estimate for
  candidate delete patterns before committing them as rules.

Rules are persisted to `protect_rules.json` / `delete_rules.json` on every
mutation and synced to GCS every 10 minutes (with hourly archives) when
running with `GCS_DATA_PREFIX` set.

### Deployment

`deploy.py` builds a Docker image, pushes to GCR, and creates/updates a GCP
VM running the dashboard. It syncs the parquet data and rule files to a GCS
staging prefix that the container downloads on cold start.

## Phase 3: Compute

```bash
# See step status
uv run scripts/ops/storage/compute.py plan

# Compute the deletion manifest
uv run scripts/ops/storage/compute.py run

# Force recompute (e.g. after changing rules)
uv run scripts/ops/storage/compute.py run --force
```

Evaluates delete rules against protect rules and the `dir_summary` table,
then collapses the result into a compact CSV manifest.

### Collapsing

The key optimization: if all children of a parent directory are marked for
deletion, the manifest emits the parent instead of listing every child. This
is applied bottom-up and repeated until stable, reducing thousands of leaf
directories to a handful of top-level prefixes.

### Output format

`purge/deletion_manifest.csv`:

```csv
bucket,prefix,object_count,total_bytes,bytes_human,storage_class_breakdown,matched_rule
marin-us-central2,checkpoints/old_experiment/,4521,98234567890,91.49 GiB,ARCHIVE:4521,checkpoints/old_experiment%
marin-eu-west4,raw/,12000,500000000000,465.66 GiB,COLDLINE:8000;ARCHIVE:4000,raw/%
```

Inspect with standard tools:

```bash
# Quick look
column -t -s, scripts/ops/storage/purge/deletion_manifest.csv | head -20

# Sort by size
sort -t, -k4 -rn scripts/ops/storage/purge/deletion_manifest.csv | head

# Filter to one bucket
grep marin-eu-west4 scripts/ops/storage/purge/deletion_manifest.csv
```

### Steps

**`compute.materialize_deletion_set`** — For each bucket, queries
`dir_summary` with keep/delete status by joining against `protect_rules` and
`delete_rules`. Collapses all-delete subtrees bottom-up. Writes the CSV
manifest and stores a fingerprint in the step marker for downstream
validation.

## Phase 4: Cleanup

```bash
# See step status
uv run scripts/ops/storage/cleanup.py plan

# Dry run (no mutations)
uv run scripts/ops/storage/cleanup.py run --dry-run

# Execute cleanup
uv run scripts/ops/storage/cleanup.py run

# Run a single step
uv run scripts/ops/storage/cleanup.py run --only cleanup.delete_cold_objects
```

### Steps

**`cleanup.enable_soft_delete`** — Enables 3-day soft-delete retention on
each source bucket. All subsequently deleted objects remain recoverable via
`gcloud storage restore` for 3 days.

**`cleanup.delete_cold_objects`** — Reads the deletion manifest from the
compute step and deletes all objects under each listed prefix. Validates the
manifest fingerprint before proceeding (raises an error if the CSV has been
modified since the compute step). Fans out over manifest prefixes with
concurrent workers.

**`cleanup.wait_for_safety_window`** — Records a settle deadline (default
72 hours). Refuses to proceed to the next step until the deadline passes.
This is a checkpoint, not a sleep — re-run the command later.

**`cleanup.disable_soft_delete`** *(optional)* — Clears soft-delete on each
bucket, permanently removing the soft-deleted objects. Only run after
confirming no important data was accidentally deleted.

### Rollback

During the soft-delete safety window:

```bash
# Restore a specific object
gcloud storage restore gs://marin-us-central2/path/to/object

# Restore an entire prefix
gcloud storage restore gs://marin-us-central2/prefix/** --all-versions
```

After `cleanup.disable_soft_delete`, deletions are permanent.

## Files

| File | Role |
|------|------|
| `db.py` | Shared DuckDB catalog: schemas, init, migrations, queries, buffers |
| `scan.py` | Adaptive parallel GCS object scanner with Rich progress display |
| `prep.py` | CLI for the prep workflow (scan + materialize) |
| `server.py` | FastAPI dashboard server + JSON API |
| `compute.py` | CLI for the compute workflow (materialize deletion manifest) |
| `cleanup.py` | CLI for the cleanup workflow (soft-delete + batch delete from manifest) |
| `deploy.py` | Docker build + GCP VM deployment for the dashboard |
| `rules.py` | Rule consolidation helpers (wildcard promotion) |
| `Dockerfile` | Container image for the dashboard server |
| `dashboard/` | Vue frontend (plain JS + HTML, no build step) |

## Cost Model

Monthly cost estimates use GCS list prices with a configurable CUD discount
(currently 30%):

| Class | US ($/GiB/mo) | EU ($/GiB/mo) |
|----------|---------------|---------------|
| STANDARD | 0.020 | 0.023 |
| NEARLINE | 0.010 | 0.013 |
| COLDLINE | 0.004 | 0.006 |
| ARCHIVE | 0.0012 | 0.0025 |

Storage class serves as a proxy for "cold": Autoclass has been enabled on all
buckets for over a year, so non-STANDARD objects are those that haven't been
accessed recently.
