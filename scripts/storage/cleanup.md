# Marin Storage Cleanup Plan

## Goal

Delete cold, unprotected objects from the `marin-*` GCS buckets to reduce storage costs.
We keep everything that's either actively used (STANDARD class) or explicitly protected.

## What gets deleted

Objects that meet **both** criteria:

1. **Storage class is non-STANDARD** (NEARLINE, COLDLINE, or ARCHIVE) — Autoclass
   moved them to a cold tier because they haven't been accessed recently.
2. **Not in the protect set** — the object's prefix doesn't match any entry in the
   curated protect-prefix lists (`protect/protect_prefixes_*.csv`).

Objects in STANDARD class are never deleted, even if unprotected.  Autoclass promotes
recently-accessed objects back to STANDARD, so STANDARD is a reliable proxy for "in use."

## Safety net: soft-delete

Instead of copying protected data to backup buckets (the previous STS-based approach),
we use GCS's native **soft-delete** feature:

- Before deleting anything, enable soft-delete with a 3-day retention window.
- All deleted objects remain recoverable via `gcloud storage restore` for 3 days.
- After the safety window, disable soft-delete to finalize.

This is cheaper (no duplicate storage for the protect set), simpler (no STS jobs, temp
buckets, or temp holds), and uses GCS's own undo mechanism for recovery.

## Cost accounting

The estimate step scans every object in each bucket and groups by storage class.
Monthly savings are computed using per-class GCS list prices with a 50% CUD discount:

| Class    | US ($/GiB/mo) | EU ($/GiB/mo) | After 50% discount |
|----------|---------------|---------------|---------------------|
| STANDARD | 0.020         | 0.023         | 0.010 / 0.0115      |
| NEARLINE | 0.010         | 0.013         | 0.005 / 0.0065      |
| COLDLINE | 0.004         | 0.006         | 0.002 / 0.003       |
| ARCHIVE  | 0.0012        | 0.0025        | 0.0006 / 0.00125    |

The estimate output shows bytes and object counts broken down by class, so you can
see exactly where the savings come from.

## Workflow steps

```
prep.resolve_listing_prefixes    Resolve wildcard protect prefixes → concrete lists
prep.build_protect_inputs        Merge direct + resolved prefixes into per-region protect set
prep.estimate_deletion_savings   Scan buckets, classify objects, compute savings by class
cleanup.enable_soft_delete       Enable 3-day soft-delete on each source bucket
cleanup.delete_cold_objects      Delete non-STANDARD unprotected objects (batch)
cleanup.wait_for_safety_window   Checkpoint: wait for soft-delete retention to pass
cleanup.disable_soft_delete      Turn off soft-delete (finalizes the purge) [optional]
```

### Running

```bash
# See current step status
uv run scripts/storage/purge.py plan

# Run all prep steps (read-only against buckets)
uv run scripts/storage/purge.py run --to prep.estimate_deletion_savings

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

**prep.resolve_listing_prefixes**
Reads `protect/protect_prefixes_classified.csv`.  For entries classified as
`sts_prefix_via_listing`, lists one level of child prefixes under the parent
and matches them against the wildcard glob to produce concrete prefix URLs.
Caches listings in SQLite for fast reruns.

**prep.build_protect_inputs**
Merges direct prefixes (from `protect_prefixes_direct.csv`) with the resolved
wildcard prefixes into a single `protect_prefixes_{region}.csv` per region and
writes `cleanup_plan.csv` as the manifest for downstream steps.

**prep.estimate_deletion_savings**
Scans every object in each bucket, classifies it as protected or deletable by
storage class, and writes per-region JSON estimates plus a summary CSV.
This is the most expensive prep step (full bucket listing) but is read-only.

**cleanup.enable_soft_delete**
Sets `--soft-delete-duration=259200s` (3 days) on each source bucket.
If soft-delete is already enabled with sufficient retention, this is a no-op.

**cleanup.delete_cold_objects**
Lists every object, skips STANDARD-class and protected objects, and batch-deletes
the rest.  Writes a `deletion_log_{region}.json` with counts and class breakdowns.

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

## Differences from previous approach (STS + lifecycle)

| Aspect | Old (STS + lifecycle) | New (soft-delete) |
|--------|----------------------|-------------------|
| Safety mechanism | Copy protect set to temp buckets via STS | GCS native soft-delete |
| What gets deleted | Non-STANDARD objects (via lifecycle rule) | Non-STANDARD unprotected objects (explicit delete) |
| Autoclass impact | Must disable/re-enable (resets learned tiering) | No disruption |
| Extra storage cost | Full copy of protect set | Soft-deleted objects retained 3 days |
| Recovery | Manual restore from backup bucket | `gcloud storage restore` |
| Complexity | ~2300 lines, 13 steps, STS + holds + lifecycle | ~900 lines, 7 steps |
| Cleanup after | Delete temp buckets | Disable soft-delete |
