# Storage Purge Plan

This document captures the current storage-purge workflow and the files produced
to support it.

## Goal

Protect explicitly approved artifacts, back them up within the same GCS region,
and then purge the remaining cold objects from the source buckets using storage
class as the proxy for "cold":

- keep the protect set from the normalized manifests
- treat non-`STANDARD` as "30+ day cold" because Autoclass has been enabled for
  over a year on the source buckets
- avoid cross-region copies
- avoid object reads; only list metadata/prefixes as needed

## Source Inputs

Human-curated manifests:

- `high-value.csv`
- `named.csv`

Derived files in this directory:

- `protect_manifest_expanded.csv`
- `protect_manifest_deduped.csv`
- `protect_manifest_classified.csv`
- `protect_manifest_issues.csv`
- `protect_manifest_sts_direct.csv`
- `sts_direct_regions/*.csv`
- `sts_via_listing_regions/*.csv`
- `sts_region_backup_plan.csv`

## Current State

### Normalized Protect Set

- `protect_manifest_deduped.csv` is the canonical normalized protect manifest.
- `protect_manifest_issues.csv` currently has no `error` rows.
- `protect_manifest_classified.csv` splits the protect rows into:
  - `sts_prefix_direct`
  - `sts_prefix_via_listing`

Current classified counts:

- `sts_prefix_direct`: 1777
- `sts_prefix_via_listing`: 1773
- `object_manifest_or_manual`: 0

### Regional STS Inputs

Direct-prefix regional CSVs:

- `sts_direct_regions/sts_direct_prefixes_eu-west4.csv` (`1701`)
- `sts_direct_regions/sts_direct_prefixes_us-central1.csv` (`28`)
- `sts_direct_regions/sts_direct_prefixes_us-central2.csv` (`24`)
- `sts_direct_regions/sts_direct_prefixes_us-east1.csv` (`8`)
- `sts_direct_regions/sts_direct_prefixes_us-east5.csv` (`8`)
- `sts_direct_regions/sts_direct_prefixes_us-west4.csv` (`8`)

Listing-based regional CSVs:

- `sts_via_listing_regions/sts_via_listing_eu-west4.csv` (`290`)
- `sts_via_listing_regions/sts_via_listing_us-central1.csv` (`296`)
- `sts_via_listing_regions/sts_via_listing_us-central2.csv` (`299`)
- `sts_via_listing_regions/sts_via_listing_us-east1.csv` (`296`)
- `sts_via_listing_regions/sts_via_listing_us-east5.csv` (`296`)
- `sts_via_listing_regions/sts_via_listing_us-west4.csv` (`296`)

Interpretation:

- `sts_direct_regions/*` can be fed directly into prefix-based same-region STS
  jobs.
- `sts_via_listing_regions/*` still require a single `gsutil ls -d` pass per
  `listing_prefix` to resolve exact concrete prefixes.

### Regional Backup Buckets

Temporary same-region backup buckets have already been created:

- `gs://marin-tmp-backup-eu-west4-purge-tmp-20260326` in `EUROPE-WEST4`
- `gs://marin-tmp-backup-us-central1-purge-tmp-20260326` in `US-CENTRAL1`
- `gs://marin-tmp-backup-us-central2-purge-tmp-20260326` in `US-CENTRAL2`
- `gs://marin-tmp-backup-us-east1-purge-tmp-20260326` in `US-EAST1`
- `gs://marin-tmp-backup-us-east5-purge-tmp-20260326` in `US-EAST5`
- `gs://marin-tmp-backup-us-west4-purge-tmp-20260326` in `US-WEST4`

Verified configuration for each:

- soft delete disabled (`retentionDurationSeconds = 0`)
- Autoclass disabled

## Full Rollout

### Phase 1: Resolve Listing-Based Families

For each regional `sts_via_listing_regions/*.csv` file:

1. Read each row's `listing_prefix`.
2. Run a metadata-only directory listing:
   - example: `gsutil ls -d gs://marin-eu-west4/tokenized/paloma/*`
3. Filter the returned directory names against the intended wildcard family
   described by:
   - `normalized_glob`
   - `concrete_prefix_hint`
4. Emit exact concrete prefixes.

Output of this phase should be a new per-region CSV of exact prefixes, similar
in shape to `sts_direct_regions/*.csv`.

Notes:

- This is the point where Paloma and the other wildcard families become
  concrete.
- The point of carrying both `listing_prefix` and `concrete_prefix_hint` is to
  do one bounded listing under the right parent and then filter locally.

### Phase 2: Backup the Protect Set

For each region:

1. Use the direct-prefix regional CSV.
2. Add the resolved-prefix CSV from Phase 1.
3. Create one same-region STS job per source bucket -> backup bucket.
4. Copy only the protected prefixes into the matching regional backup bucket.

Constraints:

- Keep copies same-region.
- Do not use STS manifests unless the rows are concrete object names. Prefix
  jobs are sufficient here because the current workflow resolves to prefixes.

### Phase 3: Protect the In-Place Keep Set

After backup succeeds:

1. Resolve the final keep set to exact object names.
2. Apply temporary holds to the in-place protected objects in the source
   buckets.

Reason:

- lifecycle delete cannot express "delete everything except these manifests"
- temporary holds give the exemption mechanism

### Phase 4: Freeze Storage Classes

On each source bucket:

1. Disable Autoclass.

This freezes the current storage-class snapshot so `STANDARD` vs non-`STANDARD`
becomes stable during the purge window.

### Phase 5: Purge Cold Unprotected Objects

On each source bucket:

1. Add a lifecycle `Delete` rule that targets non-`STANDARD` objects.
2. Wait for lifecycle to settle.
3. Remove the lifecycle rule.

Assumption:

- because protected objects are under temporary holds, the lifecycle rule can be
  broad without deleting them

Operational note:

- budget roughly 48 hours for lifecycle propagation/settling rather than
  assuming an immediate transition

### Phase 6: Return Source Buckets to Steady State

1. Re-enable Autoclass on the source buckets.
2. Optionally clear temporary holds later after confidence in the backup/purge
   result.

Expected consequence:

- when Autoclass is re-enabled, surviving objects move back to `STANDARD`
  initially; that is acceptable because only intended survivors remain

## Recommended Execution Order

1. Resolve `eu-west4` listing-based rows first.
2. Resolve the remaining US regional listing-based rows.
3. Launch same-region backup jobs for direct + resolved prefixes.
4. Validate a small restore sample from each backup bucket.
5. Apply temporary holds on source keep-set objects.
6. Disable Autoclass on source buckets.
7. Add lifecycle delete rule for non-`STANDARD`.
8. Wait for lifecycle to settle.
9. Remove lifecycle rule.
10. Re-enable Autoclass.
11. Clear holds later if desired.

## Scripts

- `generate_keep_globs.py`
  - crawl `experiments/` and infer high-value keep globs
- `normalize_protect_manifests.py`
  - normalize `high-value.csv` and `named.csv` into explicit protect globs
- `classify_protect_globs.py`
  - classify protect rows into direct vs listing-based backup strategies
- `extract_sts_direct_prefixes.py`
  - extract the `sts_prefix_direct` subset
- `emit_sts_region_inputs.py`
  - emit per-region direct-prefix CSVs, per-region listing-based CSVs, and the
    backup-bucket plan

## Warnings

- `sts_via_listing_regions/*.csv` are not ready for STS as-is.
- STS true manifest files require concrete object names, not wildcard prefixes.
- `listing_prefix` rows still need a bounded listing pass before they can be
  copied.
- Avoid bucket-wide or cross-region copy operations.
