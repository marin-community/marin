# Option C — Region-Agnostic Output Path

Investigation in support of dlwh's review on PR #5279
(`lib/marin/src/marin/execution/executor.py:1278`, "doesn't OutputName depend on region?").

## Design (C1 vs C2)

**C1 — single dual/multi-region bucket.** Replace the `marin-{region}` convention
with one `gs://marin/...` (`us`/`eu` multi-region) bucket. `marin_prefix()`
(`lib/rigging/src/rigging/filesystem.py:136`) returns the same string in every
region, so `compute_output_path` produces a path that is valid wherever the job
reschedules. Region is no longer encoded. The project has already voted against
this: `docs/tutorials/storage-bucket.md:25-26` explicitly tells users to *avoid*
multi-region buckets ("higher costs and more complex performance characteristics").
The TTL-scratch lifecycle scheme (`storage-bucket.md:70-86`, `ALLOWED_TTL_DAYS`
in `filesystem.py:79`) and `infra/configure_buckets.py` are written per-region.
The checkpoint write path (~10s of GB per save × every region) would also pay
multi-region replication egress on every checkpoint — exactly the cost
`TransferBudget` (`filesystem.py:458`) was built to police.

**C2 — region-agnostic logical scheme.** Keep per-region buckets. Make
`marin_prefix()` return a logical prefix like `marin://`; let an fsspec layer
translate `marin://experiments/foo` to `gs://marin-{local_region}/experiments/foo`
at I/O time. The existing `MirrorFileSystem` (`filesystem.py:804`, registered
at `:1046`) already does half of this: `_local_url` writes go to `marin_prefix()`,
reads scan all regions and copy on miss (`_find_in_remote_prefixes` at `:877`,
`_resolve_path` at `:915`). But it writes-then-mirrors-on-read; training
actually wants writes in the local region only and reads to prefer local with
*lazy* fall-through (no copy if a freshly-rescheduled job hasn't checkpointed yet).

## What changes

C1: rename buckets / configure replication; remove
`_REGION_TO_MARIN_BUCKET_OVERRIDES` and `REGION_TO_DATA_BUCKET`
(`filesystem.py:54,61`); rewrite `marin_temp_bucket` (`filesystem.py:191`);
delete `CrossRegionGuardedFS` (`filesystem.py:632`) and `check_path_in_region`
(`filesystem.py:272`). Migrate ~50 `gs://marin-us-central2/...` doc and
data-browser links in `docs/reports/index.md` etc.

C2: extend `MirrorFileSystem` with a "write-local, read-local-first,
fall-through-to-mirror" mode; have `marin_prefix()` return `marin://` on
workers; teach `bake_output_path`
(`lib/marin/src/marin/training/training.py:107`), `compute_output_path`
(`executor.py:1804`), `temporary_checkpoint_base_path` (`training.py:96`),
and `marin_temp_bucket` to keep the path logical. `_make_prefix_absolute_path`
in `instantiate_config` (`executor.py:1239`) must not collapse `marin://`.

## Compatibility hazards

- **LATEST-pointer race / R-M-W on checkpoints.** Levanter's
  `discover_latest_checkpoint` (`lib/levanter/src/levanter/checkpoint.py:975`)
  globs the checkpoint root and picks the max by `metadata.json` timestamp.
  Under C1 multi-region replication, region A writes step 1000, the resumed
  worker in B writes step 1100, replication is eventual, and the reader can
  see two "latest" candidates with stale metadata. The temporary-checkpoint
  deletion path (`checkpoint.py:548-583`) reads metadata, then deletes — under
  replication lag this can drop a *newer* checkpoint that hasn't replicated
  back. C2 dodges this by keeping writes single-region.
- **Cross-region guard becomes a lie.** `check_train_config_paths` (called from
  `experiments/defaults.py:676`) enforces same-region. Under C1 the invariant
  is meaningless; under C2 the check must run *after* `marin://` resolution.
- **Versioning hash includes paths.** `compute_output_path` hashes the config;
  any string-level path rewrite (C1 bucket rename) bumps every version,
  forcing re-runs of all upstream data steps. C2 is hash-stable.
- **fsspec breadth.** Tensorstore (Levanter array I/O) bypasses fsspec —
  `record_transfer` (`filesystem.py:605`) exists for exactly this reason.
  C2 needs every tensorstore call site to honor `marin://`. C1 inherits
  fsspec/tensorstore for free.
- **External links.** `docs/reports/index.md` and data-browser URLs hard-code
  `gs://marin-us-central2/...`. C1 breaks all of them; C2 leaves them alone
  (logical only on workers; published artifacts stay fully-qualified).

## Verdict

**C1: not recommended** — contradicts existing project guidance, replicates
checkpoint bytes you don't want replicated, and exposes the LATEST-pointer
race under preemption (the very scenario this PR is fixing).
**C2: needs more investigation** — `MirrorFileSystem`'s shape is close but
not identical (it copies on read; we want lazy fallback only), and
tensorstore needs a parallel shim. The cleaner answer for PR #5279 is the
third option already raised in the thread: defer `OutputName` resolution to
the worker so the path is computed under the worker's `marin_prefix()` after
preemption.

**Summary: C1 fights project conventions and risks checkpoint corruption
under replication lag; C2 is plausible only if `MirrorFileSystem` grows a
read-through-fallback mode and tensorstore gets a matching shim, so deferring
`OutputName` resolution to the worker remains the path of least resistance.**
