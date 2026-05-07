# Option A: Ship the un-resolved config and materialize on the worker

Investigates fixing PR #5279's cross-region rebake bug, where
`prepare_lm_train` (`experiments/defaults.py:644-678`) calls
`compute_output_path` (`lib/marin/src/marin/execution/executor.py:1804`) and
`bake_output_path` (`lib/marin/src/marin/training/training.py:107`) on the
*submitter*. `compute_output_path` resolves `marin_prefix()` from the
submitter's GCE metadata (`lib/rigging/src/rigging/filesystem.py:144-150`),
so `gs://marin-{submitter-region}` gets baked into
`CheckpointerConfig.base_path` and `hf_save_path`. If Iris reschedules the
worker into another region, it writes back to the original-region bucket,
re-introducing the very cross-region traffic this PR is trying to remove.

## Design

Defer every region-dependent substitution until the worker runs. The
submitter builds a `train_config` that still contains `OutputName(...)`
markers and an unbaked checkpointer. The Iris job ships *that* dataclass
(pickled by Fray, same path as today via `Entrypoint.from_callable` —
`experiments/defaults.py:600-606`). On the worker, `_run_training_on_worker`
expands placeholders against the worker's own `marin_prefix()`.

Wire format: keep the dataclasses. Today `_run_training_on_worker` already
ships placeholder-bearing configs and lets `materialize` substitute
`InputName`/`ExecutorStep` references on the worker
(`experiments/defaults.py:613-620`); extending that to also resolve
`OutputName` and to re-run `bake_output_path` is the smallest delta. A
JSON/templated-string format would force draccus serialization for every
trainer config and give us nothing the dataclass path doesn't already.

The hash is region-agnostic *by design*: `compute_version` builds
`version = {name, config, dependencies}` (executor.py:1535-1539), excluding
`self.prefix`. The hash is joined with the prefix only at
**executor.py:1548** (`os.path.join(self.prefix, step.name + "-" +
hashed_version)`), and `_dep_version` (executor.py:1602-1614) explicitly
documents that prefix is omitted "so the same logical pipeline rehashed
under a different `MARIN_PREFIX` would [not] produce a different identity."
So the submitter and the worker, given the same `train_config`, will
compute the same `name-hash` suffix; only the prefix differs.

## What changes

1. `prepare_lm_train` stops calling `compute_output_path`,
   `resolve_local_placeholders`, `bake_output_path`, `impute_run_id`, and
   `check_train_config_paths` on the submitter. It returns the unbaked
   config plus the `job_name`.
2. `_run_training_on_worker` (defaults.py:613) becomes the single
   resolution site: call `compute_output_path(job_name, train_config)` —
   which on the worker reads the worker's `marin_prefix()` — then
   `resolve_local_placeholders`, `bake_output_path`, `impute_run_id`,
   `check_train_config_paths`, and finally today's `materialize` call for
   upstream `InputName`s.
3. `job_name` must be passed alongside the config (already a parameter on
   `_submit_train_job`, defaults.py:577); thread it into
   `_run_training_on_worker`'s args list.
4. Drop the `output_path` return value from `prepare_lm_train`'s tuple
   (defaults.py:678) — it cannot be known at submit time. Audit callers;
   this is a breaking change to that API but is the whole point.

## Compatibility hazards

- **`override_output_path`**: takes precedence over the hash path
  (executor.py:1551-1560). It can legitimately be a relative path that
  `_make_prefix_absolute_path` joins onto the submitter's prefix
  (executor.py:1792-1795). Either keep it relative (worker absolutizes) or
  forbid absolute, region-qualified overrides at submit time.
- **Mirror/hardcoded `InputName`s**: `InputName.hardcoded(path)` is
  resolved against `prefix` (executor.py:1238-1239). If a hardcoded path
  is logically a us-central1 bucket, the worker in us-east5 will rewrite
  it to a us-east5 path that does not exist. Hardcoded paths are already
  the wrong abstraction for cross-region; flag and require explicit
  `gs://...` URIs for genuinely shared assets.
- **Run-id imputation from output path** (`training.py:164-168`) becomes
  worker-local. Two workers running the same job in different regions
  would derive the same run-id (the hash is region-agnostic), so W&B
  resumes still match — but this depends on the no-prefix-in-hash
  invariant holding; add a regression test.
- **Logging**: today the submitter logs the resolved `output_path`. After
  this change it can log only `job_name` + hash-deferred-to-worker. Mild
  UX regression; acceptable.
- **Dry-run / `executor.run` path**: the local-execution `Executor` still
  resolves prefix on the submitter. That's fine — single-region — but the
  rebake codepath is now exclusively on remote workers, so any new caller
  that wants a path before submission must opt into the submitter
  resolution explicitly.

## Verdict

**Recommended.** The hash is already region-agnostic by construction, so
option A is mechanically clean: move four function calls from submitter to
worker, thread `job_name` through one extra arg. The smallest viable
change is even smaller — keep `compute_output_path` as-is and have the
worker re-call it; the only required structural change is *not* baking
checkpointer paths on the submitter. Hazards (`override_output_path`,
hardcoded `InputName`s) are real but local and already smell wrong.

One sentence: ship the unbaked config and resolve `output_path` +
checkpointer paths on the worker — the version hash already excludes
`prefix`, so submitter and worker agree on the suffix and only the
regional bucket differs.
