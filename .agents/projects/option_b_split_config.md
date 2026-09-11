# Option B: Split Submitter / Worker Config Views (PR #5279)

The bug: `prepare_lm_train` calls `compute_output_path` which resolves
`marin_prefix()` on the *submitter* (`experiments/defaults.py:658-662`,
`lib/marin/src/marin/execution/executor.py:1821`), then
`resolve_local_placeholders` substitutes every `OutputName(...)` with the
joined absolute path (`experiments/defaults.py:664`,
`lib/marin/src/marin/execution/executor.py:1276-1313`). After
`bake_output_path` (`lib/marin/src/marin/training/training.py:107`) the inner
`TrainLmConfig` carries `gs://marin-us-central1/...` for
`trainer.checkpointer.base_path`, `temporary_base_path`, `hf_save_path`, and
`tracker.replicate_path`. If the worker reschedules to `us-east5`, materialize
on the worker only resolves *remaining* `InputName`/`ExecutorStep` refs
(`lib/marin/src/marin/training/training.py:365`); the already-baked output
paths still point to `marin-us-central1`. Confirmed by dlwh on
`executor.py:1270`.

## Design

Two configs flow through the system, with a hard rule: *no absolute regional
path is computed by the submitter.*

* **Stub** = the launch dataclass as user-authored: `SimpleTrainConfig` /
  `GrugBaseLaunchConfig` (`experiments/grug/base/launch.py:40-58`). Still
  contains `OutputName(...)`, `InputName(step=...)`, `VersionedValue(...)`,
  `MirroredValue(...)`, `this_output_path()`. The submitter only needs the
  stub to (a) hash for versioning and (b) pick `resources` for dispatch.
* **Concrete** = `TrainLmConfig` / `GrugRunConfig` — what `worker_fn`
  consumes. Today this is built in the submitter via `_build_train_lm_config`
  (`experiments/defaults.py:369-502`) and `_build_grug_run_config`
  (`experiments/grug/base/launch.py:93-129`).

### Submitter responsibilities (unchanged from versioning POV)
1. `compute_output_path(name, stub, prefix=None)` — runs
   `Executor.compute_version` (`executor.py:1511`) which only hashes
   `step.name + version_dict` and concatenates `prefix`. **Critical:** the
   hash is region-stable (`executor.py:1546-1548`, `_dep_version` doc at
   `:1602-1610`), so we can compute it under *any* prefix and the suffix
   `name-<hash>` is portable.
2. Drop the absolute path entirely. Send only `(name, version_hash, stub,
   build_concrete_fn, worker_fn, resources)` to the worker.
3. `_submit_train_job(...)` becomes
   `_submit(name, hash_suffix, stub, build_fn, worker_fn, resources)`.

### Worker responsibilities
On entry to `_run_training_on_worker` (`experiments/defaults.py:613`):
1. Resolve `prefix = marin_prefix()` *here* — picks up the worker's region.
2. `output_path = os.path.join(prefix, f"{name}-{hash_suffix}")` (mirrors
   `executor.py:1548`, override path bypass also handled here).
3. `stub = resolve_local_placeholders(stub, output_path)` —
   `OutputName`/`VersionedValue` resolution (`executor.py:1276-1313`).
4. `concrete = build_concrete_fn(stub, output_path=output_path)` — runs
   `_build_train_lm_config` + `bake_output_path` + `impute_run_id` *with the
   worker's prefix*.
5. `concrete = materialize(concrete)` — resolves the surviving
   `InputName`/`ExecutorStep`/`MirroredValue` refs against the worker's
   region (already correct today via `executor.py:1881`).
6. `worker_fn(concrete)`.

## What changes

* `prepare_lm_train` returns `(name, stub, hash_suffix)` instead of
  `(name, baked_inner_config, output_path)`. `_build_train_lm_config` stays
  in `marin.training` but is *called by the worker*.
* `train()` (`experiments/defaults.py:681`) sends the stub + a callable
  reference to `_build_train_lm_config` through `Entrypoint.from_callable`.
* `_run_training_on_worker` (`experiments/defaults.py:613-620`) grows from
  3 lines to ~10; signature becomes
  `(worker_fn, build_concrete_fn, name, stub, hash_suffix, override_output_path)`.
* `check_train_config_paths` (`experiments/defaults.py:676`) moves to the
  worker — that's where it can actually verify region alignment.
* Sweep (`lib/marin/src/marin/execution/sweep.py`) needs the same wire
  format; today it relies on pre-baking trials.

### Grug standalone property
Grug has its own `_build_grug_run_config` at
`experiments/grug/base/launch.py:93` and *does not* import
`marin.training._build_train_lm_config`. Option B preserves this: each
template (base / moe / modular_opt) supplies its own `build_concrete_fn` to
`_submit_train_job`. The shared infra is only the submit/worker handoff in
`experiments/defaults.py:_submit_train_job` plus the `materialize` call.
Grug tests at `tests/test_grug_launch_checkpoint_paths.py` would shift from
asserting baked paths in the submitter result to mocking
`marin_prefix()` and asserting the worker-side build.

## Compatibility hazards

1. **Pickling `_build_train_lm_config` and `_build_grug_run_config`.**
   `Entrypoint.from_callable` (`experiments/defaults.py:600`) pickles by
   qualified name. Both are currently underscore-prefixed *private*
   helpers. They must become module-level (already true) and
   importable on the worker — fine, but renaming them to public is a
   minor API ripple.
2. **`compute_output_path` still wants `prefix`.** `executor.py:1821`
   defaults to `marin_prefix()`. Submitter must pass an explicit
   `prefix=""` (or refactor the function to a `compute_output_suffix`
   that returns just `f"{name}-{hash}"`). Recommended: add
   `compute_output_suffix(name, config, override_output_path=None) -> str`
   so the submitter never even *touches* `marin_prefix()`. This also
   matches the existing region-stable hash semantics at
   `executor.py:1602-1610`.
3. **`tracker.replicate_path` is a `this_output_path()` placeholder
   inside the stub** (`experiments/defaults.py:429`). It will be
   resolved by `resolve_local_placeholders` on the worker — fine. But
   `WandbConfig` is created inside `_build_train_lm_config` *today*; if
   we keep that boundary, the placeholder is created on the worker too.
   Verify nothing else captures the wandb config before submit.
4. **Versioning consistency: smaller hash, not bigger.** The version is
   computed from the *stub*, not the concrete config. Today
   `_build_train_lm_config` materializes `eval_harness_tasks` into a
   `LmEvalHarnessConfig` (`experiments/defaults.py:396-399`); none of
   that nested structure currently affects the hash because it sits
   inside the stub `TrainLmConfig` *after* hashing happens — wait,
   today `compute_output_path(name, inner_config)` is called *after*
   `_build_train_lm_config` (`experiments/defaults.py:658-662`), so
   the concrete config *does* get hashed. **This is a real semantic
   change.** Caching identity will shift: any non-versioned field
   inside the expanded `TrainLmConfig` (mesh axes, checkpointer
   `keep` policy, tracker tags, `eval_harness_steps`) currently
   contributes to the hash via the Python object structure even when
   not wrapped in `versioned(...)`. Mitigation: require all hash-worthy
   knobs to live on the stub (`SimpleTrainConfig`/`GrugBaseLaunchConfig`)
   wrapped in `versioned(...)`. Audit `_build_train_lm_config` for
   any literal that should be `versioned()` — e.g. `steps_per_eval`
   default of `1000` (`defaults.py:435`), `save_interval=10min`
   (`defaults.py:437`). Most are already either derived from stub
   fields or are non-identity defaults, but this needs a careful pass.
5. **`override_output_path` semantics.** When a user pins
   `override_output_path` (`executor.py:1551-1560`), the submitter must
   still pass that string to the worker; the worker honors it
   verbatim. Easy, but a bug-magnet if forgotten.
6. **`marin_temp_bucket` inside `bake_output_path`**
   (`training.py:96-104`, `:124`) reads `MARIN_PREFIX` /
   `marin_region()`. Today it's called on the submitter and produces
   the submitter region's temp bucket. After option B it runs on the
   worker — *correct* (and that's actually another silent bug today
   with the same root cause).
7. **`_doublecheck_paths` already runs on the worker**
   (`training.py:334-335`, called from `_prepare_training_run`), so the
   "fail fast in submitter" check at `defaults.py:676` is currently
   double-protection. Moving that submitter call to the worker is fine.
8. **Sweep hashing per-trial.** `sweep.py` currently pre-bakes each
   trial's config. With option B each trial only needs its stub
   hashed; the worker expands. The sweep harness itself needs to
   handle list-of-stubs, not list-of-baked-configs.

## Verdict

**Recommend, with a tightening pass on hash inputs first.**

The wire-format change is small, the bug is real and silent, and worker-side
expansion is the only design that's robust to cross-region preemption *and*
to operators who rerun the same script from a workstation in a different
region than the worker pool. The single sharp edge is hazard #4: moving
expansion to the worker shrinks the hashed surface, so any non-`versioned()`
literal inside `_build_train_lm_config` that operators rely on for cache
identity will silently stop participating. That requires (a) auditing
`_build_train_lm_config` for literals that should be `versioned()` and
(b) a one-time intentional cache invalidation when the change lands. Both
are tractable.

Hazard #1 (pickling builders by qualname) and hazard #5
(`override_output_path`) are routine. Hazard #6 is a bonus fix, not a cost.
The grug standalone property is preserved cleanly — each template carries
its own `build_concrete_fn`.

One-sentence summary: split the launch-stub (used for hashing + dispatch)
from the concrete trainer config (built and baked on the worker under its
own `marin_prefix()`), passing only `(name, hash_suffix, stub,
build_concrete_fn, worker_fn, resources)` over the wire — at the cost of a
deliberate hash-surface audit.
