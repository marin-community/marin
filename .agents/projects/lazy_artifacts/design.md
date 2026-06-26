# Lazy Artifacts

_Why are we doing this? What's the benefit?_

Marin experiments build their step DAG at **import time**: module-level `ExecutorStep(...)`
definitions, `versioned()` fields, `this_output_path()`, and `InputName`/`.cd()`
cross-references. The `Executor` then content-addresses every step as `{name}-{md5}`, so any
code change silently forks a new output path, the build environment's region/prefix is baked
into the graph, and the resulting paths are unreadable. This design replaces that with **lazy
typed artifacts** addressed by an explicit `name@version`, computed on demand, with a
build-once registry instead of content-addressing. Authoring an experiment stops being coupled
to a heavyweight engine, paths become human-readable and stable, and a change to an artifact's
*fingerprinted config* becomes *loud* (a guarded conflict) rather than a silent new path. (The
fingerprint covers the config a recipe builds, not the recipe function's source — see Costs.)

## Challenges

The hard part is **separating identity from execution** without a content hash. The Executor
fused them: the md5 of the whole config tree *was* the identity, which is why region, prefix,
and the specific TPU all leaked into the address. We need a model where "what an artifact is"
(model, hyperparameters, dependency identities) bears identity, while "where/how it runs"
(output path, prefix, region, accelerator) does not — and where that line is drawn by
construction, not by convention. Second, the fingerprint that guards immutability must be
**identical across processes**, or the guard misfires (spurious conflicts, or a cache that
never hits). Third, ~113 import-time `ExecutorStep` sites across 33 files make this a
big-bang-prone migration.

## Costs / Risks

- **Migration burden.** 113 `ExecutorStep(...)` constructions across 33 files
  ([grep](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/experiments))
  are still import-time; the heaviest are `experiments/common_pile/tokenize_common_pile.py`
  (32) and `experiments/eval_datasets.py` (27).
- **Two systems coexist** during migration. The import-time guard is a *warning*, not yet an
  error ([context.py:80](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/context.py#L80)).
- **Semantic dedup is dropped** (intentionally): two steps with identical configs under
  different names build twice. The Executor used to merge them by hash.
- **The author picks the version string by hand.** No content hash means no automatic
  distinct address per config — the build-once guard catches accidental reuse instead.
- **Function/code changes are not content-addressed.** The fingerprint is the *config*
  `build_config(ctx)` produces, not the source of `recipe.fn` (e.g. `_train`, a tokenize fn, a
  `remote` wrapper). Editing logic *inside* a step fn without touching its config serves the
  cached output — same staleness the Executor's config-hash had. Bump the version (or use `dev`)
  when you change behavior the config doesn't capture.
- **Pins are an unguarded escape hatch.** `override_path` resolves to existing data and writes no
  record (no provenance, no guard); `adopt()` guarantees *pointer* immutability, not *content*
  immutability — it fingerprints the source location, so external data mutating in place, or a
  relative source under a moved prefix, is invisible to the guard.

## Design

A **lazy artifact** is an inert handle `Artifact(name, version, recipe)`; building one runs
nothing ([lazy.py](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/lazy.py#L159)).
A `Recipe(fn, build_config, deps, run_args)` says how to build it. `lower(artifact)` turns the
handle graph into the existing `StepSpec` graph, which the existing `StepRunner` runs
(cache → lock → run). The lower engine — `StepSpec`, `StepRunner`, `step_lock`, status files —
is kept verbatim; only the fat `Executor` layer (content-addressing, `versioned`, `InputName`,
`this_output_path`) is bypassed and will be deleted last.

**Identity vs. execution.** `build_config(ctx)` is a pure function of a `RunContext`, and that
context is where the line is drawn
([lazy.py:60](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/lazy.py#L60)).
Values written as **literals** (model, hyperparameters, dependency *versions*) bear identity and
enter the fingerprint. Values **pulled from `ctx`** never do: `ctx.out`, `ctx.path(dep)`,
`ctx.prefix`, `ctx.region`, and `ctx.run_arg(key)` (e.g. the TPU a dispatched job runs on). The
fingerprint is computed by re-running `build_config` under `RunContext.for_fingerprint()`, which
substitutes placeholders for everything pulled — so a prefix move, a region move, or a different
accelerator never re-fingerprints. Compute likewise rides on the fn via
`remote(fn, resources=…)`, never on the graph node.

The subtle case is the accelerator. The *logical* mesh (`MeshConfig` axes, partitioning) is written
as a **literal** in `build_config`, so it bears identity; only the *physical* TPU
(`ResourceConfig.with_tpu(...)`, pulled via `ctx.run_arg("train_resources")`) is run-only. The line
is "logical shape = what, physical hardware = where" — and it's a real judgment call: the same
logical config on a `v4-8` vs a `v5p-128` can shift numerics. So the rule is normative, not
incidental — anything that changes *what is computed* must be a literal (hence fingerprinted), and
`run_args` are reserved for placement that leaves the result equivalent. `spec.md` carries the
identity-bearing vs run-only table.

**Build-once registry.** Identity is the explicit `{prefix}/{name}/{version}`. On success, a
step writes a `.artifact_record.json` recording its fingerprint + provenance
([registry.py](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/registry.py)).
Rebuilding `name@version` from a *changed* recipe raises `ImmutableArtifactError` — bump the
version. `dev`/`-dev` versions are mutable and always rebuild. The guard runs *before* a cached
SUCCESS is served ([step_runner.py:273](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/step_runner.py#L273)).
Pre-existing data is brought in with `adopt(name, version, source)`: consumers resolve to the
source (no move, no recompute) while the alias is still registered and guarded.

The guard is **single-process-tight but not yet concurrency-tight**: it runs before the
distributed `step_lock` is acquired, and a worker that finds the step already done exits without
re-checking the record
([step_runner.py:333,341,360](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/step_runner.py#L333)).
So two workers that *first-build* the same `name@version` with different fingerprints can both pass
the guard — one wins the lock and records; the other's divergent recipe silently adopts that
output. Closing this means moving the guard inside the lock (or re-checking on the
already-done path); see Open Questions.

**Fingerprint integrity.** The guard is only as trustworthy as the fingerprint's determinism, so
the fingerprint is computed by a strict encoder
([fingerprint.py](https://github.com/marin-community/marin/blob/4221dde1789d3d7a5b6892896c376787f9388891/lib/marin/src/marin/execution/fingerprint.py))
that canonicalizes every value it understands (dataclasses, enums, paths, timedeltas, dtypes,
sets by sorted members, arrays by content) and **raises** on values with no reproducible
serialization (callables, default-`repr` objects) instead of the old silent `str(o)` fallback,
which could fork a fingerprint from one process to the next. The canonical payload is recorded
next to the fingerprint (on a dedicated `StepSpec` field, out of `hash_id` and out of logs), so a
build-once conflict reports a **field-level diff** (`learning_rate: 3e-3 -> 4e-3`) rather than two
opaque hashes. `Artifact.expected_fingerprint` optionally pins the fingerprint, checked at
`lower()` time — making the config↔identity contract explicit in review and catching a recipe
change *before* the first build.

**Migration path.** `lower()` is a pure transform with no Iris/Fray awareness, so catalogs
migrate independently: a module that today exposes module-level `ExecutorStep`s becomes a
*function* returning lazy `Dataset`/`Checkpoint` handles (mechanism stays in the library, policy
moves to the experiment). The guard at `context.py` flips from warn → error under
`MARIN_EXECUTOR_STRICT`, then default-on, then the content-addressing layer is deleted once
unused. `experiments/defaults.py` and the dataset catalogs are the high-leverage unlocks. See
`spec.md` for the canonical authoring patterns (single run, sweep+select, multi-phase chains,
data mixtures) and `research.md` for the full migration surface.

## Testing

The anti-drift gate is **materialized-config equality**: `tests/experiment/test_train_lm_golden.py`
asserts the lazy `build()` produces the same path-independent training decisions as the
`Executor`'s `default_train`, so the readable inline code and the executed config cannot drift
while both systems coexist. The registry, fingerprint, adopt, sweep, and phase-chain tests all
drive the **real** `StepRunner` against a `tmp_path` prefix (no mocks): build-once conflict,
`dev` mutability, field-level diff, `expected_fingerprint` pin, deterministic encoding,
order-independent sets, adopt resolution/guard, grid product, metric selection, and phase
lineage.

## Open Questions

- **Concurrency-tighten the build-once guard.** It currently runs before `step_lock` and isn't
  re-checked when a worker finds the step already built, so concurrent first-builds with different
  fingerprints can both pass (above). Move the check inside the lock, re-check on the already-done
  path, or accept the gap as out-of-scope for now? (This is a real bug, narrow in trigger.)
- **Where exactly does the accelerator sit on the identity line?** The logical mesh is a literal;
  the physical TPU is a run-arg. But cross-hardware numerics aren't bit-identical — do we need a
  normative per-field rule (which `ResourceConfig` fields, env vars, seeds, source revisions are
  identity-bearing), or is "logical=what, physical=where" + author judgment enough?
- **When does the import-time guard become a hard error?** Flip `MARIN_EXECUTOR_STRICT`
  default-on *before* the catalog migration completes (forcing the work) or *after* (avoiding a
  flag-day break)?
- **Should non-`dev` versions require an `expected_fingerprint` pin?** Opt-in per artifact today;
  a "production versions must pin" rule would make every identity-bearing change reviewable, at
  the cost of a re-pin on every intended edit.
- **Auto-versioning ergonomics.** Is hand-choosing version strings acceptable, or do we want a
  fingerprint-suffixed convention (`name/v1-<fp>`) for sweep-like cases so distinct configs
  coexist without manual bumps — re-introducing a hash in the path?
- **Do we want fingerprint *attribution* (tracing)?** The field-level diff says *what* changed;
  tracing would say *why* (e.g. "an upstream dep version moved, not your code"). Worth the
  subsystem, or is the diff enough?
