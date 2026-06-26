# Research: Lazy Artifacts

In-repo findings, prior art, and the migration surface that shaped the design. Code refs are
pinned to branch `weaver/adopt-executor-context-for-remai` @ `4221dde`.

## Prior art (in-repo)

- **The original extraction plan** — `.agents/projects/executor-lazy-artifact-extraction.md`. The
  user chose *full extraction* over an incremental strangler; Codex flagged it as a high-risk
  big-bang across (then) ~188 `ExecutorStep` sites, mitigated by phases that each leave the branch
  building + goldens green, and by **relocating** (not reimplementing) the Executor's load-bearing
  guts. That doc's "Delete vs relocate vs keep" table is the authority on which symbols go: delete
  `compute_version`/`versioned`/`InputName`/`this_output_path`/`executor_main`/`Executor`; keep
  `step_lock`/`StatusFile`/`run_step`/`StepRunner` verbatim. This design doc is the post-stabilization
  writeup of what landed.
- **The Executor being replaced** — `lib/marin/src/marin/execution/executor.py` content-addresses
  each step `{name}-{md5}` over the whole config tree, which is why region/prefix/compute leak into
  identity. The lazy model keeps its *lowering* engine and drops its *addressing*.

## The lazy execution engine (current shape)

- `lib/marin/src/marin/execution/lazy.py` — `Artifact(name, version, recipe, override_path,
  adopt_source, expected_fingerprint)` (`eq=False`, object identity); subclasses `Dataset`,
  `Checkpoint`. `Recipe(fn, build_config, deps, run_args)`. `RunContext` with `for_run()` /
  `for_fingerprint()` (the identity line). `Artifact.fingerprint_payload()` / `fingerprint()`,
  `adopt()`, `lower()`, `materialized_config()`.
- `lib/marin/src/marin/execution/registry.py` — `ArtifactRecord` (now with `fingerprint_payload`),
  `guard_immutable()`, `enforce_immutability()`, `is_mutable_version()`, `ImmutableArtifactError`,
  `FingerprintMismatchError`, `_diff_json()` / `_describe_change()` (`_MAX_DIFF_LINES = 20`).
- `lib/marin/src/marin/execution/fingerprint.py` — `canonical_json()`, `fingerprint_hash()`,
  `_FingerprintEncoder` (strict, deterministic; raises on callables / default-`repr`).
- `lib/marin/src/marin/execution/step_spec.py` — `StepSpec` with the new `fingerprint_payload`
  field (provenance/diff, deliberately **out** of `hash_id` and out of logs). `output_path`
  prefers `override_output_path`; `name_with_hash`/`hash_id` only used when no override.
- `lib/marin/src/marin/execution/step_runner.py` — `enforce_immutability(step)` at `:273`, called
  **before** the `STATUS_SUCCESS` cache-skip at `:277`; mutable (`dev`) returns `True` → rebuild.
- `lib/marin/src/marin/execution/remote.py` — `remote(fn, resources=…)` → `RemoteCallable` that
  submits itself to Fray; resources ride on the fn, never the step node, so they never enter the
  fingerprint.

## Migration surface (the remaining work)

- **113 `ExecutorStep(...)` call sites across 33 files** under `experiments/` are still
  import-time. Heaviest: `experiments/common_pile/tokenize_common_pile.py` (32),
  `experiments/eval_datasets.py` (27), `experiments/midtraining_datasets.py` (7),
  `experiments/evals/evals.py` (5), `experiments/defaults.py` (5). `defaults.py` is the
  bottleneck — imported by most experiments, so migrating it unlocks dependents.
- **The import-time guard** — `lib/marin/src/marin/execution/context.py:65` `check_build_context()`,
  invoked from `ExecutorStep.__post_init__`. Default: warn once per `(kind, name)` (`:80`,
  "constructed outside an executor_context(); this will become an error"). Under
  `MARIN_EXECUTOR_STRICT` (`_STRICT_ENV`, `:30`): raises `RuntimeError` (`:71`). So today the door
  is still open; the flip to default-error is a sequencing decision (Open Question 1 in `design.md`).
- **Already migrated to lazy:** `experiments/tutorials/dclm_1b_1x_inline.py`,
  `experiments/tutorials/train_tiny_sweep_tpu.py`, `experiments/grug/moe/phases_lazy.py`,
  `experiments/grug/moe/launch_lazy.py`, `experiments/pretraining_datasets/{dclm_lazy,simple_lazy,nemotron_lazy}.py`,
  `experiments/paloma_lazy.py`, `experiments/evals/uncheatable_lazy.py`.
- **Old API still live (not deprecated, just unneeded for lazy):** `versioned()` ~35 sites,
  `InputName`/`.cd()` ~30, `executor_context()` ~35 (mostly tests), `THIS_OUTPUT_PATH` ~2,
  `executor_main` 1–2, `Executor` ~9 (tests/scripts; the golden test builds the executor path).

## Canonical patterns that already exist

- **Single run** — `experiments/tutorials/dclm_1b_1x_inline.py:76` `build(*, version="v1") ->
  Checkpoint`: every training decision (model, mixture, optimizer, budget, precision, parallelism,
  evals, checkpoint cadence) is plain Levanter config inside `build_config(ctx)`; dispatch via
  `remote(run_levanter_train_lm, resources=ctx.run_arg("train_resources"))`.
- **Sweep + select** — `lib/marin/src/marin/experiment/sweep.py`: `grid(**axes)` (cartesian
  product), `sweep(trial, **axes)` (one handle per point), `select(name, version, trials, *, metric,
  mode)` (a reducer artifact depending on all trials, reads each payload at run time, writes
  winner + scores). Example: `experiments/tutorials/train_tiny_sweep_tpu.py`.
- **Multi-phase chains** — `experiments/grug/moe/phases_lazy.py`: `_phase(name, *, fn, steps,
  parent, …)` returns a `Checkpoint` whose `init_from = f"{ctx.path(parent)}/checkpoints"`; the
  pretrain→midtrain→sft→rl lineage is encoded in `deps` (parent pointer). SFT/RL fns are
  `NotImplementedError` sketches.
- **Data builders** — `lib/marin/src/marin/experiment/data.py`: `tokenized()` (pinned /
  fresh-from-HF / download-dependent), `pretokenized()` (adopt an HF-hosted Levanter cache),
  `raw_download()`, `mixture(ctx, {handle: weight}, validation=[…])` → `LmDataConfig`. `mixture` and
  `data_configs.dataset_component()` share one `DatasetComponent` assembler so the executor and
  lazy builders can't drift. Catalogs return inert handles; the **experiment** states the weights.
- **Adoption** — `lazy.py:adopt(name, version, source, kind=Dataset)`: consumers resolve to
  `source` (no move/recompute), provenance recorded at the canonical address, re-adopting from a
  different source is a guarded conflict.

## Tests that lock behavior

- `tests/experiment/test_train_lm_golden.py` — lazy `build()` decisions vs `Executor` `default_train`
  (path-independent fields: model, optimizer, batch/steps/seq, z-loss, eval cadence, nonzero
  mixture weights). The anti-drift gate.
- `tests/execution/test_registry.py` — provenance recorded, same-recipe rerun is a cache hit,
  changed recipe raises, **error names the changed field**, payload recorded, `dev` mutable,
  `expected_fingerprint` pin (match + stale).
- `tests/execution/test_fingerprint.py` — determinism + field sensitivity, dtype/timedelta/Path,
  order-independent sets, arrays-by-content, enum-by-value, rejects callable + default-`repr`.
- `tests/execution/test_adopt.py` — resolves to source, records pointer at canonical address,
  re-adopt same source idempotent, different source raises, usable as a dependency.
- `tests/experiment/test_sweep_select.py` — grid product, distinct fingerprints per point, min/max
  selection, mode + empty-sweep validation. (`tests/execution/test_sweep.py` covers the
  leader/follower lock coordination for parallel TPU sweeps.)
- `tests/experiment/test_grug_moe_phases.py` — phase chain lowers to a pretrain→rl lineage; each
  phase's `init_from` resolves via `ctx.path(parent)`.

## What surprised me / still unclear

- **The strict encoder is a recent, load-bearing fix.** Before it, the fingerprint used the
  permissive `CustomJsonEncoder`, whose silent `str(o)` fallback could serialize an unknown type
  from its memory-address `repr` → a fingerprint that drifts between processes → the build-once
  guard misfires or never caches. Determinism is a *correctness* property of the guard, not a nicety.
- **The guard is "build-once," not "config-addressed."** Two different LRs both target
  `name/v1`; the guard arbitrates by erroring on the second (now with a field-level diff), not by
  giving them distinct homes. That's the deliberate trade for readable, human-chosen addresses —
  and the root of Open Questions 2–4.
- **`adopt()` is underused** — most catalogs use `override_path` pins; `adopt()` is the
  provenance-tracking variant and hasn't seen wide adoption.
- **The phase-chain test is structural only** — it checks lowering + `init_from` resolution, not a
  real SFT/RL run (those fns are sketches). A full smoke would exercise all four phases.
- **`tokenized()` is over-broad** — one signature spanning four mutually-exclusive source modes
  (`pin` / `source` / `paths` / `raw`+`glob`) plus a `validation` flag. It reads as four functions
  in a trench coat; the codebase's own API guidance (`AGENTS.md`: separate classes/functions over
  flag dispatch) suggests splitting it. These are authoring-layer conveniences, not part of the
  low-level execution contract.

## Peer review (codex)

A codex pass over `design.md` + `spec.md` against the code surfaced five issues; all incorporated:

1. **"Recipe change is loud" overstated** — the fingerprint covers `build_config` output, not
   `recipe.fn` source. Code-only changes serve cached output. Docs now say "fingerprinted-config
   change" and name code-change staleness as a risk.
2. **Identity line too permissive for TPUs** — codex argued the accelerator (a run-arg) can shift
   numerics. *Rejected on review:* changing TPU generation (e.g. v5p→v4) does not change numerics;
   the computation is fixed by the *logical* mesh, which is already a literal and so bears identity.
   So the original line is correct. Docs reframed descriptively — `run_args` are the mechanism for
   expressing configuration that must reach the step but must not be versioned — and the prescriptive
   "normative rule" + identity table + open question were dropped as over-correction.
3. **Guard-before-lock race (real bug)** — `enforce_immutability` runs before `step_lock`
   (`step_runner.py:333` vs `:341`) and isn't re-checked on `StepAlreadyDone` (`:360`), so
   concurrent first-builds with different fingerprints can both pass. Documented as a risk + open
   question; flagged for a follow-up fix (guard-under-lock).
4. **`select` name-collision (real bug, fixed)** — `select` keyed trials by bare `t.name`
   (`sweep.py:85`), so same-name/different-version trials silently collapsed. Fixed to key by
   `name@version` with a duplicate-identity `ValueError` guard + a regression test.
5. **Pin/adopt risk language** — `override_path` is unguarded (no record); `adopt` gives
   *pointer* not *content* immutability. Spec now has a three-tier guarantee table.
