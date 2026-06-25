# Plan: replace the Executor with lazy artifacts (full extraction)

## Decision

Replace the eager-DAG + content-addressing `Executor` with **lazy typed artifacts**
addressed by an **explicit `name@version`** (no hash in the path). The user chose
*full extraction* over an incremental strangler. Codex flagged this as a high-risk
big-bang across ~188 `ExecutorStep` sites; we mitigate by executing in phases that
each leave the branch building and golden tests green, and by **relocating** (not
reimplementing) the Executor's load-bearing guts.

## Target architecture

User writes a **lazy artifact**: name, explicit version, recipe (fn + config + deps).
Building runs nothing. Demand materializes via the existing lower engine.

```python
@dataclass(frozen=True)
class Artifact:
    name: str
    version: str               # explicit; "@dev" mutable, "vN"/CalVer immutable
    recipe: Recipe             # fn, config, deps (handles), resources

class Dataset(Artifact):    tokenizer: str
class Checkpoint(Artifact): model_identity: str; tokenizer: str   # + .hf_export
class HfExport(Artifact):   ...
```

- **Reference** = `name@version` → path `{prefix}/{name}/{version}`. Interpretable, immutable.
- **Recipe fingerprint** = `{config, [dep.version for dep in deps]}` — **flat** (immediate
  dep versions only; each dep version already immutably identifies it, so no nesting →
  the `_MAX_INLINE_DEPTH` problem vanishes). Stored in a `.recipe_fingerprint` sidecar +
  registry row, **never in the path**.
- **Demand**: `marin run experiments.x` imports the module, calls `experiment()` →
  terminal handle(s) → lower → `StepRunner.run`. Notebooks: `handle.load()` / `handle.path`.

## Delete vs. relocate vs. keep

| Fate | Symbols |
|---|---|
| **Delete** | `compute_version`, `versioned`/`VersionedValue`, eager DAG walk, `canonicalize`/version-dedup, `_MAX_INLINE_DEPTH`, `InputName`/`.cd()`, `THIS_OUTPUT_PATH`/`OutputName`/`this_output_path`, `executor_main`, `Executor`, `check_build_context` strict mode |
| **Relocate** (keep behavior, new home in the lowering layer) | `instantiate_config` (config materializer), region/prefix placement (`executor.py:214,280,1183`), `ExecutorInfo` experiment+step provenance (`executor.py:847,1396`) |
| **Keep verbatim** | `step_lock`, `StatusFile`, `run_step` (+ fingerprint guard), `StepRunner` (+ pseudo-dep + path-dedup), `_run_iris_job`, `Artifact.save`/`from_path`, `artifact_registry` (+ schema/version ext) |

## Codex findings → where each is handled

| # | Finding | Phase |
|---|---|---|
| P0-1 | Fingerprint guard must live in the cache/status protocol (before SUCCESS-skip + under lock), not a higher layer | 1 |
| P0-2 | Build-once-immutability vs retry: write fingerprint **before** run; immutable only after SUCCESS | 1 |
| P0-3 | `@dev` mutable needs a force-rerun-success path (status epoching) | 1 |
| P0-4 | Region/prefix placement must move into lowering | 2 |
| P1-5 | Config materialization (`instantiate_config`) — handle-ref equivalent | 2 |
| P1-6 | `handle.path` symbolic until final prefix/region known (don't capture build-env `marin_prefix()`) | 2 |
| P1-7 | Nonblocking pseudo-deps need `StepSpec` metadata (in fingerprint, omitted from blocking `deps`) | 0 |
| P1-8 | Dedup: accept explicit-identity (path dedup); document semantic dedup is intentionally dropped | 1 |
| P1-9 | Specify flat fingerprint shape + test a deep chain | 1 |
| P1-10 | Registry: accept `@vN`/`@dev`/`@latest` + store fingerprint/git/deps/builder | 1 |
| P2-11 | Untangle `StepSpec` from `as_executor_step`/build-context guard | 0 |
| P2-12 | Replace experiment-level provenance (`ExecutorInfo`) | 2 |
| P2-13 | Migration order: catalogs/mega-files first, behind shims | 3 |

## Phases (each: branch builds, goldens green)

- **Phase 0 — Anti-drift gate + untangle the lower layer.** *(gate DONE)* Upgrade the
  golden from content-hash equality to **materialized-config equality** (path-hashes
  normalized) so it survives the identity change — done in `tests/experiment/test_train_lm_golden.py`.
  Then make `StepSpec` standalone: drop `as_executor_step` coupling to `THIS_OUTPUT_PATH`/
  `VersionedValue`/`ExecutorStep`; decouple the build-context guard; add `nonblocking_deps` +
  `recipe_fingerprint` fields.
- **Phase 1 — Tracer bullet: one vertical slice end-to-end.** Identity + config materialization
  are coupled, so prove the architecture on a single path first: `Artifact`/`Checkpoint` handle
  + `Recipe` + `lower(handle) -> StepSpec` + the minimal config materializer for the train case,
  wired so `train_lm(...)` returns a `Checkpoint` whose lowered StepSpec **materializes to the
  same config as `default_train`** (the upgraded gate, with path scheme `name/version`). This
  de-risks the whole design before scaling.
- **Phase 2 — Fan out the lowering: fingerprint, guard, registry, all step types.** Generalize
  the materializer to `tokenize`/`download`/`sft` (handle refs `dep.path`, `self.output`,
  kept symbolic until run-prefix is known). Add the flat fingerprint, the `.recipe_fingerprint`
  sidecar + guard in `run_step`/`_launch_step`, `@dev` mutable (status epoching), registry
  `@vN`/`@dev`/`@latest` + provenance row. Relocate region/prefix placement + experiment
  provenance. Prove a deep chain.
- **Phase 3 — Migrate users.** Convert the ~10 module-level catalogs to step-functions/handles
  (mega-files `tokenize_common_pile.py`, `eval_datasets.py` first). `default_train`/`default_sft`
  → shims over `train_lm`/`sft` returning the lowered step. `executor_main` → shim accepting
  handles; add `marin run` + `experiment()`.
- **Phase 4 — Delete.** Remove the content-addressing layer and dead vocabulary once unused.

## Golden-test strategy (anti-drift gate before any deletion)

For representative steps — `train_lm`, `tokenize`, a download using `this_output_path()`,
a dep referenced via `.cd()`, and `sft(init_from=ckpt)` — assert that the lazy `lower(handle)`
produces the **same output path and the same materialized config** as the equivalent
`ExecutorStep` through the current `Executor`. Lock behavior first; migrate; delete last.

## Open decisions (my picks; veto on review)

1. **Region placement:** relocate as-is into lowering now; revisit simplification against the
   `region UNSET/PINNED/ANY` work as a follow-up (don't couple the two).
2. **`executor_main`:** keep as a thin shim that accepts handles or lowered steps, so the 38
   call sites don't all change at once; `marin run` is the new front door.
3. **`@dev` mutable:** status epoching (a generation counter in the status file) over a global
   `force_run_success` flag, so immutability stays the default and only `@dev` opts out.
