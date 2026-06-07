# PLC0415 circular-import cleanup — structural fix plan

Companion to the PLC0415 ("no function-local imports") rollout for
[issue #5883](https://github.com/marin-community/marin/issues/5883).

The rollout hoisted 148 imports and left function-local imports only for
genuine "cannot hoist" cases. This doc covers the **circular-import** subset
(the `# noqa: PLC0415 # circular import: ...` sites). Each was investigated by
reading the actual import graph and empirically reproducing the cycle with a
cold-import test. **Finding: almost none are irreducible cycles** — they are
layering inversions, and several are dead or broken code. Breaking them
structurally removes the local imports entirely instead of papering over them.

`tree_util.py` (haliax) was already fixed this way: its `from
haliax.nn.array_stacked import ArrayStacked` local imports were deleted in
favour of `isinstance(x, haliax.nn.ArrayStacked)` attribute access, because
`import haliax.nn` was already a top-level dependency. That is the template.

## Summary

| # | Site | Cycle | Verdict | Removes | Effort | Risk |
|---|------|-------|---------|---------|--------|------|
| H1 | `haliax/core.py:957` | core↔partitioning | attribute access | 1 | S | Low |
| H2 | `haliax/jax_utils.py:356` | jax_utils↔core | **delete dead+broken `to_jax_shape`** | 1 | S | Low |
| H3 | `haliax/util.py:19` | core→axis→util→core | attribute access (+docstring fix) | 1 | S | Low |
| H4 | `haliax/debug.py:169` | tree_util→nn→partitioning→tree_util | reorder `__init__.py` | 1 | S | Low |
| L1 | `levanter/data/sharded_datasource.py:91` | data↔store.cache | **delete dead `build_or_load_cache` method (0 callers)** | 1 | S | Low |
| L2 | `levanter/data/_preprocessor.py:132` | _preprocessor↔sharded_datasource | move `_TransformedDataset` marker down | 1 | S | Low |
| L3 | `levanter/tracker/tracker_fns.py:18` | tracker↔tracker_fns | reorder one import (package→leaf) | 1 | S | Low |
| L4 | `levanter/data/dataset.py:100,112` | dataset↔permutation | merge perm classes into `dataset.py` | 2 | M | Low |
| L5 | `levanter/layers/attention.py:276` | attention↔flash_attention | extract `layers/attention_mask.py` leaf | 1* | M | Low–Med (hot) |
| Z  | `zephyr/{plan,shuffle,execution}.py` (9 sites) | plan↔shuffle↔execution↔runners | extract 3 leaf modules | 9 | M | Med (hot) |
| I1 | `iris/cluster/config.py:1144` | config↔providers.local.cluster | move `connect_cluster` → `cluster/lifecycle.py` | 1 | S | Low |
| M1 | `marin/profiling/ingest.py:270` | ingest↔xplane | extract `profiling/trace_summary.py` engine | 1 | M | Med |
| M2 | `marin/training/training.py:204,205` | (transitive zephyr) | hoist — already importable | 2 | S | Low |
| E1 | `experiments/defaults.py` (4 sites) | defaults↔paloma/exp1600 | extract `experiments/tokenize_helpers.py` | 4 | M | Low (33 call sites) |

`*` L5 leaves one legitimately-deferred kernel import on a cold path unless the
flash fallback is repointed (higher-risk numerics change).

Total: **26 of 26** remaining circular noqas are removable. 8 are trivial /
dead-code; the rest are bounded extractions.

---

## Tier 1 — trivial & dead-code (unambiguous, low risk)

Removes 8 noqas, fixes 2 latent bugs. No new modules, minimal call-site churn.

- **H1 `core.py`** — delete the local import; `out_sharding =
  haliax.partitioning.get_pspec_for_manual_mesh(new_axes)` (the `haliax`
  package is already imported top-level; only call site).
- **H2 `jax_utils.py`** — `to_jax_shape` here is shadowed by `axis.py`'s
  version (the one everything imports) **and is broken** (`ensure_tuple`/`Axis`
  aren't re-exported from `haliax.core`, so calling it raises). Delete the
  function. Removes the cycle and a latent bug.
- **H3 `util.py`** — add top-level `import haliax`; `is_named_array` →
  `isinstance(leaf, haliax.NamedArray)`. Also moves the misplaced docstring
  (currently after the import, so not a docstring). No call-site changes.
- **H4 `debug.py` + `nn/__init__`** — the noqa comment is *wrong*: the real
  cycle is `tree_util→nn→_src.state_dict→partitioning→tree_util`, triggered
  only because `haliax/__init__.py` imports `debug` (line 13) before
  `tree_util` (line 20). Move `import haliax.debug` to after `import
  haliax.tree_util`; then hoist debug's local import. (`haliax.debug` is a leaf
  nothing else imports early.)
- **L1 `sharded_datasource.py`** — `ShardedDataSource.build_or_load_cache` has
  **zero callers** (all real callers use the free `store.cache.build_or_load_cache`;
  `AudioIODatasetConfig` has its own method). Delete it; the only `data→store`
  back-edge goes with it.
- **L3 `tracker_fns.py`** — change line 18 from `from levanter.tracker import
  CompositeTracker, Tracker` (the *package*) to `from levanter.tracker.tracker
  import ...` (the *leaf module*, matching line 23's `DictTracker` import). The
  back-edge through the package `__init__` disappears; hoist `tracker.py`'s
  local import.
- **M2 `training.py`** — the two `levanter.adaptor` / `train_dpo` imports are
  already importable at HEAD (zephyr breaks the cycle internally); `train_lm`
  is already imported top-level at line 16 from the same chain. Hoist and drop
  the noqas. (Verify with a cold import first.)

## Tier 2 — bounded extractions (one new leaf module each)

Each is a self-contained refactor worth its own focused review.

- **L2 `_preprocessor.py`** — move the `_TransformedDataset` marker class into
  `_preprocessor.py` (next to the `_DatasetTransform` types it pairs with),
  dropping its `ShardedDataSource` annotation so the module stays a leaf.
  `sharded_datasource` then imports it downward. (~5-line move.)
- **L4 `dataset.py`** — move `PermutationDataset`/`BlockShufflingDataset` (+
  private helpers) from `permutation.py` into `dataset.py`; re-export from
  `data/__init__.py`. Removes a base↔subclass cycle.
- **Z `zephyr`** — the plan↔shuffle↔execution↔runners tangle. Extract three
  leaves and every back-edge disappears (proven end-to-end by the investigator):
  - `shard_keys.py` ← `composite_sort_key`, `deterministic_hash`
  - `worker_context.py` ← `WorkerContext`, `_worker_ctx_var`, `zephyr_worker_ctx`
  - `stage_io.py` ← worker-side stage-IO / task-result helpers + counter-key
    constants (`ShardTask`, `TaskResult`, `_write_stage_output`, …)
  Then `counters`/`runners` import the leaves instead of `execution`, and the 9
  local imports hoist. `stage_io` carries a per-shard hot path (pure
  relocation, but the module to review most carefully).
- **I1 `iris/config.py`** — move `connect_cluster` (zero `src/` callers; only
  e2e tests) into a new `iris/cluster/lifecycle.py` that may import both
  `config` and `providers.local.cluster` top-level. Update 4 test imports.
- **M1 `marin/profiling/ingest.py`** — extract the trace-summary engine
  (`_CompleteTraceEvent` + ~50 private summarizer helpers) into
  `profiling/trace_summary.py` that both `ingest` and `xplane` import down into.
  Largest mechanical move; risk is missing a helper (re-creates the cycle) —
  move the whole private region as one unit and `plc_check` both sides.
- **L5 `levanter/attention.py`** — extract `AttentionMask` / `materialize_mask`
  / `_materialize_*` into `layers/attention_mask.py`; re-export from
  `attention` to keep ~45 call sites stable. Removes the mask half of the
  cycle. The `dot_product_attention ↔ flash_attention` dispatcher/kernel
  recursion is partly irreducible: either keep one cold-path local import, or
  repoint the flash VANILLA fallback to a lower primitive (numerics review).
- **E1 `experiments/defaults.py`** — extract `default_tokenize` (+
  `default_download`, HF-bucket helpers) into `experiments/tokenize_helpers.py`.
  `paloma`/`exp1600` import the primitive downward; `defaults` then imports them
  top-level. ~33 files re-point their `default_tokenize` import (mechanical).

## Tier 3 — free
- **M2** resolves regardless of zephyr (see Tier 1).

## Recommended sequencing
1. Tier 1 in this PR (clear wins, removes 8 noqas + 2 bug fixes).
2. Each Tier 2 item as its own follow-up PR (focused, independently reviewable),
   ordered: L2 → L1-adjacent → L4 → Z → I1 → M1 → L5 → E1.
3. Re-run `./infra/pre-commit.py --all-files --fix` after each.

---

## Outcome (all done in this PR)

All of Tier 1 and Tier 2 were implemented. 24 of the 26 circular local imports
were eliminated structurally. Highlights:

- **Dead code deleted:** `haliax.jax_utils.to_jax_shape` (shadowed + broken),
  `ShardedDataSource.build_or_load_cache` (zero callers). 2 latent bugs gone.
- **Attribute access / reorder:** haliax `core`, `util`, `debug` (+`__init__`
  reorder with `# isort: skip`), levanter `tracker_fns` (package→leaf import).
- **New leaf modules:** `zephyr/{shard_keys,worker_context,stage_io}.py`,
  `levanter/layers/attention_mask.py`, `marin/profiling/trace_summary.py`,
  `experiments/tokenize_helpers.py`, `iris/cluster/lifecycle.py`.
- **Code motion:** levanter `permutation`→`dataset` (file deleted),
  `_TransformedDataset`→`_preprocessor`; marin `training` imports hoisted;
  experiments `default_tokenize` extracted (32 call sites repointed).

### Two deliberate exceptions (genuine, proven cycles in hot-path code)

Kept as documented function-local imports — these are *not* slop; each was
empirically confirmed and is irreducible without risky surgery:

1. `levanter/layers/attention.py` — `dot_product_attention` dispatches to the
   `flash_attention` kernel, which calls back into the vanilla path. The mask
   primitives were extracted to `attention_mask.py` (half the cycle removed);
   the remaining dispatcher↔kernel edge would need a numerics change to the
   flash fallback. Cold-path import.
2. `levanter/tracker/tracker.py` (`Tracker.__enter__`) — importing `tracker_fns`
   at module load pulls the wandb/trackio backends → `background` → back to
   `tracker`. Removing it means relocating the mutable global-logging singleton
   state out of `tracker_fns`, invasive surgery on the hot logging path. The
   package-`__init__` back-edges were severed (`tracker_fns`/`tensorboard`/
   `trackio`/`wandb` now import the leaf `tracker.tracker`); only the
   call-time accessor import remains.

Both are candidates for a future focused refactor if desired.
