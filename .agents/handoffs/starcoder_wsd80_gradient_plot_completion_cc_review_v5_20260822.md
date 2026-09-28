PASS_AFTER_BLOCKERS_RESOLVED

# CC review: StarCoder WSD80 gradient plot completion v5 (post-provenance-contingency)

Reviewer: `claude-opus-5[1m]`, read-only, no Bash. Account `plambdafour@proton.me`. No files edited.

## Does 7efb->377ad close my prior contingency?

Yes, and more strongly than the briefs argue. Four independent lines converge:

1. **The 5,888-field check is not a spot check.** `_config_identity` (`freeze_starcoder_wsd80_gradient_probe_20260816.py:221-284`) returns exactly 23 keys; 256 x 23 = 5,888. One of those keys is `full_train_config_sha256 = canonical_sha256(asdict(train))` - a hash over the *entire* `TrainLmConfig` tree, plus `model_config_sha256`, `optimizer_config_sha256`, and `data_config_sha256`. Reconstructing all 256 at `377ad` therefore proves structural and value identity of the whole config tree, not 23 hand-picked fields.
2. **The pinned v6 freeze script itself already reads `train.optimizer_schedule_num_train_steps`** (`:225`). That file is hash-pinned as `parent_implementation_files`, so it was authored against a tree that *had* the field. The historical dirty worktree therefore contained the 377ad change; `7efb`-clean would have raised `AttributeError`.
3. **The one optimizer-adjacent change is a verified no-op at these configs.** `lib/levanter/src/levanter/main/train_lm.py:93` defaults it to `None`, `:245-253` resolves `None -> trainer.num_train_steps` and only permits *widening*. `_config_identity` records both `configured_...` (empty) and `effective_...`, so a non-default would have shown up in the 5,888.
4. **The `_historical_config_identity` tripwire is non-vacuous.** `CheckpointerConfig.write: TensorStoreWriteConfig` exists at HEAD (`lib/levanter/src/levanter/checkpoint.py:1176`), so `starcoder_wsd80_gradient_plot_completion.py:41` genuinely discriminates the historical tree from the current one.

The diff audit's framing is slightly loose in one place: `data/dataset.py` and `data/text/datasets.py` changes *can* change which sequences are drawn, which is not fully captured by "does not change loss computation." That path is nonetheless covered - `first_batch_sha256` and the four loss moments are compared per source against pinned v6 parent documents (`starcoder_wsd80_gradient_mechanism_repair.py:1134-1147`), and data drift moves losses grossly.

## Is the staged numerical gate sufficient?

Yes. It is the correct instrument, because the residual is unprovable by inspection and the gate converts it into an empirical test with **demonstrated** sensitivity: v3 failed 2,471/9,000 comparisons at ~6e-4 cosine shifts while all 80 loss statistics were bit-exact. Stage 1's eight rows span all six labels, all three shapes, all three supports, both H5 policies, and the zero-LR path, and nothing promotes without an exact prior-stage audit (`:522-523`). Tolerance is `5e-6 * max(|obs|,|exp|,1)` (`:1097`) - the diff audit's "absolute tolerance 5e-6" is imprecise wording, not an implementation difference. Host-side `mechanism._load_release` does recompute the canonical release hash (`:721`), and the worker re-verifies the full train-config hash before any gradient work (`:405-407`).

Supersession is correct: v3 and v4 releases plus both failure markers are hash-pinned and re-verified in `_load_release`; v4 has no environment baseline and no authorization sidecar, confirming it never launched; no `complete_tables_v*` or `complete_plots_v*` directory exists for any version, so no superseded output was ever merged or rendered.

## Conditions that must still be checked after launch

**Before/at freeze**
1. Save this verdict verbatim to `.agents/handoffs/starcoder_wsd80_gradient_plot_completion_cc_review_v5_20260822.md`; it does not yet exist and `freeze()` hard-fails without it.
2. Confirm `V4_RELEASE_FILE_SHA256` (`170353ba...`) matches the on-disk v4 `release.json`. I verified the embedded `release_sha256` `46bae7e7...` but cannot hash files.
3. Confirm the v5 `full_plot_completion_manifest.csv` sha equals v4's `cc64e2396f6da460dcbbcff9850feeb22eb196be4116cb9622a459d06f39e079`. Row/group identities are release-independent, so the design is unchanged iff these match; a difference means something drifted silently.
4. Confirm `uv.lock` at `377ad` pins **libtpu 0.0.41**, not just JAX/JAXLIB 0.10.1. The diff audit only asserts the latter.
5. Confirm the v5 `historical_runtime_stack_manifest.csv` row count against v4's 881 and account for any delta from the 7efb->377ad diff.
6. Verify `gs://.../starcoder_wsd80_gradient_plot_completion_v5_20260822` does not already exist.

**Coverage gaps to close or accept explicitly**
7. `HISTORICAL_RUNTIME_PATHS` pins only 5 of the 11 `lib/*` packages. **finelog, finestore, ducky, dupekit, rigging, zephyr are unverified**, and `lib/marin` may import some at runtime. Also unpinned: `.python-version` and any per-lib lockfiles. Either extend the manifest or record why none is on the numerical/data path.
8. The `full` design module backing `_full_configs()` is not in `implementation_files`; it comes from `377ad` in the worktree and is validated only indirectly by `_audit_frozen_provenance`. Confirm that indirection is intended.
9. `7efb96842624a2e8cbab36c9a9aa6b1cb68c4922` appears only abbreviated, in contract prose. The full hash survives only via the pinned v4 `release.json`.
10. v1 and v2 are not pinned as superseded, and v2 carries a `full_launch_authorization.json`. No contamination path exists (the materializer reads only the v5 result root), but the paper chain is incomplete.

**Stage 1 - first real execution of code that has never run**
11. v4 never launched, so `_runtime_execution_observation`, `_verify_worker_runtime_packages`, `_assert_source_only_parent_statistics`, and `_verify_or_freeze_stage1_environment` are all executing for the first time. Expect worker-side failures to be instrumentation bugs, not numerics, until proven otherwise.
12. Confirm `mechanism.jax.config.jax_default_matmul_precision` resolves under JAX 0.10.1 (`starcoder_wsd80_gradient_plot_completion.py:163`) and that `_installed_distribution_versions()` does not trip its duplicate-distribution guard on the TPU image.
13. Confirm `libtpu` resolves under exactly that normalized distribution name in the image; a different name (`libtpu-nightly`) fails `_verify_worker_runtime_packages` before gradient work.
14. Read `source_only_parent_statistic_comparisons` in the Stage-1 report: expect **88** (2 rows x 2 sources x 2 geometries x 11 components). It is recorded but never asserted `> 0`.
15. Inspect `stage1_runtime_environment_baseline.json` before promoting. Outside the four pinned versions the baseline is **self-defining** - the first run declares truth. Only the ~9,000-comparison parent gate makes it meaningful, so the baseline is only trustworthy retroactively, after Stage 1 passes.

**Stages 2-4 and materialization**
16. Stages 2 and 3 contain **only** `common_target_trajectory_completion` rows (the `remaining` sort puts all 220 common rows first). Their audits will report `source_only_parent_statistic_comparisons: 0` - that is expected, not a regression. All 30 remaining source-only and 30 remaining H5-target rows land in Stage 4.
17. The v10 overlap A/B (`_historical_runtime_overlap_audit`) runs **only at materialization**, after all 288 rows are paid for. Manually compare the four common Stage-1 rows against `source_source_geometry_all_states.csv` before funding Stages 2-4; the automated cross-check is otherwise too late to save spend.
18. Full-inventory environment uniformity is enforced across all 288 rows against the Stage-1 baseline. Any transitive package drift between stages fails Stage 4 late. Keep it fail-closed and diagnose rather than relax - but plan for that failure mode.
19. Expect `source_only_parent_statistic_comparisons` = **1,408** in the final all-288 audit (32 x 44).
20. Confirm launch is submitted **from the detached `377ad` worktree**, not the main checkout. `_steps()` builds `pod_config` host-side via `_full_configs()`; running from HEAD would trip `_historical_config_identity` or the frozen-provenance audit, but that wastes a cycle.
21. Iris launch commands must include this review file and exclude the eight large visualization-only inputs, per the v4 finding.

The only claims I could not verify are the ones requiring execution or hashing: the 256/5,888 reconstruction result itself, and every file digest. Everything checkable from source is consistent, and item 1 of my analysis makes that reconstruction a materially stronger piece of evidence than the briefs claim for it.

---

PASS_AFTER_BLOCKERS_RESOLVED

# Post-review closure check (v5, items 7-10, 14)

**Item 7 - runtime stack coverage: closed.** `HISTORICAL_RUNTIME_PATHS` now pins `.python-version`, root `pyproject.toml`/`uv.lock`, and `pyproject.toml` plus `src` for all 11 `lib/*` packages: ducky, dupekit, finelog, finestore, fray, haliax, iris, levanter, marin, rigging, and zephyr. That matches the on-disk inventory exactly. No per-lib lockfiles exist, so the unpinned per-lib lockfile branch is moot rather than accepted. `EXPECTED_RUNTIME_VERSIONS` also carries libtpu 0.0.41.

**Item 8 - `_full_configs()` design module: closed.** `_full_configs()` calls `full.build_training_steps`, and `_canary_configs()` calls `canary`; `CANARY_CONFIG_PATH` and `FULL_CONFIG_PATH` are exactly those two modules and are now in `implementation_paths`. No hash conflict exists: the v6 release's `implementation_files` contains only the parent freeze and probe runtime, so nothing is double-pinned at two different digests. `HISTORICAL_RUNTIME_PATHS` and `implementation_files` are disjoint.

**Item 9 - full clean commit: closed.** `RECORDED_CLEAN_COMMIT = "7efb96842624a2e8cbab36c9a9aa6b1cb68c4922"` flows into both `ANALYSIS_CONTRACT` and `release["historical_runtime"]`, so it is covered by the contract file hash and the release self-hash rather than surviving only in v4's `release.json`.

**Item 10 - v1/v2 paper chain: closed.** Both releases and their `PRELAUNCH_SUPERSEDED.md` markers are pinned by file digest and embedded `release_sha256`, checked at freeze, recorded with status strings, and re-verified host/worker-side in `_load_release`. The markers describe the states accurately; v2's marker explains its existing `full_launch_authorization.json`.

**Item 14 - source-only count asserted: closed, and the constants are right.** The runtime asserts `{None: 1408, 1: 88, 2: 0, 3: 0, 4: 1320}[stage]`. There are 11 components, hence 2 sources x 2 geometries x 11 = 44 comparisons per source-only row. Thirty-two source-only rows total yield 1,408; Stage 1 holds two, yielding 88; Stages 2 and 3 are common-only; and Stage 4 holds the remaining 30 source-only rows, yielding 1,320.

**No launch blocker introduced.** Every new pin fails closed before TPU spend. The v5 historical runtime manifest is expected to grow beyond v4's 881 rows because six libraries and `.python-version` were added. The two launch-config builders must still match their frozen config identities in the detached `377ad` worktree before submission.
