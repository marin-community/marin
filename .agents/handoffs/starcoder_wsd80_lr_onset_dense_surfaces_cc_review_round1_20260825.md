## What I verified

Read the brief, the design generator, the frozen JSON (644 rows), the launcher, the tests, and the sources they import (`launch_starcoder_wsd_80_20_surface`, `design_starcoder_wsd80_dense_support_surfaces_20260808`, `launch_starcoder_wsd80_dense_support_surfaces`, `launch_starcoder_wsd80_lr_onset_intervention`), plus Levanter's checkpointer/trainer and Marin's `train_lm`/`mixture`, and the 20260811 confirmation artifacts.

**Schedule construction checks out numerically and semantically.** I reproduced every arm integral in `starcoder_wsd80_lr_onset_dense_surface_design_20260825.json:33-109` by hand: warmup 282 steps contributes 140.5; `no_decay` = 140.5 + 27978 = 28118.5 ✓; `decay_0p80` = 140.5 + 22326 + (2826 + 0.5) = 25293.0 ✓; `decay_0p90` = 26706.0 ✓; `decay_0p60` = 22467.0 (float32 22466.998) ✓; phase-1 integrals 1027.16 / 2826.5 / 4239.5 / 5652.0 are the correct cosine tails, and the area-match multiplier 0.888269 = 22466.998/25293 with recomputed integral bitwise-equal to `decay_0p60` (so the `abs=1e-6` assertion in `tests/test_starcoder_wsd80_lr_onset_dense_surfaces.py:77` is safe). Pre-onset schedules are identical across main arms, `no_decay` is genuinely flat (`min_lr_ratio=1.0`, same trick the audited intervention uses), and `boundary_step=22608` is untouched by the treatment.

**Fixed pair and replication are defensible and correctly sourced.** `selected_policies.csv:52-55` gives r3/m100 → tied c109 (0.70,0.70), untied c020 (0.01008,0.85968); r3/m200 → c079 (0.4,0.4), c011 (0.0,0.5). The confirmation results report (`..._confirmation_results_20260811/report.md:48-49`) gives mean gains 0.0075763 [0.0061083, 0.0090444] and 0.0104869 [0.0083333, 0.0126405] — exactly the CIs frozen into `analysis_contract`. Both blocks were Holm-significant; the paired-gain SD (0.001182) I re-derived from the CI, so the report's SNR argument for choosing option B over the 1.001B cell is correct.

**Endpoint read point is right.** `StepInfo.step = state.step − 1` (`callbacks/_core.py:76`), and the historical pipeline recorded `final_metric_step=3819` for `total_steps=3820`. So `metrics.endpoint_step = 28259` is correct, the forced final hook (`trainer.py:616`) guarantees a final eval there, and `_keep_policy`'s `every: 28259` fires exactly once at that step.

**Also verified:** support arithmetic (m100 = 1068 batches / 279,969,792 tokens; m200 = 534) and that both spine pairs wrap their support; data-stream equality across arms is real *and* audited (`launch_...:466-469`); `train_holdout_*` are pinned to `None`, so no holdout leaks in from the probe family; discovery seed is disjoint from confirmation seeds; `mixture()` keeps zero-weight components, so the p0=0 coordinates are safe; central1 pinning, `--stage` gating, hash-pinned release, and per-row artifacts (resumable, `force_run_failed=True`) are all in place; `uv.lock` does pin jax 0.11.0; no `manifests/` or `release.json` exists and the CC review file is empty, so the panel is unfrozen and unsubmitted.

## Blocker (launch-fatal, implementation)

**`save_interval=250` is an `int` where Levanter requires `Optional[timedelta]`** — `launch_starcoder_wsd80_lr_onset_dense_surfaces.py:257`.

`CheckpointerConfig.save_interval: Optional[timedelta]` (`lib/levanter/src/levanter/checkpoint.py:1171`), passed through untouched by `create()` and by Marin's `resolve_checkpointer_output_path`, then used at `checkpoint.py:587`: `last_save_time >= self.save_interval`, where `last_save_time = datetime.now() − self._last_save_time` is a `timedelta`. The checkpoint hook runs `every=1` (`trainer.py:648`) and `run_hooks` fires for `step > 1` (`trainer.py:143`), so at **step 2** — the first step that isn't a `keep`-interval multiple — this raises `TypeError: '>=' not supported between instances of 'datetime.timedelta' and 'int'`. Every one of the 644 runs dies after TPU acquisition, cache load, and JIT. `--dry-run` lowering cannot catch it, and no test asserts it: `tests/...:122` checks `checkpointer.keep` but never `save_interval`.

Secondary consequence: even with the type fixed, `250` seconds is not the intended semantic — permanent checkpoints exist only at 22608 and 28259, so rolling preemption recovery depends entirely on the time policy. Marin's default is `timedelta(minutes=10)` (`lib/marin/src/marin/experiment/train.py:65`), and every sibling launcher passes a `timedelta`. Fix is cheap and touches nothing hashed (the design JSON doesn't encode checkpointing, and the release isn't frozen).

## Scientific issues to resolve while the design is regenerated

These require editing the frozen JSON, so do them in the same pass, not after launch.

1. **No scale-invariant companion estimand.** `no_decay` will land at a much worse BPB level than `decay_0p90`. If tied-vs-untied gaps scale with level, an additive `gain_arm − gain_0p80` contrast will show an "interaction" that is only level scaling. Pre-specify a relative (or log-BPB) secondary alongside `p1_primary`; without it the panel is not cleanly interpretable in the direction results are most likely to go.
2. **The positive control is not independent, and the overlap is undisclosed.** `..._confirmation_design_20260811/run_manifest.csv` shows the historical fresh seeds were 20260821–20260825 — the first five of `CONFIRMATION_SEEDS` — at the same coordinates and support. Comparing the new `decay_0p80` gain to CI [0.006108, 0.009044] is therefore largely a cross-version reproducibility check, not a validity test, and no action is specified if it fails. Either relabel it as a reproducibility check or move to five unused seeds.
3. **No missing-metric or completeness policy.** The audited source design has a detailed one; this one has none. `p3` says "within each **complete** arm" without defining completeness, and there is no retry-until-durable rule and no rule for dropping a coordinate from `p2`. With 500 preemptible discovery rows, some will fail, and the selection rule then becomes analyst discretion.
4. **Multiplicity is unspecified** beyond "report all frozen arm contrasts" — three main-arm contrasts × two supports, plus the sensitivity arm. The source panel used Holm; pre-specify something.
5. **`policy_role` carries two incompatible taxonomies in one payload**: `coordinates[].policy_role` ∈ {tied, boundary_untied, interior_untied} (inherited) vs `runs[].policy_role` ∈ {tied, ineligible_near_tied, eligible_untied} (`design_...:178-185`). `p3`'s eligibility rule keys off the second; an analyst filtering `coordinates[]` for "untied" gets nothing. Rename one.
6. **Replication changes coordinates and support together** (m200 uses c079/c011), so it is an independent-block replication of the interaction, not a coordinate-matched support contrast. Say so in the contract.
7. **`p3`'s confirmation rows are outside the frozen inventory** (~64 rows, 4 main arms × 2 coordinates × 8 seeds, minus dedup) and need a second frozen design. The trigger is unconditional and the selection rule is deterministic, so this is *not* an outcome-contingent follow-up — but the brief's "best tied vs best untied" question is only answered by that second block, and the program size should be stated.

**Reuse verdict:** correct as declared — nothing is bitwise reusable. The historical dense rows ran under jax 0.10.1 with a different launcher, artifact path, and keep contract, and cross-arm comparability requires all five arms in one runtime. The stated reason in `reuse_contract` is right for the probe rows (holdout) but incomplete for the dense rows; the real reason is the runtime change. Note that 20 rows (`decay_0p80` × the two spine pairs × the five shared seeds) are nominal re-runs of completed work — that is the cost of the runtime pin, and it is the source of issue 2.

## Polish

- `_validate_runtime_environment()` hard-fails on any `uv.lock` byte change; the source panel only warned. A dependency bump mid-panel blocks resumption of a multi-day launch and forces a design regeneration (the hash is inside the hashed payload). Warn on `uv.lock`, hard-fail on jax/numpy/PRNG/x64.
- 644 permanent boundary checkpoints at step 22608 (~1.6 TB extra) have no documented consumer and no retention clause; the source panel deliberately kept only the forced final checkpoint. Arms diverge before the boundary, so these aren't even shareable across arms.
- `_optimizer()`'s guard at `:212` can never fire — neither `warmup` nor `lr_schedule` is replaced. Compare against frozen constants (282, `"cosine"`).
- `no_decay` records an inert `decay_steps: 5652` under `min_lr_ratio=1.0`; easy to misread.
- `release["submitted"]` is written but never read or flipped.
- No `CI` guard in `main()`, unlike both source launchers.
- The source launcher's `if starcoder_name not in data_config.components: raise` guard was dropped; currently safe, but cheap insurance given the many p0=0 coordinates.

## Missing tests

- **Checkpointer usability** — assert `isinstance(save_interval, timedelta)` (or drive `CheckpointerConfig.create(...).on_step`); this is the test that would have caught the blocker.
- `audit_materialized_runtime_configs` is never exercised; the test file re-implements a weaker version, so the cross-arm data-contract and pre-onset schedule guards are unverified.
- No test that `no_decay` is flat after warmup, nor that phase-1 LR integrals are strictly ordered (1027 < 2510 < 2826 < 4239 < 5652) — the semantic core of the treatment.
- No test pinning the confirmation seeds' relationship to the historical confirmation seeds (either disjointness or a documented, intentional match).
- No test that the fixed coordinates equal the historically confirmed selections — the provenance that makes the pair defensible is uncodified.
- No test that `_freeze_release` refuses without `VERDICT: PASS`, or that a non-central1 prefix is rejected.
- No assertion that materialized `num_train_steps == row.total_steps` and `optimizer.warmup == 282` (the intervention launcher audited both).

VERDICT: BLOCK
