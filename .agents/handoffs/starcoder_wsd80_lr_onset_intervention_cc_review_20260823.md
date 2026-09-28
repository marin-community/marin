## Re-review: StarCoder WSD80 LR-onset intervention

I re-read the brief, then verified the launcher against the levanter/marin runtime rather than against the summary. No blocker survives for the scientific design or the stage-0 training submission.

### What I verified

**Treatment is correctly specified.** `_optimizer` (`launch_starcoder_wsd80_lr_onset_intervention.py:192`) sets `decay=TOTAL_STEPS-onset` for decaying arms and `min_lr_ratio=1.0` for `no_decay`. Tracing levanter's `lr_scheduler` (`lib/levanter/src/levanter/optim/config.py:286-362`): warmup 38 steps, then `stable_steps = 3820 - 38 - decay`, so the decay boundary lands at exactly 2292 / 3056 / 3438 = 0.60T / 0.80T / 0.90T. `no_decay` keeps the 0.80T segmentation but `optax.cosine_decay_schedule(lr, 764, alpha=1.0)` (`config.py:55`) is constant, so it holds peak LR through T. The preregistered LR-matched half-peak triple checks out numerically: 3056 is 764/1528 into decay_0p60, 3438 is 382/764 into decay_0p80, 3629 is 191/382 into decay_0p90 — all exactly 0.01.

**Shared prefix is real, not just asserted.** All four arms use `optax.constant_schedule(0.02)` for every step below 2292, so bitwise identity at 0.55T (step 2101) is achievable by construction, and `_tree_sha256` (`starcoder_wsd80_gradient_probe.py:873`) is a genuine dtype+shape+bytes digest, not a summary statistic.

**Checkpoint retention resolves to the intended grid.** `_get_current_step_save_interval` (`lib/levanter/src/levanter/checkpoint.py:679`) picks the first policy with `until >= step`, so `{every: s, until: s}` saves at exactly `s` and nothing spurious; `CheckpointerConfig.__post_init__` (`:1204`) accepts the raw-dict form and its monotonicity assertions hold for the sorted 20-step grid, which means `audit_runtime_configs`' dict equality at `:669` also works. Every checkpoint the brief requires is present, and decay_0p90 has three post-onset states (3502/3629/3819), satisfying the ≥2 rule.

**Stage-0 validator plumbing is sound.** `_prepare_train_config`/`_initialize_runtime` are used read-only, matching `ProbeGroupConfig`'s established shape; `expected_restored_state_step(2101)=2102` and `schedule_step = 2102-1 = 2101` agrees with the aggregate's `expected_learning_rate`. MuonH injects both `learning_rate` and `adam_lr` as outermost hyperparams (`lib/levanter/src/levanter/optim/muonh.py:104`), so the `hyperparams_states`-excluding scalar scan at `:286` resolves to exactly one leaf each. Checkpoints land at `{output_path}/checkpoints/step-N` (`lib/marin/src/marin/training/training.py:183`), matching the constructed URI. `train_lm`'s kwargs all exist, and the calendar `VERSION` keeps the shared (non-per-user) name.

**Re-freeze is safe.** Training-step identity never reads the release (`_build_training:468`), so fixing validator code and re-freezing re-runs only the cheap validation steps — the four trainings stay cached. That is what makes the items below fixable in flight.

### Must close before stage 1 (none block stage-0 submission)

1. **The realized-LR gate is pre-treatment only.** The brief's plan says "realized LR at every retained state"; the implementation checks one state (0.55T) where all four arms are identical by construction, so the check has zero power to detect a treatment that failed to apply. The same validator job could restore 3056/3438/3629 — all retained, all present by the time it runs.
2. **Prefix identity is verified for seed 0 only.** The 28 fan-out trajectories get no identity check, so the paired design's core assumption holds verified for 1 of 8 seeds. Natural home is the probe follow-up, which reads those checkpoints anyway.
3. **`optimizer_schedule_num_train_steps` is unpinned.** The historical launcher asserts it is `None` (`launch_starcoder_wsd80_gradient_conflict_full.py:1029`); this one does not, and the 0.55T check cannot catch a horizon mismatch because `_restored_hyperparameters:302` recomputes `expected` from the runtime's own effective horizon. It is `None` in practice on this path — add the assertion.
4. **`_write_remote_json:267` byte-compares a payload containing `wall_seconds`.** A validator re-run after a post-write failure (e.g. in `_close_runtime`) fails closed and needs manual GCS deletion. The reviewed probe compares only `identity_sha256` (`starcoder_wsd80_gradient_probe.py:196`) for exactly this reason.
5. **First execution of the validator is post-training.** `--dry-run` lowers only training steps, not `validation_step`, and nothing tests `_restored_hyperparameters` or `run_stage0_aggregate_validation`. Both are unit-testable without a TPU over a synthetic opt_state pytree and synthetic arm documents.
6. Minor: `--max-concurrent` defaults to 32 while both stage limits are 4 and 28, so either stage errors unless the flag is passed; and on a bitwise mismatch the aggregate raises without recording magnitudes (`_tree_max_abs_diff` exists in the probe), though the four per-arm fingerprints do survive on disk.

### Limits of this review

No shell in this session, so I did not execute `--audit`/`--dry-run` or evaluate the schedule numerically — the schedule arithmetic above is traced from levanter's source. I also could not recompute the v10 probe runtime's hash against its pin; I confirmed instead that no LR-onset work touches that file (the only `lr_onset` reference in the repo is the launcher itself). Note that `.agents/handoffs/starcoder_wsd80_lr_onset_intervention_cc_review_20260823.md` is currently empty and `OUTPUT_DIR` does not exist, so `--freeze` must run after this review lands.

VERDICT: PASS
