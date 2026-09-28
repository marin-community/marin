# Handoff: retained-power-law surrogate, 2026-07-29

Read this with `.agents/projects/two_phase_surrogate_north_star.md`, the latest
entry in `.agents/logbooks/wsd80-mechanistic-surrogate.md`, the active registry
at `.agents/projects/two_phase_surrogate_active_registry.csv`, the historical
99-route registry at
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mechanistic_surrogate_discovery_20260719/approach_registry.csv`,
and the repo `AGENTS.md`. Everything below is measured unless explicitly
labelled otherwise.

Do not fit a new candidate until its active-registry row names the nearest
historical routes, the materially new mechanism or identification argument,
why the prior rejection does not apply, and the cheapest falsification test.

## The goal

Build a surrogate that fits the StarCoder 80/20 WSD panel much better than the current Observatory
models, with a mechanistically motivated design, and that evaluates **on par or better** on the other
swarms, 39-bucket in particular. Added conditions: no test-set selection; a Codex review each
iteration; on meeting the conditions, implement in Observatory and get a Claude Code review of that.

**Status: not met.** WSD80 is met by a wide margin. 39-bucket within-panel cross-validation is a loss.
Observatory implementation is not started and is correctly gated behind the 39-bucket condition.

## Files

All under `experiments/domain_phase_mix/exploratory/two_phase_many/`. All **untracked** — do not
`git checkout` or reset them, there is nothing to restore from.

| file | role |
|---|---|
| `retained_power_law_model_20260728.py` | the model: design matrix, shape grid, robust head, `fit` |
| `benchmark_retained_power_law_swarm39_20260728.py` | 39-bucket comparison; `--scales/--targets` run one cell per process |
| `benchmark_wsd80_incumbents_20260728.py` | WSD80 comparison; `--folds blocked\|random` |
| `starcoder_wsd80_panel_20260728.py` | WSD80 panel loader, 346 coordinates, sigma 0.004633 |
| `plot_wsd80_model_vs_surface_20260728.py` | orthographic model-vs-measurement overlay |
| `swarm39_harness_20260725.py` | shared harness; modified (see below) |
| `swarm39_models_20260725.py` | incumbent zoo, unmodified |

Harness modifications made for this work: `mixture_blocked_splits`; a `split_fn` parameter on
`fit_model`; a `head` field on `Model` so a model can supply its own estimator; and a NumPy 2.5 fix in
`fit_head` (`float()` on a size-1 array) that had been breaking every Track-B model.

Durable outputs: `reference_outputs/retained_power_law_swarm39_20260728/` and
`reference_outputs/wsd80_incumbent_benchmark_20260728/`. Note the swarm39 directory currently holds a
stale `swarm39_comparison.csv` from before per-cell filtering; new runs write
`swarm39_comparison_<scale>__<target>.csv`.

Session scratch diagnostics live in
`/private/tmp/claude-501/-Users-calvinxu-Projects-Work-Marin-marin/c4467274-ad58-4e3f-b191-61c61ce969d9/scratchpad/`
(`onese.py`, `localize39.py`, `bucketphase2.py`, `latemult.py`, `latemult_wsd80.py`,
`identify_late.py`, `solverfix.py`). That path is session-scoped and may be cleaned; the numbers they
produced are all reproduced below.

## Run status, updated 2026-07-30 18:31

Six processes were launched at 17:52 on 2026-07-29. **The two WSD80 runs completed; all four 39-bucket
cells were killed and left zero-byte logs** — no traceback, no stdout, so they were terminated rather
than crashed. Six concurrent 1620-combination grids is the likely cause. The benchmark printed nothing
until its first model finished, which is hours, so a dead run and a slow one looked identical; it now
prints at cell and model start, and cells are run two at a time.

Relaunched 2026-07-30 18:31, confirmed alive:

```
scratchpad/run2_delphi_3e18_uncheatable.log
scratchpad/run2_60m_uncheatable.log
```

Still never measured, to run once those land: both Table-9 cells (`delphi_3e18` and `60m` with
`--targets table9_macro_bpb`). A nested cell took 186 min at 720 combinations; the grid is 1620 now, so
expect 6-8 h each.

### WSD80 on the corrected grid — measured, supersedes the table below

Removing ridge 0 and restoring the five retention values **changed the blocked RMSE from 6.74 sigma to
11.65 sigma**. The 6.74 figure depended on ridge-0 solves that were rank-deficient and are now gone.
Everything decision-relevant held: median 1.4344 -> 1.4084 sigma, spearman 0.9395 -> 0.9373, optimum
unchanged at (0.075, 0.475), predicted gain unchanged at +0.007720. RMSE on this panel runs about eight
times the median, so it is dominated by a few extreme rows and is the least stable statistic reported.

| protocol | mine RMSE / median / spearman | best incumbent |
|---|---|---|
| blocked | **11.65 / 1.41 / 0.937** | 22.95 / 11.01 / 0.164 |
| random | **3.31 / 0.78 / 0.995** | 10.80 / 3.42 / 0.841 |

**Lesson worth carrying:** the pre-correction WSD80 numbers were inflated by a defect, and the same is
plausibly true of the pre-correction 39-bucket numbers in the next section. Treat every figure below as
an upper bound on quality until re-measured.

## The model

```
L = b + sum_i A_i (S_i + E0)^-a  +  sum_i B_i max(D_i - T, 0)^g  +  J * concentration  [+ ordering block]
```

`S_i` is *retained* token share, which is what makes the benefit term an interaction rather than a
reweighting:

```python
survival = exp(GATE_CLIP * tanh(retention * (w1 - w0) / GATE_CLIP))
S        = survival * (beta0 * w0) + late_multiplier * beta1 * w1
```

Why this shape matters: a model in which the schedule enters only through an additive phase-weighted
dose predicts **exactly zero** two-phase gain, because the tied-policy class already sweeps the whole
attainable dose set. That is a theorem, not a fitting artifact, and it is why the incumbent DSP family
cannot represent the WSD80 result at all.

Grid: 3 benefit exponents x 2 offsets x 3 damage exponents x 1 threshold x 5 retentions x 3 late
multipliers x 2 ordering settings = 540 shapes, x 3 ridges = **1620 combinations**.

Head: bounded `lsq_linear` (nonnegative amplitudes, free intercept), Huber IRLS with the cut set from
the residual MAD, convergence tested on *predictions* not coefficients. Penalty multipliers are 0 on
pooled family columns and 1 on bucket departures and ordering columns, so shrinkage pulls a bucket
toward its family rather than toward zero.

## Results

### WSD80 — met (346 coordinates, sigma 0.004633, blocked folds)

Measured: best tied p=0.300 gives 0.945062; best two-phase (0.100, 0.500) gives 0.935468; advantage
**+0.009594 BPB (2.1 sigma)**.

| model | RMSE (sigma) | median (sigma) | spearman | optimum | dist | predicted gain |
|---|---|---|---|---|---|---|
| **retained_power_law** | **6.74** | **1.43** | **0.94** | (0.075, 0.475) | **0.035** | **+0.007720** |
| effective_exposure | 22.95 | 11.01 | 0.16 | (0.460, 0.235) | 0.447 | 0.000000 |
| effective_exposure_geometry | 24.62 | 11.52 | 0.18 | (0.345, 0.345) | 0.290 | -0.000001 |
| separate_heads | 28.80 | 16.43 | 0.61 | (0.045, 0.300) | 0.207 | +0.036975 |

The `g(a)` profile tracks measurement across all eight fixed-aggregate fibers (0.2400 predicted against
0.2433 measured at a=0.80) with 7/7 sign agreement on the best contrast.

### 39-bucket — not met (nested OOF, 5 grouped outer folds, selection repeated inside each)

These are **pre-late-multiplier**. Both Table-9 cells have never been measured.

| cell | mine | hierarchical_phase_replay | bucket_family_grp |
|---|---|---|---|
| delphi_3e18 / uncheatable | 0.010237 (med 0.005883) | **0.008531** (0.004725) | 0.009153 (0.005391) |
| 60m / uncheatable | 0.011510 (0.006564) | **0.008732** (0.005248) | 0.009218 (0.005815) |

Held-out panels, same fits — mine wins both, and the incumbents' median error is essentially their
bias, i.e. they are systematically offset:

| cell | mine (bias) | HPR (bias) | bucket_family_grp (bias) |
|---|---|---|---|
| delphi | **0.045608** (-0.0025) | 0.053921 (-0.0258) | 0.054018 (-0.0275) |
| 60m | **0.032127** | 0.044568 | 0.045682 |

The held-out panel spans 0.9825-2.5820 BPB where the fit panel spans 1.0067-1.1654, so it is an
extrapolation test. The saturating deficit `(S+E0)^-a` extrapolates; the incumbents' unbounded
`state^power` drifts.

### Where the 39-bucket loss lives

Pinned at its selected shape on delphi/uncheatable, grouped folds:

| rows | mine | HPR |
|---|---|---|
| tied policies (n=42) | **0.003410** | 0.005298 |
| moved, all four TV quartiles | ~0.0110 | 0.0074-0.0100 |

I win tied policies by 36% and lose every quartile of moved policies. All twelve worst rows have phase
total variation above 0.39 with all 39 buckets active. **The loss is specifically on the rows the phase
machinery exists to describe.**

The late multiplier was the response to that and is the largest single improvement so far: pinned-shape
delphi/uncheatable went 0.010109 -> 0.008930 (-11.7%), moved rows -12.2%, closing the gap to HPR from
+23.1% to +8.8%. It is interior, not a grid edge (8, 16, 32 are all worse). **This has not yet been
measured nested** — that is what the running jobs will tell you.

## Do not redo these — refuted

1. **Selection variance / one-standard-error rule.** The 1-SE band holds 15 of 720 combinations, all the
   same shape family; nested sits only 1.3% above the within-fold argmin. The grid is not overfitting
   its own selection.
2. **Per-bucket ordering columns.** Hierarchical per-bucket phase response takes the design to 256
   columns on 280 rows and is 11.4% worse overall and worse on the moved rows it targeted, under either
   prior on the pooled amplitude. The contrast columns are near-collinear across buckets.
3. **Phase machinery being dead weight at 39 buckets.** False: ordering-on beats ordering-off 0.010109
   to 0.010611.
4. **Log-deficit link on WSD80** (56 sigma against 10), **continuous shape search** (3x worse
   out-of-fold than the discrete grid), **epoch-based rather than token-share benefit** (33.6 sigma
   against 10.3).

## Open leads, in the order I would try them

1. **HPR's `forgetting_rate` is a mechanism this model does not have.** HPR searches
   `late_multiplier` x `forgetting_rate`; this model now has `retention` x `late_multiplier`. Retention
   gates on the phase *contrast*; HPR's forgetting decays early exposure independently of what came
   late. Those are not the same thing and only one is present here.
2. **Pooling before vs after the nonlinearity.** `bucket_family_grp` and HPR form
   `(sum_members state)^power`; this model sums already-transformed per-bucket values. An earlier test
   put this at ~3% but ran with the wrong head (see below).
3. **`BENEFIT_OFFSETS` is too coarse.** Offset 0.02 beats both shipped values (0.01, 0.1) at every late
   multiplier, worth ~2%. **I deliberately did not adopt it**: choosing a grid value because it won on
   the full panel is the same leakage removed in Codex review 4. It needs either a structural
   justification for a denser log grid or selection on independent data.
4. The 39-bucket residual localization has only been done on delphi/uncheatable. Doing it on 60m and on
   the Table-9 targets may show the loss is not the same shape everywhere.

## Provisional results — seven eliminations that ran with the wrong estimator

`harness.fit_model` was substituting its own NNLS head for this model's robust head, so these seven
conclusions measured a different estimator and are **not** safe to rely on: regularization strength;
phase machinery affordability; family aggregation form; damage threshold form; benefit functional form;
family-specific retention; grid size. The head bug is fixed (`Model.head`). One of the seven —
grid size — was re-run under the real head and **inverted**: 720 combinations beat 2880. Assume the
other six may also invert.

## Codex review history

Four reviews, eleven findings, all verified against the code and all fixed. The two that changed
conclusions rather than code:

- **Review 3** found `harness.fit_model` never called this model's head, which invalidated the seven
  eliminations above.
- **Review 4** found target-informed grid pruning: retention values 1.0 and 5.0 had been dropped using
  full-panel sweeps on the panels the nested score is then reported on. That is the no-test-set-selection
  condition, violated. Restored to five values.

Review 4 also correctly found that ridge 0 makes the design exactly rank-deficient (every multi-member
family column is the sum of its member columns), producing predictions near 7e10 that were scored as
valid folds. Ridge 0 is removed.

**One Review-4 prescription I did not follow.** It asked for `assert solved.success` on the bounded
solve. Implemented, that fires on every `l2=1.0` configuration — but at ridge 1.0 the objective is
identical to eight significant figures at 200, 2000 and 10000 iterations with the same coefficients
pinned at their bound, so the iterate is optimal and only TRF's termination test never fires. Asserting
on the flag discards correct fits. The guard now tests fitted *scale* against the response scale
(`PREDICTION_SCALE_LIMIT`), which catches the ridge-0 divergence and passes the benign case, and trips
zero times across sampled shipped combinations. Consequence worth knowing: **every ridge-1.0 number
reported before this was an unconverged iterate**, including a 1-SE pick that sat there.

## Constraints

- **Sealed:** do not inspect anything under a path containing `targeted_pairwise`, specifically
  `delphi_3e18_targeted_pairwise_phase_order_20260724`.
- Do not reset or discard existing worktree changes; the model and benchmarks are untracked.
- Do not force a positive result. A clean identification boundary is a better outcome than an
  unsupported winner. If the 39-bucket condition cannot be met, the defensible report is that this model
  trades within-panel accuracy for cross-panel extrapolation and two-phase representability, stated as a
  trade rather than dressed up.
- `uv run python`, never bare `python`. `./infra/pre-commit.py` is the lint entry point, never
  `uv run pre-commit`.
