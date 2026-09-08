# Review request: GEN-001 + split head, single-phase

**Date:** 2026-08-24
**Scope:** the general mixture surrogate reduced to single-phase mixtures, as it would be used by the
ATOM-030 prefix search and two-stage optimisation. Two-phase behaviour is explicitly out of scope.

## Why this review

We need a surrogate that selects a good **single-phase** 39-bucket mixture. GEN-001 + the split-damage
head currently leads that task, but it carries twelve searched dimensions, a two-tier ridge and an
SVD-truncated projection, which is a lot of machinery relative to canonical DSP. Before it goes into the
prefix search we want an independent read on whether the complexity is justified and whether the
single-phase reduction is sound.

## The model, single-phase

Notation: `B = 39` buckets, mixture `w_r` on the simplex, `c0`, `c1` epochs per unit weight in each phase.
`c1/c0` is exactly constant per scale (spread ~1e-15), so one scalar `alpha_E = c0/(c0+c1)` governs
everything: 0.798138 at 60M and delphi_3e18, 0.800000 at 300M.

With `w0 = w1 = w`, every feature is a function of total epochs

    E_rb = (c0_b + c1_b) w_rb                  # c0+c1 ranges 4.801 .. 1723.889, a 359x spread
    c0_b w_rb = alpha_E * E_rb

and the model is

    yhat_r =  theta_0 + sum_f theta_f <w>_f                         # free sign
            + sum_f beta_N_f  <(E + delta)^(-gamma_f)>_f            # benefit deficit, per-class exponent
            + sum_f beta_K_f  <exp(-alpha_E E / lambda_s)>_f        # per-stratum exponential
            + sum_f beta_0_f  <D(alpha_E E; tau)>_f                 # bounded harm, scale 1
            + sum_f beta_1_f  <D(E; tau) - D(alpha_E E; tau)>_f     # bounded harm, scale 2
            + sum_b beta_d_b  (E_rb + delta)^(-gamma_f(b))          # per-bucket departure, shrunk hard

    D(e; tau) = u^tau / (1 + u^tau),  u = max(e - 1, 0) / 105,  D in [0, 1)

with all beta >= 0 and `<.>_f` the **mean** over class members.

Estimator: project out the free block with an SVD-truncated projector, column-normalise, then NNLS with
ridge `1e-3 * rho` on the `5F` pooled columns and `rho` on the `B` departures, plus a penalty row per class
shrinking `beta_1` toward `beta_0`. `theta` follows by least squares on the residual. Shape parameters by
differential evolution against out-of-fold SSE on 3 **mixture-blocked** folds.

## Code references

Model core, `experiments/domain_phase_mix/exploratory/two_phase_many/general_mixture_surrogate_20260809.py`:

| what | line |
|---|---|
| `DAMAGE_KNEE = 105.0` | 43 |
| `Panel.exposure(horizon)` | 67 |
| `Panel.early_epochs()` | 71 |
| `Panel.n_exposure_strata(requested=3)` | 74 |
| `Panel.exposure_stratum(n_strata=None)` | 83 |
| `Shape` | 115 |
| `family_sums` (MEAN, not sum) | 126 |
| `design(panel, shape, damage)` | 131 |
| `saturating_damage` | 165 |
| `damage_columns` (blended / physical / split) | 180 |
| `pooled_width` | 206 |
| `column_space` (SVD truncation) | 211 |
| `fit_head` (two-tier ridge, departure rows) | 232 |
| `unpack` / `bounds` | 292 / 310 |

Outer fit, `fit_swarm39_split_damage_20260817.py`: `DEPARTURE_BOUND` 60, `departure_pairs` 68,
`fit_variant` (differential evolution, popsize 8, maxiter 15, sobol init) 76, `predict` 123.

Panels and folds, `swarm39_harness_20260725.py`: `Panel` 95, `load_scale` 263, `mixture_blocked_splits` 487.

Benchmarks: `benchmark_single_phase_surrogates_20260824.py` (scores every model on single-phase held-out
rows, both fit panels, reports selection regret and rank correlation);
`benchmark_dsp_single_phase_ladder_20260824.py` (builds upward from canonical DSP instead).

## What is measured

Held-out rows whose two phases are identical. `regret@1` is the outcome of the top-ranked policy minus the
best observed; `rho` is Spearman.

Under label-free pooling (exposure strata), GEN-001 fitted on the single-phase panel, per cell:

| cell | n | rho | regret@1 |
|---|---|---|---|
| 60m / uncheatable (two-phase fit panel) | 174 | 0.78575 | 0 |
| 60m / table9 (two-phase fit panel) | 173 | 0.69482 | 0 |
| 300m / uncheatable | 397 | 0.96086 | 0.00582 |
| 300m / table9 | 414 | 0.91565 | 0.00646 |
| delphi_3e18 / uncheatable | 425 | 0.95436 | 0.00666 |
| delphi_3e18 / table9 | 425 | 0.93846 | 0.02854 |

Mean margin over `effective_exposure_dsp` is +0.0848 rank correlation, winning 6 of 6 cells. But the top
five configurations span only 0.014 in mean rho (separate_heads 0.9530, general 0.9423,
compact_retained_state 0.9394, bounded_saturation 0.9390, general+quality 0.9386), which at ~0.003 standard
error per cell is one tied group.

Fitting on the single-phase fit panel rather than the two-phase one is worth +0.0497 rank correlation,
helping in 72 of 80 model-cell pairs. That panel (`delphi_3e18_one_phase_fit`, `300m_one_phase_fit`, 280
rows each) is **not** wired into `load_scale`; 60M has no single-phase fit panel at all.

## Known defects — please confirm or refute rather than rediscover

1. **`near` and `late` are exactly collinear on single-phase data.** `exposure(h)` is independent of `h`
   when the mixture is constant, so the two readout blocks are identical to 4.4e-16. The design goes from
   rank deficit 4 (two-phase) to **7 of 55**, decomposing as 1 (free block: family shares sum to one) + 3
   (`<N>_f` lies in the span of the per-bucket `N_b`) + 3 (`N == L`). Only the last is single-phase-specific.
   `column_space` truncates only the FREE block, so the duplicated pair enters NNLS each carrying its own
   ridge; minimising `lambda(b1^2 + b2^2)` at fixed `b1 + b2 = s` gives penalty `lambda s^2 / 2`, so the
   **effective prior on that direction is halved**. A fix must change `design`, `pooled_width` and
   `departure_pairs` together, since the last hardcodes offsets `3F+i` / `4F+i`, and ten scripts import
   `gen.design`.

2. **Two of twelve searched dimensions are inert here.** `near_horizon` because `E(h)` does not depend on
   `h`; `damage_horizon` because the split damage form reads epochs directly and never consults it (it is
   used only by the `blended` variant). They still consume differential-evolution budget.

3. **60M metadata inconsistency.** `Panel.alpha` is 0.800000 at 60M but the exposure columns imply
   0.798138. `alpha` is only read by `Panel.aggregate`, which neither model touches, so no reported number
   is affected — but it is wrong.

## Constraint on classes

The hand-assigned semantic partition (`broad_text` 31 buckets / `tech_code` 6 / `reasoning` 2) is **banned**
and archival only. Legitimate structure is domain classification plus quality splits: the thirteen
`dolma3_cc` topics each carry a `_high` and a `_low` split, giving 13 same-domain pairs and a balanced
high / low / unsplit partition.

Domain identity cannot be the pooling: 13 of 26 groups are singletons, and a singleton's pooled column is
its own per-bucket column, taking the design to 196 columns at rank 141 and one fit to 305s from ~20s.
The same defect hits any model carrying both bucket and class blocks.

Current pooling is equal-count strata on `log10(c0+c1)` (`exposure_stratum`, 13/13/13). Whether that is
admissible is open — it is derived from token counts, not from any label, but it is not "domain plus
quality" either.

## Questions

1. Is the collinearity fix in defect 1 correct, and what is the least invasive version given ten consumers
   of `gen.design`? Is halving the effective ridge on the readout actually harmful here, or is the
   two-tier ridge already compensating by accident?
2. Does the single-phase reduction have identifiability problems we have missed, beyond the three rank
   deficiencies above?
3. Is the two-tier ridge (`1e-3 * rho` pooled, `rho` departures) defensible on its own terms?
4. Is `exposure_stratum` pooling admissible under the class constraint, or should the readout exponent be
   keyed to the quality class (high / low / unsplit) instead?
5. **The main question:** is this complexity justified? Canonical single-phase DSP is
   `b0 - sum_b a_b (1 - exp(-rho_b E_b)) + sum_b p_b softplus(log(1+E_b) - tau_b)^2`, four parameters per
   domain and no classes at all. If GEN-001's advantage is one or two mechanisms, we would rather add those
   to DSP than defend the whole object. Which of its parts would you expect to survive that test?

## Caveats on the numbers

- `effective_exposure_dsp` in the model zoo is **not** canonical DSP. It adds a family-summed benefit block
  canonical DSP lacks, drops the per-domain rate and threshold it has, and is absent from
  `packet_cross_check.csv` (which verifies only compact_retained_state, bucket_family_grp,
  hierarchical_phase_replay, separate_heads). Do not read its numbers as canonical DSP's.
- `regret@1` ties frequently — the held-out optima are coarse — so mean rank correlation is what separates
  models, and differences below ~0.01 are not resolvable at these panel sizes.

## Running the benchmarks yourself

Both scripts are committed and runnable. We started the DSP ladder but stopped it before it finished, so
there are no ladder numbers in this note — the comparison in question 5 is open, not answered.

Score every model on single-phase held-out rows, both fit panels, all three scales:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_single_phase_surrogates_20260824 --scales 60m,300m,delphi_3e18 --extended --grouping strata
```

`--grouping` is `strata` (default, equal-count exposure terciles), `domain` (26 groups; see the rank
warning above) or `semantic` (banned, archival only). `--extended` adds the eight builders that postdate
the model benchmark. The run prints a per-cell table, a cross-cell summary, each configuration against
`effective_exposure_dsp`, and which model-cell pairs the fit-panel switch hurts.

Build upward from canonical single-phase DSP:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_dsp_single_phase_ladder_20260824 --scales 60m,300m,delphi_3e18 --maxiter 50 --restarts 2
```

Rungs are `canonical`, `shared_shape` (ablation down to one rate and one threshold), `bounded_harm` (the
general surrogate's bounded harm at the same one-parameter-per-domain budget), `canonical+pairs` (tie the
two quality splits of a domain) and `bounded_harm+pairs`.

`bounded_harm` is the load-bearing rung. Our hypothesis is that DSP's unbounded quadratic penalty is what
makes it mis-select rather than mis-rank: at 3e18 Uncheatable the zoo variant's rank correlation is only
0.076 below the leader while its selection regret is twenty times worse, which is the signature of a model
that orders the bulk correctly and puts its minimum in the wrong place. These panels reach 91 epochs at the
median policy and 283x oversampling, and the head normalises columns by their TRAINING norm, so a test row
past that range is amplified quadratically. Please treat that as a hypothesis to test, not a finding.

Measured timings on one laptop core, so you can size the run: a GEN-001 fit at three classes is about 20s
and at twenty-six classes 305s; a per-domain DSP rung is about 34s at `--maxiter 25` and about 70s at 50,
while `shared_shape` is under 2s. The full ladder over three scales is roughly 50 fits.

Two practical notes. The ladder prints its tables only after every fit completes, so do not pipe it
through `head`/`tail` — that buffers the whole run and then truncates the tables, which cost us two runs.
And `--restarts 1` was used in our aborted run to save time; if a rung looks bad at one restart, treat its
number as a floor rather than its true performance and re-run that rung with more.
