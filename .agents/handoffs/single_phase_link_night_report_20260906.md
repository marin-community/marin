# Overnight report: log-deficit link, coupling, and Table-9 selection (2026-09-06/07)

Scope: everything done after Calvin went to sleep with the instruction to submit more 3e18 validation runs
like `exp_01m1vbajfrdebbsatntj4sm00z` and to try new ideas for the Table-9 optimum. All surrogate fits use the
frozen 280-row panel of `delphi_offline_selection_20260906` and its partitions; selection metrics are scored on
its frozen bank with `score_delphi_selection_20260906`. One Iris submission was made (section 4). Nothing else
was launched.

## 0. TLDR

- The bounded log-deficit link (round 2's `weibull_softplus_unscaled@log_deficit_bounded_link`) is the
  best-calibrated surrogate on the archive by a wide margin (optimism −0.005 / −0.015 BPB against WSPU's
  +0.036 / +0.070; RMSE 0.015 / 0.021 against 0.029 / 0.038) and moves the pooled Table-9 pick from rank 14
  to rank 10 (regret 0.0157 → 0.0143), but within source blocks its Table-9 regret is worse by +0.0036
  [+0.0009, +0.0071]. Uncheatable selection is unchanged. Calibration and selection come apart for the sixth
  time; the link is the strongest calibration mechanism so far.
- Its two-bucket failure (worse than DSP on 15 of 45 StarCoder curves) is entirely the held-out p = 0 corner:
  interior and p = 1 are at least as good as WSPU; exponentiating an extrapolated benefit column overshoots at
  zero exposure (4.2 predicted against 1.7 observed on the 4× replay curve).
- Codex's post-hoc product on the fitted WSPU curves is a third instance of the same null; stacking it on the
  link over-corrects. Blends of WSPU and link predictions interpolate calibration and do not change picks.
- Five validation runs were submitted: the link's three optima (Uncheatable cap 6, Table 9 caps 6 and 8;
  `/calvinxu/dm-delphi-3e18-lwspu-link-v6e8-20260906`, Fieldbook `exp_01m1vdtb243bg75c233y34chr8`; expected
  gains 0.003–0.005 BPB) and the two Table-9 optima of the link with hub interactions
  (`/calvinxu/dm-delphi-3e18-lwspu-linkhub-v6e8-20260906`, Fieldbook `exp_01m1vg9rc5yzx2htrgm9wn7x9p`;
  expected gain 0.0066 BPB at both caps).
- Two link variants with one extra parameter each (floor fraction by inner CV; hub interactions under the
  link) and a named-pair synergy variant (three bucket pairs chosen from panel out-of-fold residuals) were
  screened on the same benchmark: see sections 5 and 6.
- An 18-run resolution-V factorial around the replicated Table-9 frontier centre was designed and, after
  Calvin's approval, launched (section 7). It measures the pairwise synergies that no O(M) surrogate can
  learn from the panel.
- After the launches (section 8): shrinking unpredictable Table-9 components out of the pick is null, the
  Uncheatable value is no proxy for the Table-9 optimum (Spearman 0.48), and the ordering skill of every
  panel-fitted surrogate ends inside the 30 best-measured Table-9 mixtures (sign accuracy 0.46–0.67 against a
  0.92 noise ceiling; reversed inside the 10 best), while a held-out-source kernel on the bank orders that band
  at 0.79 and picks rank 3. The surrogate finds the basin; measured neighbours must order its floor. Neighbour
  forecasts for all queued runs are in `queued_run_forecasts.csv`.

## 1. Link on the frozen selection benchmark (`delphi_link_selection_20260906`)

Optima stratum, point policy, external bank (170 Uncheatable, 157 Table-9 coordinates). WSPU refit here
reproduces the reference package to 2e-16.

| target | model | regret@1 | best-of-5 | best-of-10 | rank | optimism | RMSE | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Uncheatable | WSPU | 0.0023 | 0.0012 | 0 | 5/170 | 0.036 | 0.029 | 0.875 |
| Uncheatable | WSPU + Codex coupling κ=1 | 0.0023 | 0.0012 | 0 | 5/170 | 0.032 | 0.028 | 0.884 |
| Uncheatable | bounded log-deficit link | 0.0030 | 0.0016 | 0 | 6/170 | −0.005 | 0.015 | 0.924 |
| Uncheatable | link chosen by inner CV | 0.0035 | 0.0012 | 0 | 7/170 | 0.015 | 0.017 | 0.935 |
| Uncheatable | bounded link + Codex coupling κ=1 | 0.0035 | 0.0012 | 0 | 7/170 | 0.012 | 0.019 | 0.915 |
| Table 9 | WSPU | 0.0157 | 0.0132 | 0 | 14/157 | 0.070 | 0.038 | 0.894 |
| Table 9 | WSPU + Codex coupling κ=1 | 0.0157 | 0.0132 | 0.0082 | 14/157 | 0.058 | 0.035 | 0.901 |
| Table 9 | bounded log-deficit link | 0.0143 | 0.0143 | 0.0132 | 10/157 | −0.015 | 0.021 | 0.922 |
| Table 9 | link chosen by inner CV | 0.0143 | 0.0143 | 0.0132 | 10/157 | 0.003 | 0.020 | 0.930 |
| Table 9 | bounded link + Codex coupling κ=1 | 0.0143 | 0.0143 | 0.0132 | 10/157 | 0.006 | 0.019 | 0.924 |

Source-block contrasts against WSPU (14 Table-9 / 15 Uncheatable blocks, descriptive bootstrap): bounded
link Table-9 regret +0.0036 [+0.0009, +0.0071], RMSE −0.015 [−0.021, −0.009], optimism −0.045 [−0.056,
−0.032]; Uncheatable regret 0.0000 [−0.0005, +0.0005]. Per component on the panel the link lowers
out-of-fold RMSE on 21 of 25 QA tasks (median 0.066 → 0.062), 13 of 19 code and 4 of 7 math (unchanged
medians), and 5 of 7 Uncheatable components (0.0117 → 0.0091).

## 2. Where the link fails on two buckets (`plot_starcoder_link_gate_20260906.py`)

Out-of-fold RMSE by position on the 45 curves: interior link 0.027 / WSPU 0.032 / DSP 0.035; p ≥ 0.8
0.088 / 0.081 / 0.100; p = 1 0.213 / 0.286 / 0.275; p = 0 0.678 / 0.312 / 0.501. Of the link's excess
squared error over WSPU, 112% is at p = 0. The 15 losing curves are nine of the ten matched-ladder curves,
five replay curves at 2× and 4× repetition, and one onset curve. Figures in
`reference_outputs/starcoder_link_gate_plots_20260906/`.

## 3. Coupling, blends, and prior work

Codex's fixed product `A[1 + (Π(1 + κΔ_b/A) − 1)/κ]` on WSPU's fitted per-bucket curves, the round-2 links,
the round-4 hub interactions and pooled law, and the August deficit links all improve calibration far from the
panel and none changes the selected optimum (memory `coupling-variants-tested`). Blends `(1−w)·WSPU + w·link`
for w in {0.25, 0.5, 0.75}: regret within noise of WSPU on both targets, optimism interpolating.

## 4. Submitted validation runs

`/calvinxu/dm-delphi-3e18-lwspu-link-v6e8-20260906` (13:17 UTC), parent us-east5-a, children v6e-8 in
us-east5-b, max_concurrent 3; candidates `lwspu_u_bl_cap06` (seed 666200), `lwspu_t9_bl_cap06`,
`lwspu_t9_bl_cap08` (seed 662009). Package
`reference_outputs/delphi_link_validation_3e18_20260906/` (SUBMISSION.md, materialization with parity
< 2e-15, dry run, `east5_launch_safety --expected-child-zone us-east5-b` passed, seven launcher tests, lint
OK). Fieldbook: experiment `exp_01m1vdtb243bg75c233y34chr8`, parent job `job_01m1ve6jzc5yxk5zmchxdc5kwb`,
runs `run_01m1ve9cayczabx9ygc6c1837w` / `run_01m1ve9d5adybcfr035vcey561` / `run_01m1ve9dytnmmce7sky0h5d1nf`.
Predicted (link) 0.9870 / 1.0841 / 1.0839; the link predicts the κ-0 WSPU policies at 0.9902 / 1.0866 /
1.0888 and WSPU predicts the link optima at 0.9546 / 1.0155 / 1.0150. TV between the link and WSPU optima
0.14 / 0.11 / 0.16; effective buckets 11.7 / 13.5 / 13.6. Controls: `wspu_uncheatable_cap06` 0.9834,
`wspu_table9_cap06` 1.0722, the cap-8 control of the same sweep, and the nine coupling runs. At submission
all twelve children were pending v6e-8 capacity (scheduler short of CPU on matching workers, autoscaler tier
block); one child had started by 13:40 UTC.

## 4b. Submitted validation runs, second batch (link + hub interactions)

`/calvinxu/dm-delphi-3e18-lwspu-linkhub-v6e8-20260906` (13:59 UTC), two Table-9 runs at seed 662009:
`lwspu_t9_bh_cap06` and `lwspu_t9_bh_cap08`, the optima of
`weibull_softplus_unscaled@log_deficit_bounded_link_total_hub` (section 5), predicted 1.0747 / 1.0737 against
its own predictions of 1.0813 / 1.0804 for the κ-0 WSPU policies (expected gain 0.0066 / 0.0067 BPB; WSPU
predicts them at 1.0196 / 1.0156). Package `reference_outputs/delphi_link_hub_validation_3e18_20260906/`
(SUBMISSION.md; parity < 1e-15; SLSQP needed a coarser-step polish, after which all restarts converged with
endpoint spreads 0.0011 / 0.0025); Fieldbook `exp_01m1vg9rc5yzx2htrgm9wn7x9p`, parent job
`job_01m1vg9rqgqac61w5c7z0rhq29`. Fourteen validation children are now queued behind v6e-8 capacity.

## 5. Link variants: floor fraction by inner CV, hub interactions under the link

Screen `delphi_link_variants_selection_20260906` (same protocol; `GridModel.head_for` now reads a
`floor_fraction` shape key, grid {0.5, 0.8, 0.95, 0.99}; the hub variant adds Scheffé products of the total
benefit signal with each bucket's under the bounded link).

| target | model | regret@1 | best-of-5 | best-of-10 | rank | optimism | RMSE | Spearman | source-block regret vs WSPU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Table 9 | bounded link | 0.0143 | 0.0143 | 0.0132 | 10/157 | −0.015 | 0.021 | 0.922 | +0.0036 [+0.0009, +0.0071] |
| Table 9 | link, floor by inner CV | 0.0151 | 0.0143 | 0.0132 | 12/157 | +0.006 | 0.021 | 0.936 | +0.0001 [−0.0019, +0.0024] |
| Table 9 | link + hub interactions | 0.0140 | 0.0132 | 0.0084 | 9/157 | −0.008 | 0.023 | 0.916 | −0.0005 [−0.0017, +0.0001] |
| Uncheatable | bounded link | 0.0030 | 0.0016 | 0 | 6/170 | −0.005 | 0.015 | 0.924 | 0.0000 [−0.0005, +0.0005] |
| Uncheatable | link, floor by inner CV | 0.0035 | 0.0012 | 0 | 7/170 | +0.009 | 0.016 | 0.936 | −0.0002 [−0.0006, +0.0001] |
| Uncheatable | link + hub interactions | 0.0030 | 0.0023 | 0 | 6/170 | −0.010 | 0.019 | 0.895 | +0.0004 [−0.0002, +0.0012] |

Panel out-of-fold: Table-9 RMSE / Spearman 0.0302 / 0.828 (WSPU), 0.0261 / 0.871 (link), 0.0256 / 0.869
(floor by CV), 0.0241 / 0.881 (link + hub). The link with hub interactions is the best Table-9 selector the
archive has seen (rank 9, best-of-10 0.0084) and the first variant whose within-block regret contrast against
WSPU is negative, although its interval still touches zero; it is the candidate materialized for validation in
section 4b. The CV-selected floor removes the fixed link's within-block regret penalty at the cost of one rank.
Every difference here is one or two neighbours on the archive, inside the seed noise of single-run
coordinates.

## 6. Named-pair synergy columns

Panel out-of-fold residuals of WSPU and the link, after regressing on the twelve highest-variance bucket
weights, correlate with three pair products: synthetic QA × synthetic instruction (partial ρ −0.43 WSPU, −0.35
link; positive synergy), synthetic math × synthetic instruction (+0.27 / +0.26), synthetic instruction × arXiv
(+0.30 / +0.28). The bank's residual structure near the frontier is different (code × synthetic QA and
synthetic QA × science-math web under the link, both negative, i.e. unmodelled positive synergy), which is
why a designed experiment there is proposed in section 7. Variants `@named_pairs` and
`@log_deficit_bounded_link_named_pairs` add signed product columns for the three panel-chosen pairs.

Screen (`delphi_link_pairs_selection_20260906`): on the panel the pairs carry out-of-fold information, as the
residual analysis promised (WSPU RMSE 0.0302 → 0.0271, Spearman 0.828 → 0.847; link 0.0261 → 0.0248,
0.871 → 0.881). On the archive they change nothing: `@named_pairs` keeps WSPU's pick on both targets
(regret 0.0157 / 0.0023) with identical calibration, and `@log_deficit_bounded_link_named_pairs` keeps the
link's calibration (optimism −0.013 / −0.004) while its Table-9 pick moves from rank 10 to 13 (regret
0.0155); source-block contrasts against WSPU are all within noise for regret. The synergies the panel can
identify are not the ones that matter near the frontier.

Also tried without refitting: the link fitted directly on the aggregate (`link_direct_macro`) has Table-9
regret 0.0140 at rank 9 with optimism −0.012 and RMSE 0.026, and Uncheatable regret 0.0023 at rank 5 with
optimism −0.005; the same one-neighbour move the reference's `wspu_direct_macro` makes, now calibrated.

## 7. Frontier factorial (launched 14:14 UTC after Calvin's approval)

`design_delphi_frontier_factorial_20260906.py` → `reference_outputs/delphi_frontier_factorial_design_20260906/`.
Centre `a1a917b1…` (26 runs, 1.0639 ± 0.0041). Five factors (code ±0.02+0.02, synthetic QA ±0.04, CC-HQ
±0.04, synthetic reasoning ±0.006×3, olmOCR ±0.015 + arXiv ±0.005), mass balanced against the 26 CC cells;
2^(5−1) with E = ABCD plus two centre replicates = 18 runs (≈6e19 FLOPs), TV 0.09–0.16 from the centre, all
rows within 16 epochs. Effect SE ≈ 0.0019 BPB, so 0.005-BPB interactions are detectable. It would give the
first measured second-order model of the frontier region, which is what round 6 said was missing. Calvin
approved it at about 14:05 UTC; it was submitted as `/calvinxu/dm-delphi-3e18-frontier-factorial-v6e8-20260906`
(Fieldbook `exp_01m1vh5s802cj685fbzcx3w475`) with `launch_delphi_frontier_factorial_3e18.py`: all rows at data
seed 662009, the second centre replicate at trainer seed 1 in its own table because the sweep loader aliases
identical mixtures. Candidate ids carry a `_cap16` suffix. Package and analysis plan:
`reference_outputs/delphi_frontier_factorial_design_20260906/SUBMISSION.md`.

## 8. Offline screens after the launches (not blocked by the validations)

Calvin's instruction on waking briefly: "try more ideas that aren't blocked by the validation results". Three
screens on the frozen benchmark, no fits beyond the existing shards, no launches.

**10a. Skill-weighted component aggregation (`evaluate_delphi_component_shrinkage_20260906.py`,
`reference_outputs/delphi_component_shrinkage_selection_20260906/`).** The Table-9 mean averages 51 components
whose out-of-fold R² ranges from below zero to 0.90; a component the surrogate cannot predict adds noise to the
pick. Each component's bank prediction was shrunk toward its panel mean by a pre-specified function of its
out-of-fold R² (soft, squared, thresholds 0.25/0.5/0.75, top half), for WSPU, the bounded link, the CV-floor link
and the link with hub interactions. Null: every soft rule keeps the pick, rank and best-of-k of its base model on
both targets; hard thresholds worsen the pick (rank 14 → 14–24) and the panel Spearman (0.83 → 0.64–0.78). The
low-skill components still carry ordering information in aggregate, and the near-frontier misordering is not
component noise.

**10b. Cross-target proxy.** The measured Uncheatable value orders the Table-9 optima at Spearman 0.48; picking
the Table-9 optimum by the measured Uncheatable value has regret 0.032 (rank 51 of 157). Dead.

**10c. Where the ordering skill ends (`analyze_delphi_top_band_ordering_20260906.py`,
`reference_outputs/delphi_top_band_ordering_20260906/`, figure `top_band_sign_accuracy.png`).** Regret is decided
inside the band of coordinates already near the optimum. On the 157 Table-9 optima sorted by measured value,
pairwise sign accuracy on pairs differing by more than one run SD:

| band | width (BPB) | WSPU | DSP | OLMix | bounded link | link + hub | rank ensemble (WSPU, DSP, OLMix) | bank kernel LOSO, TV 0.05 | noise ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 best | 0.014 | 0.24 | 0.20 | 0.24 | 0.08 | 0.16 | 0.16 | 0.64 | 0.91 |
| 30 best | 0.023 | 0.67 | 0.66 | 0.50 | 0.46 | 0.63 | 0.65 | 0.79 | 0.92 |
| 100 best | 0.061 | 0.84 | 0.85 | 0.79 | 0.81 | 0.86 | 0.87 | 0.82 | 0.97 |
| all 157 | 0.253 | 0.89 | 0.91 | 0.89 | 0.91 | 0.91 | 0.92 | 0.89 | 0.99 |

Every panel-fitted model is at chance or below inside the 30 best (source-block bootstrap intervals all cover
0.5; within-block pairs 0.9, cross-block pairs 0.46–0.66), and reversed inside the 10 best. On Uncheatable the
same 30-best band is ordered at 0.88 by WSPU (ceiling 0.96). The one predictor with skill in the Table-9 band is
the bank itself: a Nadaraya–Watson smoother on TV distance with bandwidth 0.05, each coordinate predicted from
other source blocks only, reaches 0.79 [0.62, 0.94] and ranks 3rd (regret 0.008); bandwidth 0.1 gives 0.68 and
0.2 gives 0.56. Split-half check (bandwidth chosen on half the band's source blocks, scored on the other half):
+0.06 [−0.13, +0.23] over WSPU on Table 9, better in 72% of splits; on Uncheatable −0.10, WSPU wins. WSPU plus a
kernel-smoothed residual is worse than the kernel alone (0.63), because the residual inherits each model's
optimism about its own optima.

Diagnosis inside the band: measured BPB falls with code share (Spearman −0.67), synthetic-reasoning share
(−0.53) and the 90th-percentile epochs (−0.61), and rises with Common Crawl share (+0.44); WSPU's prediction
follows code even more strongly (−0.84) and its residual grows with code share (+0.64): it over-credits code, and
its own sweep coordinates carry the largest residuals (+0.06–0.07 against +0.02–0.03 for other campaigns'
optima). Inside the band the ordering is the optimizer's curse of whichever model proposed each coordinate.

Two-stage policy (`two_stage_policy.csv`): let the surrogate choose the basin (its predicted top-k of the optima
stratum) and the held-out-source kernel choose inside it (lowest forecast among shortlist members with kernel
mass ≥ 0.5). Table-9 regret@1 with bandwidth 0.05: WSPU 0.0157 (k = 5) → 0.0082 (k = 10–30, rank 3); DSP
0.0151 → 0.0082 (k ≥ 20); hub 0.0140 → 0.0082 (k ≥ 20); the bounded link needs k = 30. Bandwidth 0.1 gives
0.0132–0.0140 at every k. On Uncheatable the policy never beats the plain pick (0.0023 → 0.0023–0.0089). The
shortlist size and bandwidth are a reported grid, not a selection; the split-half check above is the only
out-of-sample evidence for the bandwidth.

Cross-scale check (`cross_scale_band_ordering.csv`, from the round-3 canonical held-out predictions): on the
60M and 300M held-out banks the 30 best coordinates are ordered at 0.92–0.96 (ceilings 0.96–0.97) by WSPU, DSP
and OLMix on both targets, but those banks have no dense floor: their top-30 bands span 0.05–0.11 BPB of sampled
and proportional-perturbation runs, whereas the Delphi Table-9 band is 0.023 BPB of competing optima. The
failure is a property of a floor densely populated by model optima, not of the Delphi scale as such; whether a
60M or 300M floor of optima would behave the same is untested.

A five-factor response surface fitted on the bank's own near-centre optima (projections of each coordinate onto
the factorial's five factor directions, ridge regression with and without pairwise interactions, radius 0.2–0.5
in TV, held-out source blocks) orders the top-30 band at 0.66–0.72 with main effects only and at 0.52–0.66 with
interactions: the unplanned bank coordinates do not identify the pairwise terms, which is what the designed
factorial is for.

Consequences. (i) The surrogate finds the basin and cannot order its floor on Table 9; the floor must be ordered
by measured neighbours, which is what the factorial supplies and what a local kernel or quadratic on those runs
will do. (ii) For the paper's Analysis section this is the clean form of the calibration-versus-selection
statement (outline note added). (iii) `forecast_delphi_queued_runs_20260906.py` writes the bank kernel's
neighbour forecast for every queued run (`queued_run_forecasts.csv`): link Table-9 optima 1.0737 (own prediction
1.084), hub 1.0728 / 1.0729 (own 1.0747 / 1.0737), coupling 1.073 (all within TV 0.03 of the WSPU sweep), centre
1.069 (26-run mean 1.0639; the kernel is pulled by neighbours at 1.07–1.08), factorial corners 1.066–1.084 with
the best forecasts on the corners that add code and synthetic reasoning and remove CC-HQ. The morning comparison
should report realized values against both the model's prediction and this forecast.

### 8d. Prepared, not submitted: floor replicates (`reference_outputs/delphi_floor_replicates_design_20260907/`)

The measured answer to "which mixture is best" at the noise of the bank is replication. The five best
optima-stratum coordinates with at most two runs (1.0579 HPR-280 tied control, 1.0660 hpr-300m-to-3e18, 1.0663
decoupled-phase, 1.0664 aggregate-V cap sweep, 1.0678 symmetric-sepheads frontier) would each get two more runs
at data seed 662009 (trainer seeds 0 and 1): ten runs, about 3.3e19 FLOPs, giving each candidate a three-run
mean and a difference from the 26-run centre with SE about 0.0024 BPB. `design_delphi_floor_replicates_20260907.py`,
`launch_delphi_floor_replicates_3e18.py`, three passing tests, a dry run and a passed safety check are in the
package; SUBMISSION.md holds the exact command. Not submitted, because 23 children were still queued behind
v6e-8 capacity; it is Calvin's call whether ten more are excessive.

### 8e. The link's floor (2026-09-06 afternoon, with Calvin; Fieldbook `exp_01m1wg9z23vmyjx5zfr7n4v5wa`)

The link fits log(y − φ) with φ = 0.95 × the task's swarm minimum. The measured optima already sit below that
floor on code: GitHub C++ under the link's Uncheatable optimum is at its floor to four decimals (0.7609 vs 0.7605),
WSPU's optimum measured 0.748; the link's Table-9 cap-8 optimum is below the swarm minimum on 28 of 51 tasks and
below the floor on 11 (all mt_mbpp code tasks, 7–9% below the swarm minimum). Against the bank's per-task minima
(`check_delphi_link_floors_20260907.py`): the 0.95 rule is above the bank minimum on 21 of 51 Table-9 tasks (worst
excess 0.096 BPB) and 2 of 7 Uncheatable tasks. Proportional-anchored floors φ = prop − κ(prop − swarm min) with
one κ: κ 2 fails 4/51 and 4/7, κ 2.5 fails 3/51 and 3/7; the empirical κ of the bank minimum is 1.3–1.5 for code,
math and MMLU, and up to 6.6 for tasks whose swarm gap is within run noise (AO3, BBC News, jeopardy), where the
bank's targeted runs moved them 6× further than any swarm row. Aggregate κ: 2.0 (Uncheatable), 1.7 (Table-9
mean); κ-2 aggregate floors stay above the next rung's proportional. Decision: parametrize the floor relative to
proportional and the swarm gap with a group-level κ fitted continuously with the heads, add a noise-based
margin for tasks the swarm never moved, keep the extrapolation cap, and check every fit against the bank minima
and the next rung. Text entropy is a bound only.

### 8f. Frontier factorial: measured (all 18 runs, 2026-09-06 evening; `analyze_delphi_frontier_factorial_20260906.py`)

Table-9 mean, effects in BPB per ±δ step (SE 0.0019): code A −0.0065 (t −3.4), synthetic QA B −0.0080 (t −4.2),
synthetic reasoning D −0.0040 (t −2.1), CC-HQ C −0.0008, PDF/arXiv E −0.0014. All ten two-factor interactions are
within one SE (|effect| ≤ 0.0017): the response is additive at this step size around the centre, so the O(M)
form is not what fails locally; the levels are. Centre pair (trainer seeds 0/1, same data seed) 1.0642 / 1.0615,
difference 0.0027, against the 26-run mean 1.0639. Best corner `fac_ppppp` (every factor at +1) measured 1.0557,
the lowest single-run Table-9 value in the bank (HPR-280 control 1.0579), 0.008 below the centre; the fitted
model ranks it first and predicts 1.0544 for the 1.5× main-effect step and 1.0486 / 1.0408 for the corner at
1.5× / 2× (extrapolations, kernel forecasts flat at 1.066 with no support). Uncheatable (secondary): code +0.0019
and synthetic reasoning +0.0024 hurt, PDF/arXiv −0.0038 helps; the two objectives trade off along code and
reasoning. `proposal_ranking.csv` and `proposal_candidate_weights.csv` hold the six best feasible unmeasured
proposals in the launcher schema. Reading: at the best-measured region the direction to move is more code,
more synthetic QA and more synthetic reasoning, which is the direction the link's optimum took on QA (0.16
against WSPU's 0.13) and WSPU did not.

### 8g. The fitted-floor link (2026-09-07, Fieldbook `exp_01m1wg9z23vmyjx5zfr7n4v5wa`)

Implementation (`single_phase_observatory_models_20260902.py`): `LinkKind.KAPPA_FLOOR` (log-space NNLS at the floor
prop − κ(prop − swarm min), used to select shape and ridge at the prior κ 2.5) and `LinkKind.FITTED_FLOOR`
(response-space bounded least squares at a fixed κ, warm-started from the log-space fit, ridge rows rescaled by
the mean deficit, cap kept); `FittedFloorModel` chooses per task among four heads at the selected shape and
ridge by inner CV: identity, the fixed 0.95 log link, the log-space fit at a searched κ, the response-space fit
at a searched κ (bounded scalar search on log κ in [1, 8]). Anchors (`delphi_floor_anchors_20260907/anchors.csv`)
are the reliability proportional means and repeat SDs per component. Registry ids
`weibull_softplus_unscaled@fitted_floor_link` (per task), `@fitted_floor_link_group` (one κ per family),
`@fitted_floor_link_three` (no fixed-floor head), `@fitted_floor_link_permissive` (one-SE upper rule, from
DeepSeek's review). Materializer handles the model (`materialize_delphi_link_validation_20260906.py`, `--kl`).

Screen (`delphi_fitted_floor_selection_20260907`): picks identical to the bounded link (ranks 6 and 10);
optimism at the picks +0.0037 / −0.0009 (link −0.0049 / −0.0145, WSPU +0.036 / +0.070); archive RMSE 0.0145 /
0.0207; panel OOF RMSE 0.0078 / 0.0257 (best on both); within-block Table-9 regret vs WSPU +0.0013 [−0.0005,
+0.0034] (the fixed link's +0.0036 [+0.0008, +0.0069]). Choices: 27 log-space κ, 22 response-space κ, 1 identity,
1 fixed floor on Table 9; per-task κ medians code 1.49, math 1.43, MMLU 2.24, QA 2.94, science 3.86, tracking the
bank's empirical headroom (1.37 / 1.18 / 1.34 / 1.32). Fresh-run calibration
(`predict_delphi_fresh_runs_20260907.py`, the 14 validation runs of 2026-09-06, seen by no model): bias / RMSE
−0.0010 / 0.0026 (Uncheatable) and −0.0066 / 0.0079 (Table 9) against the bounded link's −0.0053 / 0.0054 and
−0.0181 / 0.0183 and WSPU's +0.0298 / 0.0301 and +0.0460 / 0.0507. Floor-region calibration (bottom decile of
each task's bank values, `analyze_delphi_floor_region_calibration_20260907.py`): Table-9 RMSE 0.041 (link 0.048,
WSPU 0.080; code tasks 0.031 / 0.047 / 0.120), paired |residual| vs WSPU −0.025 [−0.042, −0.011]; Uncheatable
0.018 (link 0.015). Bank gate: 7 of 51 Table-9 and 1 of 7 Uncheatable fitted floors above the bank minimum
(0.95 rule: 21 and 2). Band ordering unchanged. Group κ: same picks, within-block +0.0022 [+0.0003, +0.0045].

Materialized optima (KL 0, `delphi_fitted_floor_validation_3e18_20260907`): `lwspu_u_ff_cap06` predicted
0.9757 (8.0 effective buckets, hull distance 0.50), `lwspu_t9_ff_cap06` 1.0711, `lwspu_t9_ff_cap08` 1.0708 (about
what it predicts for the bounded link's measured optima, 1.0717–1.0719, so it claims no gain over them). KL 0.05
(`..._kl05_...`): 1.0000 / 1.0938 / 1.0938, 0.02–0.03 worse, hull distance 0.15; Calvin's preregistered
hypothesis is that the penalty hurts. Both launchers (`launch_delphi_fitted_floor_validation_3e18.py`,
`launch_delphi_fitted_floor_kl05_validation_3e18.py`) have passing tests, dry runs and safety checks; not
submitted. DeepSeek review in `reviews/deepseek_review_20260907.md`: the inner-CV κ is a curvature choice inside
the swarm, not evidence about the floor at the optimum; use the permissive one-SE rule; check solver convergence;
the anchors' MMLU names were wrong (bare names in the panel) and the registry fell back silently; fixes applied in
round 2.

### 8h. One head suffices; the kappa-floor optima are in validation (2026-09-07 afternoon)

Single-head screens (`delphi_single_head_selection_20260907`): the log-space κ-floor head alone
(`weibull_softplus_unscaled@kappa_floor_link`, κ per task searched up to 100 so the additive form is its large-κ
end) matches the multi-head model on selection (ranks 6 / 12, within-block Table-9 regret vs WSPU +0.0008
[−0.0004, +0.0026], the best of the links), has the best archive Spearman (0.928), the best floor-region error
(RMSE 0.039; paired |residual| vs WSPU −0.0275 [−0.0440, −0.0114]), the cleanest bank gate (2 of 51 and 1 of 7
floors above the bank minimum) and the best fresh-run calibration on the 14 unseen runs (bias +0.001 / +0.007,
RMSE 0.006 / 0.007). κ never reaches its bound. The response-space head alone fails the noisy QA tasks again and
its refit is not reproducible to 1e-8, so it is retired; the three-head and four-head models add nothing; the
permissive one-SE rule returns WSPU's optimism (+0.044 on Table 9) with no selection gain. Decision: the
kappa-floor link is the successor candidate, one convex head with one searched scalar per task.

Validation submitted (six runs, `exp_01m1wmyw6n47v8a8b0wfvsd36b`, `exp_01m1wmyxpqrmq25n4q7ck9ch1g`):
`/calvinxu/dm-delphi-3e18-lwspu-kappafloor-v6e8-20260907` (KL 0: `lwspu_u_kf_cap06` predicted 0.9711, 8.0
effective buckets, hull distance 0.50; `lwspu_t9_kf_cap06` 1.0631; `lwspu_t9_kf_cap08` 1.0623; TV 0.03–0.08 from
the three-head model's optima) and `/calvinxu/dm-delphi-3e18-lwspu-kappafloor-kl05-v6e8-20260907` (Olmix-style
KL 0.05: 0.9953 / 1.0876 / 1.0875, predicted 0.02–0.03 worse, Calvin's preregistered hypothesis). Both in the
collector (`--launch kappa_floor`, `--launch kappa_floor_kl05`); monitor running. The earlier three-head
launchers were prepared and not submitted.

### 8i. Kappa-floor validation measured (2026-09-07 20:26 UTC)

`lwspu_t9_kf_cap06` **1.0613** (predicted 1.0631): below the WSPU cap-6 control (1.0722), the bounded link's
optima (1.0651 / 1.0636), the 26-run centre (1.0639) and Olmix (1.0769); the best surrogate-proposed Table-9
mixture measured so far (single run, SD 0.0038). `lwspu_t9_kf_cap08` 1.0672 (predicted 1.0623; control 1.0736).
`lwspu_u_kf_cap06` **0.9890** (predicted 0.9711; control 0.9834; bounded link 0.9820): a 0.018 miss. Per component,
the misses are the tasks whose inner-CV κ ran large: arxiv_physics κ 17 (floor 0.35, effectively additive) predicted
0.965, measured 1.017; bbc_news (κ 9) +0.028; github_python +0.026; wikipedia (κ 7.7) +0.013; the code tasks with
κ 2–3 were within 0.003–0.026. The mixture had 19% olmOCR PDFs, 36% on two high-quality CC cells, 0.1% Wikipedia,
2.4% arXiv (8 effective buckets, hull distance 0.50). Reading: a floor too low is not safe either; large κ makes
a task additive and returns WSPU's over-extrapolation there, which is the risk DeepSeek's permissive direction
ignores. Fix under test: cap κ at 6 (the largest empirical headroom the bank has shown) and shrink flat-profile
tasks toward their group median. KL 0.05 batch: Uncheatable 0.9916 (predicted 0.9953), Table-9 runs training.

## 9. Morning checklist

1. Validation results: five new runs (`lwspu_u_bl_cap06`, `lwspu_t9_bl_cap06/08`, `lwspu_t9_bh_cap06/08`)
   behind the nine coupling runs; compare each Table-9 run with `wspu_table9_cap06` 1.0722 (seed 662009),
   the cap-8 WSPU control, and the replicated frontier 1.0639 ± 0.0041; a gain beyond one repeat SD (0.0038)
   at cap 6 would be the first measured selection gain from a calibration mechanism. The hub surrogate's own
   forecast is 1.0747 / 1.0737, the link's 1.0841 / 1.0839.
2. The frontier factorial is queued (section 7); its analysis script is `analyze_delphi_frontier_factorial_20260906.py`
   once results land.
3. If the hub variant's runs come in below 1.072, consider promoting
   `weibull_softplus_unscaled@log_deficit_bounded_link_total_hub` in the benchmark's finalist protocol (five
   repeats, StarCoder gate) before any paper claim; its two-bucket behaviour was not run tonight.
4. Registry/model changes are backward compatible (pins refreshed; 24 model tests and all launcher tests pass).
   The branch-wide `run_tests.py` (2859 passed, 23 failed, 13 collection errors) has exactly two failures that
   touch tonight's changes: the helper-pin test copied into the frozen packages
   `delphi_offline_selection_20260906/reproduction_sources/tests/` and
   `delphi_coupling_followup_20260906/reproduction_sources/tests/`, which compare the live `head_for` and
   `family_design` sources against the pins frozen inside those packages. They are provenance copies, not live
   tests; the live pin file is refreshed. The other failures (datakit store, lm_eval alias patches, starcoder
   dense-surface manifests, run-registry name patterns, frontier-fiber panel, grug eval) do not import the
   modules changed tonight.
5. Results collection is one command: `uv run collect_delphi_3e18_validation_results_20260906.py` (all three
   launches; Uncheatable from GCS endpoints, Table-9 from the W&B evaluation groups), then
   `analyze_delphi_frontier_factorial_20260906.py` for the factorial's effects table.

- Once all sixteen corners are measured, `analyze_delphi_frontier_factorial_20260906.py` also writes
  `proposal_ranking.csv` (all 32 corners of the factor box, the main-effect step, and 1.5x/2x extrapolations,
  each with the fitted prediction, feasibility under the 16-epoch cap, and the bank kernel's neighbour forecast)
  and `proposal_candidate_weights.csv` (the six best feasible unmeasured proposals in the launcher schema,
  ids `prop_<signs>_x<scale>_cap16`). Treat the scaled steps as extrapolations; the defensible next runs are the
  best unmeasured corner and the 1x main-effect step. Tested end to end on synthetic responses with known effects.
- The three parents (link, hub, factorial) and Calvin's coupling parent were preempted and restarted; at 18:54 UTC
  the executors re-created every child with new Iris job ids (the first attempts show as `killed: Parent task
  preempted`, the new ones `pending: Insufficient TPUs`). Nothing is lost, but the Fieldbook child job records
  point at the first attempts and need a refresh. By 19:00 UTC two coupling trainings had finished
  (`cwspu_t9_k025_cap06`, `cwspu_u_k05_cap06`) with their Table-9 evaluations queued.
- First measured control (19:05 UTC, `collect ... --launch coupling`, now part of the collector): Calvin's
  `cwspu_u_k05_cap06` (WSPU + coupling κ = 0.5, Uncheatable, cap 6) measured 0.9835 against the κ-0 control's
  0.9834 (run SD 0.0009): the coupled Uncheatable optimum reproduces the WSPU optimum's value, as its TV 0.008
  distance implied. `cwspu_t9_k025_cap06` finished training (inline Uncheatable 0.9993); its Table-9 evaluation
  is queued.
- Coupling results by 19:40 UTC (5 of 9 measured): the Uncheatable optima at κ = 0.25 / 0.5 / 1 measured
  0.9841 / 0.9835 / 0.9827 against the κ-0 control's 0.9834 (flat, as their TV ≤ 0.016 from the WSPU optimum
  implied; Table-9 mean 1.085–1.090). The Table-9 optimum at κ = 0.25, cap 6 measured **1.0659**, 0.0063 below
  the single-run WSPU control at nearly the same mixture (`wspu_table9_cap06` 1.0722, TV 0.005 away, same data
  seed) and 0.002 above the 26-run centre (1.0639). Two single runs differ with SD 0.0054, so this is within
  noise, and it is also 0.0075 below the neighbour forecast (1.0734): the floor is noisier than the bank's single
  runs make it look, which is the case for the replicates in section 8d. The κ = 0.25 optimum at cap 8 (20:10 UTC)
  measured 1.0735 against the cap-8 control's 1.0736 and the neighbour forecast 1.0732, so the cap-6 result reads
  as a favourable draw rather than a gain.
- Link results by 20:50 UTC: the bounded link's Uncheatable optimum `lwspu_u_bl_cap06` measured **0.9820**
  against the κ-0 WSPU control's 0.9834 at the same seed (run SD 0.0009, so the 0.0014 gain is about one SD of a
  difference of two runs), the coupling optima's 0.9827–0.9841, and the bank's best 0.9811 (DSP cap 10); the link
  predicted 0.9870 for itself and 0.9902 for the WSPU policy, so it under-predicted both by 0.005–0.007 and got
  the gain's sign right at half the size. Table-9 trainings `lwspu_t9_bl_cap08` and `lwspu_t9_bh_cap06` finished
  (inline Uncheatable 0.9963 / 1.0014); their Table-9 evaluations are running.
- Table 9 by 21:20 UTC: the link + hub optimum at cap 6 (`lwspu_t9_bh_cap06`) measured **1.0693** against its
  own prediction 1.0747, the neighbour forecast 1.0728, the single-run WSPU cap-6 control 1.0722 and the 26-run
  centre 1.0639. The three coupling cap-6 optima (κ 0.25 / 0.5 / 1, TV 0.005–0.024 from the WSPU optimum)
  measured 1.0659 / 1.0681 / 1.0718, mean 1.0686, so the WSPU cap-6 neighbourhood's level is about 1.069 ± 0.002
  and the single control run was a slightly unfavourable draw; the hub optimum sits at that level, 0.005 above
  the centre. Coupling complete (21:50 UTC): the cap-8 trio measured 1.0735 / 1.0726 / 1.0718 (mean 1.0726)
  against the cap-8 control's 1.0736. Every optimum measured tonight lands in the 1.066–1.074 band the neighbour
  forecasts named; the link's two Table-9 trainings have finished (inline Uncheatable 0.9958 / 0.9963) and their
  evaluations are queued with the hub cap-8 one.
- Factorial, preliminary (22:10 UTC, 13 of 16 corners trained, Table-9 evaluations running): on the inline
  Uncheatable mean (secondary response, main effects only, SE 0.0005) adding PDF/arXiv helps (E −0.0040, t −7.6),
  adding synthetic reasoning hurts (D +0.0025, t 4.8), adding code hurts (A +0.0018, t 3.4), synthetic QA and CC-HQ
  are flat. The design resolves 0.002-BPB effects at 4–8 SE; the Table-9 effects and the interactions need the
  remaining corners and evaluations.
- Link and hub complete (22:15 UTC). Table 9 at seed 662009: bounded link cap 6 **1.0651**, cap 8 **1.0636**;
  link + hub cap 6 1.0693, cap 8 1.0667; coupling cap 6 1.0659 / 1.0681 / 1.0718, cap 8 1.0735 / 1.0726 / 1.0718;
  WSPU κ-0 controls 1.0722 (cap 6) / 1.0736 (cap 8); 26-run centre 1.0639; matched-seed Olmix 1.0769. The link's
  two runs average 1.0644 against the WSPU sweep's 1.0729 (two-run means, difference SD 0.0038, so about 2.2 SD),
  and both sit at the replicated centre's level; the link predicted them at 1.084 (0.02 too pessimistic) and the
  neighbour forecast at 1.0737 (0.010 too pessimistic). This is the first measured Table-9 selection gain from a
  calibration mechanism, the opposite of the offline within-block contrast, which had the link worse than WSPU
  by 0.0036; the offline archive did not contain the link's own optima. Uncheatable: link 0.9820 against 0.9834.
- Decide on the prepared floor replicates (section 8d); add the link's two Table-9 optima to the replication set.
: ten runs, one command in
  `reference_outputs/delphi_floor_replicates_design_20260907/SUBMISSION.md`.
- Compare realized values with `reference_outputs/delphi_top_band_ordering_20260906/queued_run_forecasts.csv`
  (bank kernel, TV 0.05) as well as with each model's own prediction; the kernel says every queued optimum lands
  at 1.073 ± 0.001 and the factorial corners between 1.066 and 1.084.

## 10. Code changes

- `single_phase_observatory_models_20260902.py`: `GridModel.head_for` reads an optional `floor_fraction`
  shape key; `family_design` gains `interaction="named_pairs"` with `FamilyOptions.interaction_pairs`;
  helper pins refreshed for both (behaviour unchanged for existing configurations; model tests pass).
- `single_phase_observatory_registry_20260902.py`: `LINK_FLOOR_SHAPES`, `PANEL_PAIRS`, entries
  `@log_deficit_bounded_link_floor_cv`, `@log_deficit_bounded_link_total_hub`, `@named_pairs`,
  `@log_deficit_bounded_link_named_pairs`.
- New scripts: `evaluate_delphi_link_selection_20260906.py` (fits registry methods on the frozen benchmark
  and scores them; `--methods`), `materialize_delphi_link_validation_20260906.py`,
  `plot_starcoder_link_gate_20260906.py`, `design_delphi_frontier_factorial_20260906.py`,
  `experiments/domain_phase_mix/launch_delphi_link_validation_3e18.py`,
  `tests/test_launch_delphi_link_validation_3e18.py`.
- Paper: the outline's Analysis section carries the calibration-versus-selection evidence; the appendix,
  Figure 1 titles, Figure 7 floor, and Figure 9 strip were done earlier in the session.

### 8j. Flat-profile rule adopted; its optima submitted; Calvin's decision to finalize (2026-09-07 03:41 UTC)

`@kappa_floor_link_flat15` (κ ∈ [1, 6]; κ = 1.5 when the inner-CV profile is flat, i.e. the argmin lies within
10% of the log upper bound; 4 of 7 Uncheatable and 6 of 51 Table-9 tasks take the default) is the final candidate.
Screen (`delphi_kappa_floor_flat15_selection_20260907`): Uncheatable pooled pick rank 4 (regret 0.0016, the best
of any surrogate; WSPU rank 5), optimism −0.0001, archive RMSE 0.0127, Spearman 0.953; Table 9 rank 12 (0.0151),
optimism +0.008, RMSE 0.021, Spearman 0.930; panel OOF 0.0081 / 0.0258; within-block Table-9 regret vs WSPU
+0.0008 [−0.0004, +0.0026]. Fresh-run calibration over 18 unseen runs (the 14 of 2026-09-06 plus the four measured
κ-floor runs, `delphi_fresh_run_calibration_flat15_20260907`): the failed Uncheatable mixture is predicted at
0.9842 (measured 0.9890; the unbounded model said 0.9711), the KL-0.05 one at 0.9881 (0.9916); Uncheatable bias
+0.0045, RMSE 0.0054; own-target optima bias +0.0011, RMSE 0.0025; the measured Table-9 optimum 1.0613 is predicted
at 1.0635. The κ cap alone (`@kappa_floor_link_cap6`) is not enough; the flat default is what removes the
over-extrapolation.

Materialized (`delphi_kappa_floor_flat15_validation_3e18_20260907`, sha256 d99a0f9c…): `lwspu_u_kff_cap06`
predicted 0.9807 (CC-HQ 15%, literature-high 14%, olmOCR 13%, science-math-high 12%; TV 0.215 from the failed
mixture, 0.14 from the bounded link's), `lwspu_t9_kff_cap06` 1.0635 and `lwspu_t9_kff_cap08` 1.0626, both within
TV 0.012 of the measured κ-floor Table-9 optima. Calvin (03:30 UTC): finalize the single-phase procedure soon;
1.0613 on Table 9 is acceptable, no more grinding; address Uncheatable; those become the reported numbers and the
paper gets written. Submitted 03:41 UTC as `/calvinxu/dm-delphi-3e18-lwspu-kappafloor-flat-v6e8-20260907`
(launcher `launch_delphi_kappa_floor_flat_validation_3e18.py`, run ids 7,400,000+ / 7,400,100+; Fieldbook
`exp_01m1wzbhkb454cjg8k52j1t511`): the two Table-9 rows exist so that both targets are measured under the final
procedure and double as replicates at the optimum; the κ ≤ 100 Table-9 result stays as a sensitivity check.
Collect with `collect_delphi_3e18_validation_results_20260906.py --launch kappa_floor_flat`. KL 0.05 Table-9 runs
still training. Lint clean (registry line 1996 reflowed).

### 8k. Is 1.5 the best flat-profile default? Sweep 1.0–4.0 (2026-09-07 04:20 UTC)

Calvin asked. `delphi_flat_default_sweep_20260907` (registry entries `@kappa_floor_link_flat{10,20,25,30,40}`, κ ≤ 6)
and `delphi_fresh_run_calibration_flat_sweep_20260907`:

| default | U regret (rank) | U optimism | U RMSE | U Spearman | T9 regret / Spearman | fresh U bias all / own optima | U floors above bank min (worst) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.0016 (4) | −0.0008 | 0.0132 | 0.898 | 0.0151 / 0.923 | +0.0030 / −0.0001 | 5 (0.036) |
| 1.5 | 0.0016 (4) | −0.0001 | 0.0127 | 0.953 | 0.0151 / 0.930 | +0.0045 / +0.0011 | 5 (0.029) |
| 2.0 | 0.0016 (4) | +0.0012 | 0.0132 | 0.950 | 0.0151 / 0.930 | +0.0050 / +0.0025 | 4 (0.025) |
| 2.5 | 0.0016 (4) | +0.0021 | 0.0136 | 0.947 | 0.0151 / 0.930 | +0.0054 / +0.0036 | 3 (0.021) |
| 3.0 | 0.0016 (4) | +0.0028 | 0.0140 | 0.942 | 0.0151 / 0.930 | +0.0057 / +0.0045 | 3 (0.017) |
| 4.0 | 0.0030 (6) | +0.0053 | 0.0145 | 0.935 | 0.0151 / 0.930 | +0.0061 / +0.0058 | 3 (0.009) |

Table 9 does not move at all (six flat tasks with small weight). On Uncheatable the selection is flat from 1 to 3
and the calibration degrades monotonically with the default: lowering the floors of the unmoved tasks satisfies the
per-task held-out check but pushes the aggregate prediction down, so the refuted floors are not the binding
error (the over-predicted tasks are elsewhere: the code corpora). 1.5 keeps the best bank ranking (Spearman 0.953
vs 0.898 at 1.0); 1.0 has the smaller fresh-run bias by 0.0015 over 18 runs, within noise. Decision: keep 1.5; the
paper's appendix says the default is not critical between 1 and 2 and that larger values are more optimistic.
Paired source-block contrasts: every Uncheatable regret CI vs WSPU contains 0; Table 9 [−0.0004, +0.0026].

### 8l. KL 0.05 batch measured: the penalty hurts on both objectives (2026-09-07 04:50 UTC)

`/calvinxu/dm-delphi-3e18-lwspu-kappafloor-kl05-v6e8-20260907` (the unbounded-κ model's optima under an
Olmix-style KL 0.05 penalty toward proportional; Calvin's preregistered hypothesis: worse). Uncheatable cap 6
**0.9916** (unpenalized 0.9890); Table 9 cap 6 **1.0799** (1.0613), cap 8 **1.0860** (1.0672); Olmix's best 1.0769,
Olmix at KL 0.05 1.0814. Confirmed on all three, by about one run SD on Uncheatable and five on Table 9. The
penalized mixtures are TV 0.21–0.22 (Table 9) and 0.43 (Uncheatable) from the unpenalized ones: on Table 9 the
penalty took 5.6 points from synthetic QA, 5.3 from Stack-Edu + FIM and 3.4 from high-quality literature and put
7.3 into CC-HQ plus finance/entertainment/health web; the components that got worse are those buckets' clients
(basic-skills arithmetic +0.16, sciq +0.07, basic-skills coding +0.07, jeopardy +0.07, drop, mbpp, bash), while
squad, naturalqs, socialiqa and logical reasoning improved. Contrast: Olmix's own 3e18 KL sweep (July,
`delphi_one_phase_olmix_kl_sweep_3e18_20260705`) has KL 0 worse than its best by 0.008 (Table 9) and 0.012
(Uncheatable), so the penalty regularizes Olmix and only shrinks ours. Caveat on "predicted": the materializer's
value for a KL run is the penalized objective (0.9953 / 1.0876 / 1.0875); the surrogate's own predictions were
0.9852 / 1.0711 / 1.0710, and the flat-profile model predicts 0.9881 / 1.0713 / 1.0713, so the Table-9 KL runs are
under-predicted by 0.009–0.015 (the two near-identical mixtures differ by 0.006, i.e. noise). Flat15 calibration
over the 20 measured runs: Uncheatable bias +0.0048 (RMSE 0.0056), own optima +0.0011 (0.0025); Table 9 bias
+0.0020 (0.0065), own optima +0.0043 (0.0064). Paper: §6.1 sentence resolved (penalty helps Olmix, hurts ours;
Table 2 carries Olmix's KL-0 rows); outline M6 and Table R1 notes updated; Fieldbook `exp_01m1wmyxpqrmq25n4q7ck9ch1g`.

### 8m. Paper review round (GPT-6 Astra), Tier 1 and Tier 2 applied (2026-09-07 late evening)

Review in the paper folder (`review_feedback_2026-09-06.md`); my triage: nine local corrections and the claim
softenings are text (done), three objections needed offline analysis (done or running), and baseline
comparability needs runs (deferred until the methodology is frozen, Calvin's call). Text: proxy optimum
"concentrates repetition less" (Σ P_i E_i fixed), screen removed no task, non-monotonicity a capability, Table R-SNR
subtask means per run, ten repeats / eleven proportional runs / 10 df, Holm within each task's 39 deletions
(936 = 39 × 24), the 5% example is the Dolmino CC pool (0.24 epochs), Σ_r C(N_r, D_r), ladders use total
parameters, AdamH/MuonH defined (norm-preserving), DSP removed from prose, grid minima and "approximately aligned",
transfer as an observed association (repetition scalar = 95th-percentile epochs), noise section as unexplained
prediction error (nested subsets; anchor only), "optimistic on every optimum" replaces "well-calibrated in
support" (the full-aggregate optimum was predicted 0.947, measured 0.983), exposure reference vs evaluation
configuration, run counts, regret formula, MT-MBPP 17/51, compute denominator (280 × 3.4e18 = 9.5e20 ≈ one 1e21
run). Offline: predictive R² (1 − MSE/Var) in Figure R7B / Table R-fit (suite medians WSPU 0.65, Olmix 0.45; math
0.67 / 0.07); new appendix "Swarm support for the proposed mixtures" (`plot_delphi_swarm_support_20260907.py`,
Drive `a_swarm_support`); bank provenance paragraph + by-source selection table (`selection_by_source.csv`:
the final model's optimism at its pick within ±0.009 in every source, WSPU up to +0.070; regrets within 0.002 of
each other except the earlier-proposal sources); weight-coordinate ablation described narrowly and a
scale-matched `@common_inventory` variant added (harness run `single_phase_observatory_ablation_addendum_20260907`
failing on the registry probe twice, fixed; third run pending); transfer figure rank axes tightened (ranks are within
280). Not done: Figure 2 larger, Figure 3 token-budget legend, Related-Work comparison table, §6.2 shortening.
Disagreements kept: the additive-spline baseline is the "without the harm term" row; Olmix on epochs is the same
function class (E_i = c_i w_i). Outline carries the full record under "[Review round 2026-09-07 ...]".

### 8n. Flat-profile validation measured (2026-09-07 morning UTC): the final procedure's optima

`/calvinxu/dm-delphi-3e18-lwspu-kappafloor-flat-v6e8-20260907`: `lwspu_u_kff_cap06` Uncheatable **0.9832**
(predicted 0.9807, +0.0025; the calibration expected about +0.0045); `lwspu_t9_kff_cap06` Table 9 **1.0680**
(predicted 1.0635, +0.0045); `lwspu_t9_kff_cap08` Uncheatable endpoint 1.0002, native Table-9 evaluation pending.
Uncheatable: the three surrogates' optima are the same number within run noise, bounded link 0.9820, additive
WSPU control 0.9827 / 0.9834, flat15 0.9832 (aggregate repeat SD about 0.001); by component the flat mixture is
better on the two GitHub corpora (0.748 / 0.727 vs the link's 0.761 / 0.738) and worse on arXiv physics (1.032 vs
1.021) and BBC (1.085 vs 1.071). Reading: on Uncheatable the measured optimum is insensitive to the surrogate; the
models differ in prediction accuracy (flat15 own-optima bias +0.0013, RMSE 0.0025 over 23 fresh runs; WSPU
predicted 0.947 for its 0.983), not in the mixture they find. Table 9: the flat cap-6 optimum and the unbounded
model's cap-6 optimum are TV 0.011 apart and measured 1.0680 / 1.0613 (difference 1.7 repeat SDs, mean 1.0647);
by component the two differ far more than noise (21 of 51 components by more than 0.01; basic-skills string
operations +0.125, arithmetic +0.045, coding +0.038 on the flat run; squad −0.038), so the pair is a near-replicate
in mixture space and not in component space, and the basic-skills tasks are the sensitive ones. Both flat
optima remain below the WSPU controls (1.0722 / 1.0736) and Olmix's best (1.0769). Calibration of flat15 over
23 fresh runs: Uncheatable bias +0.0049 (RMSE 0.0056), own optima +0.0013 (0.0025). Decision needed from Calvin:
which Table-9 number the paper reports (the final procedure's own run 1.068, or the pair 1.061 / 1.068 with its
mean 1.065), and whether to submit the single Uncheatable KL-0.05 run of the final model (launcher trimmed to one
definition, prepared). Fieldbook `exp_01m1wzbhkb454cjg8k52j1t511`.

**8n addendum (cap 8 measured):** `lwspu_t9_kff_cap08` Table 9 **1.0685** (predicted 1.0626); its near-replicate, the
unbounded model's cap-8 optimum, measured 1.0672 (per component, 22 of 51 tasks differ by more than 0.01). The
final procedure's Table-9 optima are 1.0680 (cap 6) and 1.0685 (cap 8); with the unbounded variant's 1.0613 /
1.0672 the four near-identical mixtures span 1.061–1.069 (mean 1.066), all below WSPU (1.072 / 1.074) and Olmix
(1.077). The cap makes no measurable difference for the final model (cap 6 vs 8: 0.0005).

### 8o. Loose ends before the freeze: joint selection, noise margin, cap margin (2026-09-07)

`delphi_loose_ends_sweep_20260907` (frozen bank) + `delphi_fresh_run_calibration_loose_ends_20260907` (23 runs) +
floor checks (`delphi_link_floor_check_20260907/flat15_*`). Registry variants of `@kappa_floor_link_flat15`:
`_joint` (shape and ridge rescored once at the fitted κ instead of the prior 2.5; `FittedFloorModel.joint_rounds=1`),
`_margin2` / `_margin5` (noise margin 2 / 5 repeat SDs), `_cap025` / `_cap100` (prediction cap 0.25 / 1.0 nat).

| variant | U rank (regret), optimism, RMSE, ρ | T9 rank, optimism, RMSE, ρ | fresh U bias all / own | fresh T9 bias all / own, ρ | refuted floors T9 / U | κ changed |
|---|---|---|---|---|---|---|
| flat15 (two-stage) | 4 (0.0016), −0.0001, 0.0127, 0.953 | 12, +0.0084, 0.0209, 0.930 | +0.0049 / +0.0013 | +0.0018 / +0.0044, 0.810 | 3 / 5 | – |
| joint | 3 (0.0012), +0.0002, 0.0112, 0.949 | 12, +0.0075, 0.0205, 0.938 | +0.0051 / +0.0014 | +0.0020 / +0.0039, 0.843 | 7 / 4 | 11 T9, 3 U |
| margin 2σ | 4, −0.0001, 0.0127, 0.953 | 12, +0.0084, 0.0211, 0.930 | same | +0.0018 / +0.0045, 0.801 | 5 / 5 | 3 T9 |
| margin 5σ | 4, −0.0003, 0.0128, 0.952 | 12, +0.0083, 0.0207, 0.928 | +0.0047 / +0.0012 | +0.0016 / +0.0044, 0.835 | 2 / 5 | 8 T9 |
| cap 0.25 nat | 4, −0.0003, 0.0147, 0.952 | 12, +0.0084, 0.0209, 0.930 | +0.0048 / +0.0011 | same as flat15 | 3 / 5 | 1 / 1 |
| cap 1.0 nat | 4, −0.0001, 0.0102, 0.953 | 12, +0.0084, 0.0208, 0.931 | same | same | 3 / 5 | 0 |

Reading: no variant changes the selected mixtures on either objective or the fresh-run calibration beyond
noise. Joint selection changes the shape on 14 of 58 tasks and buys 0.001–0.002 RMSE on the bank and 0.03 Table-9
Spearman on the fresh runs, within the bootstrap; it also moves more Table-9 floors above measured runs (7 vs 3).
The cap margin only changes bank RMSE on the bad mixtures the cap truncates (0.0147 / 0.0127 / 0.0102). Decision:
keep the two-stage fit, 3σ and 0.5 nat; the appendix states the sensitivities. Llama-panel floor anchors built
(`build_llama_floor_anchors_20260907.py`; 300M Table 9 from the eleven proportional-reference runs, the rest from
the panel's proportional run with the aggregate repeat SD); the final model is being fitted on all three swarms
through the observatory harness (`single_phase_observatory_final_model_20260907`, certify + heldout).

### 8p. Simplification round (Codex brief): cap removable, per-task gamma necessary, margin nearly inert (2026-09-07)

`delphi_simplification_sweep_20260907` (old splits; to be repeated on the corrected benchmark), fresh-run
calibration over 23 runs, and each candidate's optima materialized at caps 6 / 8 and compared with the incumbent's
realized sampler counts (the incumbent re-materialized here matches its validated table exactly).

| candidate | U pick (rank, regret), optimism, RMSE | T9 pick, optimism, RMSE | fresh own-optima bias U / T9 | realized policy vs incumbent (U6 / T9-6 / T9-8) |
|---|---|---|---|---|
| incumbent (flat 1.5, 3σ, 0.5 nat) | 4 (0.0016), −0.0001, 0.0127 | 12 (0.0151), +0.0084, 0.0209 | +0.0013 / +0.0044 | – |
| no prediction cap | same, RMSE 0.0101 | same, RMSE 0.0205 | +0.0013 / +0.0044 | identical / identical / identical |
| no noise margin | same | same, RMSE 0.0210 | +0.0013 / +0.0044 | identical / TV 0.007 (20 buckets) / TV 0.006 (14) |
| γ fixed 1.5, grid at 1.5 | 4 (0.0016), −0.0095, 0.0109 | 12, −0.0021, 0.0210 | −0.0085 / −0.0052 | TV 0.085 / 0.035 / 0.039 (predicted 0.990 / 1.073 / 1.073) |
| γ fixed 2.5, grid at 2.5 | 4, +0.0025, 0.0142 | 14 (0.0157), +0.0223, 0.0230 | +0.0041 / +0.0160 | TV 0.053 / 0.034 / 0.043 |

Reading: the training-derived cap is removable as a behavior-preserving change (same picks, same calibration,
exactly the same realized policies and predictions at the optima; only bank RMSE on the truncated bad mixtures
moves). Per-task γ is necessary: a single γ for every task fails the calibration gate in both directions (1.5
pessimistic by 0.009 / 0.005, 2.5 optimistic by 0.004 / 0.016) and changes the policies. The noise margin changes
nothing on Uncheatable and moves the Table-9 policies by TV 0.006–0.007 with identical predictions; keeping it costs
one line, dropping it would need a validation run under the strict rule. Proposal: final = incumbent minus the cap.

### 8q. Olmix reproduction with the reference implementation (2026-09-07, `delphi_olmix_reproduction_20260907`)

allenai/olmix is Apache 2.0; our fitter (`olmix_loglinear_fit.py`, `fit_olmix_reference_deletion_augmented_300m.py`)
is an independent Apache-2.0 implementation of the same law and the same exact proposer (cvxpy, evaluation-weighted
mean of the per-task exp(A·w) plus constants, KL(w‖natural) by rel_entr, simplex and per-bucket caps at repetition
4; upstream uses ECOS, ours Clarabel then ECOS then SCS). The fit differs: upstream minimizes the Huber loss (δ 0.01)
with torch L-BFGS (20 steps, 300 initialisations per metric), ours with scipy from 48 multistarts.

Exact reruns of our scripts with their stored inputs: the July Uncheatable mixture (`olmix_onephase_uncheatable_d001_kl005_cap4`)
reproduces bit-for-bit (TV 0.0); the June Table-9 mixture (`olmix_onephase_table9_d001_kl0p005_cap4`) reruns to TV
0.029 (max weight difference 0.018) because its fit panel (`olmo_base_easy_one_phase_parity_panel_300m_20260628`)
was regenerated on 12 July for the native-evaluator parity work and the script changed on 3 and 13 July; the June
panel is not on disk in its original form.

On the frozen 280-run panel (the swarm exported as `swarm_ratios.csv` / `swarm_metrics.csv`, the CSVs `olmix fit`
reads), the reference implementation (installed without its stale OLMo-core pin, with the interaction-matrix plot
disabled because it fails matplotlib's layout at 58 × 39) and our fitter agree closely: TV 0.056 (Table 9, KL
0.005) and 0.122 (Uncheatable, KL 0.05), both at the cap; upstream predicted objectives 1.0959 / 0.9716, ours
1.0952 / 0.9941. Both differ from the trained Table-2 mixtures by TV 0.29 (Table 9) and 0.31–0.36 (Uncheatable):
the trained mixtures were fitted on the earlier panels (279 rows, proportional row replaced by the eleven-run mean,
no UniMax / uniform rows; June evaluator for Table 9), so the Olmix proposal is sensitive to the fit panel's
composition at the TV 0.3 level while its predicted objective moves by 0.005. Consequences: the artifact must ship
the exact fit panel of every reported Olmix mixture; the paper's Table-2 Olmix rows are reproducible from the
current data to TV 0.03 (Table 9, same script) and exactly for Uncheatable; the reference implementation gives a
mixture within TV 0.06–0.12 of ours on identical inputs.

### 8r. Corrected benchmark (proportional run pinned as calibration data) confirms the simplification round (2026-09-07)

Protocol fix per Codex: the proportional run is calibration data, pinned to the training side of every outer and inner
fold in both benchmarks (`pin_calibration`, tested), never scored out of fold; package `delphi_offline_selection_20260908`
(same panel and bank; Matern reference alternatives skipped because their frozen kernel-dof guard trips on the pinned
splits; scorer skips unfitted alternatives and unscored calibration rows). `benchmark.DEFAULT_OUTPUT` now points at it.

| model (corrected splits) | U rank (regret), optimism, RMSE, ρ | T9 rank, optimism, RMSE, ρ | fresh own-optima bias U / T9 (23 runs) |
|---|---|---|---|
| WSPU additive | 5 (0.0023), +0.037, 0.026, 0.881 | 14 (0.0157), +0.068, 0.038, 0.891 | – |
| incumbent (flat 1.5, 3σ, 0.5 nat) | 4 (0.0016), −0.0005, 0.0125, 0.958 | 12 (0.0151), +0.0078, 0.0202, 0.942 | +0.0009 / +0.0039 |
| no cap | same, RMSE 0.0099 | same, RMSE 0.0197 | same |
| no margin | same | same | same |
| γ fixed 1.5 | same picks, optimism −0.0083 | same picks, −0.0015 | −0.0072 / −0.0046 |

Same conclusions as on the old splits: drop the cap (behavior-preserving), keep per-task γ, margin inert. New fact:
on the pinned splits the incumbent's final fit differs from the validated fit (γ changed on 3 of 7 and 40 of 51
tasks, shape on 1 and 8), and its realized policies move TV 0.038 / 0.025 / 0.027 from the validated mixtures with
the same predicted values (0.9810 / 1.0638 / 1.0631). Under the strict rule those policies are candidates until
validated (three runs), or the paper reports the validated runs as the pre-fix fit's proposals and the corrected
protocol's metrics beside them.

### 8s. Standalone implementation (2026-09-07 night): `/Users/calvinxu/Projects/Work/Marin/mixture-selection`

One file, `mixture_selection.py`, extracted operation by operation from the frozen Marin path (features, pinned
mixture-blocked folds, the 168 × 5 grid scored at the provisional multiplier 2.5, bounded γ search with the 1.5
flat rule, centered NNLS with ridge rows and QR reduction, floored log-deficit link without the cap, vectorized
aggregate, seeded SLSQP multistart with polish and projection, 2,048-block rounding with exchange refinement),
plus the Olmix baseline (log-linear Huber fit, exact cvxpy proposer) and bank metrics; CLI `fit`, `predict`,
`optimize`, `evaluate`, `olmix`, `self-test`. Data: the frozen swarm as CSVs (weights with the calibration flag,
outcomes, buckets, objectives, anchors, the corrected-protocol fold table, both banks) and the reference fits /
predictions / policies (`delphi_corrected_screen_20260908`, `flat15_nocap`). Self-test: prediction parity 7e-15
(Uncheatable) and 5e-14 / 2e-13 (Table 9) on swarm and bank; realized policies identical at U cap 6 (predicted
0.9810), T9 cap 6 (1.0638), T9 cap 8 (1.0631). Those are the corrected-protocol policies (TV 0.025–0.038 from the
validated mixtures), i.e. the mixtures a validation batch of the frozen procedure would train. First commit made;
no remote yet.

### 8t. Standalone review actions, paper corrections, frozen-procedure launcher (2026-09-07 night)

Codex's review of the standalone (`.agents/handoffs/single_phase_standalone_review_20260907.md`) accepted the
method and asked for description fixes. Done: cap switch removed from the standalone (commit `14fda28`, parity
self-test unchanged: 1e-13, identical policies), fold-table validation, no subset fitting, diagnostics and laws
written beside outputs, MANIFEST. Paper: Methods 4.3/4.4 and Appendix A.6 now describe the fitting rule as
implemented (BPB-scored CV, NNLS on log-deficits, boundary default, calibration pinning, no cap, floors as
regularizers, upper bound six a development choice with bank ratios up to 8.2, SLSQP local); Overleaf `dd60a3b`.
Launcher for the frozen procedure's three proposals prepared and validated (dry run, safety check, tests);
awaiting approval. Details: freeze handoff Section 12.

### 8u. Frozen procedure validated at 3e18; fairness round submitted (2026-09-07 11:35 PDT)

`/calvinxu/dm-delphi-3e18-lwspu-frozen-v6e8-20260908` (calibration-pinned flat kappa-floor link, no cap; policies =
the standalone's `data/reference_policies.csv`): Uncheatable 0.9814 (predicted 0.9810; flat incumbent 0.9832, WSPU
0.9834, Olmix best 1.0022), Table 9 cap 6 1.0642 (1.0638; incumbent 1.0680, WSPU 1.0722, Olmix 1.0769), cap 8 1.0682
(1.0631; incumbent 1.0685). Pre-registered gate (freeze handoff Section 13) passed on all rows. Per component the
Uncheatable run beats the incumbent on six of seven (Wikipedia +0.0014); the surrogate's per-component misses are the
usual ones (BBC News +0.033 measured above prediction, arXiv physics -0.016 below). Cap 6 below cap 8 on Table 9
again (third pair: 1.0613/1.0672, 1.0680/1.0685, 1.0642/1.0682). Unconstrained optimization reproduces the Uncheatable
and cap-8 policies exactly, so the paper's headline rows are the unconstrained optima (Table 2 updated, Overleaf
`ce2f0dc`). Fairness round (Calvin's brief, conditional approval): seed-matched repeats
`/calvinxu/dm-delphi-3e18-fairness-repeats-v6e8-20260908` (12 runs) and KL ablation
`/calvinxu/dm-delphi-3e18-lwspu-kl-ablation-v6e8-20260908` (16 runs), both v6e-8; packages
`delphi_fairness_repeats_3e18_20260908` and `delphi_kl_ablation_3e18_20260908`.

### 8v. Per-bucket shapes tie the shared shape; MARINER named; Table 2 and Table 8 re-based (2026-09-07 evening)

Per-bucket (rate, power, threshold) by coordinate descent on the frozen procedure (`@kappa_floor_link_flat15_nocap_per_bucket_shape`):
OOF Spearman within 0.006 of the shared-shape model on every panel-target (see freeze handoff Section 13); the
shared shape stays. The method's paper name is MARINER (Mixture Allocation by Repetition-aware, Inventory-Normalized
Epoch Regression); WSPU stays a code name. Table 2 reports three-seed means ± SD at 3e18 for ours and Olmix's best
settings; Olmix's KL 0.05 Uncheatable ladder policy is being repeated (`dm-delphi-3e18-olmix-kl005-repeats-v6e8-20260908`).
