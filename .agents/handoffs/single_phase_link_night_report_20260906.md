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
