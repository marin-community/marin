# Handoff for review: freezing the single-phase mixture-selection procedure (2026-09-07)

Purpose: an independent review of the evidence behind the procedure we intend to freeze for the one-phase
data-mixing paper, of the decisions taken this week, and of the changes still proposed. The reviewer has no
conversation context; everything needed is in this file or in the files it points to. Nothing here has been
launched or edited by the reviewer's session; all paths are relative to the Marin repository root unless noted.
Numbers are bits per byte (BPB); lower is better. "Table 9", "T9" and "the suite mean" all mean the OlmoBaseEval
Easy 51-component macro mean; "Uncheatable" is the byte-weighted seven-corpus Uncheatable Eval aggregate.

## 1. What to review

1. Is the final procedure (Section 2) sound, and is its description in the paper faithful to the code?
2. Does the evidence (Sections 3 to 6) justify freezing it now, rather than after the changes listed in Section 7?
3. Are the comparisons with Olmix and with the earlier additive surrogate fair as stated, and what would make
   them fairer at small cost?
4. Which of the pending runs (Section 8) are worth their compute, and in what order?
5. Anything we are misreading. The two readings we rely on most are in Section 4.4 (the measured Uncheatable
   optimum does not depend on the surrogate) and Section 4.5 (near-identical Table-9 mixtures are not
   component-level replicates).

## 2. The procedure to be frozen

Setting: Qwen3 360M / 1.6B tokens at 3e18 FLOPs ("Delphi 3e18"), 39 buckets of a 6.99T-token Dolma 3 + Dolmino
partition, simulated epoching against the exposure reference 6.33T tokens, fit swarm of 280 runs (238 Dirichlet
mixtures, proportional / uniform / UniMax baselines, 39 leave-one-bucket-out deletions) plus 10 proportional
repeats. Two objectives: Uncheatable (7 components, byte weights) and the suite mean (51 components, unweighted).

Steps, with the code that implements each:

1. Reliability screen: per-task noise SD from the proportional repeats; deletion t-statistics, Holm within each
   task's 39 deletions. Removed no task at this setting. (`plot_table9_deletion_matrix_20260905.py`,
   `table9_snr_table_20260905.py`, both in `experiments/domain_phase_mix/exploratory/two_phase_many/`; every
   script below lives there unless a path is given.)
2. Surrogate per task ("WSPU" response functions, floored log-deficit head; registry id
   `weibull_softplus_unscaled@kappa_floor_link_flat15` in `single_phase_observatory_registry_20260902.py`,
   model classes in `single_phase_observatory_models_20260902.py`):
   - prediction f_t(w) = phi_t + exp(eta_t(w)), eta_t = c_t - sum_i alpha_ti b(E_i) + sum_i beta_ti h(E_i),
     alpha, beta >= 0; b(E) = 1 - exp(-(rho E)^kappa) (Weibull benefit), h(E) = softplus(log(1+E) - tau)^2
     (delayed harm); E_i(w) = D_tgt w_i / P_i materialized epochs.
   - floor phi_t = prop_t - max(gamma_t (prop_t - min_r y_rt), 3 sigma_t): prop_t = proportional mean over the
     11 proportional runs, min over the 280 swarm runs, sigma_t = repeat SD. Anchors:
     `reference_outputs/delphi_floor_anchors_20260907/anchors.csv` (58 rows; built by
     `build_delphi_floor_anchors_20260907.py`). In the code the multiplier is called `kappa`; the paper calls
     it gamma_t because kappa is the Weibull power.
   - gamma_t per task by a bounded scalar search on log gamma in [1, 6] (24 evaluations, xatol 0.02) of the
     3-fold mixture-blocked inner-CV RMSE; if the argmin lies within 10% of the log upper bound the profile is
     flat and gamma_t = 1.5 (`FittedFloorModel._search_kappa`, `FLAT_PROFILE_FRACTION = 0.1`). Flat default
     applied to 4 of 7 Uncheatable and 6 of 51 Table-9 tasks.
   - shape (rho, kappa, tau) from a 168-point grid and ridge from {0, 1e-3, 1e-2, 0.1, 1} by the same inner
     CV, scored in log-deficit space at a provisional multiplier of 2.5 (`FITTED_FLOOR_KAPPA_PRIOR`), then
     gamma_t refined with that shape fixed. Amplitudes and intercept by NNLS on log(y - phi_t).
   - prediction cap: eta_t <= max training log-deficit + 0.5 nat (`LINK_CAP_MARGIN`).
3. Aggregate: sum_t a_t f_t(w) with the evaluation weights a_t (byte shares; 1/51).
4. Optimizer: SLSQP from five starts (proportional plus four swarm mixtures, projected into the feasible set),
   polish pass, best feasible endpoint, rounding to the sampler grid with an exchange refinement; per-bucket
   epoch caps (policy: 6 for Uncheatable; 6 and 8 for Table 9; the optima converge before the cap on
   Uncheatable, 5.86 epochs, and at 7.47 under cap 8 on Table 9); no KL penalty
   (`optimize_delphi_matched_policies_20260906.py`, `materialize_delphi_link_validation_20260906.py`).
5. Validation: fresh 3e18 runs of the proposed mixture at the data seed of the matched Olmix comparator
   (Uncheatable 666200, Table 9 662009), same recipe as the swarm; launchers
   `experiments/domain_phase_mix/launch_delphi_kappa_floor_flat_validation_3e18.py` and siblings; results
   collected by `collect_delphi_3e18_validation_results_20260906.py --launch <name>`.

Paper text (Methods 4.3 and 4.4, Appendix A.6) in the Overleaf project `6a9d5d8e336776396e15003e` (commit
`aaa2863`) and in the Drive working copy
`~/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/`
(`sections/methods.tex`, `sections/appendix.tex`, source of truth `outline.md` sections M5 and M6).

## 3. How the procedure was chosen (offline evidence)

All offline screens use the frozen selection benchmark `reference_outputs/delphi_offline_selection_20260906`:
the 280-row panel with fixed outer and inner partitions, and a held-out bank of 408 (Uncheatable) / 247 (Table 9)
mixture coordinates measured at 3e18, of which 170 / 157 are "optima" proposed by earlier surrogates. Metrics:
regret at one (measured value of the bank member the model ranks first minus the bank minimum), rank of that
pick, optimism (measured minus predicted at the pick), RMSE and Spearman over the bank, paired source-block
bootstrap contrasts. Evaluate with
`evaluate_delphi_link_selection_20260906.py --methods <ids> --no-composite --output-dir <dir>`.

Candidate models and their bank results (rank out of 170 / 157 optima; regret; optimism at the pick; Spearman):

| model | Uncheatable | Table 9 | package |
|---|---|---|---|
| WSPU, additive (identity link) | rank 5, regret 0.0023, optimism +0.036, rho 0.875 | rank 14, 0.0157, +0.070, 0.894 | any package, `weibull_softplus_unscaled` |
| Bounded log-deficit link, floor 0.95 x swarm min | rank 10 (T9); optimism -0.005 / -0.015 | | `delphi_link_selection_20260906` |
| Kappa-floor link, kappa <= 100 (one head) | rank 6, 0.0027, +0.009, 0.928 | rank 12, 0.0151, +0.009 | `delphi_single_head_selection_20260907` |
| Kappa-floor, kappa <= 6, flat default 1.5 (final) | rank 4, 0.0016, -0.0001, 0.953 | rank 12, 0.0151, +0.008, 0.930 | `delphi_kappa_floor_flat15_selection_20260907` |

Within-block Table-9 regret vs WSPU for the final model: +0.0008 [-0.0004, +0.0026]; every Uncheatable regret
interval contains zero. Panel out-of-fold RMSE 0.0081 / 0.0258.

Flat-default sweep (`delphi_flat_default_sweep_20260907`, defaults 1.0 to 4.0, kappa <= 6): selection identical
for 1.0 to 3.0 on both objectives; bank optimism -0.0008 / -0.0001 / +0.0012 / +0.0021 / +0.0028 / +0.0053 and
fresh-run Uncheatable bias +0.0030 / +0.0045 / +0.0050 / +0.0054 / +0.0057 / +0.0061 for 1.0 / 1.5 / 2.0 / 2.5 /
3.0 / 4.0; bank Spearman 0.898 at 1.0, 0.953 at 1.5, then falling. Refuted floors (a held-out run below the
floor) on Uncheatable: 5 / 5 / 4 / 3 / 3 / 3 of 7 tasks. Reading: lowering the floors of unmoved tasks satisfies
the per-task check but makes the aggregate more optimistic, so 1.5 was kept.

Retired variants (all screened, none adopted): response-space NLS head (fails noisy QA tasks, not reproducible
to 1e-8), fixed-floor head, three- and four-head selection by inner CV, group-pooled kappa, permissive one-SE
rule (returns WSPU's optimism), kappa cap alone without the flat default (`delphi_kappa_floor_cap6_selection_20260907`).
Details: `.agents/handoffs/single_phase_link_night_report_20260906.md` Sections 8e to 8k.

Held-out floor check (`check_delphi_link_floors_20260907.py`, `reference_outputs/delphi_link_floor_check_20260907/flat15/`):
the final model's floors lie above the best held-out run on 5 of 7 Uncheatable tasks (four at the default;
AO3 by 0.029, BBC 0.015, arXiv physics 0.014, arXiv CS 0.005, Wikipedia 0.004) and 3 of 51 Table-9 tasks
(mt_mbpp_go 0.022, logical reasoning 0.016, piqa 0.001). The held-out bank improved those Uncheatable tasks by
1.8 to 8.2 times the swarm's own gain over proportional. We accept the conservative side because the
unbounded model, which passed this check almost everywhere, missed its Uncheatable optimum by 0.018 (Section 4.2).

Top-band ordering (`analyze_delphi_top_band_ordering_20260906.py`): no panel-fitted surrogate orders the 30
best-measured Table-9 mixtures (pairwise sign accuracy 0.46 to 0.67 against a 0.92 noise ceiling); a
Nadaraya-Watson kernel over measured neighbours does (0.79). The surrogate finds the basin; it does not order
the floor of it.

## 4. The 3e18 validation runs (September 6 and 7)

All runs: same recipe as the swarm, v6e-8 in us-east5-b, data seed 666200 (Uncheatable) or 662009 (Table 9),
trainer seed 0, evaluated at step 3006 (Uncheatable inline; Table 9 by the native evaluator). Proportional repeat
SD at this setting: about 0.001 (Uncheatable aggregate), 0.0038 (Table-9 mean). Measured tables:
`reference_outputs/<package>/measured_results.csv` and `measured_table9_components.csv`.

### 4.1 Uncheatable optima (cap 6)

| model that proposed the mixture | measured | predicted by its own model | predicted by the final model | package |
|---|---:|---:|---:|---|
| WSPU additive, matched-seed control (kappa 0.25 / 0.5 / 1 coupling variants) | 0.9841 / 0.9835 / 0.9827 | 0.947 | 0.9836 / 0.9836 / 0.9834 | `delphi_coupling_validation_3e18_20260906` |
| Bounded log-deficit link | 0.9820 | 0.9870 | 0.9833 | `delphi_link_validation_3e18_20260906` |
| Kappa-floor, kappa <= 100 | 0.9890 | 0.9711 | 0.9842 | `delphi_kappa_floor_validation_3e18_20260907` |
| Kappa-floor, kappa <= 100, KL 0.05 penalty | 0.9916 | 0.9852 (penalized objective 0.9953) | 0.9881 | `delphi_kappa_floor_validation_kl05_3e18_20260907` |
| Final (flat 1.5) | 0.9832 | 0.9807 | 0.9807 | `delphi_kappa_floor_flat15_validation_3e18_20260907` |

Comparators: proportional 1.038 (mean of 11), Olmix best 1.002 (KL 0.1, cap 4), Olmix default 1.004 (KL 0.05),
bank best 0.9811. The unbounded model's miss came from tasks the swarm never moved (arXiv physics kappa 17, BBC
9, Wikipedia 7.7) becoming additive; the mixture had 19% olmOCR PDFs and 0.1% Wikipedia.

### 4.2 Table-9 optima

| model | cap 6 measured (predicted) | cap 8 measured (predicted) | package |
|---|---:|---:|---|
| WSPU additive, matched-seed controls | 1.0722 (1.008); coupling variants 1.0659 / 1.0681 / 1.0718 | 1.0736 (1.004); 1.0735 / 1.0726 / 1.0718 | `delphi_coupling_validation_3e18_20260906` |
| Bounded log-deficit link | 1.0651 (1.0841) | 1.0636 (1.0839) | `delphi_link_validation_3e18_20260906` |
| Link plus hub interactions | 1.0693 (1.0747) | 1.0667 (1.0737) | `delphi_link_hub_validation_3e18_20260906` |
| Kappa-floor, kappa <= 100 | 1.0613 (1.0631) | 1.0672 (1.0623) | `delphi_kappa_floor_validation_3e18_20260907` |
| Kappa-floor, kappa <= 100, KL 0.05 | 1.0799 (1.0711; penalized 1.0876) | 1.0860 (1.0710) | `delphi_kappa_floor_validation_kl05_3e18_20260907` |
| Final (flat 1.5) | 1.0680 (1.0635) | 1.0685 (1.0626) | `delphi_kappa_floor_flat15_validation_3e18_20260907` |

Comparators: proportional 1.199, Olmix best 1.0769 (KL 0.005, cap 4), Olmix KL 0 1.0851, 26-run replicated
frontier centre 1.0639 (SD 0.0041), best measured mixture in the bank 1.0557 (a factorial corner, Section 6).

### 4.3 Calibration on the fresh runs

`predict_delphi_fresh_runs_20260907.py` scores each model's fold -1 fit on the 23 measured runs above, none of
which any model saw. Final model: Uncheatable bias +0.0049 (RMSE 0.0056) over all runs and +0.0013 (0.0025) on its
own-target optima; Table 9 bias +0.0018 (0.0065) and +0.0044 (0.0062); Spearman 0.964 / 0.810. WSPU additive on
the earlier 14 runs: +0.030 / +0.046 (RMSE 0.030 / 0.051). Files: `reference_outputs/delphi_fresh_run_calibration_flat15_20260907/`.

### 4.4 Reading one: on Uncheatable the measured optimum does not depend on the surrogate

Three different models (additive WSPU, bounded link, final) proposed mixtures 0.14 to 0.17 apart in total
variation and measured 0.9827 to 0.9834, 0.9820 and 0.9832: the same value within about one run SD. What
separates the models is the prediction (0.947 against 0.983 for WSPU; 0.987 against 0.982 for the bounded link,
whose fixed floor was too high; 0.981 against 0.983 for the final model).
Per component the final mixture is better on the two GitHub corpora (0.748 / 0.727 vs the link's 0.761 / 0.738)
and worse on arXiv physics (1.032 vs 1.021) and BBC (1.085 vs 1.071). Question for the reviewer: is 0.982 to
0.983 the floor of this objective at 3e18 under cap 6 (the bank best is 0.9811), and should the paper say so?

### 4.5 Reading two: near-identical Table-9 mixtures are not component-level replicates

The final model's cap-6 optimum and the unbounded model's are 0.011 apart in TV and measured 1.0680 / 1.0613
(1.7 repeat SDs); at cap 8, 0.012 apart, 1.0685 / 1.0672. Per component the pairs differ by more than 0.01 on 21
and 22 of 51 tasks (SD of the differences 0.023 and 0.025), concentrated on the basic-skills tasks (string
operations +0.125, arithmetic +0.045, coding +0.038 on the flat cap-6 run; squad -0.038). The four mixtures span
1.061 to 1.069 (mean 1.066), all below WSPU (1.072 / 1.074) and Olmix (1.077). Open decision: report the final
model's own run (1.068) or the four-run band with its mean and spread. We lean to the band.

### 4.6 KL penalty

An Olmix-style KL(w || proportional) penalty with coefficient 0.05 was tested on the unbounded model's optima
under a preregistered hypothesis (worse). Measured worse on all three runs (Section 4.1, 4.2: +0.003 Uncheatable,
+0.019 Table 9). The penalized mixtures sit 0.21 to 0.22 (Table 9) and 0.43 (Uncheatable) from the unpenalized
ones; on Table 9 the penalty took 5.6 points from synthetic QA, 5.3 from Stack-Edu plus FIM and 3.4 from
high-quality literature into CC-HQ and finance / entertainment / health web, and the components that got worse
are those buckets' clients. Olmix's own 3e18 KL sweep (July; `delphi_one_phase_olmix_kl_sweep_3e18_20260705`,
run ids 662000 to 662015, delta 0.01, cap 4) goes the other way: KL 0 is worse than its best by 0.008 (Table 9)
and 0.012 (Uncheatable). The final model's penalized Table-9 mixtures coincide with the measured ones (TV 0.006),
so the only informative remaining KL run is the final model's Uncheatable one (prepared, Section 8).

## 5. Ablations and mechanisms (offline, three swarms)

Observatory harness (`benchmark_single_phase_observatory_20260902.py`, certify tier), out-of-fold Spearman on the
Qwen3 360M/1.6B, Llama 200M/6B and Llama 160M/1.2B swarms (Uncheatable / Table 9), from
`reference_outputs/single_phase_observatory_benchmark_20260902/aggregate_metrics.csv` and the addendum
`single_phase_observatory_ablation_addendum_20260907` (which reproduces the earlier rows to 0.001):

| row | Qwen3 | Llama 200M/6B | Llama 160M/1.2B |
|---|---|---|---|
| WSPU additive | 0.951 / 0.828 | 0.951 / 0.932 | 0.937 / 0.876 |
| without the harm term | 0.902 / 0.747 | 0.903 / 0.882 | 0.881 / 0.833 |
| harm reads scrambled exposures | 0.899 / 0.741 | 0.853 / 0.873 | 0.876 / 0.829 |
| weights in place of epochs | 0.914 / 0.801 | 0.944 / 0.901 | 0.908 / 0.865 |
| one pool size for every bucket (new) | 0.918 / 0.810 | 0.946 / 0.912 | 0.911 / 0.852 |
| permuted bucket sizes | 0.916 / 0.763 | 0.941 / 0.892 | 0.918 / 0.826 |
| Olmix | 0.899 / 0.808 | 0.848 / 0.832 | 0.857 / 0.876 |

Caveats we have written into the paper after a reviewer's objection: E_i is a fixed rescaling of w_i, so the
coordinate carries no information by itself; the weight row never reaches the harm threshold (log(1+w) < tau)
and is therefore also close to the no-harm row; the two scale-matched rows isolate the per-bucket pool sizes.
The final model has not yet been run through this harness (it needs floor anchors for the two Llama panels; only
the Qwen3 anchors exist).

## 6. Other measured evidence used in the decisions

- Frontier factorial (`delphi_frontier_factorial_design_20260906`, 18 runs, resolution V around the replicated
  Table-9 centre 1.0639): main effects per step code -0.0065, synthetic QA -0.0080, synthetic reasoning -0.0040,
  CC-HQ -0.0008, PDF/arXiv -0.0014 (SE 0.0019); no interaction beyond one SE; all-plus corner 1.0557. Near the
  optimum the response is additive in these directions; the panel surrogates got the levels wrong, not the form.
- Olmix comparators at every rung of the compute ladder (Table 2 of the paper): trained at Olmix's own cap 4 with
  KL 0.05 (Uncheatable) and 0.005 (Table 9); the additive WSPU cap-6 optima are being scaled on the same seeds
  (`launch_delphi_one_phase_wspu_scaling.py`, Fieldbook `exp_01m1tqcvakk96x9t94ak327739`, 4 jobs succeeded,
  upper rungs waiting on the v6e pool). Those ladder mixtures are the additive model's, 0.17 (Uncheatable) and
  0.08 (Table 9) in TV from the final model's.
- Additive WSPU at Olmix's cap 4 (bank, `delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902`): Table 9
  1.0840, Uncheatable 0.9841; Olmix at cap 4: 1.0769 / 1.002. So at matched cap the additive model loses on
  Table 9 and wins on Uncheatable; the final model has not been measured at cap 4 (predicted 1.073 / 0.972 by the
  unbounded variant).

## 7. Loose ends we propose to close before the freeze (offline, no training)

Status (updated the same day, after the reviewer copy was sent): items 1 and 2 are closed by
`delphi_loose_ends_sweep_20260907` and `delphi_fresh_run_calibration_loose_ends_20260907`: rescoring the shape grid
at the fitted gamma_t (registry `@kappa_floor_link_flat15_joint`) changes the shape on 14 of 58 tasks and gives
Uncheatable rank 3 / regret 0.0012 (vs 4 / 0.0016), bank RMSE 0.0112 vs 0.0127, fresh Table-9 Spearman 0.843 vs
0.810, own-optima bias +0.0014 / +0.0039 vs +0.0013 / +0.0044, and moves more Table-9 floors above measured runs
(7 vs 3); a noise margin of 2 or 5 SD and a cap of 0.25 or 1.0 nat leave every pick and the fresh-run calibration
unchanged (the cap only changes bank RMSE on the bad mixtures it truncates: 0.0147 / 0.0127 / 0.0102). We keep
the two-stage fit, 3 SD and 0.5 nat. Item 5 is done (`build_llama_floor_anchors_20260907.py`; the final model is
being fitted on all three swarms in `single_phase_observatory_final_model_20260907`). Items 3 and 4 are decisions
for the freeze.

1. Joint selection of shape and gamma_t: the grid is scored at the provisional multiplier 2.5, then gamma_t is
   refined with the shape fixed. Rescoring the grid at the fitted gamma_t costs under an hour per screen. Adopt
   only if fresh-run calibration or bank selection moves beyond the bootstrap.
2. Sensitivity of the 3 sigma noise margin and the 0.5 nat cap: never varied; they bind only on tasks the swarm
   barely moved. Two values each, about 15 minutes per variant.
3. One cap policy for the headline (cap 6 for both objectives; cap 8 as a sensitivity row). The final model
   measured 1.0680 / 1.0685 at caps 6 / 8.
4. One held-out bank for every reported selection metric: the paper's ablation table and learning curves used
   the 5 September registry (409 / 248 coordinates), the floor work the 6 September frozen benchmark (408 / 247).
5. Floor anchors for the two Llama swarms (proportional repeats exist), so the final model can be fitted on all
   three swarms for the ablation table and transfer results.

## 8. Runs proposed after the freeze (need approval; none submitted)

- True replicates of the two headline optima at a second trainer seed: two runs.
- The final model's Uncheatable optimum under KL 0.05: one run, launcher prepared
  (`experiments/domain_phase_mix/launch_delphi_kappa_floor_flat_kl05_validation_3e18.py`, trimmed to the
  Uncheatable definition; candidate table `delphi_kappa_floor_flat15_validation_kl05_3e18_20260907`, predicted
  0.989, TV 0.07 from the measured 0.9916 run).
- Baseline fairness: Olmix at caps 6 and 8 with its best KL on Table 9 and at cap 6 on Uncheatable (three
  runs); the final model at cap 4 on both objectives (two runs); ridge-tuned Olmix by the same inner CV, offline.
- Ten floor-replicate runs plus two link-optimum replicates designed earlier
  (`reference_outputs/delphi_floor_replicates_design_20260907/`, launcher prepared), lower priority.
- Not proposed: relaunching the compute ladder for the final model's mixtures (about 2.6e21 FLOPs).

## 9. Known limitations to keep visible

- The held-out bank was development data for the response family (frozen 2 September) and for the floor and link
  (chosen on the frozen bank 7 September, source panels as the resampling unit). The independent evidence is the
  two Llama swarms (family fitted anew) and the 23 runs trained after each freeze.
- Noise is measured at one anchor (proportional), with repeats that share materialized subsets and differ only in
  data seed.
- The learning-curve study and the SNR / fit-quality tables were computed for the additive WSPU; rerunning them
  for the final model requires a fast solver (the WSPU study took seven hours with one) or a smaller design.
- Olmix's Huber delta and iteration limit follow its repository and were not tuned by inner CV, whereas WSPU's
  ridge is.

## 10. Bookkeeping

Fieldbook research line `exp_01m1wg9z23vmyjx5zfr7n4v5wa`; launches `exp_01m1wmyw6n47v8a8b0wfvsd36b` (kappa-floor
KL 0), `exp_01m1wmyxpqrmq25n4q7ck9ch1g` (KL 0.05), `exp_01m1wzbhkb454cjg8k52j1t511` (final model), factorial
`exp_01m1vh5s802cj685fbzcx3w475`. Running narrative: `.agents/handoffs/single_phase_link_night_report_20260906.md`
(Sections 8e to 8n). Deck: `.agents/handoffs/slides/single_phase_observatory_benchmark_20260902/slides.md`.
Memory notes (Claude): `kappa-floor-link-20260907`, `top-band-ordering-20260906`, `frontier-factorial-design-20260906`,
`coupling-variants-tested`, `delphi-link-validation-20260906`. Paper review already received and applied (GPT-6
Astra, `review_feedback_2026-09-06.md` in the paper folder; Tier 1 and 2 applied on 7 September, recorded in
`outline.md` under "[Review round 2026-09-07 ...]").

## 11. Standalone implementation, for review (added 2026-09-07 night)

`/Users/calvinxu/Projects/Work/Marin/mixture-selection` (git, commit `b1707f7`): `mixture_selection.py` is the
authoritative implementation of the procedure to be frozen, with the data it was fitted on and a parity self-test.
What it implements is the outcome of the simplification round and the protocol fix of Sections 7 and 8:

- the floored log-deficit surrogate with per-task gamma by inner CV in [1, 6] and the flat-profile default 1.5
  (fixed gamma failed the calibration gate); the three-sigma noise margin (inert, kept); the two-stage shape
  selection at the provisional multiplier 2.5 (the joint refinement bought nothing beyond noise);
- no training-derived prediction cap (identical picks, calibration and realized policies without it);
- the calibration protocol: the proportional run pinned to training in every fold, never scored;
- cap 6 as the headline policy; no KL penalty.

Two switches remain in the interface and are not part of the procedure: `fit --cap-margin` reproduces the
historical fits that carried the cap (the validated runs were proposed with it), and `optimize --kl` reproduces
the KL ablation. Everything else from the development registry (fixed gamma, joint rounds, margin and cap sweeps,
group pooling, response-space heads, alternative links) is absent.

Parity (`python mixture_selection.py self-test`, against `delphi_corrected_screen_20260908` /
`weibull_softplus_unscaled@kappa_floor_link_flat15_nocap`): swarm and bank predictions to 7e-15 (Uncheatable) and
5e-14 / 2e-13 (Table 9); realized policies at Uncheatable cap 6 and Table-9 caps 6 / 8 identical to the
reference materialization (predicted 0.9810 / 1.0638 / 1.0631). Those policies are TV 0.025-0.038 from the
validated mixtures (Section 8 of the report addendum), so the frozen procedure's own validation batch is its three
`optimize` outputs.

Questions for the reviewer: (1) does the file implement the paper's Methods 4.3-4.4 and Appendix A.6 as written,
and where do they disagree; (2) is anything in the fitting path still a development leftover rather than part of
the procedure; (3) is the calibration protocol implemented as intended (`final_inner_folds`, `pin_calibration`,
the fold table) and is `data/splits.csv` the right thing to ship versus recomputing folds; (4) is the Olmix
baseline faithful to allenai/olmix's exact proposer (it agrees with the upstream package to TV 0.06-0.12 on the
same inputs; the law fit uses scipy multistart instead of torch L-BFGS); (5) what is missing for the artifact to
stand on its own for a reader of the paper (provenance, naming, README).

## 12. Actions on the standalone review (2026-09-07 night)

Standalone (`/Users/calvinxu/Projects/Work/Marin/mixture-selection`, commit `14fda28`): the `--cap-margin` switch is
removed (Calvin: the artifact needs no backward compatibility; Marin keeps `cap_margin` for the historical fits);
supplied fold tables are validated (legal labels, calibration row pinned, nonempty folds); `fit_task` no longer
accepts a row subset, so folds always index the swarm; `optimize` writes its start diagnostics and `olmix` its
fitted laws beside the CSVs; `data/MANIFEST.json` records input hashes, the Marin revision and the environment;
README and docstring describe the fitting rule as implemented. Parity self-test after the change: predictions to
1e-13, the three policies identical.

Paper (Overleaf `dd60a3b`, outline updated): Methods 4.3 no longer says positive amplitudes guarantee an initial
decrease, no longer calls the upper bound six an observed maximum (the banks reach 8.2 on bbc_news), describes the
default as a boundary rule, states that CV scores predictions in BPB while NNLS fits log-deficits, states the
calibration pinning, drops the half-nat cap sentence and calls the floor a regularizer; 4.4 calls SLSQP local;
Appendix A.6 the same, with the margin binding on 6 of 51 Table-9 heads (verified from the reference fits) and
the cap sensitivity replaced by "removed after changing no selection and no optimized mixture".

Validation of the frozen procedure's own proposals (Codex: proceed): launcher
`experiments/domain_phase_mix/launch_delphi_frozen_procedure_validation_3e18.py` (+ test, 5 passed) reads
`delphi_corrected_screen_20260908/materialized_flat15_nocap/runtime_materialization/candidate_weights.csv`
(sha `e3a90c96…`), numerically identical to the standalone's `data/reference_policies.csv`; run ids 7,402,000 /
7,402,100, prefixes `lwspufp_` / `t9p`. Dry run and the east5 safety check passed; the exact command is
`reference_outputs/delphi_frozen_procedure_validation_3e18_20260908/submission/launch_command.sh`. Not submitted:
needs Calvin's approval (three v6e-8 runs, predicted 0.9810 / 1.0638 / 1.0631). The corrected-protocol KL 0.05
Uncheatable policy is also available from the standalone (`optimize --kl 0.05 --cap 6`: surrogate prediction
0.9898, penalized objective 0.9979, TV 0.32 from the KL-0 policy, max epoch 4.6); no launcher for it yet.

Cap convergence of the frozen fits (standalone `optimize`, caps 4 / 6 / 8 / 10 / 12 / 16 / 32, 2026-09-07 night):
Uncheatable is interior at 5.25 epochs from cap 6 on and caps 6–32 give the identical policy (cap 4 binds on three
buckets, predicted 0.9821 vs 0.9810); Table 9 binds on two buckets at cap 6 (1.0638), is interior at 7.50 epochs at
cap 8 (1.0631) and caps 8–32 give the identical policy. The cap-6 and cap-8 Table-9 policies and the cap-6
Uncheatable policy equal the launcher's candidate table exactly. Calvin (2026-09-07): no KL run for now; evidence
for the frozen procedure first.

Submitted 2026-09-07 04:50 PDT with Calvin's approval: Iris parent `/calvinxu/dm-delphi-3e18-lwspu-frozen-v6e8-20260908`
(three v6e-8 children, run ids 7,402,000 / 7,402,100+), Fieldbook `exp_01m1xv9n8j3v1e9175xf5te225`, package
`reference_outputs/delphi_frozen_procedure_validation_3e18_20260908/SUBMISSION.md`. These runs produce the paper's
reported numbers for the frozen procedure.

## 13. Overnight plan (2026-09-07 night, Calvin asleep): review gate and the fairness round

Calvin's instruction: monitor `/calvinxu/dm-delphi-3e18-lwspu-frozen-v6e8-20260908` to completion, review, and
proceed with the fairness round only if performance is acceptable with no material regression against the matched
incumbents. No KL runs of Olmix (its 3e18 sweep already covers KL 0 to 0.5, best 0.1 Uncheatable = 1.0022 and
0.005 Table 9 = 1.0769).

Pre-registered gate (each frozen-procedure run against its matched incumbent, the flat-validation mixture at the
same data seed and trainer seed 0: Uncheatable 0.9832, Table 9 cap 6 1.0680, cap 8 1.0685; repeat SD about 0.001
Uncheatable, 0.004 Table 9):
- pass: measured <= incumbent + 2 SD (Uncheatable <= 0.9852; Table 9 <= 1.0765 / 1.0770) on every row, and below the
  Olmix comparators (1.0022 / 1.0769) and the additive WSPU (0.9834 / 1.0722) within the same margin;
- fail: any row above its threshold, or a prediction miss larger than the flat validation's (0.0025 / 0.0045 / 0.0059);
  then nothing is submitted and the report explains.

Unconstrained check (Calvin's brief): with no cap at all (`optimize --cap 1e9`) the standalone reproduces the
selected Uncheatable policy (max 5.25 epochs) and the Table-9 cap-8 policy (7.50 epochs) exactly, so the headline
procedure is the unconstrained solution and the caps are sensitivity evidence; the Table-9 cap-6 row is the capped
comparison, cap 8 the predeclared tie-breaker.

Prepared, validated (dry run, east5 safety check, tests), not submitted:
- `launch_delphi_fairness_repeats_3e18.py`, 12 runs, `dm-delphi-3e18-fairness-repeats-v6e8-20260908`: Olmix KL 0.1
  (Uncheatable) and KL 0.005 (Table 9) as their exact runtime mixtures (Levanter block quantization of the trained
  weights, TV 0.010 / 0.008 from the continuous weights; `olmix_quantization.md`) and our three proposals, at
  (666200 / 662009, trainer seeds 1, 2) for every policy plus trainer seed 0 for the Olmix policies, all on v6e-8
  (the July Olmix runs were on v5p-8 at their own run-id data seeds). Three matched seed pairs per objective.
- `launch_delphi_kl_ablation_3e18.py`, 16 runs, `dm-delphi-3e18-lwspu-kl-ablation-v6e8-20260908`: the frozen fits'
  proposals at KL 0.005 to 0.5 (standalone `optimize --kl`), Uncheatable cap 6 and Table 9 cap 8, at the control seeds;
  `policy_summary.csv` holds the penalized objectives; raw predictions: Uncheatable 0.9817 (0.005) to 1.0103 (0.5),
  Table 9 1.0633 to 1.1163; max epochs fall from 5.24 to 2.78 and 7.36 to 3.26.
Candidate tables `delphi_fairness_repeats_3e18_20260908/candidate_weights.csv` (sha a6c72640) and
`delphi_kl_ablation_3e18_20260908/candidate_weights.csv` (sha 53b5f4b4); run ids 7,403,000+ and 7,404,000+.

Interim (2026-09-07 10:25 PDT): the frozen Uncheatable proposal measured 0.9814 (predicted 0.9810; incumbent
0.9832, additive WSPU 0.9834, Olmix 1.0022), gate row passed; per component it is below the incumbent on six of
seven (Wikipedia +0.0014). The two Table-9 runs were still training after 5.5 hours.
Interim (11:15 PDT): Table-9 cap 6 measured 1.0642 (predicted 1.0638; incumbent 1.0680, WSPU 1.0722, Olmix 1.0769),
gate row passed; cap-8 training done, its evaluation running.

Final (11:35 PDT): gate PASS on all three rows (`review_delphi_frozen_procedure_validation_20260908.py`, package
`review.md`): Uncheatable 0.9814 (predicted 0.9810), Table 9 cap 6 1.0642 (1.0638), cap 8 1.0682 (1.0631); every
row below its incumbent (0.9832 / 1.0680 / 1.0685), the additive WSPU (0.9834 / 1.0722 / 1.0736) and Olmix
(1.0022 / 1.0769). Cap 6 again measured below cap 8 on Table 9 (1.0642 vs 1.0682, about one repeat SD), against the
surrogate's ordering; the repeats settle it, cap 8 stays the predeclared tie-breaker. Submitted the fairness round:
`/calvinxu/dm-delphi-3e18-fairness-repeats-v6e8-20260908` (12 runs, Fieldbook exp_01m1yjft203bdg75hg0rdv4b5g) and
`/calvinxu/dm-delphi-3e18-lwspu-kl-ablation-v6e8-20260908` (16 runs, Fieldbook exp_01m1yjg60x0tz6tjrzj80pxm6e).

Shape-sharing ablation (Calvin, 2026-09-07 afternoon; expected to do worse; goes into Table 8): registry entry
`weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_shape` = the frozen procedure with one
(rate, power, threshold) per bucket from the same 168-shape grid, chosen by coordinate descent on the inner-CV error
from the shared optimum (at most two sweeps, ridge re-selected), then the parent's per-task floor search
(`models.PerBucketShapeGridModel`, `_fitted_floor(per_bucket_shapes=True)`). Harness run
`single_phase_observatory_shape_sharing_20260908` (certify tier restricted to the three 39-bucket panels with the
new `--panels` / `--curves none` options; five folds, seed 20260902; the frozen entry fitted alongside as the
re-based Table-8 baseline). Prior evidence: the canonical DSP's per-bucket shapes tied one shared shape in the
2026-09-02 matched ablation (16/22 units) and did not survive it.

Fairness repeats measured (2026-09-07 18:30 PDT, `delphi_fairness_repeats_3e18_20260908/fairness_summary.csv`): over
three matched seeds ours 0.9825 ± 0.0010 vs Olmix (KL 0.1) 1.0033 ± 0.0014 on Uncheatable (paired −0.0208 ± 0.0014
SE, every seed); suite: ours unconstrained 1.0678 ± 0.0052 vs Olmix (KL 0.005) 1.0830 ± 0.0071 (paired −0.0152 ±
0.0029); ours cap 6 1.0658 ± 0.0031 (cap 6 − uncapped −0.0020 ± 0.0017, inconclusive; cap 8 = unconstrained is the
headline as predeclared). Calvin's decision: report the unconstrained results; the epoch cap is not a swept
hyperparameter (optional only); Olmix tunes cap and KL, we tune neither. Paper: Table 2 rows now three-seed means
with a dagger, new Table `tab:seeds`, §6.1/§6.2 rewritten (Overleaf `cc9ffe2`).

Olmix KL 0.05 repeats (Calvin, 2026-09-07 18:49 PDT): Table 2's Uncheatable ladder policy (cap 4, KL 0.05) had one
3e18 run; three v6e-8 repeats at 666200 x trainer seeds 0/1/2 submitted as
`/calvinxu/dm-delphi-3e18-olmix-kl005-repeats-v6e8-20260908` (Fieldbook `exp_01m1zba83hhqz089fr4z8q4tvx`, package
`delphi_olmix_kl005_repeats_3e18_20260908`) so that row can show mean ± SD like the KL 0.1 row. Table 2 now shows
mean ± SD at 3e18 wherever runs were repeated (proportional 11 runs; Olmix best KL and ours 3 seeds) and orders the
Olmix rows by KL from high to low (Overleaf `8377cd9`).

Olmix KL 0.05 repeats measured (2026-09-07 23:45 PDT, parent succeeded, 2 preemptions): Uncheatable 1.0030 /
0.9997 / 1.0022 at trainer seeds 0/1/2, mean 1.0017 ± 0.0018 (the June v5p-8 ladder run measured 1.0039); paired
difference of our unconstrained optimum to this policy −0.0192 ± 0.0016 SE, against −0.0208 ± 0.0014 to the KL 0.1
policy. `summarize_delphi_fairness_repeats_20260908.py` now folds the package in (`fairness_summary.csv/md`); Table 2's
KL 0.05 row reads 1.002 ± 0.002 and `tab:seeds` carries the row.

Editorial round 3 (2026-09-08, Codex's third review): new composite motivation figure
`plot_motivation_composite_figure_20260908.py` (2x2: downsampling; N scaling; D scaling with MARINER's and Olmix's
heads fitted per curve in-sample; fixed-TPP diagonal; configurations table `tab:a-motivation` in the ladders appendix)
replaces the old Figures 2, 5 and 6; R1 scaling figure into Results (shared legend, no DSP track, no placeholder stars)
and Table 1 into the appendix; noise figure = R^2 panel only (`plot_fit_error_vs_snr_20260905.py --variant r2_only`);
`plot_delphi_swarm_support_20260907.py` now reads the frozen-procedure validation package (unconstrained optima;
Uncheatable outside the sampled range on literature high and olmOCR PDFs only). Prose: Background premise paragraph
restored (three problems), lab catalogue moved to Related Work, Methods claims narrowed (empirically motivated
allocation; no 'harm bounds repetition'; floor equation shows the 3-sigma margin; positive-deficit convention in
`app:fitting`), Results states end-to-end procedure comparison, ablation findings separated, KL appendix reworded
against the measured table. Main text ends a third down page 9 with Intro/Related/Analysis/Conclusion still stubs.
Details in the outline's round-3 note.

Learning-curve study closed (2026-09-08 00:36 PDT): MARINER + Olmix phase complete (1,680 fits, 10 draws; 59 draw-0
records refitted after a 19:58 registry edit had changed the protocol hash); the per-bucket-shape phase was stopped at
Calvin's request (15.6x per-fit cost, 10-20 h, not informative). Paper numbers: MARINER reaches Olmix's full-swarm
out-of-fold correlation at 80 / 100 runs (Uncheatable / suite), Olmix's full-swarm held-out regret at 60 / 80, 95% of its
own full-swarm correlation at 120 on both; full-swarm correlation 0.967 / 0.888 vs 0.894 / 0.802. Figures r6 + appendix
pair regenerated from `learning_curve_mariner_delphi_3e18_20260908` (`report.md`, `efficiency.csv`).

Three-component optimum submitted (Calvin, 2026-09-08 01:04 PDT): the frozen procedure re-targeted at the byte-weighted
AO3 + BBC News + Wikipedia aggregate with the standalone `mixture_selection.py` (objective `uncheatable_worsened`, no cap,
no KL; 5.63 epochs max, code sources at zero; predicted 1.136 on the trio, 1.108 on full Uncheatable), trained at data seed
666200, trainer seeds 0/1/2 on v6e-8: Iris `/calvinxu/dm-delphi-3e18-three-component-optimum-v6e8-20260908`, Fieldbook
`exp_01m200pjbcye298amh8fjsbhry` / `job_01m200qy55kb6a2tpyqzh45g9d`, package
`delphi_three_component_optimum_3e18_20260908`, launcher `launch_delphi_three_component_optimum_3e18.py` (+ test),
collector entry `three_component_optimum`. Table R-components already carries MARINER's component means (AO3 unchanged,
BBC News +0.044, Wikipedia +0.015) and Figure R4 the two frozen optima; Tables R-objectives and A-subset wait for the
runs (`collect ... --launch three_component_optimum`).

Table 10 coordinate rows (2026-09-08 02:30 PDT): registry parents `linear_exposure` and `olmix_loglinear_taskwise_exposure`
(source_model_ids empty so the coverage test holds; `OlmixTaskwiseModel` gained a `coordinate` field), harness run
`single_phase_observatory_coordinate_rows_20260908`. Least squares on epochs == least squares on weights exactly; Olmix's law on
raw epochs collapses (0.68/0.37/0.82/0.65/0.79/0.60) because its multistart Huber solver is tuned to [0, 1] inputs; both rows are
in the appendix table with that explanation. Appendix J (Codex, `sections/surrogate_properties.tex`) reviewed: math verified;
suggestions sent to Calvin (kappa counts: kappa = 1 on all 7 Uncheatable tasks and 28/51 suite tasks; drop the manual newpage;
paragraph labels resolve to the section number; the orange placeholder macro was removed). Drive copy built locally; not pushed
pending Calvin's decisions on Appendix J.

Shape-sharing follow-ups (2026-09-08 05:00 PDT): `PerBucketShapeGridModel.free_keys` + registry entries
`..._per_bucket_{threshold,rate,power}` (one shape parameter per bucket, the other two shared); run
`single_phase_observatory_per_bucket_partial_20260908`: all three within 0.010 of MARINER (ties), rows in Table 10.
Threshold-grid sensitivity: entry `..._fine_threshold` (tau in {0, 0.5, ..., 6}), harness run
`single_phase_observatory_fine_threshold_20260908` in flight; standalone refit + re-optimization in
`delphi_fine_threshold_sensitivity_20260908` (optima move by TV 0.11 / 0.04, max epochs 5.25 -> 5.14 and 7.50 -> 7.93;
11 suite tasks pile up at tau = 0). Overleaf pushed through e3e1e71 + later pushes; Appendix J (Codex) is on Overleaf.

Three-component optimum measured (2026-09-08 morning): trio aggregate 1.1102 ± 0.0008 (predicted 1.136), full Uncheatable
1.2080 ± 0.0011 (predicted 1.108), OBE 1.439 ± 0.004; all three components improve vs proportional and the full optimum;
code components 1.523 / 1.433. Tables R-objectives and A-subset, the subset appendix paragraph and the Results limitation
sentence updated; Fieldbook closed out; pushed.

Threshold-per-bucket as a revision candidate (2026-09-08): bank regret identical to MARINER (0.0016 / 0.0151), offline
optima within TV 0.06 / 0.05 and cross-predictions within 0.0005 BPB (`delphi_per_bucket_threshold_optima_20260908`);
no validation submitted, procedure stays frozen; appendix fitting paragraph states it.
Fine threshold grid (tau 0..6 by 0.5): OOF Spearman ties (largest +0.013), bank regret slightly worse (0.0035 / 0.0157 vs
0.0016 / 0.0151); row + caption sentence in Table 10; grid kept. Runs `single_phase_observatory_fine_threshold_20260908`.

Shape-sharing ablation measured (2026-09-07 evening, `single_phase_observatory_shape_sharing_20260908`, certify tier
on the three 39-bucket panels, `--curves none`, pinned protocol, 1740 fits, no failures): out-of-fold Spearman
MARINER 0.960 / 0.878 (Qwen3 U / T9), 0.964 / 0.934 (Llama 200M), 0.952 / 0.910 (Llama 160M); per-bucket shapes
0.964 / 0.876, 0.964 / 0.932, 0.952 / 0.904: a tie (|Δρ| ≤ 0.006, RMSE within 0.0006), not the loss Calvin expected;
117 extra parameters buy nothing, so the shared shape stays. Table 8 now has the MARINER baseline (well above the
additive variant on the suite: 0.878 vs 0.828 at Qwen3, 0.910 vs 0.876 at 160M) and the per-bucket row; the additive
rows keep a todo to be re-based. Harness fixes: `--panels` / `--curves` options, `aggregate_predictions` over scored
rows (the pinned run has no OOF prediction), report tolerates a missing Screen promotions table.

Re-basing the paper's development evidence on MARINER (Calvin, 2026-09-07 evening; preregistered hypothesis for the
shape-sharing learning curve: per-bucket shapes are much less sample-efficient and never beat MARINER):
- Learning curves under the pinned protocol: `learning_curve_mariner_fits_20260908.py` (models mariner,
  mariner_per_bucket, olmix on identical subsets; a subset of size k = k random runs of the 279 non-calibration runs
  plus the pinned proportional run in every training set; sizes 20..279; records in
  `reference_outputs/learning_curve_mariner_delphi_3e18_20260908/records`). Running: mariner + olmix (10 draws, 12
  sizes, 1680 jobs) then mariner_per_bucket (5 draws, sizes 20/40/80/120/160/240/279). Scoring and plotting:
  `learning_curve_metrics_20260905.py --study mariner`, `plot_learning_curve_20260905.py --study mariner` (the
  metrics script now pairs a primary model against every comparator; `primary_better_fraction`, `comparator`).
  Verify timings at k=20: MARINER ~1 s per component fit, per-bucket ~15 s; the per-bucket half is the slow one.
- Table 8 re-base: registry entries `@kappa_floor_link_flat15_nocap_{no_harm,row_scrambled_harm,permuted_inventory,
  weight_coordinate,common_inventory,signed_head,outcome_permutation}` (`_fitted_floor` now takes options / shapes /
  head_kind / ridge_grid); harness run `single_phase_observatory_mariner_ablations_20260908` (three 39-bucket panels,
  pinned, plus `olmix_loglinear_taskwise` and `linear_weight` under the same protocol), 5 workers alongside the
  learning curve's 10.
- Still to do after those land: Table 8 rows, Figure R6 + appendix learning-curve figures, the R7 / Table R-fit /
  SNR tables from MARINER's pinned component fits, the Uncheatable component table from the repeat means, the
  three-component optimum (needs new runs).

Table 8 re-based (2026-09-07 21:20 PDT, `single_phase_observatory_mariner_ablations_20260908` + the shape-sharing run;
every row under the pinned protocol, `table8_spearman.csv`): MARINER 0.960/0.878/0.964/0.934/0.952/0.910; harm
removed 0.920/0.754/0.881/0.904/0.899/0.850; scrambled harm 0.919/0.748/0.833/0.894/0.891/0.843; weights
0.919/0.828/0.951/0.920/0.933/0.886; permuted sizes 0.905/0.817/0.951/0.915/0.924/0.853; one pool size
0.888/0.830/0.953/0.923/0.936/0.880; signed head 0.965/0.875/0.954/0.935/0.966/0.935 (a tie or better on OOF ρ, worse
OOF regret on 200M Uncheatable; nonnegativity kept for interpretability); additive variant (no floor)
0.956/0.841/0.964/0.925/0.936/0.875; Olmix 0.899/0.821/0.889/0.882/0.856/0.877; linear 0.861/0.716/0.769/0.693/
0.813/0.756; permutation −0.240/0.009/−0.058/0.227/0.059/0.086. Paper Table 8 + §6.5 rewritten (Overleaf 23991f8).

Paper compression (2026-09-07 evening, Codex's editorial review, Calvin: keep the downstream noise analysis with
Figure R7): Background / Methods / Setup / Results rewritten to about 800 / 1,200 / 600 / 1,700 words (detex counts,
captions included); eleven floats in appendix section `app:moved`, the deletion statistic in the appendix reliability
section, the full twelve-row ablation table in the fitting appendix (`tab:a-ablations-full`), a five-row table in
the main text; Figure 1 = the v22 pipeline schematic; title "MARINER: Compute-Budgeted Data Mixing with an
Epoch-Aware Parametric Surrogate". Main text ends on page 8 with the Introduction, Related Work, Analysis and
Conclusion still uncompiled. Overleaf `b5b3b0e`.

KL ablation measured (2026-09-07 22:30 PDT, `delphi_kl_ablation_3e18_20260908/kl_ablation_summary.csv`): the penalty
never helps MARINER; Uncheatable degrades monotonically from 0.9829 (λ 0.005) to 1.0147 (λ 0.5) against 0.9814;
Table 9 stays within noise up to λ 0.075 (1.0657–1.0777, control 1.0682) except 0.05 (1.0830), then 1.0906 / 1.1052 /
1.1256. Raw predictions track the degradation (misses +0.001 to +0.012). Appendix table `tab:a-kl-ablation`; §6.1
sentence (Overleaf d832dd1); Fieldbook exp_01m1yjg60x0tz6tjrzj80pxm6e closed.
