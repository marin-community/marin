# Single-phase Observatory benchmark, round 2 (2026-09-02)

Second improvement round on the round-1 successor `weibull_softplus_unscaled`, under the round-1 protocol
(`single_phase_observatory_benchmark_cc_report_20260902.md`): Screen with matched ablations and controls on 38
family-macro units, promotion, Certify, external heldout selection, the 45-curve StarCoder gate, a five-repeat
finalist, and read-only Codex and DeepSeek reviews. User requirements for the round: try the mechanisms in payoff
order (shared shapes, bucket interactions, heteroskedastic or significance-informed weighting, quality-axis
families, cross-scale sharing, StarCoder grid and link fixes) and beat canonical DSP on every two-bucket
StarCoder curve. Fieldbook experiment `exp_01m1ge7ye6hz2epd0mjkbkrvt8`; plan in
`.agents/projects/single_phase_round2_plan_20260902.md`.

## 0. Answers

1. Which round-2 mechanisms help? At Screen none improves `weibull_softplus_unscaled` on the 38 family-macro
   units: shared shapes (three sharing units), both interaction forms, quality-axis pooling, the significance
   prior, the Huber head, the wide grid, and unbounded refinement are null or worse; the pairwise-interaction
   diagnostic is negative out of fold on every panel. The bounded log-deficit link is the one mechanism that beats the successor at Certify and in the five-repeat finalist on every tabular cell (finalist 60M 0.0059 / 0.0176, 300M 0.0040 / 0.0116, Delphi 0.0075 / 0.0260, dclm 0.102, high quality 0.064; 134 / 60 Certify-scope units), and the one that loses on the two-bucket StarCoder curves (worse than the benchmark's DSP on 15 of 45 out of fold). The CV-selected link matches it: in the finalist it leads on five of eight cells and trails on Delphi Table 9 and high quality, with no interval between the two excluding zero. Neither was promoted by the frozen rule (no interval excluded zero at Screen); both were advanced to Certify and the finalist by operator choice on their Screen tabular results, a post-hoc choice recorded in section 8.
2. Is DSP beaten on all two-bucket StarCoder settings? Out of fold the successor beats the benchmark's DSP on 40
   of 45 curves and every family macro; in-sample against the tied-curves page's four-parameter per-curve fit,
   only wide grid plus refinement reaches 45 of 45, and that variant collapses out of fold. Hull-bounded
   refinement on the standard grid is the variant that is safe out of fold (worse than DSP on 6 of 45) and beats
   the page's DSP in-sample on 43 of 45. No single variant satisfies both readings on every curve.
3. Prospective test: the 12 successor-proposed Delphi coordinates do not beat the pre-sweep bank on either target; six of the eight tabulated frozen models select one of them, and seven of the eight overestimate the gain at them by 2 to 7 percent (OLMix is pessimistic).
4. Round-2 candidate (not a successor: the plan requires both reviews and the StarCoder gate, which is not met): the link chosen per fit by inner CV (`@link_by_cv`) recovers the bounded link's tabular
   gains at Certify (60M 0.0068 / 0.0179, 300M 0.0038 / 0.0111, Delphi 0.0078 / 0.0263, dclm 0.104, high quality
   0.068; 123 / 54 Certify-scope units against the successor, p = 2e-7; Delphi Uncheatable contrast -0.0016 with the
   interval excluding zero) and halves the curve loss (worse than the benchmark's DSP on 10 of 45 out of fold;
   bounded link 15, successor 5). Finalist (five repeats): 0.0058 +- 0.0008, 0.0176 +- 0.0005, 0.0039 +- 0.0002, 0.0113 +- 0.0003, 0.0074 +- 0.0003, 0.0263 +- 0.0012 on the 39-bucket cells, 0.1017 +- 0.0027 (dclm), 0.0652 +- 0.0028 (high quality); contrasts against the successor -0.0020*, -0.0015, -0.0017, +0.0001, -0.0016*, -0.0032, -0.0109, -0.0277 (60M and Delphi Uncheatable intervals exclude zero); against canonical DSP -0.0045, -0.0056, -0.0015, -0.0014, -0.0022, -0.0038, -0.2989, -0.1189*; against OLMix every Uncheatable cell and both Michael cells exclude zero.


## 1. Code changes (all verified to reproduce 40 sampled round-1 shards bit-for-bit)

- Models: `LinkKind.LOG_DEFICIT_BOUNDED` (linear predictor capped at the largest training log-deficit plus
  `LINK_CAP_MARGIN` = 0.5 nats, cap stored on the fitted head); `FamilyOptions.interaction`
  (`total_square`, `family_products`: signed column pairs so NNLS can pick either direction);
  `FamilyOptions.quality_axis` (`benefit`, `harm`, `both`: columns pooled by quality bin across families) with
  `shuffled_quality` as the capacity control; `FamilyOptions.component_ridge` (per-component, per-bucket ridge
  multipliers keyed by bucket name); `Features.buckets_names` and `Features.component`; `Fitted.cv_table`
  (inner-CV RMSE for every shape and ridge); `GridModel.refine` (Nelder-Mead over log rate, power, and
  threshold from the grid argmin on the inner-CV objective, at most 80 evaluations). `GridModel.refine_bounded` clips the
  refined parameters to the candidate grid's hull. `GridModel.link_candidates` tries every (shape, ridge) candidate
  under each listed link and keeps the inner-CV winner (stored on the fit and used by predict and refinement).
- Harness: the fitted component is passed into `Features`; shards store the CV table; `--stage shared` selects
  one shape per sharing unit (target group, panel, or the three 39-bucket panels) by summing each component's
  CV error divided by its repeat SD (training response SD where no repeat SD exists) and refits every head with that shape and its own best ridge (the normalization mixes repeat SDs where they exist, the Delphi Table 9 components, with training-response SDs elsewhere, so panel- and scale-level sharing weights components by noise-data availability; shared shards are bound to their sharing unit's member set); Holm-corrected sign-test p-values (`sign_test_p_holm`, one family per
  comparator kind and metric); cache generations 3 and 4 recorded before the edits; the promotion rule now also
  covers successor ablations and the row-scrambled controls; the StarCoder inputs are loaded through the frozen
  module `single_phase_observatory_starcoder_inputs_20260902.py`.
- Registry: 15 successor ablations and controls (`@huber_head`, `@wide_grid`, `@interaction_total`,
  `@interaction_family`, `@quality_benefit`, `@quality_both`, `@quality_both_shuffled`,
  `@log_deficit_bounded_link`, `@ablation_prior`, `@scrambled_prior`, `@shared_shape_target`,
  `@shared_shape_panel`, `@shared_shape_scale`, `@refined_shape`, `@wide_grid_refined`) and the budget-matched
  `@exp_benefit_matched` from the DeepSeek disposition. The significance prior reads
  `reference_outputs/domain_ablation_pvalue_matrix_with_training_eval_20260623/domain_ablation_cell_pvalues.csv`
  (300M, one bucket deleted at a time against the proportional mixture with repeats): a bucket column gets ridge
  multiplier 10 for a metric whose two-sided deletion p-value is at least 0.05, and 1 otherwise; Table 9
  components map to `lm_eval/<task>_5shot/bpb` where that exists; the scrambled control permutes the multipliers across buckets per metric. Two defects found by the Codex review limit this ablation: the design cache keyed designs without the fitted component, so within a worker one component's multipliers could be reused by later components (fixed; both prior entries refit at Screen and Certify), and the p-value matrix is computed from the 300M deletion runs that are rows of the 300M fit panel, so its 300M cells use outer-test responses and are not interpretable without cross-fitting (not done; 60M and Delphi cells are free of this leak). Only 6 of the 51 Table 9 components map to a matrix metric (the aggregate MMLU components have bare names), so the prior mostly acts on the Uncheatable heads.
- Helper-source pins (`tests/data/single_phase_observatory_helper_pins.json`): a listed models-module helper may
  change only with a `DESIGN_REVISIONS` bump or a deliberate pin refresh after the reproduction check.

## 2. Diagnostics before the ablations

- Interaction structure. A ridge model of all 741 pairwise products of per-bucket Weibull benefits, fitted out
  of fold to the successor's Certify residuals on the same mixture-blocked folds, has negative out-of-fold R^2 on
  every 39-bucket panel (median -0.30 at 60M, -0.21 at 300M, -0.63 at Delphi; 0 of 58 components above 0.05;
  permuted-residual baseline -0.08 to -0.09). Pairwise interactions carry no out-of-fold information at 242 to
  280 mixtures.
- StarCoder in-sample reference. The tied-curves page fits canonical DSP per curve with four free continuous
  parameters on all points and no cross-validation; its in-sample RMSE is about 6.6 times below the benchmark's
  inner-CV DSP fits on the same curves. The gate therefore has two parts: in-sample fit and argmin under that
  protocol, and the benchmark's out-of-fold RMSE and regret.

## 3. Screen (38 family-macro units, worse / better against `weibull_softplus_unscaled`)

| Entry | Worse / better | p (sign) | Holm | Reading |
|---|---|---|---|---|
| `@wide_grid` | 4 / 0 | 0.125 | 1 | identical on every tabular cell; out-of-fold curves collapse (coupled 0.261 vs 0.038, dense 0.161 vs 0.081, fixed 0.167 vs 0.061) |
| `@interaction_family` | 8 / 10 | 0.82 | 1 | tie |
| `@interaction_total` | 17 / 21 | 0.63 | 1 | tie on tabular cells; dense-horizon family explodes (1.44) |
| `@ablation_prior` (refit after the cache fix) | 2 / 7 | 0.18 | 1 | 300M 0.0138, 60M 0.0202, Delphi 0.0219 against 0.0139 / 0.0205 / 0.0221; its 300M cells leak test outcomes |
| `@scrambled_prior` (refit) | 3 / 6 | 0.51 | 1 | control indistinguishable from the prior |
| `@log_deficit_bounded_link` | 14 / 24 | 0.14 | 1 | tabular better everywhere (300M 0.0112, 60M 0.0183, dclm 0.146, high quality 0.096), no explosion, curves worse (coupled 0.087, dense 0.112, matched 0.025) |
| `@huber_head` | 21 / 17 | 0.63 | 1 | tie |
| `@shared_shape_target` | 20 / 14 | 0.39 | 1 | worse on Michael (0.178 vs 0.162) |
| `@shared_shape_panel` | 23 / 11 | 0.058 | 1 | worse |
| `@shared_shape_scale` | 22 / 12 | 0.12 | 1 | worse (300M 0.0183) |
| `@quality_benefit` | 23 / 10 | 0.035 | 1 | worse |
| `@quality_both` | 26 / 8 | 0.0029 | 0.25 | worse |
| `@quality_both_shuffled` | 25 / 9 | 0.0090 | 0.72 | control equally worse: the quality columns are capacity |
| `@exp_benefit_matched` | 26 / 12 | 0.034 | 1 | Weibull kept (from the DeepSeek disposition) |
| `@refined_shape` | 23 / 15 | 0.26 | 1 | ties every tabular cell; out-of-fold curves collapse (dense 2.5, matched 2,829) |
| `@wide_grid_refined` | 23 / 15 | 0.26 | 1 | ties every tabular cell; curves 0.31 / 0.36 / 0.33 / 0.015 |
| `@refined_bounded` | 20 / 18 | 0.87 | 1 | refinement clipped to the grid hull: tie everywhere, curves 0.039 / 0.083 / 0.059 / 0.014 (no collapse) |
| `@wide_grid_refined_bounded` | 21 / 17 | 0.63 | 1 | curves still 0.27 / 0.24 / 0.27: the wide grid's hull, not the refinement, is what extrapolates badly |
| `@link_by_cv` | 14 / 23 | 0.19 | 1 | keeps the tabular gains (300M 0.0114, 60M 0.0179, Michael 0.147 / 0.098); curves between the two links |

Holm values are computed over the family of every model-versus-parent contrast present in the Screen report (one family per comparator kind and metric), so they move slightly whenever an entry is added; the values above are from the final report. The 38 units are correlated (18 anchors on 3 panels, 16 Michael tasks on 2 panels, 4 curve families), so the sign tests are descriptive orderings, not calibrated p-values. No round-2 mechanism improves the successor on the 38 units at Screen. Shared shapes, quality-axis pooling, and
the significance prior are null or negative; interactions are null; the bounded log-deficit link splits by
source (tabular better, curves worse).

## 4. StarCoder gate

The tied-curves page fits canonical DSP per curve with four free continuous parameters chosen on a three-fold
interleaved inner-CV objective (`fit_rung` with `interleaved_folds`), so the like-for-like in-sample comparison is
the benchmark's own CV-selected fit refit on all points. Under that protocol every candidate loses in-sample RMSE to
the page's DSP on most curves: successor 33 of 45 (median RMSE ratio 1.56), bounded log link 33 (1.91), bounded
refinement 35 (1.29), wide grid plus bounded refinement 28 (1.06), CV-selected link 32 (1.84); argmin regret is
worse than the page's DSP on 12 to 19 curves. An earlier comparison that selected the candidates' shape and ridge on
the training loss (successor 6 of 45, wide grid plus refinement 0 of 45) was not like-for-like and is withdrawn.
Out of fold under the benchmark's protocol the successor beats the benchmark's DSP on 40 of 45 curves and on every family macro (the one family-level exception at the per-model-curve split is the fixed-model 1B curve, 0.1030 against 0.1012); the wide grid and unbounded refinement collapse on held-out mixture blocks (shapes leave the grid:
power 1.6 or 0.002, rate 731); hull-bounded refinement is worse than DSP on 6 of 45, the CV-selected link on 10,
the bounded log link on 15. The in-sample gate ("beat DSP on every two-bucket curve") is therefore not met by any
variant; the out-of-fold gate is met by the successor on the family macros and on 40 of 45 curves. Hull bounds
were introduced after unbounded refinement collapsed on these same curves, and curve tasks use one partition at
every tier, so the bounded variant is a post-hoc repair on the gate curves, not an independent replication.
Per-curve tables: `starcoder_gate_round2_oof.csv`, `starcoder_gate_round2_insample.csv`,
`starcoder_gate_round2_link_oof.csv` beside the benchmark outputs (`starcoder_gate_round2_insample_objective_wide.csv`
is the withdrawn training-objective comparison, kept for the record).

## 4a. Prospective test: the refreshed heldout registry

On 2026-09-02 the user's completed `weibull_softplus_unscaled` epoch-cap sweep added 12 fit-panel-disjoint Delphi
3e18 coordinates (epoch caps 2 to 8, complete Uncheatable and 51-component Table 9) to
`reference_outputs/single_phase_heldout_benchmark_20260902/` (542 runs, 234 Delphi; bank 171 Uncheatable and
158 Table 9 coordinates). The heldout stage was rerun for the frozen set only (16 parents, 2 references, both
successors); fit shards are unaffected because the fit panels' inputs do not include the registry, and the
Delphi repeat SD is unchanged. Round-2 candidates were not scored on these rows, which stay prospective for the
frozen models and become development evidence the moment they inform selection.

- No proposed coordinate beats the pre-sweep bank: Uncheatable best 0.9811 against new 0.9834 to 1.0091;
  Table 9 best 1.0579 against new 1.0722 to 1.1292.
- Six of the eight tabulated models (canonical DSP, DSP-concentration, bucket-family GRP, Weibull family onset, both successors) now select one of the new coordinates as their argmin over the enlarged bank (Uncheatable regret 0.0023 to 0.0035, rank 5 to 7 of 171; Table 9 regret 0.0143 to 0.0157, rank 10 to 14 of 158); OLMix and GRP pairs do not (OLMix 0.0086 / 0.0190, GRP pairs 0.209 / 0.107). The successor's Delphi Uncheatable regret moves from 0.0012 to 0.0023. The table covers eight frozen models, not all twenty.
- Predictions at the proposed points are optimistic for seven of the eight tabulated models: Uncheatable bias -0.019 to -0.031 BPB (successor -0.030, canonical DSP -0.031, GRP pairs -0.019), Table 9 -0.027 to -0.067 (successor -0.041, DSP -0.067, GRP pairs -0.027); OLMix is pessimistic (+0.011 / +0.025). Within the 12 points the ranking is good for all but GRP pairs on Uncheatable (Spearman 0.92 to 0.99, GRP pairs 0.24; Table 9 0.89 to 0.99). The DSP and Weibull families rank their own proposals well and overestimate the gain at them by 3 to 7 percent; the successor is neither better nor worse than DSP here.
- Table: `reference_outputs/single_phase_observatory_benchmark_20260902/prospective_delphi_epoch_cap_analysis.txt`.

## 5. Certify, heldout, finalist

Certify (top-level tables, `paired_model_contrasts.csv`, `pooled_screen_contrasts_certify_scope.csv`; 194
Certify-scope units) and the five-repeat finalist (`finalist/`):

- `@log_deficit_bounded_link` is the best tabular model at both tiers. Certify aggregate RMSE against the
  successor: 60M 0.0068 / 0.0179 against 0.0076 / 0.0211, 300M 0.0039 / 0.0114 against 0.0052 / 0.0116, Delphi
  0.0081 / 0.0261 against 0.0092 / 0.0302, dclm 0.101 against 0.117, high quality 0.067 against 0.086 (contrasts
  -0.0019 / -0.0021 / -0.0015 / +0.0002 / -0.0013 / -0.0051 / -0.0153 / -0.0196); Certify-scope units 134 better /
  60 worse (Holm 4.9e-6). Finalist: 0.0059 +- 0.0008 / 0.0176 +- 0.0006 (60M), 0.0040 +- 0.0002 / 0.0116 +- 0.0004
  (300M), 0.0075 +- 0.0005 / 0.0260 +- 0.0009 (Delphi), 0.102 +- 0.003 (dclm), 0.064 +- 0.004 (high quality): ahead of every round-1 finalist on six of eight cells and tied on the other two, while the CV-selected link (below) leads it on five of eight cells with no interval between them excluding zero; 25-fold contrasts against the successor
  -0.0020 (60M Uncheatable, interval excludes zero), -0.0013, -0.0016, +0.0005, -0.0015, -0.0038, -0.011, -0.029;
  against canonical DSP all negative (-0.12 on high quality excludes zero); against OLMix every cell excludes zero.
- Its weakness is the two-bucket StarCoder curves: out of fold it is worse than the benchmark's DSP on 15 of 45
  curves (successor 5, bounded refinement 6) and loses 14 / 24 units to the successor at Screen, so it fails the
  StarCoder gate that the successor passes on the family macros.
- Bounded refinement ties the successor at Certify (within 0.0003 on every 39-bucket cell; 103 / 91 units) and
  no longer collapses on the curves; the significance prior, refit after the design-cache fix, is within 0.0004 of the successor on every aggregate (29 of 39 Certify-scope units better, p = 0.003, but its 300M cells leak test outcomes and the scrambled control is 21 / 18, so the unit result is not interpretable);
  quality-axis pooling is worse (117 / 73, Holm 0.047).
- Heldout selection for the frozen models on the refreshed registry is in section 4a; round-2 candidates were not
  scored on it.



## 6. Reviews

Codex (gpt-5.6-sol, maximum reasoning, read-only, scope: uncommitted changes) reviewed the round on 2026-09-02
21:38 to 22:15 (`single_phase_observatory_round2_codex_review_20260902.md`). Five P1 and eight P2 findings, all
verified against the code or the tables and all accepted:

- P1. The report named `@link_by_cv` a successor before the reviews and the StarCoder gate: relabelled a candidate
  (section 0).
- P1. The design cache keyed designs by model, panel features, and shape but not by the fitted component, so the
  component-dependent significance-prior designs could be reused across components inside a worker: the cache key
  now includes the component; both prior entries were refit at Screen and Certify and the tables regenerated.
- P1. The 300M significance prior is derived from deletion runs that are rows of the 300M fit panel, so its 300M
  cells use outer-test responses: recorded as not interpretable on 300M; cross-fitting is open.
- P1. The tied-curves page's DSP fit is CV-selected, so the training-objective in-sample comparison was not
  like-for-like: withdrawn and replaced by the CV-objective comparison (section 4), under which no variant beats
  the page's DSP on most curves.
- P1. Shared-shape shards were keyed per task, not per sharing unit: the shard now records and checks a hash of the
  unit's member set.
- P2. Correlated units at Certify scope (caveat added); mixed normalization scales in shared aggregation (stated);
  prospective claims overstated (corrected to six of eight tabulated models and to OLMix's positive bias);
  aggregate-MMLU aliases missing (coverage disclosed, 6 of 51); the Holm family depends on report filtering
  (stated, values refreshed); the CV-link entry's candidate count omitted the link dimension (fixed, 1,680
  evaluations); bounded refinement is a post-hoc repair on the gate curves (labelled); helper pins ignored module
  constants (the pin now records the resolved constants).

DeepSeek: DEEPSEEK_ROUND2_PLACEHOLDER


## 7. Limitations the reviews added

- Search budget: the wide grid (360 shapes), the CV-selected link (two links per candidate, 1,680 inner-CV
  evaluations) and the refinement entries (up to 80 extra evaluations) have no matched-budget control, unlike the
  quality-axis and prior entries; their Screen readings are confounded with capacity.
- The `@link_by_cv` shards record 840 candidates in their diagnostics because they were fitted before the count
  was corrected; the fits themselves are unaffected.
- The Holm family is whatever model-versus-parent contrasts are present in the report; it is not frozen.
- The gate and prospective-analysis scripts now live in the tree as
  `starcoder_gate_round2_20260902.py` and `prospective_heldout_analysis_20260902.py`.

## 8. Post-hoc choices

- Advancing `@log_deficit_bounded_link`, `@link_by_cv`, `@refined_shape`, `@wide_grid_refined`, `@refined_bounded`,
  `@wide_grid_refined_bounded` and `@ablation_prior` to Certify, and the first two to the finalist, was an operator
  choice on Screen tabular results; the frozen promotion rule promoted none of them.
- Hull-bounded refinement was designed after unbounded refinement collapsed on the gate curves.
- The CV-selected link was added after the bounded link's tabular win and curve loss were known.

## 9. Open items

- Cross-fit the significance prior inside each outer training fold (its 300M cells currently leak test responses).
- A per-curve in-sample fit that matches the tied-curves page's DSP under the same CV objective needs a
  continuous three-parameter optimizer with the same inner folds as the page; the grid-plus-refinement candidates
  reach a median in-sample RMSE ratio of 1.06 to 1.56 against it.
- Recompute the Screen inference at the panel or target-cluster level, or report the sign counts descriptively.
- Score the round-2 candidates on the refreshed Delphi registry only after the candidate choice is frozen; doing so
  converts the 12 prospective coordinates into development data.
- The shared-shape normalization needs one definition per unit (repeat SD is available only for the Delphi Table 9
  components) before the shared-shape null can be read as more than a tie.

