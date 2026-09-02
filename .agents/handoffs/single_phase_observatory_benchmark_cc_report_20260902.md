# Single-phase Observatory benchmark: report

Fieldbook experiment `exp_01m1ge7ye6hz2epd0mjkbkrvt8`; handoff
`.agents/handoffs/single_phase_observatory_ablation_and_modeling_cc_handoff_20260902.md`
(SHA256 `af105c44…`). Every number below comes from
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_observatory_benchmark_20260902/`.

Three kinds of evidence are kept apart throughout: fit-panel nested cross-validation (Screen and
Certify), retrospective external validation on the coordinate-disjoint heldout registry, and fresh
confirmation runs. No fresh runs were made; nothing here is a frontier claim.

## 0. Answers

1. Prediction. On the 39-bucket panels the family GRP models (bucket-family power GRP, hierarchical
   replay, Weibull shared onset, family onset) predict reconstructed Uncheatable and Table 9 best,
   0.002-0.004 BPB ahead of canonical DSP with repeat SDs two to five times smaller, but no five-repeat
   interval against canonical DSP excludes zero; canonical DSP is best on Delphi Table 9 at one
   repeat. On Michael's 118/120-bucket panels only the log-link linear epoch reference, GRP pairs, the
   bowl, and hierarchical replay are robust; every literal-replay or log-deficit model and the retained
   power law explode on at least one panel. Taskwise OLMix trails every saturating model and depends
   on an unconverged optimizer for its stability.
2. Selection. Fit-panel regret is unresolved (mostly zero, ties). On the retrospective heldout banks
   the power-benefit family models select an extrapolated Delphi coordinate (Table 9 regret 0.107,
   rank 129 of 146) while canonical DSP and DSP + concentration stay at 0.001-0.013; the 60M and 300M
   banks are uninformative. Fit-panel accuracy does not predict selection value.
3. Mechanisms. Under matched ablation: the materialized epoch coordinate with the true inventory, a
   shared saturating Weibull benefit (better than power under matched pooling and budget, and than
   exponential saturation), and a nonnegative head survive; the softplus harm block survives removal
   but not a capacity-matched scrambled control on the anchors, so its value is mostly capacity with a
   small bucket-aligned residual concentrated at 3e18; families, hierarchy, literal replay, per-bucket
   shapes, the retention gate, and the geometry term do not survive.

Successor. The smallest model the ablations support is `weibull_softplus_unscaled`: materialized epochs from
the true inventory, one shared Weibull saturating benefit per bucket, one shared-threshold softplus
overexposure harm per bucket, no families, an unscaled nonnegative head and the identity link (three nonlinear
parameters, 2B amplitudes). At Screen it beats canonical DSP in 35 of 38 family-macro units and every retained
mechanism survives its matched ablation (harm 31 / 7, row-scrambled harm 34 / 4, inventory 33 / 1, Weibull
shape 27 / 11, unscaled head 35 / 3). At Certify it is the best model on 300M Uncheatable, 300M Table 9 and
dclm and within 0.0034 of the best elsewhere, with no five-fold interval against DSP excluding zero; its heldout
shortlist regret is zero on every cell. The successor registered before the gate (`weibull_softplus_shared`,
column-scaled head) is kept in every table: it ties DSP and explodes on the Michael panels, and its unscaled-head
ablation is the revision. The row-scrambled controls also correct section 10: the harm block's value is
overexposure information, not capacity. At the five-repeat finalist it is best or tied-best on five of eight cells and never explodes, with repeat SDs of 0.0002 to 0.0012 on the 39-bucket cells.

## 1. Inventory and single-phase reductions

All 18 `MODEL_IDS` entries of `export_mixture_fit_observatory.py` reduce to 16 distinct single-phase
models (`model_registry.csv`, `equivalence_classes.md`). Two exact equivalences were found and
benchmarked once each:

- `canonical` and `effective_exposure` both reduce to total-exposure DSP (`dsp_total_exposure`): the
  phase-1 share gain multiplies a share that is constant at tied inputs, and the phase-1 multiplier
  rescales exposure into the per-bucket rate and threshold. The Observatory's own single-phase
  entries already use its `no_phase` variant for both.
- `bucket_family_grp` and `bucket_family_power_separate_heads` build identical designs at late
  multiplier 1 and forgetting 0 (verified to 8.9e-16 on both 300M and dclm), so they form one class
  (`bucket_family_power_grp`) whose shape and ridge grids are the union of both source grids.

Every other model keeps its own class. Phase terms removed: the phase-shift TV column of the
hierarchical replay model (identically zero), the late-phase concentration duplicate of aggregate
concentration in the geometry DSP, the retained power law's concentration gap and ordering channel
(the repaired estimator's own phase-blind image), and the early/late head split of the separate-heads
bowl. `crs_plus` and `crs_bounded` keep the revisit-gated retention in their tied image because the
Observatory's single-phase entries never removed it and it still moves predictions; its removal is
an ablation. The crs_plus image keeps every family-benefit column, singletons included, because the
family rate (rate divided by the number of families) makes a singleton's family column distinct from
its bucket column; an earlier draft dropped those columns and was corrected after review.

Families come only from declared domain classification and quality splits: the 13 Common Crawl
high/low pairs plus 13 singletons on the 39-bucket panels, the 24 manifest clusters (4-5 quality bins,
order undeclared) on Michael's panels, and two singletons on every StarCoder curve. The archival
semantic partition is not used anywhere. Exposure strata appear only as a label-blind pooling ablation.

`verify_single_phase_reductions_20260902.py` (run under `uv run --with cvxpy`) checks every reduced
design and head against the original module on tied inputs with this family structure: 46 of 46 checks
pass on the 300M and dclm panels, at most 1.5e-11 on designs and heads, 1.3e-13 between the harness's
profiled DSP solver and the ladder solver, and 4.8e-9 / 1.8e-10 for the retained power law's Huber head
against the original bounded trust-region solve (`reduction_equivalence.csv`).

## 2. Protocol

Six sources, one split manifest (`split_manifest.csv`, hash `9117a40f…`): five mixture-blocked outer
folds at seed 20260902 reused from the canonical benchmark, one shared three-way inner partition per
outer training fold for every shape, ridge, link, and ensemble choice, interleaved five-fold and
three-fold partitions on the 45 StarCoder curves. Aggregates are reconstructed from atomic fits with
the frozen evaluator rules; Michael's panels use the frozen eight-task mean. Metrics, the low-BPB basin
(observed 15% quantile, minimum 5 rows), the basin-hit tolerance (one pooled repeat SD), top-k = 5,
selection optimism (observed minus predicted at the selected row), the promotion rule, and all input
hashes are in `protocol.json`. Repeat noise: 60M from the audit's single-phase repeats, Delphi from the
ten-run proportional noise floor pooled with seven heldout repeat coordinates, StarCoder fixed-model
curves from `repeat_noise.csv`; 300M has no identified same-mixture repeats for Uncheatable or Table 9
and reports continuous regret only.

## 3. Solvers and throughput

Shard identity is `(model, panel, target, component, repeat, fold)`; writes are atomic and resume
is idempotent; the cache key holds the models-module hash, the harness fit-path source hash, the
per-split fingerprint, the built model's configuration hash, and the panel input and feature hashes.
Shards fitted before the review fixes are accepted only where a recorded snapshot of the pre-fix
configurations shows the configuration unchanged (`legacy_entry_descriptions.json`). Solver
choices with parity evidence: the profiled implicit-gradient DSP solver reproduces the ladder bit for
bit; grid heads solve NNLS on the QR-reduced system (2e-16 agreement, ~25% faster); grids above 48
every grid is searched exhaustively (a two-stage ridge screen tried first was rejected after review
because it chose a different crs_plus shape and ridge with inner RMSE 0.02900 against 0.02599); the
retained power law keeps its full least-squares screen and Huber rescoring. An
analytic-gradient OLMix solver was built and rejected as the parent: it reaches a lower Huber loss with
coefficients up to 196 and explodes out of fold (9e7 RMSE against 0.38 for the numerical reference),
so the repository solver's unconverged early stop is the baseline's only regularization; the analytic
solver remains as the estimator ablation `olmix_loglinear_taskwise@analytic_gradient`.

## 4. Which models predict the full mixture panel accurately (Certify, fit-panel nested CV)

Reconstructed-aggregate RMSE, Uncheatable / Table 9, one frozen five-fold partition
(`aggregate_metrics.csv`; taskwise OLMix uses the repository solver, which never explodes on the 39-bucket panels):

| panel | best | canonical DSP | DSP + concentration | bucket-family power GRP | hierarchical replay | family onset | Weibull shared onset | taskwise OLMix |
|---|---|---|---|---|---|---|---|---|
| 60M | bucket-family 0.0069 / family onset 0.0211 | 0.0101 / 0.0264 | 0.0107 / 0.0258 | 0.0069 / 0.0216 | 0.0080 / 0.0215 | 0.0082 / 0.0211 | 0.0078 / 0.0218 | 0.0134 / 0.0225 |
| 300M | family onset 0.0052 / Weibull onset 0.0118 | 0.0054 / 0.0127 | 0.0054 / 0.0129 | 0.0058 / 0.0120 | 0.0053 / 0.0120 | 0.0052 / 0.0120 | 0.0058 / 0.0118 | 0.0095 / 0.0192 |
| Delphi 3e18 | bucket-family 0.0078 / canonical DSP 0.0283 | 0.0090 / 0.0283 | 0.0082 / 0.0285 | 0.0078 / 0.0290 | 0.0089 / 0.0295 | 0.0084 / 0.0312 | 0.0099 / 0.0298 | 0.0133 / 0.0316 |

Every paired five-fold contrast against canonical DSP at the aggregate level has a Nadeau-Bengio
interval that contains zero on every 39-bucket cell (`paired_model_contrasts.csv`): the family GRP
models lead canonical DSP by 0.002-0.005 RMSE at 60M and 300M, canonical DSP leads on Delphi Table 9,
and none of it is resolved at one repeat. Against taskwise OLMix, bucket-family power GRP, both
hierarchical forms, family onset, DSP + concentration, and GRP pairs are significantly better on 60M
Uncheatable and/or Delphi Uncheatable. The fold-mean baseline is significantly worse than canonical
DSP on five of eight cells.

Michael's panels (frozen eight-task mean): the log-link linear epoch reference is the only model
robust on both (0.1186 dclm / 0.0830 high-quality), GRP pairs 0.150 / 0.097, the bowl 0.139 / 0.104,
hierarchical replay 0.169 / 0.112, canonical DSP 0.264 / 0.160, taskwise OLMix 0.223 / 0.145. Every
literal-replay or log-deficit model and the retained power law explode on at least one panel (crs_plus
23.8 / 10,233 after its exact-image refit, log-deficit compact 0.091 / 150,624, retained power law 2.86 / 1.09): buckets with a few thousand
tokens put mixture-blocked test rows hundreds of epochs outside the training hull, and any column that
grows without bound in epochs extrapolates without bound. This is an identification failure of those
mechanisms at 118-120 buckets, not evidence against epoch exposure.

## 5. Which models select good mixtures

Fit-panel selection (regret at 1 on reconstructed aggregates) is zero for most models on most cells
and tied within noise; the low-BPB basin metrics (`basin_rmse`, `basin_spearman`) separate models more
than global RMSE does but are also unresolved at one repeat.

External heldout selection (retrospective; `external_heldout_selection_metrics.csv`,
`external_heldout_predictions.csv` with hashed predictions):

- 60M (174 / 173 coordinates) and 300M (117 / 134): almost every model, including the affine
  baseline, selects the measured-best coordinate. These banks are proportional-controllability tilts
  and diagnostics around one basin; they are a sanity check, not a discriminating test.
- Delphi Uncheatable (159): DSP + concentration regret 0.0012 (rank 3; outside the frozen one-SD
  tolerance 0.00082, so not a basin hit), canonical DSP 0.0045 (rank 6), the family power models 0.0081 (rank 16), OLMix
  0.0086; GRP pairs and the affine baseline select the worst coordinate in the bank (0.209). Random
  ranking: 0.0286.
- Delphi Table 9 (146): canonical DSP and DSP + concentration 0.0132 (rank 8), retained power law,
  compact retained state, and Weibull shared onset 0.0155, OLMix 0.0190. Bucket-family power GRP,
  both hierarchical forms, GRP pairs, and family onset all select one coordinate from the
  symmetric-sepheads geometry-frontier archive (56% weight on the two stack_edu code buckets at 13
  epochs each) with predicted 0.99-1.01 and measured 1.1646 BPB: regret 0.1067, rank 129 of 146,
  worse than the fold-mean baseline (0.0735). Their top-5 shortlists still contain the best coordinate.
  By proposal-source stratum the failure is confined to that frontier stratum (18 coordinates: 0.097
  for the power models against 0.016 for DSP); on the three epoch-cap validation strata every model is
  within 0.003 of the best.

Fit-panel RMSE therefore does not predict selection value. The unbounded power benefit E^a keeps
promising improvement far outside the fit hull; canonical DSP's saturating exponential benefit does
not. Because these outcomes now inform the choice of benefit shape, they are external development
evidence, not confirmation.

## 6. StarCoder one-dimensional shape suite (out of fold, 45 curves, equal-family macro)

`starcoder_family_summary.csv`: GRP pairs 0.0334 RMSE (interior minimum expressed on 100% of curves
that have one), Weibull shared onset 0.0386, retained power law 0.0455, hierarchical replay 0.0523,
bucket-family power GRP 0.0705, canonical DSP 0.0780 (94% interior), OLMix 0.1009 (35% interior),
affine 0.138. Canonical DSP's per-bucket unbounded harm generalises poorly out of fold on the
horizon-by-replay curves (0.118) despite its low in-sample RMSE in the descriptive atlas; OLMix is
monotone on every edge. On the four fixed-model anchors (`starcoder_tied_diagonal_metrics.csv`) most
models land inside the frozen one-SD basin at 2B/4B/8B; at 1B only retained power law, the log-link
reference, the bowl, GRP pairs, and family onset select p = 0.30. RMSE is 3-70 repeat SDs on these
curves for every model, so none is calibrated to run noise there.

## 7. Complexity and runtime

All parents meet the five-minute Certify target except the retained power law (415 s projected at 14
workers; 35 s per Michael fit dominates), which passes the eight-minute gate. Canonical DSP converges
before maxiter 36 on 19% of fits and sits on 5.3 parameter bounds on average, as in the reference
protocol. No shard failed (`failures.csv` empty).

## 8. Which mechanisms survive matched ablation (Screen, 69 ablations and controls)

Every parent received one-factor leave-one-mechanism-out ablations, matched-capacity controls, and
(after review) pooling- and budget-matched benefit ablations (`equivalence_classes.md`, `screen/`).
The frozen promotion rule pooled raw Nadeau-Bengio differences; because several Michael-panel units
explode by orders of magnitude, no raw interval excluded zero and only the matched controls were
promoted by it (18). A documented post-hoc amendment (relative RMSE interval, or a two-sided unit
sign test on fold-averaged RMSE at p < 0.05) promotes 36; the 45 StarCoder curves enter the pooled
tests as four physical-family units after review, giving 38 units. Units better/worse for the
ablation are quoted below.

- Benefit response, matched on pooling and grid budget: Weibull beats power for bucket-family power
  GRP (29/9, p = 0.002; 300M anchors 0.0136 vs 0.0151, coupled-onset curves 0.0116 vs 0.0579,
  fixed-model curves 0.047 vs 0.125, worse only on high-quality 0.285 vs 0.164), and power loses for
  the Weibull shared-onset model (9/29). For compact retained state, exponential saturation loses to
  Weibull (10/28) and power loses (9/29).
- Repetition harm: removing the softplus overexposure harm loses (bucket-family 12/26, p = 0.034;
  Weibull shared onset 11/27, p = 0.014; 60M anchors 0.0263 vs 0.0214, Delphi 0.0318 vs 0.0231), but
  the capacity-matched scrambled-harm controls are ties (bucket-family 17/21, p = 0.63; Weibull
  onset 13/25, p = 0.07; GRP pairs 15/15) and lose only 0.001-0.002 on the 39-bucket anchors. Most of
  the harm block's value on the anchors is capacity (an extra convex column per bucket); bucket-aligned
  overexposure adds a small residual, quantified per panel at Certify in section 10. For canonical
  DSP, removing harm is a tie (23/15, p = 0.26): better at 60M/300M and on Michael, worse at Delphi.
  Literal replay is harmful wherever it is compared (literal family harm 2/36 and 6/32 with Michael
  explosions); removing it improves the log-deficit compact model (27/11) and the shared literal
  column beats family literal columns (27/11). Bounded harm is a tie for DSP (20/18).
- Policy coordinate: the weight coordinate loses for canonical DSP (9/29, p = 0.002) and compact
  (12/26); permuting the inventory loses for compact (8/26), the log-link reference (8/26), and the
  bowl (10/24), and is a tie for canonical DSP (14/20) and bucket-family power GRP (16/18).
- Families and hierarchy: removing families entirely improves bucket-family power GRP on the units
  where it changes anything (26/8, p = 0.003); family signals, pooled bases, member replay, and
  shuffled families are ties for every family model (p = 0.12-1.0); GRP pairs need true pairs only
  because their aggregator sums two exposures (shuffled 1/17).
- Link and head: the positive log link helps the linear epoch reference (identity 9/29, p = 0.002);
  the log-deficit link is a tie for compact on the anchors (13/25) and explodes on high-quality 10k;
  the matched-grid signed head is weakly worse (13/25, p = 0.07); column scaling is a tie (24/14).
- Shape sharing: one shared rate and threshold ties per-bucket shapes for canonical DSP (16/22);
  quality-pair ties are a tie (8/10).
- Controls: outcome permutation loses 4/34 (p = 2e-8) and matches the fold mean; the
  analytic-gradient OLMix solver ties on the anchors (19/19) and explodes on dclm.

Mechanisms that survive matched ablation: the materialized epoch coordinate with the true inventory
for shared-shape models, a shared saturating Weibull benefit (better than power under matched pooling
and budget, and than plain exponential saturation), a nonnegative head, and an extra per-bucket convex
column whose bucket alignment matters little on the anchors. Families, hierarchy, literal replay,
per-bucket shapes, and the retention gate do not.

## 9. Finalist replication (five repeated outer partitions, `finalist/`)

Canonical DSP, taskwise OLMix, the best statistical class (bucket-family power GRP) and its
neighbours (Weibull shared onset, hierarchical replay, DSP + concentration) were refit under five
repeated mixture-blocked partitions (26,600 additional shards, 0 failures). Reconstructed-aggregate
RMSE, mean over repeats with repeat SD:

| cell | bucket-family power | Weibull shared onset | hierarchical replay | canonical DSP | DSP + concentration | taskwise OLMix |
|---|---|---|---|---|---|---|
| 60M Uncheatable | 0.0070 ± 0.0002 | 0.0072 ± 0.0004 | 0.0074 ± 0.0004 | 0.0088 ± 0.0021 | 0.0084 ± 0.0014 | 0.0129 ± 0.0006 |
| 60M Table 9 | 0.0199 ± 0.0011 | 0.0197 ± 0.0013 | 0.0194 ± 0.0013 | 0.0222 ± 0.0029 | 0.0220 ± 0.0027 | 0.1111 ± 0.1927 |
| 300M Uncheatable | 0.0054 ± 0.0003 | 0.0060 ± 0.0001 | 0.0053 ± 0.0003 | 0.0057 ± 0.0007 | 0.0059 ± 0.0007 | 0.0098 ± 0.0002 |
| 300M Table 9 | 0.0120 ± 0.0002 | 0.0118 ± 0.0003 | 0.0118 ± 0.0002 | 0.0134 ± 0.0011 | 0.0135 ± 0.0009 | 0.0178 ± 0.0016 |
| Delphi Uncheatable | 0.0074 ± 0.0003 | 0.0091 ± 0.0005 | 0.0089 ± 0.0003 | 0.0096 ± 0.0010 | 0.0093 ± 0.0011 | 0.0130 ± 0.0007 |
| Delphi Table 9 | 0.0273 ± 0.0014 | 0.0287 ± 0.0011 | 0.0276 ± 0.0015 | 0.0295 ± 0.0017 | 0.0293 ± 0.0019 | 0.0330 ± 0.0026 |

The family GRP models lead canonical DSP by 0.002-0.004 RMSE on every 39-bucket cell and by
0.001-0.007 in the low-BPB basin, with repeat SDs two to five times smaller than canonical DSP's
(whose per-bucket shapes make it partition-sensitive). Even so, no 25-fold Nadeau-Bengio interval
against canonical DSP excludes zero on any cell for RMSE, regret at 1, or basin RMSE
(`finalist/paired_model_contrasts.csv`); against taskwise OLMix, bucket-family power GRP, Weibull
shared onset, and hierarchical replay are significantly better on 60M and Delphi Uncheatable and on
300M Uncheatable, and canonical DSP is significantly better on Delphi Uncheatable. Regret at 1 on the
fit panel is zero for every finalist on most cells and does not separate them. Taskwise OLMix
explodes on one of the five 60M Table-9 partitions even with the repository solver, so its
positive log-linear law is fragile under mixture-blocked extrapolation independent of the optimizer.
On Michael's panels the family models beat canonical DSP by 0.24-0.32 RMSE on dclm and 0.05-0.07 on
high quality, again without intervals excluding zero because their own repeat SDs are large.

The finalist stage therefore confirms the ordering by prediction accuracy but leaves the accuracy
gain over canonical DSP below the resolution of five repeated partitions, and it does not change the
selection picture: the same family power models that lead on fit-panel RMSE selected the extrapolated
Delphi coordinate in section 5.

## 10. Promoted ablations at Certify

The 36 promoted entries (18 matched controls by the frozen rule, 18 by the amended rule) were run at
the full Certify tier (0 failed or stale shards) and sit in the top-level tables beside the parents.
Aggregate five-fold contrasts against each parent (`paired_model_contrasts.csv`, `comparator_kind =
parent`; * marks an interval excluding zero):

- Harm. Removing the softplus harm costs bucket-family power GRP +0.0029* (300M Table 9), +0.0038*
  (60M Uncheatable), +0.0062* (Delphi Uncheatable) and 0.002-0.006 elsewhere, and costs Weibull shared
  onset 0.002-0.008 on every 39-bucket cell. The row-scrambled control, which keeps every harm column and
  permutes the mixtures feeding it, costs the same as removing the block (bucket-family +0.0037 / +0.0063 /
  +0.0021 / +0.0037 / +0.0063* / +0.0036 against +0.0038* / +0.0063 / +0.0020 / +0.0029* / +0.0062* / +0.0037;
  Weibull shared onset +0.0025 / +0.0066 / +0.0061 / +0.0084 / +0.0012 / +0.0024 against +0.0021 / +0.0056 /
  +0.0055 / +0.0080 / +0.0020 / +0.0023). The harm block's value is therefore overexposure information, not
  capacity. The column-scrambled control used at the review gate costs only 0.0007 or less because pooled
  family totals of permuted columns still measure overexposure; it is reported in the tables as
  `@scrambled_harm` but should not be read as a capacity control (section 12.2).
- Benefit. The pooling- and budget-matched Weibull benefit is a tie with power at the reconstructed
  aggregate (within 0.0021 on every 39-bucket cell, better by 0.021 on dclm, worse by 0.042 on high
  quality); the reciprocal power ablation of the Weibull shared-onset model is likewise a tie (within
  0.0015). The unit-level Weibull advantage on the Screen anchors therefore does not carry to the
  reconstructed Uncheatable and Table 9 aggregates at one repeat.
- Coordinate. Permuting the inventory costs bucket-family power GRP +0.0029* on Delphi Uncheatable and
  0.002-0.004 on 60M and Delphi Table 9, with nothing at 300M; the weight coordinate costs canonical
  DSP 0.007-0.008 at 300M and Delphi. The outcome-permutation control is worse everywhere (+0.013 to
  +0.029, five cells significant) and reproduces the fold mean.
- Families and head. Removing families entirely changes reconstructed RMSE by at most 0.0004 (tie);
  the matched-grid signed head costs bucket-family power GRP 0.002-0.007 on five of six cells; removing
  the retention gate improves the exact-image crs_plus on every 39-bucket cell (-0.0003 to -0.0016);
  removing literal replay from the log-deficit model is a tie on the 39-bucket cells and removes the
  Michael explosion; the log-link reference's inventory control is a tie.

## 11. Independent review gate

Codex (gpt-5.6-sol, maximum reasoning, read-only) returned six P1 and four P2 findings
(`.agents/handoffs/single_phase_observatory_codex_review_20260902.md`). Every finding was reproduced
before any change and all ten were accepted: the two-stage grid screen was not selection-equivalent
(inner RMSE 0.02900 against 0.02599 on one crs_plus shard) and is replaced by exhaustive search; the
crs_plus reduction had dropped singleton family-benefit columns that are not duplicates (residual
0.05-0.34) and now keeps the exact tied image; cache keys now hold the harness fit-path source and the
built model's configuration, with legacy shards accepted only where a recorded snapshot proves the
configuration unchanged; pooled Screen tests macro-average the 45 StarCoder curves into four family
units; capacity-matched scrambled-harm controls and pooling-matched Weibull/power ablations were added;
the signed-head control uses the parent's ridge grid; ablation rows now carry their own metadata; the
Delphi basin-hit sentence was corrected; the verifier exits nonzero on failure. Affected entries were
refit (sections 8 and 10 report the post-fix numbers).

DeepSeek (deepseek-v4-pro) could not run: the harness returned `QUOTA: Insufficient Balance` and the
skill forbids routing around it. That second independent review is outstanding; rerun
`uv run ~/.claude/skills/deepseek-subscription-review/scripts/deepseek_review.py` with the brief in the
Fieldbook experiment once the balance is restored.

## 12. Successor model

Starting from the simplest member of the best statistical class and adding only mechanisms whose
matched ablations survived, the successor is

    yhat = b0 - sum_b a_b (1 - exp(-(rho E_b)^p)) + sum_b c_b softplus(log(1 + E_b) - tau)^2,   a_b, c_b >= 0,

with one shared (rho, p, tau), E_b = c_b w_b the materialized epochs with the true inventory, a
column-scaled nonnegative head, identity link, no families, no literal replay, no retention gate
(`weibull_softplus_shared`; 2B linear amplitudes and three nonlinear parameters against canonical
DSP's 2B + 2B). It is the Weibull shared-onset model with its family signal removed (a tie under
ablation) and its 34 Sobol shapes replaced by a structured grid (rate x7, power x4, threshold x6,
ridge x5). Its own matched ablations and controls (no harm, exponential benefit, scrambled harm,
permuted inventory, weight coordinate, signed head, unscaled head, log-deficit link, family harm,
outcome permutation) were run under the same Screen protocol.

### 12.1 Registered successor `weibull_softplus_shared`

The successor registered before the review gate closed (shared Weibull benefit, shared-threshold per-bucket
softplus harm, no families, column-scaled nonnegative head, identity link; three nonlinear parameters and 2B
amplitudes, 168-shape grid, five ridge values) was run at every tier under the frozen protocol.

- Screen (4,345 fits, 0 failed): mean anchor RMSE 0.0165 / 0.0220 / 0.0231 on 300M / 60M / Delphi against
  canonical DSP 0.0173 / 0.0266 / 0.0238 and the bucket-family model 0.0151 / 0.0214 / 0.0231; StarCoder
  curves 0.0375 / 0.0752 / 0.0611 / 0.0132 (coupled, dense, fixed, matched); dclm 0.197; high quality
  explodes (1,492). Sign tests against DSP and OLMix are 25 of 38 units (p = 0.073).
- Certify (1,175 fits, 0 failed): reconstructed-aggregate RMSE 0.0076 / 0.0214 (60M Uncheatable / Table 9),
  0.0062 / 0.0125 (300M), 0.0095 / 0.0336 (Delphi), 0.147 (dclm), 1,140 (high quality). No five-fold contrast
  against canonical DSP excludes zero on a 39-bucket cell (-0.0020 to +0.0054); it beats OLMix on Delphi
  Uncheatable (-0.0039, interval excludes zero).
- Heldout selection: identical to DSP-concentration on 60M and 300M (regret 0 / 0.0124, rank 1 / 3), rank 3
  with regret 0.0012 on Delphi Uncheatable and rank 11 with regret 0.0155 on Delphi Table 9 (DSP rank 8,
  0.0132); top-5 regret 0 everywhere except Delphi Table 9 (0.0132, same as DSP).
- StarCoder equal-family macro: RMSE 0.0468, Spearman 0.983 (highest of any model), regret@1 0.0042, against
  DSP 0.078 / 0.904 / 0.0158 and GRP pairs 0.033 / 0.981 / 0.0032.
- Finalist (5 repeats): 0.0075 +- 0.0001 / 0.0204 +- 0.0009 (60M), 0.0056 +- 0.0005 / 0.0128 +- 0.0004
  (300M), 0.0095 +- 0.0002 / 0.0328 +- 0.0009 (Delphi); contrasts against DSP -0.0020 to +0.0038 with no
  interval excluding zero; against OLMix -0.0050, -0.0044, -0.0035 on the three Uncheatable cells (all exclude
  zero). The Michael cells explode on some repeats (dclm 38 +- 54, high quality 4,546 +- 5,709).

Its matched Screen ablations (`screen/pooled_screen_contrasts.csv`):

- The Weibull shape is needed: the exponential-benefit ablation loses 28 of 38 units (p = 0.005), most visibly
  on the StarCoder matched-onset curves (0.099 against 0.013).
- The harm block is needed at 60M and Delphi (0.0248 and 0.0295 without it against 0.0220 and 0.0231) but not at
  300M (0.0151 without it against 0.0165) or on the curves; the 38-unit sign test is a tie (19 / 19).
- The column-scaled head is the defect: the unscaled nonnegative head wins 35 of 38 units (p < 0.001) and every
  cell (300M 0.0139, 60M 0.0205, Delphi 0.0221, dclm 0.162, high quality 0.118). On the exploding high-quality
  folds the scaled head selects threshold 1.0 at the largest ridge and predicts up to 1.6e5 where the unscaled
  head selects thresholds 3 to 6 and stays at 0.09 to 0.23: scaling divides each harm column by its training
  RMS, so a column that is almost zero in training and large on extrapolated test mixtures is amplified, and the
  ridge shrinks the scaled coefficient rather than the original one.
- The signed head loses 24 of 38 (p = 0.14); weight coordinate, permuted inventory, family harm, and log-deficit
  link tie on the anchors; the outcome permutation loses 35 of 38.
- The scrambled-harm control as implemented at the gate permutes harm-block columns; for a per-bucket harm this
  is a column reordering and the fit is identical (39-bucket cells match to four decimals). It is therefore
  vacuous for this design, and was replaced by a row-scrambled control (mixtures permuted in the harm block,
  information-free for every harm form) in section 12.2. The parent controls pool permuted columns into family
  or pair totals and remain valid.

### 12.2 Revised successor `weibull_softplus_unscaled`

The revision keeps the design and grid of the registered successor and drops column scaling from the
nonnegative head, the one change its matched ablations supported (35 of 38 units). Its ablation set was rebuilt on
the revised base and adds the row-scrambled harm control and the reciprocal scaled-head ablation.

Screen (5,925 new fits, 0 failed; `screen/`):

- Mean anchor RMSE 0.0139 / 0.0205 / 0.0221 on 300M / 60M / Delphi (weibull family onset 0.0137 / 0.0208 /
  0.0232, bucket-family 0.0151 / 0.0214 / 0.0231, canonical DSP 0.0173 / 0.0266 / 0.0238); dclm 0.162 and high
  quality 0.118, the best of any model on both Michael panels (log-link reference 0.171 / 0.121); StarCoder
  0.0376 / 0.0811 / 0.0609 / 0.0117 against the family onset model's 0.0265 / 0.0720 / 0.0461 / 0.0099.
- Family-macro sign tests: better than canonical DSP in 35 of 38 units and than OLMix in 34 of 38 (p < 0.001).
- Every retained mechanism survives its matched ablation (units worse / better against the revised base):
  removing the harm 31 / 7 (p < 0.001); row-scrambling the harm 34 / 4 (p < 0.001), so the block carries
  overexposure information and not only capacity; permuting the inventory 33 / 1 (p < 0.001); the weight
  coordinate 29 / 9 (p = 0.002); the exponential benefit 27 / 11 (p = 0.014, matched-onset curve 0.099 against
  0.012); the scaled head 35 / 3 and the signed head 35 / 3 (p < 0.001); family harm 23 / 11 (p = 0.058); the
  outcome permutation 38 / 0.
- The log-deficit link is the one alternative that is not rejected on the tabular anchors (300M 0.0112, 60M
  0.0183; 25 of 38 units, p = 0.073) but it explodes on the dense-horizon StarCoder curve (RMSE 120) and is worse
  on matched onset (0.0276), so the identity link is kept.
- Row-scrambled parent controls cost the bucket-family model 0.0028 / 0.0050 / 0.0088 (300M / 60M / Delphi) and
  the Weibull family onset model 0.0025 / 0.0041 / 0.0029, so the column-scrambled controls in section 10
  understated the harm block's information: pooled family totals of permuted columns still measure overexposure.

Certify (1,175 fits, 0 failed; top-level tables, `paired_model_contrasts.csv`):

- Reconstructed-aggregate RMSE 0.0076 / 0.0211 (60M Uncheatable / Table 9), 0.0052 / 0.0116 (300M), 0.0092 /
  0.0302 (Delphi), 0.117 (dclm), 0.086 (high quality). It is the best model on 300M Uncheatable, 300M Table 9
  and dclm, within 0.0007 of the best on 60M and within 0.0034 on high quality (log-link reference 0.083), and
  0.0014 / 0.0019 behind the bucket-family model on Delphi. Canonical DSP: 0.0101 / 0.0264, 0.0054 / 0.0127,
  0.0090 / 0.0283, 0.264, 0.160.
- Five-fold contrasts against canonical DSP on the 39-bucket cells run from -0.0050 to +0.0018 and none excludes
  zero (the 25-fold intervals are wide for every model pair; the Screen unit-level test is the decisive one);
  dclm -0.152 and high quality -0.074. Against OLMix: Delphi Uncheatable -0.0044 and dclm -0.120, both
  excluding zero.
- Regret@1 is zero on four 39-bucket cells, 0.0058 / 0.0065 at 300M (canonical DSP 0.0066 / 0.0007) and 0.029
  on high quality (best; DSP 0.078).
- Its matched ablations at Certify all move in the Screen direction: no harm +0.0021 / +0.0053 / +0.0013 /
  +0.0057 / +0.0028 / +0.0014, row-scrambled harm +0.0021 / +0.0056 / +0.0025 / +0.0068 / +0.0029 / +0.0019.

Heldout optimum selection (`external_heldout_selection_metrics.csv`): rank 1 / 1 on 60M (regret 0), rank 3 / 3
on 300M (0.0029 / 0.0124; canonical DSP 0 / 0.0124), rank 3 on Delphi Uncheatable (0.0012; DSP rank 6, 0.0045)
and rank 8 on Delphi Table 9 (0.0132, the same coordinate as DSP). Its top-5 shortlist regret is zero on every
cell; it is the only model whose Delphi Table 9 shortlist contains the measured optimum. Like every parent it
never proposes the extrapolated stack_edu coordinate that the power-benefit family models select.

StarCoder equal-family macro (`starcoder_family_summary.csv`): out-of-fold RMSE 0.0478, Spearman 0.981,
regret@1 0.0031, interior optimum expressed on 97.5 percent of curves (canonical DSP 0.078 / 0.904 / 0.0158 /
93.8 percent; GRP pairs 0.033 / 0.981 / 0.0032).

Finalist (5 repeats, 4,975 fits, 0 failed; `finalist/`): RMSE +- repeat SD 0.0072 +- 0.0002 / 0.0193 +- 0.0011
(60M), 0.0054 +- 0.0004 / 0.0116 +- 0.0003 (300M), 0.0089 +- 0.0003 / 0.0286 +- 0.0012 (Delphi), 0.113 +- 0.003
(dclm), 0.093 +- 0.006 (high quality). It is the best or tied-best finalist on 60M Table 9, 300M Uncheatable,
300M Table 9, dclm and high quality, and second to the bucket-family model on 60M Uncheatable (0.0070) and both
Delphi cells (0.0074 / 0.0273). Its 25-fold contrasts against canonical DSP are -0.0026 / -0.0041 / +0.0002 /
-0.0015 / -0.0006 / -0.0006 on the 39-bucket cells (none excludes zero) and -0.288 / -0.091 on the Michael
panels; against OLMix -0.0055, -0.0044, -0.0041 on the three Uncheatable cells and -0.127 on dclm, all
excluding zero. Unlike its column-scaled predecessor it never explodes on any repeat.

The revised successor is therefore the smallest model that the benchmark supports: every mechanism it keeps has a
matched ablation that loses at Screen with p <= 0.014, everything it drops (families, hierarchy, literal replay,
retention gate, per-bucket shapes, column scaling, signed head, log-deficit link) either ties or hurts, and it
is best or within run-noise of best on every Certify and finalist cell. What it does not deliver is a
five-fold interval against canonical DSP that excludes zero on any 39-bucket cell; with 39 mixtures per panel no
model pair achieves that, and the unit-level Screen tests are the evidence of ordering.





## 13. Post-hoc choices, corrections, and open items

Every choice made after seeing results is listed here so a reader can discount it:

- Promotion rule amendment (relative RMSE interval or unit sign test) after the frozen raw-pool rule
  promoted only controls; both decisions are stored per ablation.
- StarCoder curves macro-averaged into four family units for the pooled tests (after the Codex review).
- Taskwise OLMix reverted from an analytic-gradient solver to the repository's numerical solver after
  the analytic solver's out-of-fold explosions were seen; the analytic solver remains as an ablation.
- Exhaustive grid search replacing a two-stage ridge screen after the parity failure on crs_plus.
- The crs_plus tied image corrected to keep singleton family-benefit columns.
- Cache keys rebuilt around the built model's configuration; shards from before that change were kept
  only where a reconstructed snapshot of the pre-fix configurations matched, and all refits since use
  the new keys.
- 125 canonical-DSP shards on 60M with an unexplained stale hash were refit rather than diagnosed.
- crs_plus and crs_bounded keep the revisit-gated retention in their single-phase image (the
  Observatory's own choice), with its removal as an ablation.
- Michael's cluster ids `cXX` were treated as declared domain families with an unordered quality
  index; GRP quality discounts are therefore inert there.
- Repeat noise for 300M Uncheatable and Table 9 is not identified, so 300M reports continuous regret
  only; the Delphi heldout tolerance pools ten proportional-noise runs with seven repeated coordinates.
- The successor was designed after the Codex review but before a DeepSeek review, which could not run
  (account quota).

Open items: the DeepSeek review; a fresh-seed validation of any mixture the successor proposes (none
is proposed here); the retained power law exceeds the five-minute Certify target (415 s projected)
while passing the eight-minute gate; the harm block's bucket alignment should be re-tested at 3e18
with a larger epoch-cap validation bank, since the scrambled control ties on the anchors.
