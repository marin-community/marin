# Related work on repetition-aware data mixing: what is new for the single-phase surrogate (2026-09-03)

Six papers read in full by independent readers (equations transcribed from source); each mapped against the
Observatory successor (`weibull_softplus_unscaled`: per-bucket Weibull benefit + softplus log-quadratic harm in
materialized epochs, nonnegative amplitudes, one shared shape per target component, panel-only fit) and its three
known failure modes: optimism far from the panel, unidentified Common Crawl harm, Table-9 frontier misranking.

## Paper notes

**Sedova et al., Scaling Laws for Mixture Pretraining Under Data Constraints (2605.12715v2).** Two-source
mixtures (finite target pool + never-repeated generic pool), 101M-805M params, >2000 runs. Law: r = hD/D_target,
D_T = D_target(1 + r1(1 - e^{-(r-1)/r1})), D_eff = (1-h)D + tau D_T, L = E + A/D_eff^alpha + gamma h; Huber fit with
row weights max(r h, eps). Harm enters only as saturation plus a repetition-independent share penalty gamma h; the
post-onset rise is not modelled. Optimal r grows with budget to 15-30 for small pools; larger models overfit
sooner. Validation: second half of steps, held-out size; median predicted-h error 0.07-0.13 in log10 except the
quality-tier setup (0.47, its worst case). No seeds.

**InfoLaw (2605.02364v1).** Six CC quality buckets, 27 fit runs, 252M-1.2B. info = sum_d f_d M_d log K (1 -
e^{-lambda(N) R_d / log K}), R_d = w_d K / M_d, f_d = e^{-theta d}, L = alpha info^{-beta}; five global scalars.
Saturation only, no harm term; the interior optimum comes from the simplex constraint. Fits 16x repetition of the
top-5 % CC bucket with 0.15 % mean loss error and recommends 7-20x at the optimum. Same-scale ranking of 25 random
recipes only Pearson 0.76. Buckets 2-5 are never repeated (their identifiability gap matches ours). No seeds.

**Repetition Mismatch (2606.07597v2).** 30M-1.17B, two or three sources, in-domain loss. Proxy sweeps at 1/16 to
1/2 of the target budget with the unique support subsampled by the same factor so repetitions match; optima
transfer with weight error 0.05-0.15 against 0.65-0.85 for share-matched proxies. Optimal repetitions about 5 at
>= 757M, 11 at 124M, 24 at 30M. Three seeds on ten configs: loss range <= 0.0085. Our Delphi panel is already
repeat-aware by construction (the simulated-epoch materializer fixes a subset per bucket).

**The Finetuner's Fallacy (2603.16177v2).** OLMo-1B, 200B tokens, one specialized domain (about 300M tokens) at
0-10 % plus web, post-finetuning objective. Loss = train power law + train-test gap power law, exponents linear in
the fraction (b = delta b_s + (1-delta) b_g), gap amplitude alpha1 delta^alpha2 e^{alpha3 delta}. Degradation starts
at 17-23 epochs for a 300M source but a 30M source is negative after 4 epochs at 0.1 %: harm at fixed epochs
depends on source size. Late-phase specialization (SCPT) beats every single-phase fraction for a 30M source.

**Scaling Domain Data Repetition (2608.14071v1).** 348M-1.85B, one high-quality domain (code, math, wiki,
medical) repeated 1-7 times against never-repeated web, in-domain loss; about 400 single runs, no seeds. Quadratic
fit per (domain, size, share); optimal epochs 5-6 (math), 4-5 (code), 3-4 (wiki, medical) at constant LR; optimal
epochs correlate -0.944 with the domain's attainable loss and 0.02 with its share (epochs, not share, is the harm
coordinate); earlier LR decay tightens the optimum; at fixed data the optimum falls from 6-7 to 2 from 348M to
1.85B.

**Data Mixing as Mixture Experiment (2608.23922v1).** RegMix's 512 runs at 1M params, 17 Pile domains. Scheffe
first- and second-order polynomials, lasso-selected 63 of 136 interactions, the largest all pile_cc x domain pairs;
Spearman 0.975 (1B held-out) against 0.879 first-order and 0.962 LightGBM. Model-robust I-optimal designs place
support at 0, 0.5 and 1 and reach the 512-run ordering with about 350 runs in simulation. No repetition, no
extrapolation test, no seeds.

## What the papers agree on that bears on us

- Epochs, not share, is the coordinate of harm (Domain Repetition, Sedova, Repetition Mismatch): supports our design.
- Tolerance to repetition is not one number: it falls with model size, with earlier decay, with lower unique-token
  count, and with a domain's attainable loss; high-quality CC tolerates 7-20x (InfoLaw), math 5-6 (Domain
  Repetition), wiki and medical 3-4, small sources far less (Finetuner's Fallacy). One shared harm threshold across
  39 buckets is the assumption every paper argues against; a threshold driven by a per-bucket covariate is the
  common ground.
- Every paper never repeats its web data, so none identifies web over-exposure harm; the two that reach high
  repetition in-panel (InfoLaw, Sedova) do it by subsampling the unique support at fixed weight.
- Web x domain interactions carry the non-additive structure in the one many-domain study (Scheffe); all others use
  two or three sources.

## Test list (ordered by expected value against the failure modes; final fit stays panel-only unless marked)

| # | Change | Source | Targets | Status vs rounds 1-3 | Cheap test | Cost |
|---|---|---|---|---|---|---|
| 1 | Linear share penalty: add nonnegative columns w_b (one per bucket) beside the Weibull benefit and the softplus harm | Sedova gamma h | CC harm identified from weight variation in-panel; concentration optimism | new (round-1 `linear_weight` was a weights-only reference) | Screen units; bank LOSO and split-half; watch far-panel bias | hours |
| 2 | Covariate-driven harm onset: tau_b = tau0 + tau1 x_b with x_b = log inventory or a bucket loss/quality proxy (2 global parameters); calibrate the covariate choice on the dose curves, fit amplitudes on the panel | Domain Repetition (-0.944), Finetuner's Fallacy (size), Repetition Mismatch | CC harm; Table-9 harm over-charged on small clean buckets | new; the dose curves already give per-bucket knees for Uncheatable | leave-one-bucket-out on the dose curves; bank split-half; LOSO | hours |
| 3 | Pooled effective-data law: L = E + A (sum_b tau_b U_b (1 + rho(E_b)))^{-alpha} + sum_b gamma_b w_b, shared r1 and alpha | Sedova | far-panel optimism (concave pooling of total effective data) | new model class; round-2 `total_square` was the only pooled term and tied | Screen; LOSO frontier rank; far-panel bias | one to two days (nonlinear fitter) |
| 4 | Hierarchical harm amplitude: shared harm column plus ridge-shrunk per-bucket deviations | InfoLaw shared lambda | CC harm borrowed from over-exposed buckets | planned in round 3, not run | Screen; bank LOSO | hours |
| 5 | CC-hub interactions: products of the CC-HQ (or total CC) benefit signal with each bucket's signal, signed pairs, ridge | Scheffe pile_cc x domain pairs | Table-9 frontier composition (8 % CC-HQ with high-epoch special buckets) | round-2 quality-pair and total-square interactions were null or negative; hub form untested | Screen; LOSO frontier rank on Table 9 | hours |
| 6 | Benefit in unique tokens: F(min(w_b T, N_b)) with harm still in epochs | Repetition Mismatch, Finetuner's Fallacy | small-bucket value | round-1 `weight_coordinate` (all terms in weights) was worse; hybrid untested | Screen | hours |
| 7 | Per-type epoch caps in the proposal DP (wiki/medical-like 2-4, math/code 4-6, CC lower) instead of one cap | Domain Repetition | search policy, not the surrogate | new; DP supports per-bucket bounds | rank bank compositions under both policies | hour |
| 8 | Task-family hierarchical pooling of component fits, shrinkage set by component signal-to-noise | InfoLaw single lambda (extreme case); our component analysis | Table-9 variance across components (not the shared bias) | new | Screen; bank split-half | day |
| 9 | Row weights max(E_b h_b, eps) in the head fit | Sedova | harm region fitted | new; `_nonnegative_solve` already takes row weights | Screen | hour |
| 10 | Nonparametric reference (gradient boosting) in the registry | Scheffe (LightGBM 0.99 at 1M) | baseline sanity | absent | Screen and bank; expect extrapolation failure | hours |

Needs new training runs (owner's call):

- A. Over-expose CC at panel-typical weights by subsampling the unique support (1/2, 1/4, 1/8) of two or three CC
  buckets, as InfoLaw, Sedova and Repetition Mismatch do; the only direct identification of CC harm. About 6-12 runs.
- B. Replicate seeds at the Table-9 centre, the HPR-280 control and any proposal (the bank has 7 replicated
  coordinates out of 247; single-run noise 0.004).
- C. Next swarm designed model-robust I-optimal over the epoch-cap polytope (support at concentrated mixtures)
  instead of Dirichlet draws; cheap preview: an I-optimal 200-subset of the current 280 against random subsets.
- D. Log per-bucket held-out BPB on the training buckets so harm can be measured as a train-test gap
  (Finetuner's Fallacy) in future panels.

Already covered or not worth testing: saturation-only ablation (`@no_harm`, round 1); quality-axis pooling and
pair-discount families (rounds 1-2); share-matched proxy transfer (our materializer already matches epochs);
compute-band optima and log-ratio weight error (evaluation metrics we do not need); Spearman-objective fitting
(our out-of-fold rank is already 0.9 in-panel; the failure is out of region); mixture-process models (single budget).

Readers' outputs (with local copies of the papers) are in the session scratchpad; equations above were checked
against them.
