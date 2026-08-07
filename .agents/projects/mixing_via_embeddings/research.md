# Background Research Brief — mixture optimization via embedding-space featurization

- Effort: high (3 parallel tracks: core external anchors, adversarial/adjacent external, internal Marin)
- Stop rule: stopped when new sources stopped changing the ranked hypotheses; remaining unverified items flagged in the source ledger
- Date: 2026-07-05

## Question

Our RegMix-style mixture surrogates take bucket-indexed weight vectors as input, so the surrogate is only
defined for a fixed bucket vocabulary: adding a bucket, re-clustering domains, or relabeling quality
invalidates the fit and forces a new proxy sweep. Can we instead featurize a mixture by the **distribution
it induces over a frozen embedding-space basis** (micro-cluster histograms / kernel mean embeddings), so
the surrogate `g(features(mixture)) → metric` transfers to new or changed buckets without re-sweeping?
The candidate first experiment: re-featurize the existing swarm runs and test leave-one-bucket-out (LOBO)
retrodiction.

## Current Marin Context

- **The RegMix/swarm program lives on branch `calvin/swarm-olmo3-regmix-test`** (PR
  [#2393](https://github.com/marin-community/marin/pull/2393), open; nothing named regmix is on `main`).
  Main asset: the **qsplit240 swarm** — ~240–242 proxy runs, two phases, **39 hand-defined domains**
  (`experiments/domain_phase_mix/domains.py` on that branch: Dolma 3 + Dolmino + Nemotron splits + SFT
  sets), run at 60M and 300M with Dirichlet + vertex-biased weight sampling
  (`experiments/domain_phase_mix/experiment.py`, `weight_sampler.py`). Target metric: primarily
  `eval/uncheatable_eval/bpb`; also the #5416 aggregate and OLMoBaseEval Easy/Uncheatable (#6602/#6603).
- **Surrogates**: literal RegMix (LightGBM on weights,
  `experiments/domain_phase_mix/exploratory/regmix_regression.py`) is the weak baseline; production fits
  are **DSP** (parametric benefit/saturation/penalty form with epoch/exposure features; OOF Spearman
  **0.914** on uncheatable BPB) and **GRP** (power-family-penalty), fitters under
  `exploratory/two_phase_many/`. Optimized-mixture CSVs in
  `experiments/domain_phase_mix/assets/delphi_optimized_mixtures/`; checkpoints under
  `gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/`.
- **Datakit buckets (current, on main)**: store partitions on (domain cluster × quality bucket) — Luxical
  192-dim embeddings → FAISS spherical k-means **K=5000 micro-clusters** with agglomerative coarse views
  at K=1000 and K=40 (`experiments/datakit/cluster/domain/v0/{train,assign,summarize}.py`), × 5 quality
  buckets at fixed cutoffs (`experiments/datakit/store/datakit_store.py:84-89`) → up to 200 buckets
  (`datakit_store.py:28-34`). ~113 normalized sources (`lib/marin/src/marin/datakit/sources.py`).
- **The featurization substrate already exists**: every datakit source is embedded (Luxical map-only
  zephyr pipeline, `experiments/datakit/embeddings/luxical/pipeline.py`) and assigned to the frozen K=5000
  vocabulary; **per-source cluster histograms are already computed** (`domain/v0/summarize.py`). Corpus
  mismatch caveat: the qsplit240 swarm's 39 domains are a different corpus than datakit — re-featurizing
  those runs needs embed+assign over the swarm domains (cheap: histogram estimation needs only a document
  sample per domain, not full-corpus inference).
- **Bucket churn is live motivation**: quality buckets are domain-biased (source predicts oracle label at
  AUC 0.852, #6739; type-aware relabel validated small-scale) and the domain-cluster hill-climb (#6491,
  PR #6476) wants a stronger embedder at K=40 — both would redefine buckets and strand any bucket-indexed
  surrogate.

## Internal Prior Work

- **#6326 (closed) — MDE-style checkpoint-likelihood features failed**: 900-column feature set had
  effective rank ~22 ≈ the mixture/exposure geometry; projecting from `phase_log_exposure` reproduced the
  features (median R² ≈ 0.9998); DSP stayed stronger (0.914 vs 0.86). **This is the direct internal
  precedent for the central failure mode**: a re-featurization of the same sweep that is (nearly) a
  reparameterization of the weights adds nothing.
- **Swarm-branch logbooks** (`.agents/logbooks/` on the swarm branch):
  `partition-swarm-data-mixing-theory.md` — partition-expressivity analysis (coarse partitions limit
  achievable mixtures; exactly the gap embedding-space parameterization addresses); repetition-aware
  literal forms are weak (shared half-life Spearman 0.51; per-domain 0.925 but 39 fragile params);
  `swarm-transfer-calibration.md` — 60M→300M identity Spearman only **0.777**; anchored `y60 + Δ(scale)`
  beats a continuous law; early-rank instability (60M Spearman 0.298 at 22% progress vs final) means proxy
  runs must run near-full-length — that is the re-sweep cost this proposal attacks.
- **Domain-cluster hill-climb** (#6491): v0 K=40 coherence ≈ 0.75 by LLM intruder test, ~1.8× better than
  WebOrganizer labels; next lever is a stronger fp32 embedder — i.e., the embedding space itself is
  expected to change, so the design should treat "re-featurize under a new basis" as a first-class
  operation, not an exception.
- No internal attempt at continuous/embedding-space mixture parameterization found; no LOBO/leave-one-domain
  retrodiction on the swarm; closest are #6326 and the partition-expressivity theory notes.

## External Prior Art

Anchors (all verified against arXiv/full text unless flagged):

- **RegMix** (Liu et al., arXiv:2407.01492, ICLR 2025): 512 × 1M-param proxies × 1B tokens over 17 fixed
  Pile domains; LightGBM on raw weights; 97.12% Spearman rank transfer to 1B/25B models. Input
  dimensionality tied to the domain set; no new-domain mechanism; own stated limitation is the
  domain-partition assumption.
- **Domain2Vec** (Zhang et al., arXiv:2506.10952, ICML 2025) — **the existence proof for our featurization**:
  vectorize any dataset as a distribution over 260 meta-domains (k-means over bge-small embeddings +
  a Qwen2-1.5B classifier at 74.7% accuracy); regress performance on `V_train · r` (histogram of the
  mixture) instead of one-hot weights; explicit claim of no-cost adaptation to unseen training sets /
  validation sets / dataset counts; ~0.26% of DoReMi FLOPs for comparable quality. Validated only ≤1B
  scale, coarse taxonomy, no classifier-error-propagation analysis.
- **CLIMB** (Diao et al., NVIDIA, arXiv:2504.13161, NeurIPS 2025) — the cautionary mirror image:
  stella_en_400M_v5 embeddings, FAISS k-means K=1000 → pruned/merged to ~20 superclusters, LightGBM on
  **positional** cluster-weight vectors, 112 proxy runs over 3 iterations. No content featurization → no
  transfer story; re-clustering implies re-sweeping. Notably they *collapsed* 1000 clusters to ~20 to make
  search tractable — regression over fine histograms is the unsolved part.
- **WebOrganizer** (Wettig et al., arXiv:2502.10341, ICML 2025): 24 topics × 24 formats, RegMix over
  constructed domains (512 × 50M proxies). Key result for us: re-weighting domains to imitate
  FineWeb-Edu's induced domain distribution recovers **73% of its MMLU gain / 84% on average** — a
  domain-histogram match captures most of a pointwise quality filter. Explicitly cautions predictions are
  noise-sensitive and scale transfer uncertain.
- **Chameleon** (Xie et al., arXiv:2505.24844, ICML 2025): kernel ridge leverage scores over learned
  domain embeddings; explicitly transfers to new domains at ~1% of proxy-retraining cost — but a scoring
  heuristic, not a learned mixture→metric predictor; no what-if capability.
- **CausalMix** (arXiv:2607.01104, Jul 2026): our motivating problem verbatim ("regression fit on old
  proxy runs no longer predicts when the pool shifts"); pool statistics as covariates + CausalForestDML on
  512 proxy runs; claims extrapolation to unseen pools. Post-training-flavored setting (Qwen2.5,
  OpenDataArena); covariates are scalar quality metrics, not embedding distributions. [abstract-depth]
- **Olmix** (Ai2, arXiv:2602.12237): names the evolving-domain-set problem; "mixture reuse" heuristic
  (keep old ratios, recompute only affected domains) matches full recomputation with 74% less compute over
  5 sequential domain-set updates. **This is the cheap baseline LOBO must beat.** [abstract-depth]
- **MDE** (Google, arXiv:2502.15950): adding per-domain expert cross-entropy features to the regressor
  substantially beats weights-only regression — external proof that richer mixture featurization improves
  sample-efficiency (contrast with our #6326 variant, which failed because the features were
  weight-derivable).
- **Aioli** (Chen et al., arXiv:2411.05735, ICLR 2025): no existing method consistently beats stratified
  sampling; mixture→loss is well-fit by simple laws (R² 0.969) but methods set law parameters inaccurately
  from cheap static estimation. Regression target is learnable; proxy→target parameter transfer is the
  bottleneck.
- **DoReMi / DSIR / Data Mixing Laws / MATES**: DoReMi (arXiv:2305.10429) — group-DRO weights are
  proxy-size unstable (280M proxy → Pile-CC 0.67 vs 1B proxy → <0.20). DSIR (arXiv:2302.03169) — hashed
  n-gram importance resampling toward a chosen target; KL-reduction correlates r=0.82 with downstream;
  assumes the target distribution is already right. Data Mixing Laws (arXiv:2403.16952) — exponential law
  + nested scaling laws; O(M×K) params; rankings depend on scale and steps. MATES (arXiv:2406.06046) —
  pointwise influence model (BERT fine-tuned on one-step-influence labels), continuous per-example
  selection, no mixture-level interactions; connects to our MATES ICL-influence POC.
- **Theory + adjacent featurization**: two-stage sampled distribution regression is provably consistent —
  regressing a scalar on a kernel mean embedding of a *sampled* distribution is exactly our setting (Szabó
  et al., arXiv:1402.1754, arXiv:1411.2066; improved bounds arXiv:2308.14335). Bag-of-Prototypes
  (arXiv:2303.13251) — histogram over a frozen k-means codebook as a dataset representation, beats
  global-mean/Fréchet representations for dataset-level prediction (vision). R&B (arXiv:2505.00358) —
  ModernBERT + k-means regrouping; performance is U-shaped in cluster count; silhouette score is their
  granularity heuristic; balancing via online gradient Gram matrix (0.009% overhead) with no proxy sweeps
  at all — a rival philosophy.
- **Practical priors**: embedding-at-scale precedents — SemDeDup (OPT-125M embeddings, K≈11k for C4,
  K≈√N heuristic), D4 (re-cluster after dedup — duplicates distort cluster geometry), CLIMB (stella-400M
  over 1.2T tokens), FineWeb-Edu (Snowflake-arctic-embed-m + linear head, ~6k H100-hours over FineWeb).
  Proxy budgets: RegMix 512 runs / 17 domains; CLIMB 112 / ~20 clusters; CausalMix 512.

**Novelty check (negative result)**: no paper found that feeds fine micro-cluster histograms or kernel
mean embeddings of mixtures into a *learned performance predictor* for pretraining mixture optimization.
Domain2Vec is closest (coarse 260-meta-domain histogram); CLIMB has fine clusters but a positional
predictor. As of 2026-07, the specific combination appears unoccupied — and so does any
design-of-experiments treatment of mixture proxy sweeps (no D-optimal/active-design paper for LLM
mixtures found despite targeted searches).

## Negative / Failed Leads

- Internal #6326: weight-derivable features are a reparameterization; no gain over DSP.
- DCLM (arXiv:2406.11794): fastText lexical filtering beat BGE-embedding classifiers, SemDeDup, AskLLM,
  perplexity filters (~3.5 pt Core) — semantic embeddings alone are not sufficient statistics for
  pretraining value.
- Data-Quality Illusion (arXiv:2510.00866): classifier-based quality filtering works by *excluding*
  low-quality data, not by matching the high-quality set; quality classifiers learn spurious features —
  "distance to HQ clusters" is not a reliable value signal.
- Proxy reliability: arXiv:2512.24503 — proxy→target recipe-rank Spearman often <0.75, sometimes ~0/negative
  at standard LRs (near-perfect at tiny LRs — actionable protocol check); DataDecide (arXiv:2504.11393) —
  small-scale ranking gets only ~80% of pairwise decisions right at 1B, and no scaling-law method beat that.
- Aioli: cheap static estimation of mixing-law parameters transfers poorly to the target run.
- No LLM mixture-DoE literature exists to lean on for the identifiability question (genuine gap).

## Evidence Map

#### Claim: On a fixed bucket set, histogram featurization is information-equivalent to bucket weights; its value is exclusively in generalization to new/changed buckets
- Support:
  - Linear algebra: the mixture histogram is `h = V·w` with `V` the (cells × buckets) bucket-composition
    matrix — a fixed linear map of `w`. Rank ≤ #buckets.
  - Internal #6326: weight-derivable features (effective rank ≈ mixture geometry) added nothing over DSP.
  - Domain2Vec regresses on exactly `V_train · r` and gets its wins from *transfer*, not in-distribution fit.
- Contradictions:
  - MDE (Google) shows richer features can improve *sample-efficiency* even in-distribution — but their
    features (expert losses) are not weight-derivable; histograms on a fixed bucket set are.
- Directness to Marin: exact — same sweep (qsplit240), same trap (#6326).
- Confidence: high (mathematical + replicated internally).
- Action: design the experiment around LOBO/new-direction transfer, and always report the
  weights-baseline on identical train/test splits; in-distribution parity is a sanity check, not a result.

#### Claim: Distributional featurization can transfer a mixture surrogate to unseen buckets
- Support:
  - Domain2Vec (2506.10952): no-cost adaptation to unseen training sets at ≤1B scale, 0.26% of DoReMi FLOPs.
  - Chameleon (2505.24844): domain-embedding representation transfers to new domains at ~1% cost.
  - CausalMix (2607.01104): covariate-based predictor extrapolates to unseen pools [abstract-depth].
  - WebOrganizer: matching an induced domain histogram recovers 73–84% of a quality filter's gain.
  - Distribution-regression theory (1402.1754/1411.2066): the estimation problem is well-posed.
- Contradictions:
  - Nobody has done it with fine (10³–10⁴ cell) histograms; CLIMB collapsed 1000→~20 clusters to cope.
  - Classical mixture DoE: extrapolation off the sweep's design support is model-assumption-bound; the
    qsplit240 Dirichlet sweep was not designed to span new-bucket directions.
- Directness to Marin: high for the mechanism; medium for scale (all external validation ≤1B).
- Confidence: exploratory — plausible mechanism with one direct existence proof, unreplicated in our regime.
- Action: run LOBO retrodiction on qsplit240 (H2 below) before building anything.

#### Claim: Semantic embeddings miss value axes that move our metric (quality, dedup, noise)
- Support:
  - DCLM: fastText beat BGE-embedding classifiers for filtering.
  - Data-Quality Illusion: quality-classifier signal is spurious-feature-laden and task-specific.
  - Internal: datakit quality buckets are domain-confounded (AUC 0.852 source→label, #6739); D4: duplicates
    distort embedding-cluster geometry.
- Contradictions:
  - FineWeb-Edu: a linear head on frozen embeddings carried enough signal to curate 1.3T tokens.
  - Luxical + K=40 won the intruder-test coherence eval vs WebOrganizer (#6491) — topically the space is good.
- Directness to Marin: high — our own quality relabel work exists because of this confound.
- Confidence: replicated.
- Action: cluster-only histograms are the primary features; the quality axis is gated behind a
  per-domain audit (the web-text fasttext scorer is OOD on SFT/math/code and fixed cutoffs may
  degenerate there — see review addenda) and enters via the H4 ablation; add bucket-stats features
  (size/dedup/length/loss-mask) that embeddings cannot see but remain computable for new buckets.

#### Claim: Proxy→target scale transfer, not surrogate fit, bounds what any featurization can deliver
- Support:
  - Internal: 60M→300M Spearman 0.777; early-rank instability 0.298 at 22% progress.
  - arXiv:2512.24503: proxy rankings <0.75 Spearman at standard LRs, flip with hyperparameters.
  - DataDecide: ~80% pairwise-decision ceiling; DoReMi proxy-size weight flip (0.67 → <0.20).
  - Data Mixing Laws: mixture rankings depend on model size and steps.
- Contradictions:
  - RegMix: 97.1% Spearman 1M→1B (their setting); tiny-LR protocol gives near-perfect transfer (2512.24503).
- Directness to Marin: exact — measured on our own swarm.
- Confidence: replicated.
- Action: scope the proposal to "reuse the sweep across *bucket* changes"; scale transfer stays whatever
  DSP/anchored-delta already delivers. Don't claim the featurization fixes scale transfer.

#### Claim: Simple reuse heuristics are the bar to beat, not full re-sweeps
- Support:
  - Olmix (2602.12237): keep-old-ratios + recompute-affected matches full recomputation at 26% compute.
  - Chameleon: new-domain adaptation at ~1% cost with no predictor at all.
  - DSIR/DA²-style alignment: training-free distribution matching as a zero-run baseline.
- Contradictions: none found — these baselines are cheap and must be included.
- Directness to Marin: high — trivially implementable on our sweep.
- Confidence: exploratory (abstract-depth for Olmix).
- Action: H3 evaluates *proposed mixtures* against (a) token-proportional weight for the new bucket,
  (b) Olmix-style reuse, (c) DA² alignment — not only against "re-run the sweep". (These are mixture
  proposers, not metric predictors, so they belong in H3's policy comparison, not H2's rank test —
  see review addenda.)

## Recommended Next Experiments

#### 1. H2 — content→value retrodiction on qsplit240 (the decisive test; two stages, see addenda)
- Minimum experiment: embed+assign a ~100k-document sample per swarm domain into the frozen datakit K=5000
  vocabulary (map-only zephyr job; histograms converge fast on samples) → per-domain **token-weighted**
  histograms `V`. Then two stages. **H2a**: fit per-domain per-phase response parameters (incumbent
  DSP/GRP structure, with uncertainty) and leave-one-domain-out predict them from content histograms
  across the other 38 — the direct test of "content predicts domain value". **H2b**: held-out-dose
  retrodiction — per pre-registered domain k and per scale, train `g` only on runs where k's dose ≈ 0
  (full, physically correct `h = V·w` featurization throughout), predict the high-dose/vertex runs.
  In both stages, compare semantic `V` against shuffled-column and matched-random-projection controls
  and against the weights-surrogate, on identical paired splits.
  *(Superseded protocol note: an earlier draft proposed dropping domain k's column and training on
  `h = V_{-k}·w_{-k}`. Review found this biased — outcomes were produced with k present
  (omitted-variable bias) and `w_{-k}` is no longer a distribution. The honest estimand on this sweep
  is held-out-dose extrapolation, not zero-shot; see review addenda.)*
- Baseline/control: shuffled/matched-random featurizations and the weights-model on identical splits;
  the #6326-style control — R² of predicting `h` from `w` (expected ~1 by construction, reported for
  honesty about what in-distribution parity means).
- Expected signal: semantic featurization beats both non-semantic controls with paired-bootstrap CI
  separation on most eligible held-out domains (H2b) and across the 39 LODO folds (H2a).
- Falsifier: semantic ≈ shuffled/matched-random, or H2a shows content does not predict domain response
  parameters — then `g(V·w)` transfer has no chance; stop before any live runs.
- Cost/risk: no proxy training; one map-only embedding job over domain samples + CPU fitting. Risks:
  (a) qsplit240's Dirichlet design may not support held-out-dose extrapolation — report content novelty
  (cone distance of `V_k`) and per-test-run convex-hull design support; (b) ~20 test runs/split →
  Spearman CI ≈ ±0.3 — enumerate split feasibility per scale before fitting anything.
- Sources: Domain2Vec 2506.10952; internal #6326; swarm branch `exploratory/two_phase_many/`;
  `domain/v0/{train,assign}.py`; review addenda below.
- Confidence: exploratory.

#### 2. H1 — Information audit (prerequisite sanity check, same infra as H2)
- Minimum experiment: numerical rank and condition spectrum of `V`; linear reconstruction error of `w`
  from `h`; nested-CV fit comparison of histogram-featurized vs weights-featurized learners on
  identical splits over the full 242-run set.
- Baseline/control: DSP OOF Spearman 0.914 as a reference line (not a parity target — DSP's parametric
  per-domain form cannot ingest histograms directly; see review addenda).
- Expected signal: near-parity for comparable learner classes; any gap is attributed among compression,
  conditioning, and model-class mismatch (each measured separately).
- Falsifier: large unexplained fit loss → the histogram compresses away identity information the
  surrogate needs (e.g., two content-similar domains with very different value — a quality-axis
  failure, see claim 3).
- Cost/risk: trivial once V exists.
- Sources: #6326 protocol; DSP fitters.
- Confidence: stable prediction (linear-map argument), with the parity claim appropriately weakened.

#### 3. H3 — Live new-bucket validation (only if H2 passes)
- Minimum experiment: pick a genuinely new bucket (e.g., a datakit source outside the 39 swarm domains, or
  a post-relabel quality bucket). Optimize the mixture with the histogram surrogate; run 2–4 proxy models
  at 60M/300M: g-optimum vs Olmix-reuse vs token-proportional vs (budget permitting) a mini-sweep optimum.
- Baseline/control: Olmix-style reuse is the primary comparator.
- Expected signal: g-optimum ≥ reuse baselines on uncheatable BPB at 300M.
- Falsifier: g-optimum indistinguishable from token-proportional, or worse than reuse.
- Cost/risk: a few 300M proxy runs (the thing we are trying to amortize — keep to ≤4).
- Sources: Olmix 2602.12237; swarm launch scaffolding.
- Confidence: exploratory.

#### 4. Ablations (piggyback on H2)
- Basis granularity K ∈ {40 primary, 1000, 5000} (R&B's U-shape says don't assume finer is better; 242
  runs cannot identify a free 5000-cell response — fine granularities only via smoothness-regularized
  models); quality-score features per cell on/off (gated on the scorer audit); KME (mean Luxical
  embedding) vs histogram vs both; per-phase histograms vs pooled; kernel geometry
  (Hellinger/Jensen–Shannon vs Euclidean).
- Falsifier for the "fine basis" story: K=40 matches K=5000 on held-out-dose — then the coarse view
  suffices and the design simplifies.
- Sources: R&B 2505.00358; Bag-of-Prototypes 2303.13251; Szabó et al. distribution regression.

## Hypothesis Queue Update

- Add: H1 (information audit), H2a (content→domain-response LODO), H2b (held-out-dose retrodiction),
  H3 (live new-bucket validation, policy comparison), H4 (granularity/feature ablations) — as above.
- Revise: the original framing "refit the surrogate on histograms and check held-out-run R²" is
  insufficient — on a fixed bucket set it is a linear reparameterization (see claim 1). The original
  LOBO column-drop protocol is superseded (biased; see addenda). The queue centers on H2a → H2b with
  semantic-vs-control comparisons.
- Falsify / stop: if H2a fails (content does not predict domain response), or H2b fails across most
  eligible domains after the design-support check, stop; fall back to Olmix-style reuse + targeted
  mini-sweeps (active-learning direction remains open).
- Promote: nothing yet (no experiment run).

## Review addenda (2026-07-05)

Two independent pre-publication reviews reshaped the protocol: an internal architect pass and an
external Codex (GPT-5.5) pass acting as research colleague. Convergent findings (both reviewers):
Olmix-reuse/token-proportional are mixture *proposers*, not metric predictors — comparing them by
test-run Spearman was a category error (moved to H3 as policy-regret comparison); the LOBO
column-drop protocol contradicted the design's run-split and is statistically biased; `SwarmRun`
needed model scale, phase durations, and token counts (scale pooling would silently exploit the 0.777
cross-scale ceiling); p≫n at K=5000 needs pinned regularization with K=40 primary; the web-text
quality scorer is OOD on SFT/math domains (quality axis gated behind a per-domain degeneracy audit);
span diagnostics must use convex-hull/cone distance, not linear span (linear span over-covers and is
near-zero by construction once any train run touches the domain).

Codex-only contributions adopted into the design: (1) **exposure features are themselves
bucket-indexed** — per-domain epoch/exposure features quietly reintroduce the bucket vocabulary and
leak the dose split; split into transferable-global vs bucket-indexed (the latter diagnostic-only);
(2) **token-weighted histograms** — mixture weights govern sampled tokens, so document-uniform
histograms are the wrong measure, badly so for SFT/math/code (document length, packing, loss
masking); (3) **honest estimand renaming** — "held-out-dose retrodiction", not zero-shot LOBO;
(4) **semantic-shuffle and matched-random-projection controls** as the criterion (the claim is
"semantic alignment beats equally-regularized non-semantic reparameterizations", not "histograms beat
weights"); (5) **H2a two-stage domain-response experiment** as the cheapest decisive premise test;
(6) bucket-stats features (size/dedup/length/loss-mask fractions) that stay computable for new
buckets; (7) content-hash basis identity; (8) synthetic-recovery tests must include must-fail cases.

Architect-only contributions adopted: split-feasibility enumeration before any fit; pre-registered
domain eligibility; paired bootstrap CIs and stored per-run predictions; explicit phase reducer for
dose; the middle dose band is discarded by declared thresholds (absolute, not quantiles); DSP-parity
claim weakened in H1 (DSP cannot ingest histograms; attribute any gap among compression/conditioning/
model class).

Post-review author addition (basis sensitivity, 2026-07-05): the experiments depend on the frozen
basis only as a quantizer, but a negative result under one basis cannot distinguish "premise false"
from "basis blind". Two changes: (1) a mandatory **cluster-free arm** (token-weighted RFF mean maps ≈
MMD-kernel regression over raw embeddings) runs alongside the clustered arm in every experiment,
localizing failures to the codebook vs the embedder vs the premise; (2) a pre-committed
**interpretation rule** — killing the premise requires the H2a gate to fail under two `MixtureBasis`
versions (different embedders) and in the cluster-free arm.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| RegMix 2407.01492 | paper | arxiv.org/abs/2407.01492 | weights-only regression, 512-run budget, rank invariance | high | full text |
| Domain2Vec 2506.10952 | paper | arxiv.org/abs/2506.10952 | histogram featurization transfers to unseen sets | high | ≤1B scale; Spearman 0.9743/0.6657 medium confidence |
| CLIMB 2504.13161 | paper | arxiv.org/abs/2504.13161 | fine clusters + positional predictor = no transfer | high | per-proxy token budget unverified |
| WebOrganizer 2502.10341 | paper | arxiv.org/abs/2502.10341 | histogram-matching recovers 73–84% of filter gain | high | 80K vs 100K annotation count unresolved |
| Chameleon 2505.24844 | paper | arxiv.org/abs/2505.24844 | new-domain adaptation at ~1% cost, no predictor | high | |
| CausalMix 2607.01104 | paper | arxiv.org/abs/2607.01104 | pool-shift failure + covariate fix | medium | abstract-depth; post-training setting |
| Olmix 2602.12237 | paper | arxiv.org/pdf/2602.12237 | mixture-reuse baseline, 74% compute saving | medium | abstract-depth |
| MDE (Google) 2502.15950 | paper | arxiv.org/abs/2502.15950 | richer features beat weights-only regression | high | |
| Aioli 2411.05735 | paper | arxiv.org/abs/2411.05735 | laws well-specified, parameters set badly; stratified hard to beat | high | |
| DoReMi 2305.10429 | paper | arxiv.org/abs/2305.10429 | proxy-size weight instability | high | 280M vs 1B flip partly secondhand via RegMix |
| DSIR 2302.03169 | paper | arxiv.org/abs/2302.03169 | distribution-matching baseline; KL↔downstream r=0.82 | high | |
| Data Mixing Laws 2403.16952 | paper | arxiv.org/abs/2403.16952 | rankings scale/step-dependent; O(M×K) params | high | run counts not pinned |
| MATES 2406.06046 | paper | arxiv.org/abs/2406.06046 | pointwise influence alternative; no mixture interactions | high | pooling detail unconfirmed |
| Small-runs reliability 2512.24503 | paper | arxiv.org/abs/2512.24503 | proxy rank transfer <0.75; tiny-LR fix | high | |
| DataDecide 2504.11393 | paper | arxiv.org/abs/2504.11393 | ~80% pairwise-decision ceiling | high | |
| DCLM 2406.11794 | paper | arxiv.org/abs/2406.11794 | fastText beat embedding classifiers | high | |
| Data-Quality Illusion 2510.00866 | paper | arxiv.org/abs/2510.00866 | quality ≠ proximity to HQ set in embedding space | high | |
| R&B 2505.00358 | paper | arxiv.org/abs/2505.00358 | U-shape in cluster count; silhouette heuristic; online gradient rival | high | |
| SemDeDup 2303.09540 / D4 2308.12284 | paper | arxiv | K≈√N; re-cluster after dedup | high | |
| Distribution regression 1402.1754 / 1411.2066 / 2308.14335 | paper | arxiv | two-stage sampled KME regression is consistent | high | exact theoretical setting |
| Bag-of-Prototypes 2303.13251 | paper | arxiv.org/abs/2303.13251 | frozen-codebook histogram as dataset representation | high | vision |
| Scaling Laws for Optimal Data Mixtures 2507.09404 | paper | arxiv | law extrapolates across scale, within domain simplex | medium | |
| Survey 2604.16380 | paper | arxiv.org/pdf/2604.16380 | partition-scheme transferability is open | medium | |
| PR #2393 / swarm branch | Marin code | `calvin/swarm-olmo3-regmix-test`, `experiments/domain_phase_mix/` | qsplit240 assets, DSP/GRP fitters | high | branch-only, not on main |
| #6326 | GitHub issue | marin#6326 | weight-derivable features fail | high | retrospective TL;DR |
| Swarm logbooks | logbook | `.agents/logbooks/` (swarm branch) | transfer 0.777; early-rank instability; partition expressivity | high | |
| Datakit cluster/store code | Marin code | `experiments/datakit/cluster/`, `experiments/datakit/store/datakit_store.py` | K=5000 basis, per-source histograms, 200 buckets | high | on main |
| #6491 / PR #6476 | GitHub issue/PR | marin#6491 | cluster coherence; embedder churn expected | high | |
| #6739 | GitHub issue | marin#6739 | quality buckets domain-confounded | high | |
| Unverified leads | paper | RegMix-D 2606.18663, FastMix 2606.14971, Topic-over-Source 2502.16802, CAMEL 2603.08022, HERMES 2607.02266 | adjacent, not load-bearing | low | abstract-only |

## Handoff

- Suggested issue `Prior work` block: Domain2Vec (2506.10952) = existence proof for histogram-featurized
  mixture regression transferring to unseen sets; CLIMB (2504.13161) = fine embedding clusters but
  positional predictor (no transfer); Chameleon (2505.24844) + Olmix (2602.12237) = cheap
  new-domain baselines to beat; internal #6326 = weight-derivable features are a reparameterization
  (the control every result must pass); distribution-regression theory (1402.1754) = the estimation
  problem is well-posed.
- Suggested logbook entry: this brief (research.md) + design.md in the same directory.
- Open questions: (1) how far outside the qsplit240 design span are realistic new buckets (measure, don't
  assume); (2) does the two-phase structure need per-phase histograms or do pooled + exposure features
  suffice; (3) is Luxical (192-dim, int8) good enough as the frozen basis or should H2 be repeated under
  the hill-climb's stronger embedder; (4) who owns re-running embed+assign when the basis version bumps.
- Stop reason: new sources stopped changing the hypothesis ranking; remaining unknowns are empirical
  (H1/H2), not literature gaps.
