# Mixing via embeddings: bucket-independent mixture surrogates

_Why are we doing this? What's the benefit?_

Our data-mixture surrogates (DSP/GRP/LightGBM fit on the ~242-run qsplit240 swarm,
[PR #2393](https://github.com/marin-community/marin/pull/2393)) take **bucket-indexed weight vectors**
as input, so every fit is welded to one fixed bucket vocabulary. Bucket definitions churn constantly —
the quality relabel (#6739), the domain-cluster hill-climb (#6491), new sources — and every
redefinition strands the sweep and forces new proxy runs, which must run near-full-length because
early-checkpoint rankings are unstable (60M Spearman 0.298 at 22% progress). This design re-featurizes
a mixture by the **distribution it induces over a frozen embedding-space basis** — a token-weighted
histogram over the existing datakit micro-cluster vocabulary — so the surrogate `g(features) → metric`
is defined for *any* bucket whose contents we can embed. Adding or redefining buckets then costs one
map-only embedding job instead of a sweep. Validation is retrodictive on the existing runs: **zero new
proxy training** before the premise is proven.

## Background

Full brief: [research.md](research.md) (including review addenda). Domain2Vec (arXiv:2506.10952) is
the existence proof — regress performance on the mixture's meta-domain histogram `V·w` and adapt to
unseen datasets without refitting — but only at ≤1B over a coarse 260-cell taxonomy. CLIMB
(arXiv:2504.13161) is the counterexample: embedding-derived clusters but a *positional* predictor, so
re-clustering re-triggers its 112-run sweep — exactly our current trap. Cheap published baselines
exist (Olmix mixture-reuse, Chameleon leverage scores) that any deployment must beat. Two-stage
sampled distribution regression theory (arXiv:1402.1754) says regressing a scalar on a sampled
distribution is well-posed. Internally, #6326 is the cautionary precedent: features derivable from the
weights are a reparameterization and add nothing.

## Challenges

_What's hard?_

**The linear-reparameterization trap.** On a fixed bucket set the mixture histogram is `h = V·w` with
`V` the (cells × buckets) composition matrix — a fixed linear map of the weights. In-distribution fit
parity is therefore expected and uninformative (the #6326 failure mode); extrapolation to a new column
of `V` is carried entirely by the model's *smoothness prior over content space*. The success criterion
is therefore not "histograms beat weights" but "**the true semantic `V` beats matched non-semantic
reparameterizations**" — per-column cell-permutation controls (mass-profile-matched, simplex-valid)
and shuffled domain↔column mappings are mandatory in every experiment.

**Honest estimand.** With ~242 Dirichlet runs, nearly every run touches every domain, so no
retrodiction on this sweep is truly zero-shot. The primary estimand is **held-out-dose retrodiction**:
train only on runs where domain k's exposure is ≈ 0, predict runs where it is dominant, with correct
(full) featurization throughout — never the column-drop variant, which biases outcomes produced with
domain k against features that pretend it wasn't there. Design-support diagnostics (convex-hull
distance per test run, content novelty of `V_k` vs the cone of the others) are first-class outputs.

**Exposure features are themselves bucket-indexed.** DSP's per-domain exposure/epoch features live in
the same vocabulary we are trying to escape, and a held-out domain's exposure leaks the dose split.
Exposure splits into *transferable* features (scale, phase lengths, total tokens, per-cell exposure,
bucket size/repetition summaries — all computable for a new bucket) and *bucket-indexed* features
(per-domain epochs), which are excluded from the primary transfer model and reported only as a
diagnostic ceiling.

**Measure mismatch.** Mixture weights govern sampled *tokens*; a histogram from uniformly sampled
*documents* is a different measure, badly so for SFT/math/code with very different document lengths.
`V` is estimated token-weighted under the swarm tokenizer's actual eligibility (truncation, loss
masking). Relatedly, embeddings miss quality/dedup/repetition axes (DCLM; #6739), so cluster-only
histograms are the primary features, the web-text quality scorer is treated as an uncalibrated
descriptor pending a per-domain audit, and per-bucket size/dedup/length summaries are added as
features that remain computable for new buckets.

## Costs / Risks

- The premise may be false — content may not predict marginal training value once format, duplication,
  and repetition are accounted for. The precursor experiment (H2a) is designed to reveal this for
  ~zero compute; the fallback is Olmix-style reuse plus targeted mini-sweeps, and the featurization
  infra still serves the hill-climb thread.
- Held-out-dose retrodiction is weaker than a genuinely novel bucket; a passing result still needs one
  small live validation (H3) before production use.
- This does nothing for proxy→target *scale* transfer (60M→300M Spearman 0.777) — that ceiling binds
  regardless of featurization; results are computed per scale, never pooled.
- Statistical power is thin (~20 test runs per split → Spearman CI ≈ ±0.3): all comparisons are paired
  with bootstrap CIs, split feasibility is enumerated before any model is fit, and domain eligibility
  is pre-registered without looking at outcomes.
- The area is hot (CausalMix, Olmix, Chameleon, all 2025–26): partly scoop risk, partly validation.

## Design

_How are we doing this?_

**Frozen basis.** The datakit Luxical vocabulary: 192-dim embeddings
([`pipeline.py`](https://github.com/marin-community/marin/blob/a20220e4bab5acae88b52b0dab0cc9b5b65a8d03/experiments/datakit/embeddings/luxical/pipeline.py)),
FAISS spherical k-means K=5000 with K=1000/40 coarse views
([`train.py`](https://github.com/marin-community/marin/blob/a20220e4bab5acae88b52b0dab0cc9b5b65a8d03/experiments/datakit/cluster/domain/v0/train.py)).
Basis identity is content-hashed and versioned; re-featurizing under a new basis (e.g. the
hill-climb's stronger embedder) is a first-class, cheap operation. **K=40 is the primary granularity**
(242 runs cannot identify a free 5000-cell response; R&B's granularity U-shape); K=1000/5000 enter
only via smoothness-regularized models and ablations.

**Featurization.** Per swarm domain
([`domains.py`](https://github.com/marin-community/marin/blob/bf26b666a/experiments/domain_phase_mix/domains.py)):
sample documents, embed, assign, accumulate **token-weighted** cluster histograms (a column of `V`).
Run features per phase: `h = V·w` at the chosen granularity, optional mean-embedding moments,
transferable exposure features, bucket-stats features. Predictors are chosen for their extrapolation
prior, not raw capacity: kernel ridge on histogram distances (Hellinger/Jensen–Shannon) and ridge on
K=40 cells as primaries; LightGBM-on-cells only as a stress ablation. A **cluster-free arm**
(token-weighted random-Fourier-feature mean maps over raw embeddings ≈ MMD-kernel regression) runs
alongside the clustered arm in every experiment: it bypasses the k-means codebook entirely, so
disagreement between the arms localizes a failure to the codebook rather than the embedder or the
premise.

**Experiments, gated.**

1. **H1 — information audit** (sanity): rank/conditioning of `V`, reconstruction of `w` from `h`,
   nested-CV fit comparison of histogram vs weights features on identical splits, and the #6326
   control (R² of `h` from `w`, trivially ~1 by construction — reported to keep everyone honest about
   what in-distribution parity does and doesn't mean).
2. **H2a — does content predict domain value?** (decisive premise test, cheapest): from the existing
   sweep, estimate each domain's phase-specific marginal-value/saturation parameters with uncertainty
   (incumbent DSP/GRP structure); leave-one-domain-out predict those parameters from content
   histograms across the other 38, in both the clustered and cluster-free arms; score against
   semantic-shuffle and matched-random controls. If content cannot predict domain-level response
   here, stop — `g(V·w)` transfer has no chance. **Interpretation rule (pre-committed):** a negative
   H2a under one basis is "inconclusive pending a second basis," not "premise falsified"; the kill
   decision requires failure under two bases (e.g. Luxical + the hill-climb's candidate embedder) and
   in the cluster-free arm, so a bad codebook or a weak embedder cannot masquerade as a refuted
   premise. H2a is cheap enough (one featurization pass + CPU fits per basis) that this costs little.
3. **H2b — held-out-dose mixture retrodiction** (the response-surface test): for pre-registered
   domains k, train on bottom-dose runs, test on vertex/high-dose runs, per scale, paired against
   weights-features and the control featurizations; success = semantic `V` beats controls with CI
   separation on most eligible k, with design-support diagnostics reported per test run.
4. **H3 — live validation** (only if H2a+H2b pass): one genuinely new bucket; the surrogate *proposes*
   a mixture; ≤4 proxy runs at 60M/300M vs Olmix-style reuse and token-proportional proposals
   (policy-regret comparison — proposers are evaluated here, not as H2 predictors).
5. **H4 — ablations** (piggyback): granularity {40, 1000, 5000}; quality axis on/off (after the
   scorer audit); histogram vs KME vs both; per-phase vs pooled; kernel geometry.

Code lives at `experiments/datakit/mixture_features/` on this branch as self-contained scripts (in the
style of the swarm's `exploratory/regmix_regression.py`), reading swarm run histories from W&B and
mixture configs from the swarm branch as *data*, never importing branch code.

## Testing

_Agents make mistakes — how do we catch them?_

- Unit tests on featurization algebra: `h = V·w` composition, token-weighting, invariance of `h` under
  bucket relabeling/permutation, per-phase ordering, int8 dequantization round-trip.
- Synthetic recovery test for the retrodiction harness: plant a known smooth `g*` over content space,
  generate fake outcomes, verify the protocol recovers held-out-dose rankings, that the shuffled-`V`
  control fails, and that a planted content-blind `g*` (value depends on bucket identity only) makes
  the semantic model fail — the harness must be able to return "no" (mirrors `dsp-synthetic-recovery`).
- Split feasibility enumeration (per-domain eligible train/test counts, per scale) runs before any
  model fit and is checked into the results artifact.
- Histogram stability: bootstrap over source shards (not documents), sample-size sensitivity
  (100k vs 10k docs), and per-domain quality-scorer degeneracy report before quality features unlock.
- All fits store per-run predictions (not just correlations) so any headline number can be re-derived.

## Open Questions

- Is held-out-dose + merged-domain pseudo-new-buckets (train with two domains fused into one column,
  split at test) enough to claim "new bucket" transfer, or does the claim have to wait for H3?
- Negative results require a second basis by the pre-committed rule; does a *positive* result under
  Luxical int8 192-dim also need confirmation under the hill-climb's candidate embedder before H3, or
  is the semantic-vs-control separation enough on its own?
- Retrodiction target: `eval/uncheatable_eval/bpb` only, or also the #5416 aggregate — must the result
  hold on both?
- Where does this live long-term given the swarm assets are still branch-only (PR #2393 unmerged)?
