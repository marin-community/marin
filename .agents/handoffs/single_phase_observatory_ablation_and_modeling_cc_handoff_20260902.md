# Handoff: single-phase Observatory ablation and model synthesis

## Objective

Build a reproducible benchmark of every distinct single-phase model represented in the Mixture Fit
Observatory. Use matched ablations to determine which response shapes, exposure features, damage terms,
links, pooling rules, and estimators improve prediction. After the complete benchmark is ready, obtain
independent read-only reviews from Codex and DeepSeek, verify their findings, and then build the smallest
mechanistically defensible successor model supported by the ablations.

The result must answer three separate questions:

1. Which models predict the full mixture panel accurately?
2. Which models select good mixtures, especially in the low-BPB basin?
3. Which mechanisms survive matched ablation and external validation?

Do not infer question 3 from a model name or from question 1 alone.

## Source of truth

Use Fieldbook experiment `exp_01m1ge7ye6hz2epd0mjkbkrvt8` for all decisions, artifacts, validations,
and handoffs.

Start with:

```bash
fieldbook db where --json
fieldbook experiment context exp_01m1ge7ye6hz2epd0mjkbkrvt8
fieldbook session start \
  --experiment exp_01m1ge7ye6hz2epd0mjkbkrvt8 \
  --agent claude-code \
  --intent "benchmark and ablate single-phase Observatory models" \
  --json
```

The worktree already contains unrelated edits and generated artifacts. Preserve them. Do not revert,
overwrite, or fold them into this task without checking ownership.

## Current completed work

### Canonical DSP baseline

`benchmark_single_phase_componentwise_canonical_dsp_20260902.py` is the reference implementation of the
benchmark protocol and the optimized profiled-gradient canonical-DSP solver. It completed 6,450 atomic
component/fold fits over five outer repeats in 18.4 minutes with 16 local workers. Its first complete
five-fold repeat covered all 42 Michael tasks and finished in 4.9 minutes. The new benchmark uses only the
frozen eight-task Michael subset below, so five minutes is the wall-clock target for one model's complete
`Certify` run.

Artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_componentwise_canonical_dsp_20260902/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_componentwise_canonical_dsp_20260902/protocol.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_componentwise_canonical_dsp_20260902/component_metrics.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_componentwise_canonical_dsp_20260902/aggregate_metrics.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_componentwise_canonical_dsp_20260902/olmix_model_comparison.csv`

The aggregate baseline is:

| panel | target | RMSE | Spearman | mean fold regret at 1 |
|---|---|---:|---:|---:|
| 60M, 39 buckets | Uncheatable | 0.00819 | 0.935 | 0.00349 |
| 60M, 39 buckets | Table 9 | 0.02208 | 0.886 | 0.00246 |
| 300M, 39 buckets | Uncheatable | 0.00556 | 0.958 | 0.00296 |
| 300M, 39 buckets | Table 9 | 0.01312 | 0.906 | 0.00443 |
| Delphi 3e18, 39 buckets | Uncheatable | 0.00974 | 0.956 | 0.00260 |
| Delphi 3e18, 39 buckets | Table 9 | 0.03026 | 0.841 | 0.00445 |
| DCLM 10k | native 42-task mean | 0.53483 | 0.875 | 0.04830 |
| high-quality 10k | native 42-task mean | 0.27093 | 0.760 | 0.07755 |

Full canonical DSP is a useful baseline on the 39-bucket swarms. It is underidentified on the 118- and
120-bucket swarms: each atomic fit has \(4B+1\) fitted quantities and an outer training fold has about 290
rows. Do not interpret that failure as evidence against epoch exposure.

### Leakage-filtered external registry

`prepare_single_phase_heldout_benchmark_20260902.py` built a coordinate-audited registry of one-phase
validation runs. It excludes every coordinate in the current fit panels and keeps repeated seeds separate
at run level.

Artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/manifest.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/heldout_runs.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/heldout_coordinates.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/heldout_components.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/heldout_coordinate_components.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/source_inventory.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/target_coverage.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_heldout_benchmark_20260902/excluded_rows.csv`

The registry contains 530 eligible runs over 467 unique coordinates:

| panel | runs | coordinates | Uncheatable aggregates | Table-9 aggregates | full Uncheatable payloads | full Table-9 payloads |
|---|---:|---:|---:|---:|---:|---:|
| 60M, 39 buckets | 174 | 174 | 174 | 173 | 173 | 56 |
| 300M, 39 buckets | 134 | 134 | 117 | 134 | 117 | 134 |
| Delphi 3e18, 39 buckets | 222 | 159 | 222 | 209 | 217 | 209 |

The Delphi set includes 35 recent epoch-cap validations: 11 shared-shape DSP, 8 aggregate-V, and 16
full-canonical DSP runs. Thirteen full-canonical rows do not yet have Table-9 endpoints. Never impute an
atomic component or use an aggregate as a substitute for a missing component.

These are retrospective external validations. Models must never fit on them. If their outcomes influence
model or hyperparameter choice, call them external development evidence, not confirmation. A new model's
confirmatory claim requires a frozen candidate and a fresh validation run.

### Existing OLMix comparison

`reference_outputs/olmix_swarm_single_phase_dsp_20260901/report.md` contains the current complete Michael
proxy-swarm comparison. A taskwise ridge log-link on `log1p(epoch exposure)` reaches RMSE/Spearman
0.14276/0.940 on DCLM and 0.11314/0.942 on high-quality, versus 0.31508/0.647 and 0.22888/0.774 for the
current exact-OLMix macro baseline. The saturating DSP benefit head is statistically indistinguishable
from the linear epoch head on RMSE. Inventory-permutation controls support correctly assigned
inventory-indexed curvature, but they do not uniquely identify epoch exposure as the cause. Preserve this
distinction.

### StarCoder one-dimensional shape suite

`inventory_starcoder_single_phase_curves_20260902.py` resolves the historical StarCoder tables into a
run-deduplicated registry. Its core gate contains 45 endpoint Programming Languages BPB curves and 1,058
observations across four physical families: three coupled-onset curves, 28 horizon-by-replay curves, four
fixed-model token-ladder curves, and ten matched-model-size/token-budget curves. Every curve varies only
the tied StarCoder fraction `p`, has at least 13 distinct weights, and spans at least 0.8 of the simplex
edge.

Artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/manifest.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/curve_inventory.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/curve_memberships.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_single_phase_curve_inventory_20260902/target_observations.csv`

`fit_starcoder_all_tied_curves_canonical_dsp_20260902.py` is the descriptive capacity reference. Canonical
DSP has lower full-data RMSE than exact OLMix on all 45 curves; median RMSE is 0.00521 versus 0.09142 BPB.
This does not replace the shared out-of-fold benchmark. Use its plots and cached fits to check adapters and
diagnose response-shape failures:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_all_tied_curves_canonical_dsp_20260902/index.html`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_all_tied_curves_canonical_dsp_20260902/report.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_all_tied_curves_canonical_dsp_20260902/metrics.csv`

The inventory also contains 31 checkpoint curves from the same training runs. Keep those outside the core
endpoint gate; any trajectory analysis must group splits by physical training run to prevent state leakage.

## Benchmark panels and targets

Use the following six complete benchmark sources:

| panel id | rows | buckets | atomic targets |
|---|---:|---:|---:|
| `60m_39bucket` | 242 | 39 | 7 Uncheatable + 51 Table 9 |
| `300m_39bucket` | 280 | 39 | 7 Uncheatable + 51 Table 9 |
| `delphi_3e18_39bucket` | 280 | 39 | 7 Uncheatable + 51 Table 9 |
| `dclm_10k` | 363 | 118 | frozen 8-task subset from 42 available tasks |
| `high_quality_10k` | 363 | 120 | frozen 8-task subset from 42 available tasks |
| `starcoder_wsd80_tied_curves` | 45 curves / 1,058 observations | 2 | Programming Languages BPB per physical curve |

Fit every atomic objective independently, then reconstruct aggregates. Do not fit the aggregate label in
the primary benchmark.

- Uncheatable is the fixed seven-component byte-weighted micro-BPB. Use the payload-specific weights in
  the canonical protocol. The legacy 60M/300M payload and Delphi payload differ slightly.
- Table 9 is the fixed unweighted mean of 51 components.
- For Michael's swarms, fit and report only the frozen unweighted eight-task mean below. The historical
  42-task and 41-task overlap results remain context, but they are outside the new `Certify` contract and
  must not be mixed into its leaderboard.
- For StarCoder, fit and score all 45 endpoint curves separately. Report per-curve metrics, then
  macro-average within each of the four physical families and macro-average the four family scores so the
  28 horizon-by-replay curves do not dominate by count.
- Keep Uncheatable and Table 9 as separate paper-primary objectives. Do not average them into a new scalar.

Prominently report these atomic anchors without replacing the aggregates:

- Uncheatable: GitHub Python and GitHub C++.
- Table 9: MT-MBPP Python, MT-MBPP C++, Minerva Math Geometry, and HellaSwag.

The frozen Michael subset is identical in both proxy swarms:

1. `codex_humaneval/gold_bpb_3shot`
2. `mt_mbpp_python/gold_bpb_3shot`
3. `mt_mbpp_cpp/gold_bpb_3shot`
4. `gsm8k/gold_bpb_5shot`
5. `naturalqs_open/bpb_5shot`
6. `sciq/rc_5shot`
7. `mmlu_stem/rc_5shot`
8. `mmlu_humanities/rc_5shot`

This subset was chosen before the Observatory comparison to cover code, math, factual QA, science QA, and
two MMLU families. Do not replace tasks based on which model fits them best. Report the eight atomic tasks
as well as their unweighted mean.

### Mandatory StarCoder one-dimensional shape suite

The normalized source is `reference_outputs/starcoder_single_phase_curve_inventory_20260902/`. Use the 45
rows in `curve_inventory.csv` with `state_role=endpoint` and `primary_target_ready=true`, then join
observations through the normalized membership and target tables; do not reconstruct the suite by globbing
historical outputs. Treat each physical curve as an independent one-degree-of-freedom fit. Keep the four
physical families separate in reporting before computing the equal-family macro-average.

For every curve, report out-of-fold RMSE, Spearman, calibration, and the actual BPB and regret of the
minimum selected from its out-of-fold predictions. Also report whether the model can express an interior
minimum when the measured curve has one. Full-panel fitted curves are descriptive only and must not replace
the out-of-fold metrics. A model that cannot represent these smooth one-dimensional curves or recover their
sampled optimum basin has failed a basic response-shape sanity check, even if it ranks well on a
high-dimensional panel.

The four fixed-model token-ladder curves are the highlighted anchors within this larger suite. Their source
is
`reference_outputs/starcoder_wsd80_fixed_model_tied_diagonal_20260730/results_20260731/tied_diagonal_observations.csv`.
They contain the same 21 regular tied mixtures, from 0% to 100% StarCoder in 5-point increments, at each of
1B, 2B, 4B, and 8B materialized tokens. Within one token budget, the Nemotron share is \(1-p\), so the input
has exactly one degree of freedom. Fit and score each token budget separately; do not pool budgets unless
the model explicitly conditions on training scale.

The historical two-dimensional surface the project densely sampled before moving to the 39-bucket Delphi
setting is
`reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_observed_metrics.csv`. It began as a
107-coordinate WSD80 surface and was later augmented with fixed-aggregate fibers to 346 unique
\((w_0,w_1)\) coordinates. Its tied rows were subsequently completed into the cleaner 4 x 21 diagonal
above, which is the benchmark input.

For each fixed-model budget, additionally compare the selected weight with the measured grid minimum and
the frozen one-SD basin in `tied_optima.csv`; report RMSE divided by the corresponding repeat SD in
`repeat_noise.csv`.

The 45-curve StarCoder suite is mandatory but does not enter the paper-primary aggregate leaderboard. The
31 checkpoint curves, sparse StarCoder controls, Common Crawl topic swarms, and replay constant-mixture
sweeps remain supplemental mechanism and transfer panels.

## Observatory model inventory

The canonical registry is `MODEL_IDS` in
`experiments/domain_phase_mix/exploratory/two_phase_many/export_mixture_fit_observatory.py`.

Benchmark these 18 entries or document an exact single-phase equivalence:

| model id | display name | Observatory status |
|---|---|---|
| `linear` | Linear | visible baseline |
| `olmix_loglinear` | OLMix log-linear | visible baseline |
| `canonical` | Canonical DSP | visible |
| `effective_exposure` | Effective-exposure DSP | visible |
| `effective_exposure_geometry` | Effective-exposure DSP + geometry | visible |
| `separate_heads` | Separate heads | visible |
| `grp` | GRP, regularized | visible |
| `compact_retained_state` | Compact retained state | visible |
| `bucket_family_grp` | Bucket-resolved family GRP | visible |
| `hierarchical_phase_bucket_replay` | Hierarchical phase replay | visible |
| `crs_plus` | Compact retained state + family | visible |
| `crs_bounded` | Compact retained state, bounded | visible |
| `hpr_band` | Hierarchical phase replay band ensemble | visible |
| `bucket_family_power_separate_heads` | Power + separate heads | visible |
| `retained_power_law` | Retained power law | visible diagnostic |
| `bucket_family_power_separate_heads_family_onset` | Power + separate heads, family onset | hidden diagnostic |
| `bucket_family_weibull_shared_onset` | Weibull GRP, shared onset | hidden diagnostic |
| `bucket_family_weibull_family_replay` | Weibull GRP, family replay | hidden diagnostic |

### Single-phase reduction contract

Do not obtain a single-phase model by setting phase-0 and phase-1 weights equal while retaining duplicate
or inert parameters. For every model:

1. Derive its exact phase-tied image algebraically.
2. Remove schedule-sensitive columns that become zero, constant, or duplicate.
3. Remove nonlinear search dimensions that no longer affect predictions.
4. Refit all remaining shape and regularization hyperparameters inside each outer training fold.
5. Record the identifiable linear rank and nonlinear degrees of freedom.
6. Test prediction equivalence between the reduced implementation and the original model on tied inputs.

Known equivalence: `canonical` and `effective_exposure` reduce to the same total-exposure DSP when phase
freedom is removed. Benchmark this restriction once and list both source model ids in its equivalence
class. Find and document any other exact equivalences instead of spending compute on duplicate fits.

Create a machine-readable registry with at least:

```text
source_model_id, visible, single_phase_model_id, equivalence_class,
active_mechanisms, removed_phase_terms, allowed_metadata,
identifiable_linear_rank, nonlinear_dof, solver_id, hyperparameter_grid
```

An unsupported model is not silently omitted. Either implement its identifiable restriction or mark it
unsupported with the exact mathematical or data-contract reason.

### Family and metadata constraint

The archival `broad_text` / `tech_code` / `reasoning` semantic partition is banned.

If a model uses families, they may come only from declared domain classification and quality splits. In
the 39-bucket Dolma 3 + Dolmino panel, this permits the thirteen Common Crawl high/low domain pairs and an
explicit high/low/unsplit quality relation. Do not invent a family for singleton buckets. For Michael's
panels, use only source metadata already present in their manifests; if no matching domain/quality
relation exists, use the no-family restriction.

Exposure-count strata may be tested as a label-blind pooling ablation. Do not call them semantic families
or use their result as evidence for domain semantics.

## Required ablations

First express each distinct restriction as a small set of mechanism flags. At minimum, cover:

1. Policy coordinate: mixture weight versus materialized epoch exposure (E_b=c_bw_b).
2. Benefit response: affine, `log1p`, exponential saturation, inverse power, and Weibull where present.
3. Repetition harm: none, unbounded DSP damage, bounded damage, learned onset, and literal replay where
   present.
4. Link: identity, positive log-link, and bounded or deficit link where present.
5. Shape sharing: per-bucket shape, one shared shape, and permitted domain/quality shrinkage.
6. Geometry: no geometry versus aggregate concentration. Phase TV and late-phase concentration vanish in
   single phase.
7. Head: signed versus nonnegative amplitudes, with matched regularization.
8. Estimator: squared versus Huber loss, ridge strength, and band ensemble versus selected member where
   these choices differ between models.

For every full model, run one-factor leave-one-mechanism-out ablations. Add a matched-capacity control when
removing a mechanism changes feature count, rank, sign constraints, or tuning budget. Inventory
permutation is required before attributing gains to inventory-aware epoch curvature. Outcome permutation
is a negative control, not a model candidate.

Every distinct parent model reaches the full `Certify` tier. Every one-factor ablation reaches `Screen`.
Promote an ablation to `Certify` when it changes a paper-primary metric beyond repeat dispersion or is
needed to attribute a parent's improvement. Publish the complete promotion rule before inspecting the
ablation results.

## Common benchmark protocol

Use one shared split manifest for every model:

- five mixture-blocked outer folds;
- one frozen outer partition at seed `20260902` for `Certify`;
- three mixture-blocked inner folds for every fitted shape, ridge, link, or ensemble choice;
- coordinate-level grouping so repeated seeds never cross folds;
- identical train/test rows and component availability for paired model contrasts.

For StarCoder, assign the ordered weights within each of the 45 physical curves to one deterministic shared
five-fold partition and keep all curves as separate fits. Use the same fold assignment for every model so
all curve and optimum-selection contrasts are paired.

Five repeated outer partitions are a finalist replication stage, not part of per-model `Certify`. Run
them only for canonical DSP, taskwise OLMix, the best statistical equivalence class, and any ablation needed
to identify its winning mechanism.

Reuse the exact panel loaders, aggregation contracts, and fold definitions from
`benchmark_single_phase_componentwise_canonical_dsp_20260902.py`. Export the split assignments and their
hash. No model may choose its own easier folds.

### Tiers

Use three resumable tiers:

1. `Smoke`: one outer fold on the six named 39-bucket atomic anchors, all eight frozen Michael tasks, and
   the 1B StarCoder tied curve. It validates the adapter, reduction, gradients, cache key, and output schema.
2. `Screen`: one complete five-fold run on the same six 39-bucket anchors and eight Michael tasks. This
   tier also includes all 45 StarCoder endpoint curves. It compares mechanisms quickly; it makes no
   reconstructed Uncheatable or Table-9 macro claim.
3. `Certify`: one complete five-fold run on all 58 components of each 39-bucket panel and the same eight
   Michael tasks in each proxy panel, plus all 45 StarCoder endpoint curves. This is the tier used for the
   all-model leaderboard.

One `Certify` run contains \(5(3\times58+2\times8+45)=1{,}175\) component-fold fits. The previous
one-repeat canonical benchmark used 1,290 fits and took 4.9 minutes, so the target remains at most about
five minutes per distinct model on this host. If `Smoke` projects beyond five minutes, profile and optimize before
launching the full matrix. A run exceeding eight minutes fails the throughput gate and needs a documented
bottleneck analysis. Do not obtain speed by reducing folds, starts, convergence tolerances, or target
coverage without recording a separate protocol.

### Metrics

Report per atomic target, reconstructed aggregate, panel, repeat, and fold:

- RMSE in native units;
- RMSE divided by same-mixture repeat SD where repeat noise is identified;
- Spearman rank correlation;
- calibration intercept and slope;
- regret at 1 and preregistered top-k regret;
- selection optimism, defined as predicted improvement minus measured improvement for the selected row;
- low-BPB-basin RMSE and rank correlation under one frozen basin definition;
- fit time, convergence status, boundary hits, effective rank, and fitted degrees of freedom.

Use paired fold-level contrasts against canonical DSP and taskwise OLMix. Use the corrected five-fold
variance with the realized test/train ratio because the blocked folds can be unbalanced. Treat these
five-fold intervals as screening uncertainty. Use the five-repeat finalist stage for the final uncertainty
claim. Report point estimates and intervals; do not turn a small rank difference into a winner when
selection regret or calibration is unresolved.

For the external 39-bucket registry:

- score `heldout_coordinates.csv` for mixture-level ranking and selection;
- use `heldout_runs.csv` only for run-level noise and seed sensitivity;
- use component tables only where the relevant completeness flag is true;
- report each scale and proposal source separately before any pooled summary;
- never tune on the external rows inside an outer fold;
- label a result retrospective if it influenced model selection.

### External heldout optimum-selection test

Optimum selection on the coordinate-disjoint 39-bucket heldouts is a co-primary benchmark outcome, not an
optional diagnostic. For each `(model, panel, objective)`:

1. Select every model and hyperparameter using only the canonical fit panel and its nested-CV results,
   then refit on the complete canonical fit panel. Do not use any heldout label, component, proposal source,
   or rank in this choice.
2. Predict every eligible coordinate with an observed aggregate for that objective. For componentwise
   models, predict the atomic objectives and reconstruct Uncheatable or Table 9 with the frozen aggregation
   rule before selection.
3. Freeze and hash the row-level predictions before joining them to heldout outcomes.
4. Select the predicted minimum
   \(\hat w_H=\arg\min_{w\in H}\hat L(w)\) over the complete heldout bank \(H\). Report its measured
   coordinate-mean BPB, its measured rank and percentile, and heldout regret
   \(L(\hat w_H)-\min_{w\in H}L(w)\).
5. Also report the best measured BPB and regret among the model's predicted top 5 and top 10 coordinates.
   This measures whether a small validation shortlist would recover the observed heldout optimum basin.
6. Report a best-basin hit only under a panel/objective-specific equivalence tolerance frozen from
   independent same-mixture repeat noise before model scores are inspected. If that tolerance is not
   identified, report continuous regret and do not invent a binary success label.

Run the selection test separately for Uncheatable and Table 9 at 60M, 300M, and Delphi 3e18. Report both
the pooled eligible bank and proposal-source strata, because the bank is not a uniform simplex sample.
Compare every model with canonical DSP, taskwise OLMix where defined, and a deterministic random-ranking
baseline. The primary external-selection comparison is regret at 1; RMSE and Spearman cannot substitute
for it.

This test asks whether a fitted surrogate can choose a good already-materialized mixture outside its fit
panel. It does not establish that its continuous simplex argmin is good, and the best heldout coordinate is
not the unknown global optimum. The registry is retrospective: several outcomes predate this protocol and
have informed the research program. Once a heldout result influences model construction or selection, use
it only as external development evidence. A continuous optimum or frontier claim still requires a frozen
candidate and fresh training validation.

## Solver requirements

Build one benchmark harness with model adapters rather than one script per model. Preserve these
properties:

- atomic shard identity `(model, panel, target, component, repeat, fold)`;
- atomic writes and idempotent resume;
- input, source-code, split, model-registry, and protocol hashes in every cache key;
- deterministic starts and deterministic outputs;
- no nested process/BLAS oversubscription;
- explicit failure rows rather than silently missing metrics.

Reuse the profiled-gradient canonical solver. For other nonlinear models:

1. Separate nonlinear shape search from linear-head fitting.
2. Precompute exposure tensors and candidate designs by panel, fold, and shape.
3. Batch independent component right-hand sides when that is algebraically exact.
4. Differentiate the profiled objective or use bounded low-dimensional search instead of repeated generic
   differential evolution when possible.
5. Cache compiled JAX kernels by bucket count and model shape if JAX produces a measured speedup.
6. Parallelize across shards at one level only.

Any optimized solver needs parity tests against the current reference on small deterministic fixtures:

- feature matrix equality;
- objective and gradient equality;
- prediction equality;
- selected hyperparameter equality when the optimum is separated;
- convergence and boundary diagnostics;
- interrupted-run resume without recomputation.

Do not accept a faster solver solely because aggregate metrics look similar.

## Required benchmark artifacts

Write one versioned output directory containing:

- `model_registry.csv`
- `equivalence_classes.md`
- `split_manifest.csv`
- `protocol.json`
- `component_fold_metrics.csv`
- `component_metrics.csv`
- `aggregate_fold_metrics.csv`
- `aggregate_metrics.csv`
- `paired_model_contrasts.csv`
- `external_heldout_predictions.csv`
- `external_heldout_selection_metrics.csv`
- `starcoder_one_dimensional_curve_metrics.csv`
- `starcoder_tied_diagonal_metrics.csv`
- `complexity_and_runtime.csv`
- `ablation_promotions.csv`
- `failures.csv`
- `report.md`

The report must distinguish fit-panel nested CV, retrospective external validation, and any future fresh
confirmation. Include the full model and ablation inventory, including failed or equivalent entries.

Record each artifact and validation in Fieldbook. Resolve the existing next action
`note_01m1geh14fexxfv53dadch6b5g` only after the target registry, split manifest, metric definitions, and
hashes are frozen.

## Independent review gate

Do not begin successor-model synthesis until the complete parent-model benchmark and promoted ablations
are ready.

Run two fresh read-only reviews sequentially. Do not give either reviewer the other review.

### Codex review

Use the `codex-subscription-review` skill, pinned to `gpt-5.6-sol` at maximum reasoning. Point it to the
handoff, implementation, protocol, and final artifacts. Ask it to verify:

1. every Observatory model is represented by an exact identifiable restriction or an explicit exclusion;
2. all fold, tuning, aggregation, and heldout boundaries are leakage-safe;
3. optimized solvers are mathematically equivalent to their references;
4. paired uncertainty and selection metrics are computed correctly;
5. claimed mechanism effects follow from matched ablations rather than capacity or tuning differences;
6. the proposed successor is the minimum sufficient model.

The local wrapper is:

```bash
uv run ~/.claude/skills/codex-subscription-review/scripts/codex_review.py \
  --instructions "Review the completed single-phase Observatory benchmark described in .agents/handoffs/single_phase_observatory_ablation_and_modeling_cc_handoff_20260902.md. Inspect the code and stored artifacts. Check registry completeness, single-phase equivalence, leakage, solver parity, paired statistics, ablation attribution, and whether the proposed successor is the minimum sufficient model. Return blockers first and distinguish verified findings from inferences. Do not edit files."
```

### DeepSeek review

Use the `deepseek-subscription-review` skill, pinned to `deepseek-v4-pro` at maximum reasoning. Give it the
same evidence and ask these pointed questions:

1. Are any single-phase reductions still rank-deficient or carrying inert parameters?
2. Do any apparent mechanism gains disappear under a matched feature-count, regularization, link, or
   nonlinear-search control?
3. Are results consistent across model scale, bucket count, target family, and the low-BPB basin?
4. Does retrospective heldout selection overstate evidence for optimum prediction?
5. Which smallest formula is actually supported, and what result would falsify it?

The local wrapper is:

```bash
uv run ~/.claude/skills/deepseek-subscription-review/scripts/deepseek_review.py \
  "Review the completed single-phase Observatory benchmark described in .agents/handoffs/single_phase_observatory_ablation_and_modeling_cc_handoff_20260902.md. Inspect the implementation and artifacts. Audit identifiability, matched ablations, cross-panel consistency, retrospective-heldout use, and the minimum supported mechanistic formula. Return blockers first. Do not edit files."
```

Save both responses under `.agents/handoffs/`. Reproduce every load-bearing finding before changing the
implementation. Record agreements, disagreements, fixes, reruns, and final disposition in Fieldbook.

## Successor-model synthesis

After the review gate passes, start from the simplest model in the best statistical equivalence class.
Add only mechanisms whose matched ablation improves at least one paper-primary objective without causing
an unresolved regression on the other scales or on selection regret.

The likely design space is deliberately narrow:

- shared-shape epoch-exposure benefit;
- a `log1p` or saturating response, chosen by matched evidence;
- bounded or literal repetition harm only if it improves the high-TPP/3e18 regime;
- a bounded or log-deficit link only if it improves low-basin calibration;
- domain/quality shrinkage only where declared metadata exists and a no-family control is weaker;
- per-task atomic heads followed by fixed aggregate reconstruction.

Avoid a new union of every winning feature. Prefer \(O(B)\) linear amplitudes and a small number of shared
nonlinear shapes. Per-bucket nonlinear shapes require enough rows to identify them and should not be the
default on the 118/120-bucket panels.

Benchmark each successor candidate with the same `Screen` and `Certify` protocol. Freeze the final
formula, hyperparameter grids, optimizer constraints, and proposal objective before scoring the external
registry. Optimize Uncheatable and Table 9 separately. If the model proposes a new production mixture,
materialize it under the existing epoch-cap constraints and validate it with fresh seeds before claiming
an improved frontier.

## Completion criteria

This handoff is complete when:

1. every Observatory entry has an audited single-phase restriction and equivalence class;
2. every distinct parent model has complete frozen-five-fold `Certify` metrics or a reviewed hard exclusion;
3. every mechanism claim has a matched ablation and uncertainty estimate;
4. the benchmark runs in about five minutes per model and resumes at shard granularity;
5. Codex and DeepSeek reviews are saved and all verified blockers are resolved;
6. one minimal successor model, or a justified target-specific pair, is benchmarked under the same
   protocol and included in the five-repeat finalist stage;
7. every 39-bucket parent and finalist has the frozen external heldout argmin, rank, regret-at-1, and
   shortlist metrics; external evidence is labeled correctly and any frontier claim has fresh validation;
8. every parent model has out-of-fold fit and sampled-optimum-selection metrics on all 45 one-dimensional
   StarCoder endpoint curves, with equal-family summaries and the four fixed-model anchors reported
   separately;
9. Fieldbook contains the artifacts, validations, decisions, failures, and final handoff.
