# Official RegMix replay changes both validation paths

Executing the released regression and proposal cells on the existing Qwen3 360M/1.6B-token swarm (nominally 3e18 training FLOPs per run) produces different RegMix endpoints for both objectives. The corresponding 50% MARINER–RegMix blends also change substantially. Existing endpoint and midpoint measurements cannot be relabeled as measurements of these new mixtures.

If the paper adopts this reference procedure, validate its two endpoints and two midpoints, initially with the same data seeds and trainer seed 0 used for the existing RegMix paths. Keep the current measurements as results for the adapted baseline. No training has been submitted in this replay.

## Mixture changes

Total variation is half the sum of absolute changes in bucket weights; it is the fraction of mixture mass reassigned. Distances below compare runtime-rounded mixtures against the old trained ones.

| Objective | New endpoint | New midpoint | Sampler-only endpoint | Component-head endpoint |
|---|---:|---:|---:|---:|
| Uncheatable | 48.68% | 24.32% | 47.02% | 45.85% |
| OlmoBaseEval Easy | 47.51% | 23.88% | 48.44% | 48.54% |

The sampler-only diagnostic retains the old frozen trees and substitutes the reference candidate distribution, count, RNG and top-128 averaging. The component-head diagnostic applies reference fitting separately to the former 7/51 component responses and aggregates them with the frozen weights. Both diagnostics produce large changes, so the primary result is not contingent on switching to direct aggregate heads. The diagnostics do not independently apportion the effects of each protocol change.

![Predictions along old and recomputed RegMix paths](official_replay_paths.png)

Gold points are existing trainer-seed-0 measurements. Each right panel uses a different endpoint mixture from the left panel. Only its MARINER endpoint has a measurement; its other predictions are unvalidated.

## What was reproduced

The source is the [official notebook at commit dd9d1c3](https://github.com/sail-sg/regmix/blob/dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6/regression_fitting/regression.ipynb). Fitting cell 13 and averaging cell 18 execute unchanged. Candidate cell 16 executes with only the dataset-specific prior literal replaced.

- Raw mixture weights and raw responses; no standardization, floor or log link.
- LightGBM learning rate 0.01, seed 42, up to 1,000 iterations, default leaf settings, and early stopping after three rounds without improvement, monitoring the notebook's L1 and L2 metrics.
- The early-stopped model is retained. There is no full-data refit.
- The candidate pool has 100,000 draws using legacy NumPy seed 42 and corpus-token proportions as Dirichlet concentrations. The proposal averages the 128 lowest-predicted candidates. No polish or best-candidate safeguard is added.

The notebook provides separate 512-training and 256-validation CSVs but no split algorithm. Our replacement dataset therefore uses a prespecified random 187/93 split of the 280 existing designs, with the proportional anchor in training. The 93 validation outcomes are used for early stopping and are not independent evaluation data. No extra calibration repeats, retrospective-bank outcomes or prospective validation outcomes enter fitting. The primary response columns are the two scalar objectives: Uncheatable combines seven component BPBs with its frozen task weights, and OlmoBaseEval Easy averages 51 component BPBs. The secondary sensitivity fits each component separately before applying those same weights.

Other necessary data/runtime adaptations are the 39-bucket token prior, exact archived weights instead of the original collector's five-decimal exports, and downstream rounding to the existing 1/2048 sampler grid. The repository has no regression environment lock. This is source-exact execution on substituted inputs, not a claim to reproduce historical library behavior. Versions are pinned in the script and recorded in `input_manifest.json` (LightGBM 4.7.0, NumPy 2.3.5).

Independent verification refitted both primary heads from the saved input CSVs using the original cell. Model text, all 100,000 candidate predictions and top-128 indices match exactly. Best iterations are 460 for Uncheatable and 170 for the suite. See `independent_verification/verification.json`.

## Predictions do not establish an improvement

The reference objective fits are more pessimistic than the old fits at the three previously measured mixtures per objective. This comparison uses the same seed-0 measurements, and is a retrospective calibration check. Uncheatable retains the seven frozen task weights: the new midpoint evaluation computes component BPB as total loss bits divided by scored bytes, while the historical endpoints use the earlier batch-averaged component estimator. Across 98 saved evaluations, the largest observed estimator difference was 0.000054 BPB in the weighted objective; no empirical offset is applied. OlmoBaseEval Easy uses its unchanged native evaluator.

| Objective | Mixture | Measured BPB | Old RegMix prediction | Reference RegMix prediction |
|---|---|---:|---:|---:|
| Uncheatable | mariner | 0.981424 | 1.008356 | 1.037254 |
| Uncheatable | endpoint | 1.000331 | 1.013784 | 1.034792 |
| Uncheatable | midpoint | 0.987572 | 1.005856 | 1.035887 |
| OlmoBaseEval Easy | mariner | 1.068202 | 1.101763 | 1.180860 |
| OlmoBaseEval Easy | endpoint | 1.086836 | 1.122041 | 1.175938 |
| OlmoBaseEval Easy | midpoint | 1.075381 | 1.104587 | 1.176416 |

The suite reference fit predicts 1.180860 BPB at MARINER's mixture and 1.188817 at its own returned proposal. Thus following the reference code does not eliminate the original apparent inconsistency. The sampler searches a finite candidate pool and returns an average, which need not minimize the fitted predictor. Even the old trained RegMix proposal receives a lower prediction than the new returned proposal on both objectives. There is no basis to infer the new proposals' measured performance from these predictions alone.

For the four newly proposed validation points:

| Objective | Point | Reference RegMix prediction | Frozen MARINER prediction | Measured |
|---|---|---:|---:|---|
| Uncheatable | endpoint | 1.035484 | 1.025648 | Unmeasured |
| Uncheatable | midpoint | 1.034594 | 0.988722 | Unmeasured |
| OlmoBaseEval Easy | endpoint | 1.188817 | 1.132000 | Unmeasured |
| OlmoBaseEval Easy | midpoint | 1.181691 | 1.072671 | Unmeasured |

At the same proxy training budget and available bucket subsets used in the existing swarm, the two new endpoint mixtures repeat their most-exposed bucket about 10 times. The changes are much larger than runtime rounding (roughly 0.2% total variation), so the old midpoint measurements do not approximate the new mixtures merely because both are 50% blends.

## Reproduction and handoff

Run from the Marin checkout:
```bash
uv run --offline --no-sync --with lightgbm==4.7.0 python experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/regmix_official_rerun_20260913/rerun_official.py
```

The script freezes its input manifest and reuses completed fits when that manifest is unchanged. Raw source cells, model artifacts, input hashes, the split, candidate-pool hash, continuous/rounded weights, cross-predictions and path predictions are saved beside this report. `PROTOCOL.md` was written before fitting. `policy_weights.csv` contains the primary and diagnostic proposals; `proposal_comparison.csv` identifies the `official_objective` primary result.

Historical measurement provenance and the Uncheatable metric bridge are documented in `HISTORICAL_MEASUREMENTS.md`. The figure script is `plot_official_replay.py`. The manuscript and its plotted measurements remain unchanged until a baseline-protocol decision and new validation.
