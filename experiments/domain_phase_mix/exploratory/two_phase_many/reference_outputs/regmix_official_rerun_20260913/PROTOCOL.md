# Official RegMix regression and proposal rerun

Frozen before fitting on 13 September 2026. The purpose is to decide whether the RegMix endpoint and MARINER–RegMix midpoint require new validation runs. This round performs local regression, proposal generation and comparison only. It does not submit training or replace published measurements.

## Reference and primary procedure

Use notebook commit `dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6`: https://github.com/sail-sg/regmix/blob/dd9d1c3b2d7c1756b1a90f0ad7603068e9856cc6/regression_fitting/regression.ipynb . Execute fitting cell 13 verbatim with the replacement dataset. Execute candidate cell 16 with only the domain-prior literal replaced, and execute averaging cell 18 verbatim. No standardization, inner-fold tuning, full-data refit, candidate polishing, or best-candidate safeguard. Preserve both L1 and L2 early-stopping metrics, 1,000 maximum iterations, learning rate 0.01, seed 42, and patience three.

- Dataset: the existing frozen 280 unique Qwen swarm mixtures, in canonical order. Extra calibration repeats are excluded. No bank or prospective endpoint/midpoint outcomes enter fitting or selection.
- Split adaptation: the notebook supplies separate 512-training and 256-validation CSVs but no split recipe. Randomly permute the 279 non-calibration rows with legacy NumPy RandomState(42), put the first 93 in validation, keep the proportional anchor in training, and preserve original row order within both splits. This gives 187 training and 93 early-stopping validation runs. Freeze the resulting IDs and input hashes before fitting. Validation is used for early stopping and is not independent evidence.
- Primary response adaptation: two raw objective columns, each computed with its frozen component weights. These replace the notebook's selected scalar metric. Predict and optimize each objective head directly.
- Secondary head sensitivity: run the same official fitting cell on the 58 raw component responses, then aggregate with frozen objective weights. This isolates the former per-component head construction. It is not selected based on validation results.
- Candidate prior: frozen corpus token counts divided by their sum, in the same 39-bucket order as the regression inputs. Match np.random.seed(42) and np.random.dirichlet(prior,100000). Average the top128 candidates exactly as in cell18.
- Additional sampler-only diagnostic: apply this same candidate pool and averaging to the old frozen fitted trees. This separates candidate-search changes from refitting.
- Downstream runtime adaptation: save the continuous proposal, then use the existing 1/2048 count allocator for training-compatible endpoints and exact50% blends with the frozen MARINER endpoint. Save continuous/rounded predictions and rounding distances separately.
- Environment: use and record NumPy2.3.5, LightGBM4.7.0, sklearn1.8.0, SciPy1.17.0 and pandas2.2.2. The reference repository has no regression environment lock; this is source-exact execution on substituted data, not a historical bitwise replay. Keep CPU threads at one for the existing mixed-OpenMP environment.
- Export boundary: use the archived true mixture weights. The notebook consumes its CSV inputs raw; we do not reproduce the upstream collector's five-decimal export rounding.

## Comparisons and decision

Report total variation in endpoints and corresponding midpoints, per-bucket allocations/epochs, both old/new predictors and frozen MARINER predictions at all old/new endpoints and midpoints, and the early-stopping iteration counts. Overlay predicted paths and distinguish measured historical points from unmeasured new proposals. A changed runtime mixture has no interchangeable existing measurement; whether to train it follows from the size of the change and the intended baseline claim. Do not choose whichever protocol yields the desired outcome.
