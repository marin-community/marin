# Debugging log for Olmix subset fit instability

Investigate why `two_phase_many_olmix_loglinear_subset_optima` validated badly, and whether this conflicts with the existing full-swarm Olmix datapoint among the 242-run panel.

## Initial status

Most Olmix subset-fit predicted optima validated very poorly, despite there already being an Olmix datapoint in the many-domain table.

## Hypothesis 1

The subset-fit script is not reproducing the same objective / fitting procedure / regularization settings as the original imported Olmix baseline datapoint.

## Changes to make

Read the subset-fit code, the original Olmix baseline source code, and the summary artifacts. Compare what data, objective, and optimization settings each path uses.

## Results

Confirmed:

- The literal Olmix row in `two_phase_many.csv` is `baseline_olmix_loglinear` with `run_id=240` from `pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_bpb`.
- That row was launched from `launch_two_phase_many_olmix_baseline.py` and its source objective is `lm_eval/mmlu_5shot/bpb`, not `eval/uncheatable_eval/bpb`.
- Its actual `eval/uncheatable_eval/bpb` is `1.107874`, so it is not evidence that an uncheatable-fit Olmix raw optimum should work.
- The uncheatable-specific Olmix fit path is `two_phase_many_olmix_loglinear_uncheatable.py`, which is a separate refit not present as a literal observed row in `two_phase_many.csv`.

Conclusion: the existing Olmix datapoint in the 241-row swarm is the wrong objective for this comparison.

## Hypothesis 2

The subset-fit failures are caused by optimizer instability / local-minimum selection in the nonconvex Olmix fit, not by a mismatch in the training rows.

## Changes to make

Verify that the full-swarm uncheatable fit and the subset benchmark use the same 241 observed rows, then compare seed sensitivity of the fit and deployment solve.

## Results

Confirmed:

- `two_phase_many_observed_runs.TWO_PHASE_MANY_CSV_PATH` and the subset benchmark CSV are exactly the same file: `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many.csv`.
- Both fit paths see the same `241` completed rows, same metrics, same weights, and same domain order.
- The subset benchmark packet is actually `242` rows because it appends `baseline_stratified` to the `241` completed CSV rows.
- On that exact `242`-row packet, seed `0` reproduces the bad `k242` row exactly:
  - predicted raw optimum `0.807187`
  - chosen observed run `run_00200` with actual `1.068761`
  - nearest observed TV `0.691218`
  - phase-1 max weight `0.990925`
  - `37/39` phase-1 domains below `1e-4`
- The `241`-row helper fit without appended stratified is also pathological (`predicted=0.821914`), so the issue is not caused by the extra row.
- The Olmix summary field `fullswarm_chosen_run_name=run_00200` is the best *observed* run under the surrogate, not the raw deployed optimum. That observed choice remains reasonable even when the raw optimum is pathological.

Seed sweep on the same full `241` rows:

- seed `0`: bad basin, `pred=0.821914`, phase-1 nearly one-hot, sparse support
- seed `1`: stable basin, `pred=1.076453`, dense support, nearest TV `0.337331`
- seed `2`: stable basin, `pred=1.083581`, dense support
- seed `3`: stable basin, `pred=1.078501`, dense support
- seed `4`: intermediate, `pred=1.046992`, still dense support
- seed `5`: same bad basin as seed `0`

All these seeds still pick observed `run_00200` as the best observed run under the surrogate. The large difference is only in the raw continuous optimum.

Conclusion: the subset failures are primarily a solver-basin problem. `k140` and `k180` happened to land in a stable basin; `k242` with seed `0` landed in a collapsed phase-1 basin and produced a catastrophic deployment.

## Hypothesis 3

The `1.0687` Olmix number used in presentation tables is real as a historical run metric, but the exported row may be attaching the wrong phase weights to it.

## Changes to make

Inspect how `two_phase_many_all_60m_1p2b.csv` is built for `baseline_olmix_loglinear_uncheatable_bpb`.

## Results

Confirmed:

- `build_two_phase_many_all_60m_1p2b.py` appends `baseline_olmix_loglinear_uncheatable_bpb` as a validated run whose metrics come from the historical W&B run in `pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_uncheatable_bpb`.
- But its `phase_weights_fn` is `_olmix_uncheatable_phase_weights()`, which calls `load_fit_from_local_results(...)` and recomputes the weights from current local code/data at export time.
- The exported `1.068716` row in `two_phase_many_all_60m_1p2b.csv` has phase weights with `phase1_max = 0.989348`, `phase1_support_lt_1e-4 = 37`.
- Those weights are almost identical to the bad current `k242` seed-0 subset fit on the 242-row packet (mean-phase TV distance `0.020052`).

Conclusion: the historical validated metric and the currently exported phase weights are very likely decoupled. That means the `1.0687` number is probably trustworthy as a historical evaluation, but comparing its exported weights directly against the subset-validation weights is not trustworthy until provenance is fixed.

## Hypothesis 4

The bad Olmix subset deployments are caused by our softmax + L-BFGS-B solve, not by the fitted surrogate itself.

## Changes to make

Implement a paper-faithful two-phase exact proposer in Marin that optimizes directly on the phase simplices with CVXPY and weighted KL, then rerun the local subset benchmark and compare against the current solver.

## Results

Confirmed:

- The released Olmix source code fits the log-linear surrogate with multistart LBFGS + Huber loss, so the fit side is already nonconvex.
- The released source code does use a direct CVXPY proposer over simplex weights with KL regularization; our existing Marin solve was the part that diverged materially.
- A new exact two-phase proposer was added to `two_phase_many_olmix_loglinear_sl_verb.py` and benchmarked on the same nine subset sizes.
- The exact-solver benchmark writes to:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_olmix_loglinear_subset_cvxpy_curve_points.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_olmix_loglinear_subset_cvxpy_summary.json`
- Comparison against the current L-BFGS-B-based subset summary shows the two solvers are effectively identical on this packet:
  - chosen observed run changed on `0/9` subset sizes
  - mean absolute change in predicted optimum value: `0.000060`
  - max absolute change in predicted optimum value: `0.000138`
  - mean phase-TV distance between old and exact deployments: `0.001350`
- The pathological subset sizes remain pathological under the exact solver:
  - `k020`: old `0.267129`, exact `0.267170`, validated `1.396656`
  - `k242`: old `0.807187`, exact `0.807228`, validated `2.534581`
- The good subset sizes remain good under the exact solver:
  - `k140`: old `1.079932`, exact `1.079929`, validated `1.067286`
  - `k180`: old `1.078177`, exact `1.078180`, validated `1.067173`

Conclusion: the optimizer parameterization is not the main source of the bad Olmix subset validations. Once the surrogate is fit, our current softmax + L-BFGS-B solve and a paper-faithful two-phase CVXPY solve converge to essentially the same deployment on this packet. The remaining instability is in the surrogate fit / basin selection, not in the final solve.

## Hypothesis 5

The good Olmix trajectory in the baseline-scaling plots and the bad full-swarm Olmix point in the raw-optimum convergence plot are different artifacts with the same informal label.

## Changes to make

Trace the deployed `baseline_olmix_loglinear_uncheatable_bpb` run lineage, find the saved fit artifact, and compare it with the subset-optimum benchmark artifact used in the convergence plot.

## Results

Confirmed:

- The deployed scaling-plot Olmix run is `baseline_olmix_loglinear_uncheatable_bpb`, defined in `experiments/domain_phase_mix/two_phase_many_olmix_loglinear_uncheatable.py`.
- It was launched by `experiments/domain_phase_mix/launch_two_phase_many_olmix_uncheatable_bpb_baseline.py`, which fit the Olmix surrogate at launch time, wrote a fit summary, then trained exactly those phase weights.
- The launch-time fit summary exists at `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_uncheatable_bpb/fit_summary-e50206/olmix_uncheatable_fit_summary.json`.
- That saved fit summary is sane and dense:
  - predicted objective `1.059208`
  - regularized objective `1.068907`
  - best observed run `run_00125`, value `1.057199`
  - nearest observed run `baseline_proportional`, value `1.091835`, nearest TV `0.382463`
  - phase-0 max weight `0.194080`, phase-0 support below `1e-4`: `0/39`
  - phase-1 max weight `0.359690`, phase-1 support below `1e-4`: `0/39`
- The baseline-scaling plot uses this deployed run name at every scale:
  - 20M/2.6B: `1.076860`
  - 60M/1.2B: `1.068716`
  - 100M/6B: `0.956062`
  - 340M/10.4B: `0.871321`
  - 900M/24B: `0.809275`
- The bad convergence-plot point is not that run. It is the separate subset-optimum artifact `baseline_olmix_loglinear_optimum_k242_uncheatable_bpb`, produced by `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_olmix_loglinear_subset_optima.py` and deployed through `experiments/domain_phase_mix/two_phase_many_olmix_loglinear_subset_optima.py`.
- The `k242` subset-optimum artifact is a collapsed raw optimum:
  - predicted optimum `0.807187`
  - regularized objective `0.874994`
  - validated BPB `2.534581`
  - nearest observed `run_00125`, nearest TV `0.691218`
  - phase-0 support below `1e-4`: `3/39`
  - phase-1 support below `1e-4`: `37/39`
  - phase-1 max weight `0.990925`

Conclusion: the good Olmix baseline did not come from the bad `k242` raw-optimum convergence artifact. It came from a launch-time saved fit summary with dense phase weights. The `k242` convergence result is still valid evidence that Olmix raw continuous optima can be unstable, but it should not be interpreted as the deployed Olmix baseline.

Paper-facing conclusion:

- Olmix-style log-linear fitting is a useful learned baseline, but its unconstrained raw optimum is highly fit/optimizer-basin sensitive in this two-phase 39-domain setting.
- For baseline-scaling comparisons, use the stable saved Olmix fit that was actually launched and replayed across scales.
- Do not claim that generic unconstrained Olmix optimization is stable here; the stronger claim is only that a good Olmix-style selected fit is a reasonable learned baseline comparison.

Additional risk:

- `qsplit240_replay.py` and `build_two_phase_many_all_60m_1p2b.py` still recompute `baseline_olmix_loglinear_uncheatable_bpb` phase weights from local code/data instead of loading the saved launch-time fit summary.
- This is unsafe because historical metrics and recomputed phase weights can become decoupled. The saved fit summary should be treated as the source of truth for this deployed baseline unless we intentionally rerun and relaunch Olmix.

## Future Work

- [ ] Fix `build_two_phase_many_all_60m_1p2b.py` so validated Olmix rows use the historical run's actual phase weights or saved fit summary, not a fresh local recomputation.
- [ ] Fix `qsplit240_replay.py` so replay panels use the saved `baseline_olmix_loglinear_uncheatable_bpb` fit summary or frozen phase weights, not a fresh local recomputation.
- [ ] Add a local mirrored copy or embedded digest of `olmix_uncheatable_fit_summary.json` so paper-plot and packet code can reproduce the deployed Olmix mixture without depending on live GCS.
- [ ] Make the Olmix benchmark multi-start / multi-seed and select a solution with explicit support / TV guards instead of freezing one seed-0 basin.
- [ ] Distinguish clearly in tables between the old MMLU-fitted Olmix baseline and the uncheatable-refit Olmix baseline.
- [ ] Consider treating Olmix as an observed-run ranker only unless the raw optimum is constrained by support / TV regularization.
