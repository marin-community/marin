# Debugging log for PIQA 5-shot BPB raw-optimum pathology

Investigate why the no-L2 GRP raw optimum for `lm_eval/piqa_5shot/bpb` looks pathological even though the fit metrics are decent.

## Initial status

The fitted no-L2 GRP objective for `lm_eval/piqa_5shot/bpb` reports:

- `oof_spearman = 0.747`
- `cv_rmse = 0.0264`
- best observed PIQA BPB `= 1.322685`
- predicted-observed run `run_00218` at `1.333735`
- raw predicted optimum `= 1.230235`
- nearest observed to the raw optimum `run_00079` at `1.383801`
- nearest-TV distance `= 0.658`

The plotted raw optimum puts most of phase 1 into `cc/food_and_dining_low` and `synth_qa`.

## Hypothesis 1

The raw optimum is only pathological because the continuous optimizer found a bad local minimum.

## Changes to make

- Re-run the raw continuous optimizer with multiple random starts and seeds.

## Results

The optimizer converged to essentially the same solution across seeds:

- objective values all around `1.230234`
- phase-1 max weight always about `0.60`
- phase-1 entries below `1e-4` remained very high (`30` to `35`)
- nearest-TV stayed large (`0.65` to `0.70`)

Conclusion: this is not a local-search accident. The fitted surrogate itself prefers this off-manifold regime.

## Hypothesis 2

The surrogate is over-rewarding a small set of broad-text/synthetic features while the corresponding penalties are too weak to keep the optimum near observed mixtures.

## Changes to make

- Inspect fitted GRP coefficients and the contribution breakdown for the raw optimum versus the best observed run.

## Results

The fitted nonlinear parameters are:

- `eta = 9.73`
- `lam = 6.1e-06`
- `beta = 0.898`
- `a_broad_text = 0.181`
- `a_tech_code = 0.214`
- `a_reasoning = 1.162`
- `tau_broad_text = 0.923`
- `tau_tech_code = 5.216`
- `tau_reasoning = 8.0`

Important implications:

- `lam` is effectively zero, so phase-0 retained exposure is barely discounted by phase-1 allocation.
- `eta` is very large, so phase-1 exposure gets amplified strongly.
- The only nonzero family-total reward is for `broad_text`.
- The broad-text family-group penalty coefficient is tiny (`0.0015`), while the reasoning penalty is huge (`349.3`), so the model strongly avoids reasoning but does not meaningfully punish broad-text concentration.

Largest rewarded singleton/pair coefficients:

- `food_and_dining` pair
- `synth_qa`
- `common_crawl_hq`
- `health` pair
- `electronics_and_hardware` pair

Largest raw-optimum contribution improvements relative to the best observed run:

- lower broad-text penalty
- larger `food_and_dining` pair contribution
- larger `synth_qa` contribution
- larger `synth_instruction` contribution
- larger `health` pair contribution
- larger `synth_thinking` contribution

The surrogate therefore learns a coherent but narrow hack:

1. Put phase 1 into a small number of broad-text/synthetic domains that PIQA correlates with in-sample.
2. Exploit the large `eta` multiplier on phase-1 exposure.
3. Avoid the heavily penalized reasoning family.
4. Accept a large move off the observed mixture manifold because there is no deployment regularizer in the raw optimum solve.

## Conclusion

The PIQA-BPB raw optimum is pathological primarily because:

- the metric is narrow enough that the surrogate can improve by over-specializing,
- the no-L2 fit learns very high phase-1 leverage (`eta`) with almost no retention discount (`lam`),
- broad-text penalties are too weak relative to the rewarded broad-text/synthetic signal features,
- and the raw optimizer has no trust-region/deployment regularizer to keep the solution near observed mixtures.

The issue is not just optimization noise; it is a stable off-manifold preference of this particular no-L2 objective fit.

## Future work

- [ ] Compare against a deployment-regularized or hull-trust-blended optimum for PIQA BPB.
- [ ] Add an explicit support / nearest-TV penalty when fitting metric-objective surrogates for narrow tasks.
- [ ] Check whether `piqa_5shot/bpb` remains pathological under the scale-aware fits once more matched-scale data arrives.
