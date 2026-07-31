# Debugging log for genericfamily-tuner-gap

Understand how the original hardcoded GenericFamily-RetainedTotal-Tuned nonlinear parameters were obtained, and determine whether we can reproduce them or a validated-optimum-similar mixture with a principled tuning procedure.

## Initial status

We have in-repo GRP implementations and tuners, plus ChatGPT Pro artifacts under /Users/calvinxu/Downloads/GenericFamily-RetainedTotal-Tuned. There is a gap between our current tuning procedures and the original hardcoded tuned parameters / deployed validated GRP mixture.

## Hypothesis 1

The gap is explained by differences in objective, parameterization, initialization, and/or additional selection heuristics encoded in the ChatGPT Pro artifacts.

## Changes to make

- Inspect the external artifact bundle and identify the exact scripts, hardcoded numbers, and any reconstruction logic.
- Compare those against our in-repo GRP tuner and baseline implementations.
- Add instrumentation or small helper code only if needed to compare objectives or recovered optima.

## Future Work

- [ ] Check whether historical starts or teacher matching are being used
- [ ] Check whether the deployed GRP mixture is a convex hull / anchor combination rather than a direct tuned optimum
- [ ] Check whether alternative tuning objectives recover a mixture closer to the deployed GRP one

## Results

The ChatGPT Pro bundle does not contain a real nonlinear tuner. The self-contained script hard-codes:

- `pair_params`
- `broad_params`
- `broad_beta_params`
- `tuned_generic_params`

and only evaluates them plus a convex-hull deployment step. The deployment summary in the bundle matches the in-repo tuned GRP baseline exactly:

- predicted value `1.0436372226731572`
- anchor coefficients:
  - `best_observed = 0.02587662289245359`
  - `validated_global = 0.9175270003197237`
  - `validated_pair = 0.0015381110202398686`
  - `baseline_proportional = 0.05505826576758283`

I added [investigate_genericfamily_tuner_gap.py](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/surrogate_search/investigate_genericfamily_tuner_gap.py), which benchmarks plausible outer-loop procedures against two targets:

- closeness to the published nonlinear parameter vector
- closeness to the published convex-hull deployment mixture

The key result is that exact parameter recovery is not the useful target unless we start from the published params themselves. But a generic recovery of the deployed mixture is possible:

- Best overall, unsurprisingly:
  - `current_from_tuned_lbfgsb`
  - starting from the published tuned params, `L-BFGS-B` returns a nearly identical vector and deployment mixture

- Best discovery-style procedures excluding the tuned start:
  - `both_from_broad_beta_powell`
  - `cvregret_from_broad_beta_powell`

These do **not** recover the published nonlinear params. Their parameter-space distances are large (`~11.5` in packed z-space), but they do recover a deployment hull very close to the published one:

- `both_from_broad_beta_powell`
  - deployment TV to target: `0.004607`
  - deployment coeff L1 to target: `0.019599`
  - deployment predicted value: `1.043666`
  - coeffs:
    - `best_observed = 0.017236`
    - `validated_global = 0.924415`
    - `validated_pair = 0.004449`
    - `baseline_proportional = 0.053900`

- published target coeffs:
  - `best_observed = 0.025877`
  - `validated_global = 0.917527`
  - `validated_pair = 0.001538`
  - `baseline_proportional = 0.055058`

Interpretation:

- The nonlinear parameters are not well identified.
- Multiple very different nonlinear parameter vectors produce almost the same augmented-data convex-hull deployment recommendation.
- The gap is therefore not “we can’t recover the deployed GRP behavior”; it is “the published nonlinear params are one representative in a broad equivalent basin.”

The most plausible generalizable recipe so far is:

- start from `CCPairBroadBeta-RetainedTotal`
- optimize a regret-aware CV + anchor objective
- use `Powell`
- deploy via the augmented-data convex hull, not the raw unconstrained optimum

This reproduces the *deployment* much better than the raw nonlinear params.
