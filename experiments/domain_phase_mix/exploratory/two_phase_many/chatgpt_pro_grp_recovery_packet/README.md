# GRP Recovery Packet

This packet is a self-contained handoff for improving the many-domain GRP
functional form, nonlinear fitting procedure, convergence behavior, calibration,
and deployment regularization.

## What this packet contains

- `code/grp_packet.py`
  - self-contained GRP reference implementation
  - nonlinear parameter packing/unpacking
  - current anchor-aware tuning objective
  - raw unconstrained optimum search
  - convex-hull deployment optimization
- `code/run_convergence.py`
  - convergence evaluation across subset sizes
  - supports `raw_retuned`, `top8actual_hull`, and `all_observed_hull`
- `code/run_deployment_variant_benchmark.py`
  - observed-only regularizer benchmark
- `code/run_optimizer_benchmark.py`
  - compares `L-BFGS-B`, `Powell`, and `Nelder-Mead`
- `code/plot_fixedparam_subset_validation.py`
  - visualizes the calibration gap from the original fixed-parameter subset sweep
- `data/many_domain_packet.npz`
  - narrow packet export: `y`, `w`, `c0`, `c1`, `domain_names`, `run_names`
- `data/subset_indices_feature_bayes_linear.json`
  - exact observed-run subsets used for the convergence study
- `data/current_reference_state.json`
  - current tuned params, broad-beta start params, validated anchors, and current deployed GRP reference
- `data/fixedparam_subset_validation_results.csv`
  - realized results from the first subset sweep where nonlinear GRP params were held fixed
- `reference_outputs/`
  - current repo outputs copied in for reference

## Current picture

- The current deployed GRP baseline validates well.
- If we retune the nonlinear GRP parameters on all datapoints and optimize the raw surrogate directly,
  `Powell` and `L-BFGS-B` converge to almost the same optimum, but that optimum looks worse than the
  deployed GRP mixture and sits farther from the observed swarm.
- In the original subset sweep, keeping nonlinear parameters fixed and only refitting the linear head
  validated badly. The calibration gap is large and sometimes catastrophic.
- Retuning the nonlinear parameters per subset fixes the ranking story a lot:
  retrospective `Regret@1` is good by about `k >= 80`.
- But the absolute predicted BPB of the deployed optimum is still not calibrated well enough.
- Observed-only deployment regularization helps. Among the regularizers benchmarked so far,
  `top8_actual_hull` is the current preferred deployment rule:
  nearly tied on predicted BPB realism with the full observed hull, but a bit more stable and simpler.

## Main research questions

1. Can we find a general nonlinear fitting procedure that produces a deployment close to the validated GRP one,
   or better, without privileged reconstruction tricks?
2. Can we close the calibration gap so the predicted BPB of the optimum mixture is actually trustworthy?
3. Can we improve the functional form enough that the raw optimum is good, or otherwise design a better
   observed-only deployment regularizer?

## Important caveat

The current retuning objective uses two already-validated anchor mixtures:

- `validated_global`
- `validated_pair`

These are included because they are part of the current repo procedure. They are useful as a reference,
but they are not a clean basis for the final general method. A better answer would use only information
available at subset size `k`, or clearly separate any anchor-based reconstruction procedure from the
general deployment rule.

## How to run

All scripts are standalone and use inline `uv` metadata.

Examples:

```bash
uv run code/run_optimizer_benchmark.py
uv run code/run_convergence.py --variant top8actual_hull
uv run code/run_deployment_variant_benchmark.py
uv run code/plot_fixedparam_subset_validation.py
```

## Recommended reading order

1. `REQUEST_TO_CHATGPT_PRO.md`
2. `code/grp_packet.py`
3. `reference_outputs/fixedparam_subset_validation_results.csv`
4. `reference_outputs/optimizer_benchmark.csv`
5. `reference_outputs/convergence_top8actual_hull.csv` or the repo-copied current convergence outputs
