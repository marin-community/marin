We are revisiting the many-domain GRP surrogate and need a fully reproducible answer this time.

Please work only from the attached packet and send back the whole implementation, not just tuned coefficients or a prose description.

## What we want

We want an improved GRP-style procedure for the two-phase many-domain packet that addresses:

1. nonlinear parameter fitting
2. convergence with subset size
3. calibration of the predicted optimum BPB
4. deployment regularization

The current state is:

- the deployed GRP baseline validates well in practice
- the raw retuned all-data optimum found by `Powell`/`L-BFGS-B` looks worse than the deployed one
- fixed-nonlinear-param subset optima validated badly
- retuned-per-subset ranking is much better, but predicted optimum BPB is still not calibrated enough
- observed-only hull deployment helps, and `top8_actual_hull` is the current best simple observed-only rule we tested

## Constraints

- Please avoid privileged reconstruction tricks.
- Do not use teacher matching to the published/hardcoded nonlinear parameter vector.
- Do not use tie-breakers or objectives that explicitly reward closeness to the current tuned params.
- If you use the two validated anchors (`validated_global`, `validated_pair`), treat that as a separate reconstruction path, not the main general method.
- The main recommended method should use only information available at subset size `k`, or clearly justify any extra information it needs.

## Deliverables

Please return:

1. the full updated code, in complete files, for every file you changed or added
2. a short explanation of the exact fitting algorithm:
   - objective
   - optimizer
   - starting points
   - bounds / transforms
   - regularization
   - deployment rule
3. the exact command(s) to run the updated convergence / deployment / optimizer benchmarks
4. a summary of what improved and what did not

## Strong request

Please do not respond with only:

- tuned numbers
- pseudocode
- partial diffs
- “here is the idea”

We need the full runnable thing so we do not lose the tuning procedure again.

## Preferred target

The best answer would give us a method that:

- produces a deployment mixture close to the validated deployed GRP, or better
- has better subset-size convergence
- reduces the calibration gap for predicted optimum BPB
- does not need special off-swarm anchors in the final recommended procedure

## Files to focus on

- `code/grp_packet.py`
- `code/run_convergence.py`
- `code/run_deployment_variant_benchmark.py`
- `code/run_optimizer_benchmark.py`
- `data/current_reference_state.json`
- `data/fixedparam_subset_validation_results.csv`
- `reference_outputs/`

If you think the functional form itself should change, please implement that in full and keep the old path available for comparison.
