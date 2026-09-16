# Final review: OLMix single-phase surrogate and state-dependent exhaustion

Review these two completed exploratory analyses as an independent, read-only statistical and mechanistic reviewer. Do not edit either worktree. Inspect the reports, implementation, and focused tests directly.

## Workstream A: Michael Ryan's OLMix proxy swarms

Main worktree: `/Users/calvinxu/Projects/Work/Marin/marin`

- Report: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/olmix_swarm_single_phase_dsp_20260901/report.md`
- Promotion gate: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/olmix_swarm_single_phase_dsp_20260901/promotion_gate.json`
- Benchmark: `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_olmix_swarm_single_phase_dsp_20260901.py`
- Tests: `tests/test_benchmark_olmix_swarm_single_phase_dsp_20260901.py`

The benchmark uses two independent complete 363-mixture proxy swarms, 42 endpoint tasks, five repeated geometric outer folds, and nested shape tuning. The simple linear epoch-exposure log-link is the proposed candidate. It improves direct macro-BPB RMSE over the exact 48-start summed-Huber scalar-macro OLMix fit in both swarms, with corrected confidence intervals below zero, and beats an inventory-permutation control. The saturating DSP response is unresolved against the linear response. This is model-development evidence only, not fresh optimum validation.

Assess:

1. Is `olmix_exact_macro` a fair reproduction of the incumbent for the primary direct macro target, or does task-level fitting, normalization, loss weighting, selection, or hyperparameter treatment create an unfair comparison?
2. Are outer-fold geometry, nested tuning, repeated-fold corrected intervals, and selection regret sufficient to rule out leakage or a favorable split artifact?
3. Does the inventory permutation meaningfully identify epoch exposure, or could the gain still be generic regularization/capacity?
4. Is promoting the simpler linear epoch-exposure head to fresh prospective validation defensible? State exactly what can and cannot be claimed now.

## Workstream B: crossed-prefix state-dependent data exhaustion

Side worktree: `/Users/calvinxu/Projects/Work/Marin/marin-delphi-y0-y1-surrogate-20260827`

- Report: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_crossed_prefix_data_exhaustion_20260901/report.md`
- Benchmark: `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_crossed_prefix_data_exhaustion_20260901.py`
- Tests: `experiments/domain_phase_mix/exploratory/two_phase_many/test_benchmark_delphi_crossed_prefix_data_exhaustion_20260901.py`

The candidate mechanism is that phase-0 exposure changes the marginal phase-1 value of each bucket by exhausting fresh material and increasing repeat damage. The smallest tested head has nonnegative fresh-benefit and repeated-damage blocks, known-prefix intercepts, fixed one-epoch nonlinear shapes, and action-blocked evaluation. A 0.75 blend of cumulative and state-dependent exposure has the best point RMSE, but its corrected interval versus cumulative crosses zero, it loses leave-one-prefix-out transfer, and no common action beats tied continuation. The full state-exhaustion head is worse.

Assess:

1. Are the fresh/repeated features identified well enough after phase-1 exposure and prefix intercepts to call this evidence for the mechanism?
2. Is the cumulative/state blend mechanistically meaningful or merely a flexible interpolation selected post hoc?
3. Do the action-blocked and leave-one-prefix-out diagnostics support any useful modeling update now?
4. What is the smallest next falsifiable model or experiment that could distinguish state-dependent exhaustion from ordinary cumulative exposure without semantic bucket partitions?

## Required verdict

Separate blocking defects from nonblocking limitations. Verify concrete claims against files. End with exactly `APPROVE` if the reports and restrained conclusions are defensible as written, or `BLOCK` followed by the minimum required corrections.
