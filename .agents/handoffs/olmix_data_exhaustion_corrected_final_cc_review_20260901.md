# Final review: corrected OLMix single-phase and Delphi exhaustion analyses

Please independently audit the two corrected exploratory analyses below. This is a read-only scientific and implementation review. Inspect the code and stored outputs directly; do not edit anything.

## A. OLMix proxy-swarm single-phase challenger

Main worktree: `/Users/calvinxu/Projects/Work/Marin/marin`

- Benchmark: `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_olmix_swarm_single_phase_dsp_20260901.py`
- Tests: `tests/test_benchmark_olmix_swarm_single_phase_dsp_20260901.py`
- Report and machine-readable outputs: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/olmix_swarm_single_phase_dsp_20260901/`

The earlier review blocked because the promoted linear head was compared with an unmatched saturating inventory scramble. The corrected analysis now includes a matched `linear_epoch_log_link_permuted_inventory` control and uses the matched saturating pair (`dsp_benefit_log_link` versus `dsp_permuted_inventory`) as the mechanism witness. The gate promotes the linear model package against exact OLMix only if RMSE and held-fold selection regret improve in both complete swarms; it separately requires the saturating exposure head to beat its matched inventory scramble in both pools. It does not use the High Quality linear-inventory contrast to claim a linear-link-specific mechanism because that corrected interval crosses zero.

The ridge grid is widened to `3e-4` through `3e3`. The report now explicitly discloses the remaining boundary-selected primary rows and limits the interpretation accordingly. Please verify all numbers, leakage controls, corrected intervals, gate semantics, and whether fresh prospective validation of the linear epoch-exposure package is defensible without over-attributing the result to exposure alone.

## B. Delphi crossed-prefix state-dependent data exhaustion

Side worktree: `/Users/calvinxu/Projects/Work/Marin/marin-delphi-y0-y1-surrogate-20260827`

- Benchmark: `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_crossed_prefix_data_exhaustion_20260901.py`
- Tests: `experiments/domain_phase_mix/exploratory/two_phase_many/test_benchmark_delphi_crossed_prefix_data_exhaustion_20260901.py`
- Report and machine-readable outputs: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_crossed_prefix_data_exhaustion_20260901/`

The earlier review blocked because tied abstention had zero regret by construction and therefore supplied an impossible promotion criterion. The corrected gate removes that criterion and treats universal tied dominance as a limitation of this panel. The report now states that repeated exposure is poorly identified, that the fresh/repeated split is nearly binary, that `state_exhaustion` is the weight-1 endpoint of the blend path, and that leave-one-prefix-out is anchored by the held prefix's tied outcome.

The requested free sensitivity is now present: leave each of the 50 common actions out once and apply the corrected factor `1/50 + 1/49`. It finds no benefit for blend weights 0.25, 0.50, or 0.75, while the pure state-exhaustion head is significantly worse than cumulative exposure. Please verify the implementation and determine whether the honest conclusion is to reject this parameterization while leaving the broader exhaustion mechanism unresolved due to identification.

## Review contract

1. Reproduce or check the load-bearing numerical claims from the stored CSV/JSON artifacts.
2. Verify that the two original blocking defects are actually fixed in code, prose, and machine-readable gates.
3. Identify any remaining defect that would reverse either decision or make the Fieldbook record misleading.
4. Distinguish blockers from nonblocking limitations.
5. End the response with exactly `APPROVE` or `BLOCK` on its own line.
