# Review brief: three completed data-mixing experiments

Please independently review the numerical analyses and scientific interpretation of three completed experiments.
This is a read-only review. Look for incorrect comparisons, post-selection errors, overclaims, missing controls, and
the most defensible next decision.

## 1. Delphi one-phase shared-shape DSP epoch-cap sweep

Fieldbook: `exp_01m15xtnsdcf15sg18qdveg595`

Primary artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_dsp_epoch_cap_sweep_20260828/results.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_summary.json`

Exact result: 11 runtime-distinct policies, each with one common trainer/data seed and exact step-3006 endpoints.
Uncheatable improves almost entirely from cap 2 to cap 4, then plateaus; the best row is cap 10 at 0.981100, only
0.001621 below cap 4. DSP gets the five-cap Uncheatable ordering exactly right but is 0.039754 BPB RMSE too
optimistic in level. Table-9 is U-shaped with a best same-target row at cap 6, but DSP predicts monotonic improvement
through cap 12: Spearman -0.086 and 0.039483 BPB selection regret. The overall best DSP Table-9 row is actually the
Uncheatable-targeted cap-6 policy at 1.079741.

## 2. Delphi aggregate-linear-V one-phase challengers

Fieldbook: `exp_01m1c29bdb0tds6kqmhgzatsnn`

Primary artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_surrogate_challenger_validations_20260831/results.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_surrogate_challenger_validations_20260831/measured_results.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_surrogate_challenger_validations_20260831/measured_summary.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_one_phase_surrogate_challenger_validations_20260831/best_table9_component_delta_vs_dsp.csv`

Exact result: eight policies, caps 4/6/8/10 by two targets, one common seed. For Uncheatable, aggregate-V's best cap-8
row is 0.987830, 0.006730 worse than DSP's best; it predicts cap 10, with 0.000327 regret and Spearman 0.8. For
Table-9, aggregate-V's cap-8 row is 1.066446, 0.013295 better than the best DSP row across the two fresh sweeps.
It improves 34/51 Table-9 components versus that DSP row, with median component delta -0.005923. However, the
model predicts cap 10 rather than cap 8, with 0.007922 selection regret, Spearman 0.4, and 0.065814 absolute RMSE.
Candidates are 0.325-0.440 TV from nearest fit-panel support.

Proposed interpretation: aggregate-V finds a genuinely better Table-9 policy family, but its constrained cap argmin
and absolute calibration remain unreliable. This is not a global frontier claim and requires paired confirmation.
DSP remains the stronger Uncheatable policy generator; there is no single universally superior head.

## 3. StarCoder WSD80 coupled phase/LR-onset dense surfaces

Fieldbook: `exp_01m19sep9vs99vb9wxq6e14vvy`

Primary artifacts:

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/results.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/arm_summary.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/matched_coordinate_deformation.csv`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_coupled_onset_dense_surface_results_20260901/observations.csv`
- `experiments/domain_phase_mix/starcoder_wsd80_coupled_onset_dense_surface_design_20260830.json.gz`

All 375 exact step-28259 endpoints are present: 125 common aggregate/fiber coordinates for each coupled 0.60T,
0.80T, and 0.90T phase-boundary/LR-decay onset. The preregistered raw-grid gain is best tied BPB minus best eligible
untied BPB. Discovery gains are +0.003673, +0.006674, and +0.006154 BPB respectively, so they do not satisfy the
registered order `gain_0p60 >= gain_0p80 >= gain_0p90`. The best absolute programming endpoint is the 0.80T untied
row at 0.780577. A fixed historical c109-tied versus c020-untied comparison gives -0.001026, +0.004906, and
+0.002573, also non-monotonic. The selected untied minima are shallow: best-to-second gaps are 0.000476, 0.000523,
and 0.000907 BPB. Surface Spearman relative to 0.80T is 0.715 at 0.60T and 0.905 at 0.90T. The 0.90T programming
optimum regresses C4 by 0.120262 BPB relative to its selected tied policy.

This experiment intentionally couples phase-2 duration and LR-decay duration; it cannot attribute effects to either
alone. The predecessor fixed-boundary experiment is the LR-only control. The frozen contract reserves eight fresh
seeds for three selected tied/untied pairs (48 runs if all policies are unique), but the discovery screen is negative
for the directional hypothesis and the minima are winner-selected.

## Questions

1. Are the three result summaries numerically and statistically defensible?
2. Is “Table-9 model-family success but cap-argmin failure” the right reading of aggregate-V, or does one-seed
   winner selection make that too strong?
3. Does the StarCoder discovery result actively argue against earlier coupled onset increasing two-phaseness, or is
   it merely inconclusive before the 48-run confirmation?
4. Given the negative ordered screen, should the full 48-run confirmation still be spent, reduced to a smaller
   fixed-policy diagnostic, or stopped under the frozen contract?
5. What is the highest-value cross-experiment conclusion for mixture optimization, and what claim should explicitly
   not be made?
