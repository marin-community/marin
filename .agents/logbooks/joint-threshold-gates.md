# Joint Threshold Gates: Research Logbook

## Scope

- Goal: Test whether scale-dependent acquisition thresholds, inspired by Gu et al. phase-transition data-mixing results, explain residual errors in the current joint mixture/scale law.
- Primary metric: `eval/uncheatable_eval/bpb` on the canonical `analysis_dataset/nd_scale_runs.csv`.
- Constraints: Local-only modeling. No launches. Preserve the existing MCT-LRQ anchor behavior at corrected `100M/6B`.

## Baseline

- Date: 2026-05-07
- Code refs:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/joint_model_refreshed_20260426/mct_lrq_no_barrier_canonical/REPORT.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
- Baseline: canonical `mct_lrq69_drop_no_barrier`.

## Experiment Log

### 2026-05-07 - Threshold Gate Residual Sprint

- Hypothesis: A scale-dependent gate of the form
  `sigmoid(k * (log exposure_g(w) - log tau_g * (N/N0)^(-nu)))`
  may capture domains/families that become useful only after crossing a scale-dependent frequency threshold.
- Command:
  ```bash
  .venv/bin/python experiments/domain_phase_mix/exploratory/two_phase_many/run_threshold_gate_joint_law_sprint_20260507.py
  ```
- Config:
  - Anchor: corrected `100M/6B`, `(N0, D0) = (102,648,576, 5,999,951,872)`.
  - Threshold exponents: `nu in {0.75, 1.154791, 1.5}`.
  - Threshold quantiles: `{0.25, 0.50, 0.75}` over anchor-row group exposure.
  - Gate steepness: `{3, 6, 10}`.
  - Partitions: `dense_vs_broad`, `quality_three`, `current_source`, `canonical`, `all`.
- Result:
  - Canonical residual diagnostic: best score came from `current_source`, specifically `canon_resid_current_source_q0.75_k3_nu0.75`.
  - Compared with canonical baseline score `0.044305`, the best residual gate score was `0.032697`.
  - The improvement mostly came from the four-row 900M diagnostic and some fixed-340M movement; this is too small a support set to treat as a promotion.
  - Full local standalone threshold-gated models still raw-optimize to hard simplex corners, so threshold gates alone do not fix raw optimum quality.
- Interpretation:
  - Threshold gates are a plausible residual feature, especially with source-aware partitions.
  - The current evidence is exploratory. A serious candidate should graft constrained threshold gates into the exact canonical MCT-LRQ implementation, not the compact standalone approximation.
  - Any promotion candidate must report constrained optima and avoid relying on the four 900M rows as the main validation signal.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/threshold_gate_joint_law_20260507/REPORT.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/threshold_gate_joint_law_20260507/csv/canonical_residual_threshold_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/threshold_gate_joint_law_20260507/plots/canonical_residual_threshold_pred_actual.html`

### 2026-05-10 - Domain-Aware Scaling Synergy Sprint

- Hypothesis: The first- and second-order terms from `Domain-Aware Scaling Laws Uncover Data Synergy` may explain residual mixture/scale interactions better than threshold gates.
- Paper adaptation:
  - First-order term `z_k = u_k log(u_k D)` becomes `u_k log(D/D0)` after anchoring at fixed mixture and `D0`.
  - The paper's gamma identifiability constraint is implemented by centered share features, so the global data-scaling term remains separate.
  - Second-order terms use centered source-group pairwise soft-min features:
    `softmin_tau(log(1 + u_a D), log(1 + u_b D)) - softmin_tau(log(1 + u_a D0), log(1 + u_b D0))`.
- Command:
  ```bash
  .venv/bin/python experiments/domain_phase_mix/exploratory/two_phase_many/run_domain_aware_synergy_joint_law_sprint_20260510.py
  ```
- Variants tested:
  - First-order domain heads with power and exact log-D scale factors.
  - First-order current-source and canonical-family heads.
  - Current-source entropy scale feature.
  - Current-source second-order soft-min features with `tau=0.1` and `tau=1`.
  - Combined first-order, entropy, and second-order current-source features.
- Result:
  - Canonical residual diagnostic was negative. The unchanged canonical `mct_lrq69_drop_no_barrier` remained best with score `0.04431`.
  - Best paper-style residual correction was `canon_resid_paper_first_order_domain_log` with score `0.04519`.
  - The correction slightly improved the tiny 900M diagnostic and fixed-340M-all RMSE, but worsened seed7 holdout, fixed-340M holdout, and random supplement RMSE.
  - Full local anchored ablations were more positive: `paper_full_current_source_power_tau0.1` improved local selection score from `0.15191` to `0.13193`, but this local model is not the exact canonical MCT implementation.
- Interpretation:
  - Paper-style features are structurally meaningful but do not currently justify replacing or extending the canonical law.
  - In this single-target setting, first-order domain synergy collapses to per-domain/per-group data-scaling heads; this is only weakly identifiable without more independent scale rows.
  - Second-order co-occurrence is too flexible at the domain level; source-group soft-min terms are the only plausible version to keep exploring.
  - If this direction is revisited, use constrained current-source entropy/first-order corrections in the exact canonical implementation and require raw-optimum diagnostics.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/run_domain_aware_synergy_joint_law_sprint_20260510.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/domain_aware_synergy_joint_law_20260510/REPORT.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/domain_aware_synergy_joint_law_20260510/csv/canonical_residual_synergy_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/domain_aware_synergy_joint_law_20260510/plots/canonical_residual_domain_aware_synergy_pred_actual.html`

### 2026-05-14 - Repetition-Aware Variable-Size Law Sprint

- Hypothesis: The repetition-aware mixture scaling law from arXiv:2605.12715 can be adapted to Marin's ND scaling panel by treating each domain as a target-like component and replacing the paper's scalar target fraction with domain-level exposure and repetition estimates.
- Paper form evaluated:
  `L = E + C/N^beta + B N^delta / D_eff^alpha + gamma h`,
  with `D_eff` built from domain exposure and an exponential repetition discount.
- Marin adaptation:
  - Domain exposure: `h_i = 0.8 w_0i + 0.2 w_1i`.
  - Domain repeat estimate: `r_i = w_0i c_0i + w_1i c_1i`, using simulated-epoch multipliers from the ND packet.
  - Effective domain data: `D_i,eff = h_i D r_i,eff / r_i`.
  - Variants: scale-only, equal domain value, positive per-domain value weights, signed linear mixture head, and per-domain `r_1`.
- Command:
  ```bash
  uv run --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with kaleido python -m experiments.domain_phase_mix.exploratory.two_phase_many.fit_repetition_aware_variable_size_law_nd
  ```
- Data:
  - Source: `experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`.
  - Metric: `eval/uncheatable_eval/bpb`.
  - Rows: 641 labeled ND rows.
- Result:
  - Scale-only Chinchilla-size baseline: grouped OOF RMSE `0.0330`, Spearman `0.8080`.
  - Equal-domain repetition-aware law: RMSE `0.0288`, Spearman `0.8635`.
  - Positive per-domain value weights with shared `r_1`: RMSE `0.0262`, Spearman `0.8920`.
  - Signed linear head with shared `r_1`: Spearman remained high (`0.8873`) but RMSE was poor (`0.0850`) from severe calibration instability.
  - Signed linear head with per-domain `r_1`: best grouped OOF RMSE `0.0257`, Spearman `0.8957`, but optimizer stopped at the function-evaluation budget.
- Interpretation:
  - Repetition-aware effective data is real signal on this panel; even the 7-parameter equal-domain version improves materially over scale-only.
  - Positive per-domain value weights are the most stable multi-domain extension.
  - The most flexible 123-parameter variant is a useful diagnostic baseline, but it is not a canonical replacement for DSP without optimum and perturbation-geometry validation.
  - Leave-one-scale-out remains uneven, especially for held-out `300m_6b` and `60m_1p2b`, so this form does not solve cross-scale transfer by itself.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/fit_repetition_aware_variable_size_law_nd.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/repetition_aware_variable_size_law_nd_20260514/report.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/repetition_aware_variable_size_law_nd_20260514/grouped_oof_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/repetition_aware_variable_size_law_nd_20260514/leave_scale_summary.csv`

### 2026-05-15 - Variable-Scale DSP Screen

- Hypothesis: The best path to a variable-size DSP is not to collapse DSP into one scalar `D_eff`, but to add an `(N,D)` scale head and center DSP mixture features against proportional at the same exact `(N,D)`.
- Form:
  - Baseline scale head: `g(N,D)=E+C(N/N0)^(-beta)+B(N/N0)^delta(D/D0)^(-alpha)`.
  - Centered DSP features: `Delta S_i = S_i(w)-S_i(w_prop)`, `Delta P_i = P_i(w)-P_i(w_prop)`.
  - Variable-scale DSP: `y_hat = g(N,D)-n^kappa_b sum_i a_i Delta S_i+n^kappa_p sum_i p_i Delta P_i`.
  - Exposure-scaled diagnostic additionally uses `(D/D0)^omega z_i` inside DSP exposure.
- Practical note:
  - A full finite-difference retune over 80+ domain nonlinear parameters was too slow for interactive iteration.
  - This screen freezes the effective-exposure DSP geometry from the existing 300M fit and retunes only global scale/amplitude nonlinear parameters plus the profiled linear head.
- Command:
  ```bash
  uv run --with numpy --with pandas --with plotly --with scipy --with scikit-learn --with kaleido python -m experiments.domain_phase_mix.exploratory.two_phase_many.fit_variable_scale_dsp_nd
  ```
- Result:
  - `dsp_vs_centered_no_amp`: grouped OOF RMSE `0.02680`, Spearman `0.92853`.
  - `dsp_vs_centered_shared_amp`: grouped OOF RMSE `0.02712`, Spearman `0.92334`.
  - `dsp_vs_centered_split_amp`: grouped OOF RMSE `0.02633`, Spearman `0.92739`, regret-at-1 `0.02324`.
  - `dsp_vs_centered_exposure_scaled`: grouped OOF RMSE `0.02658`, Spearman `0.93039`.
- Interpretation:
  - Centered variable-scale DSP is a strong improvement over the standalone repetition-aware mixture scaling-law adaptations in rank.
  - Split amplitude is best by grouped OOF RMSE/regret, but it has worse leave-130M and leave-60M behavior.
  - The no-amplitude centered form is the most stable leave-one-scale-out candidate in this screen.
  - Next step, if promoting this direction: implement analytic/autodiff gradients and retune full DSP geometry across ND, then run optimum and perturbation-gradient diagnostics.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/fit_variable_scale_dsp_nd.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/variable_scale_dsp_nd_20260515/report.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/variable_scale_dsp_nd_20260515/grouped_oof_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/variable_scale_dsp_nd_20260515/leave_scale_summary.csv`

### 2026-05-15 - MCT-DSP Hybrid Screen

- Hypothesis: MCT can be improved by keeping its centered `(N,D)` scale scaffold while replacing the LRQ mixture anchor with the stronger effective-exposure DSP domain geometry.
- Forms tested:
  - `mct_dsp_anchor`: centered MCT scale scaffold plus frozen effective-exposure DSP anchor.
  - `mct_dsp_split_amp`: centered DSP benefit/penalty interactions with fitted N-dependent amplitudes.
  - `mct_dsp_tau_shift`: global `eta_N log(N/N0)+eta_D log(D/D0)` shifts to DSP penalty thresholds.
  - `mct_dsp_apple_sat`: Apple-style shared-`r_1` repetition discount on saturation exposure only.
- Command:
  ```bash
  uv run --with numpy --with pandas --with plotly --with scipy --with kaleido python -m experiments.domain_phase_mix.exploratory.two_phase_many.fit_mct_dsp_hybrid_nd
  ```
- Result against canonical `mct_lrq69_drop_no_barrier`:
  - MCT reference: seed7 holdout RMSE `0.01008`, fixed-340M RMSE `0.00614`, random supplement RMSE `0.01233`, leave-900M RMSE `0.01018`.
  - Best hybrid on seed7-family splits was `mct_dsp_tau_shift`: seed7 holdout RMSE `0.01282`, fixed-340M RMSE `0.01299`, random supplement RMSE `0.01267`.
  - Best leave-900M hybrids were anchor/Apple saturation at RMSE `0.01596`; `mct_dsp_tau_shift` worsened to RMSE `0.02311`.
  - `mct_dsp_split_amp` improved train fit but collapsed out of sample: seed7 holdout RMSE `0.03899`, random supplement RMSE `0.04377`, leave-900M RMSE `0.14059`.
- Interpretation:
  - This first hybrid does not beat canonical MCT on the established validation protocol.
  - A penalty-threshold scale shift is the only useful additive idea from this screen, but it trades off badly against leave-scale transfer.
  - The split-amplitude variant is a clear overfit warning: adding expressive DSP-scale interactions without stronger regularization or full geometry retuning is unsafe.
  - Next plausible step is not to promote this hybrid; it is to implement full ND DSP retuning with analytic/autodiff gradients, then re-run optimum and perturbation-gradient diagnostics.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/fit_mct_dsp_hybrid_nd.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mct_dsp_hybrid_nd_20260515/report.md`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mct_dsp_hybrid_nd_20260515/metric_summary.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mct_dsp_hybrid_nd_20260515/img/mct_dsp_hybrid_metric_comparison.html`
