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
