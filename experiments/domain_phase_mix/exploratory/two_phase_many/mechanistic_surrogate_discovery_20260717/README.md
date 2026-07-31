# Mechanistic surrogate discovery

This directory is the rerunnable source for the 2026-07-17 local model-discovery effort. It evaluates one- and two-phase BPB surrogates without reading the sealed adversarial stress panel or submitting training jobs.

Run every script from the Marin repository root. PEP 723 headers provide script-local dependencies:

```bash
export PYTHONPATH=.

uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/freeze_baseline_gate.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/audit_algebra.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/screen_portfolio.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/analyze_failure_modes.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/audit_oof_identification.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/audit_deployment_support.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/audit_closest_candidate_stability.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/audit_raw_optima.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/synthesize_discovery.py
```

The complete audit includes the following additional independent screens. Run them after `freeze_baseline_gate.py` and before `synthesize_discovery.py`:

```bash
scripts=(
  audit_baseline_family_transfer.py
  audit_additive_cancellation.py
  audit_data_provenance.py
  audit_calibration_pareto.py
  audit_candidate_effective_df.py
  audit_convex_support.py
  audit_convex_support_directions.py
  audit_design_identifiability.py
  audit_collision_transfer.py
  audit_hyperparameter_equifinality.py
  audit_intervention_source_transfer.py
  audit_model_disagreement_warning.py
  audit_multivariate_invariant_upper_bound.py
  audit_oof_uncertainty_transfer.py
  audit_phase_information_transfer.py
  audit_phase_tied_restrictions.py
  audit_policy_determinacy.py
  audit_ridge_calibration_path.py
  audit_raw_optimum_crossfit.py
  audit_raw_optimum_support_path.py
  audit_scalar_invariant_sufficiency.py
  audit_series_residual_structure.py
  audit_shape_transfer.py
  audit_shared_transition_laws.py
  audit_support_stratified_baselines.py
  audit_trimmed_calibration.py
  audit_worst_policy_feature_decomposition.py
  bootstrap_candidate_differences.py
  bootstrap_heldout_calibration.py
  evaluate_reverse_kl_transfer.py
  screen_collision_limited_acquisition.py
  screen_distinct_data_scaling.py
  screen_family_collision_hazard.py
  screen_gradient_flow_bowl.py
  screen_importance_ess_scaling.py
  screen_kish_collision_invariant.py
  screen_nested_support_invariants.py
  screen_nested_two_level_prior.py
  screen_phase_boundary_adaptation.py
  visualize_worst_heldout_policies.py
)
for script in "${scripts[@]}"; do
  uv run "experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/$script"
done
```

Mechanism-specific screens are separate scripts so a rejected mechanism cannot silently alter another family's derivation. Their statuses and exact evidence are in `approach_registry.md`. All generated artifacts are written under:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/
  mechanistic_surrogate_discovery_20260717/
```

The immutable acceptance gate is content-addressed by `frozen_gate/frozen_manifest.json`. The final synthesis refuses to run if its digest changes. Scripts that read external evidence apply the sealed-input assertion from `freeze_baseline_gate.py`, and synthesis applies it to every scanned screen artifact.
The final manifest records the SHA-256 digest of every protocol and evidence input, every scanned screen-metric artifact, every Python and Markdown source file in this directory, and every synthesized output except the manifest itself. `validate_deliverables.py` verifies complete source/output coverage and rejects input, source, or output drift.

The final deliverables are in `final_synthesis/`:

- `approach_registry.csv`: normalized registry with premise, state, transition, response, and rejection evidence.
- `baseline_metrics.csv`: complete frozen baseline table across applicable panels.
- `candidate_metrics.csv`: comparable serious-candidate metrics.
- `acceptance_gate_evaluation.csv`: conjunctive frozen-gate scorecard.
- `final_report.md`: verdict and evidence.
- HTML calibration, convex-support, additive-cancellation, StarCoder, baseline-transfer, hyperparameter-equifinality, disagreement-warning, ridge-path, worst-policy, and gate figures.

After synthesis, validate all frozen hashes, registry fields, row counts, target coverage, and gate outcomes with:

```bash
uv run --script experiments/domain_phase_mix/exploratory/two_phase_many/mechanistic_surrogate_discovery_20260717/validate_deliverables.py
```

No deployment KL penalty or trust region enters the raw-optimum audit. No post-hoc calibration, nearest-neighbor feature, ensemble, or heldout-derived output correction is used.
