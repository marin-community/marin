# Proportional Controllability 300M

## Purpose

Run a 300M/6B-only diagnostic around `baseline_proportional` to compare:

- full leave-one-domain-out coverage ablations, and
- per-domain central log-tilt perturbation pairs around proportional.

This is a single-seed diagnostic. Deletions are nonlocal boundary interventions
and should be interpreted as coverage ablations, not clean local derivatives.
The log-tilt pairs are closer to local projected-gradient probes, but they are
still single-seed estimates.

Related prior experiment:
`.agents/logbooks/proportional-perturbation-scale-transfer.md`, especially the
`+0.05` domain-bump panel from
`/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728`.

## Design

- Scale: historical `300m_6b`, displayed as corrected `100M/6B`.
- Baseline: reuse existing `baseline_proportional` from
  `pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b`.
- Source experiment prefix:
  `pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m`.
- Family: `proportional_controllability_300m_6b`.
- Run IDs: `800000..800116`.
- Phase mode: `both_phases`; every run has `phase_0 == phase_1`.
- Training: `v5p-8`, `us-east5-a`, target final checkpoint step `22887`,
  perplexity-only training eval, final checkpoint and HF export preserved.

### Domain deletions

For each domain `j`, run:

`w_j = 0`, and `w_i = p_i / (1 - p_j)` for `i != j`.

This yields exactly `39` runs. The TV distance from proportional is `p_j`.

### Central log-tilts

For each domain `j`, construct a singleton domain direction `v^j` and run a
plus and minus endpoint:

`w_i^+ = p_i exp(+alpha v_i) / sum_k p_k exp(+alpha v_k)`

`w_i^- = p_i exp(-alpha v_i) / sum_k p_k exp(-alpha v_k)`

with `alpha = 0.10`, `sum_i p_i v_i = 0`, and
`sum_i p_i v_i^2 = 1`.

This yields exactly `78` runs: `2` endpoints for each of `39` domains. Domain
direction IDs use slugs such as `domain_dolma3_cc_art_and_design_high`, while
the manifest preserves the original domain name in `target_domain`.

## Planned Analysis

- Compare deletion effects against the existing `+0.05` domain-bump effects.
- Orient every metric as utility; BPB/loss metrics are sign-flipped.
- Report raw deletion contrast:
  `Delta_j^del = U(w_without_j) - U(p)`.
- Report deletion-implied score:
  `q_j^del = -(1 - p_j) Delta_j^del / p_j`.
- Report log-tilt directional derivative:
  `d_v = (U(w_plus) - U(w_minus)) / (2 alpha)`.
- Flag small-domain deletion scores as noise-amplified because the rescaling
  divides by `p_j`.
- Produce plots for deletion effects by domain mass, deletion-vs-bump sign
  agreement, deletion-implied `q` vs bump-implied `q`, log-tilt directional
  derivatives, and metric-family coverage-criticality.

## 2026-05-20 Implementation Notes

- Added launcher:
  `experiments/domain_phase_mix/launch_proportional_controllability_300m.py`.
- Added eval candidate builder:
  `experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py`.
- Registry integration should preserve intervention metadata and checkpoint
  provenance under family `proportional_controllability_300m_6b`.
- Before live launch, run dry-run validation and Claude Code review using the
  subscription workflow.

## 2026-05-20 Initial Dry-Run Validation Superseded

Command:

```bash
uv run python experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  --dry-run \
  --tpu-region us-east5 \
  --tpu-zone us-east5-a \
  --max-concurrent 8
```

Results from the initial selected-direction design:

- Dry-run wrote manifests under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/`.
- Training rows: `69`.
- Intervention rows: `69`.
- Intervention counts: `39` `domain_deletion`, `30` `central_log_tilt`.
- Run IDs: `800000..800068`.
- Scale: all `300m_6b`.
- Phase columns: `39` domains in `phase_0` and `39` in `phase_1`.
- Phase sums: max error `2.22e-15` for each phase.
- `phase_0 == phase_1`: max absolute delta `0.0`.
- Target final checkpoint step: `22887`.
- `num_train_steps`: `22888`, so the final checkpoint step is `num_train_steps - 1`.
- Duplicate run names: `0`.
- Deletion TV check: max absolute error between `tv_distance` and `base_mass` is `1.04e-16`.
- Log-tilt directions: `15`, each with exactly `minus,plus`.

This validation was superseded before launch. We decided the clean comparison
should include a central log-tilt pair for every domain, not just `15` selected
directions. The corrected design has `117` total runs: `39` deletions and `78`
log-tilt endpoints.

## 2026-05-20 Corrected 117-Run Dry-Run Validation

After discussion, the selected-direction log-tilt design was replaced with a
fully symmetric per-domain design. The corrected manifest now contains:

- `39` leave-one-domain-out coverage ablations.
- `78` central log-tilt endpoints: plus and minus for each of `39` domains.
- `117` total 300M/6B training runs.

Dry-run command:

```bash
uv run python experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  --dry-run \
  --tpu-region us-east5 \
  --tpu-zone us-east5-a \
  --max-concurrent 8
```

Dry-run result:

- Manifest directory:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/`.
- Training rows: `117`.
- Intervention rows: `117`.
- Intervention counts: `39` `domain_deletion`, `78` `central_log_tilt`.
- Run IDs: `800000..800116`.
- Scale: all `300m_6b`.
- Phase columns: `39` domains in `phase_0` and `39` in `phase_1`.
- Phase sums: max error `2.3314683517128287e-15` for each phase.
- `phase_0 == phase_1`: max absolute delta `0.0`.
- Minimum phase weight: `0.0`, only from deletion target domains.
- Target final checkpoint step: `22887`.
- `num_train_steps`: `22888`, so the final checkpoint step is `num_train_steps - 1`.
- Duplicate run names: `0`.
- Duplicate run IDs: `0`.
- Deletion TV check: max absolute error between `tv_distance` and `base_mass` is `1.04e-16`.
- Log-tilt directions: `39`, exactly one per domain.
- Log-tilt pairs: every direction has exactly `minus,plus`.
- Log-tilt target domains: all `39` domains covered, missing `0`.

Validation commands for the corrected design:

```bash
uv run python -m py_compile \
  experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_metric_registry.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py

./infra/pre-commit.py \
  experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_metric_registry.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py
```

Both validation commands passed for the corrected `117`-run design. The live
launch still needs the planned Claude Code review gate.

## 2026-05-20 Log-Tilt Visualization

Generated a visualization of the `78` central log-tilt mixtures relative to the
proportional baseline:

- Interactive HTML:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_weight_relative_heatmap.html`.
- Static PNG:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_weight_relative_heatmap.png`.
- Per-target summary CSV:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_target_multiplier_summary.csv`.

The heatmap value is `log2(w_tilt / w_proportional)` for each plotted domain.
The largest relative target upweights happen for very small proportional-mass
domains, as expected under fixed `alpha = 0.10` KL-local coordinates.

Also generated materialized-weight audit artifacts:

- Interactive HTML:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_materialized_weight_heatmap.html`.
- Static PNG:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_materialized_weight_heatmap.png`.
- Full `78 x 39` materialized weight matrix:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_materialized_weights_matrix.csv`.
- Target-domain `w+`/`w-` summary:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_materialized_target_weights.csv`.

The materialized heatmap values are raw phase weights in percent, not ratios.
Rows sum to `100%` with max absolute numerical error
`2.4158453015843406e-13`.
