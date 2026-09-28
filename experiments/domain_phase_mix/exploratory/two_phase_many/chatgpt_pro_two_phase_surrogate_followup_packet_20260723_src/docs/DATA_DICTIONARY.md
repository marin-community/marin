# Data Dictionary

Canonical tables use one row per checkpoint observation. Each phase contains `phase_0_weight::<bucket>` or `phase_1_weight::<bucket>` columns normalized to sum to one. The catalog's `c0` and `c1` vectors convert phase weights to simulated exposure: \(e_i^{(0)}=c_i^{(0)}w_i^{(0)}\) and \(e_i^{(1)}=c_i^{(1)}w_i^{(1)}\).

## Tables

| File | Meaning |
|---|---|
| `data/canonical/delphi_3e18_one_phase_fit.csv` | 280 phase-tied policies |
| `data/canonical/delphi_3e18_two_phase_fit.csv` | 280 two-phase policies with exact aggregate-matched one-phase counterparts |
| `data/canonical/delphi_3e18_heldouts.csv` | Completed append-only development observations available at build time |
| `data/delphi_3e18_pair_index.csv` | Exact fit-pair mapping, phase TV, aggregate-match error, and observed phase deltas |
| `data/catalog.json` | Bucket order, family labels, exposure multipliers, targets, and row counts |

## Common Metadata

- `row_id`: packet-unique observation identifier.
- `policy_class`: one-phase, tied, or two-phase policy label.
- `split`: fit or heldout.
- `training_series`: experimental or proposal population; this is an evaluation stratum, not a model feature.
- `proposal_target`: objective used to generate a candidate.
- `candidate_kind`: design or selection stratum.
- `group_id`: coordinate grouping key used to prevent leakage across repeated or paired observations.
- `fit_panel_overlap`: whether the coordinate exactly overlaps the fit panel.
- `uncheatable_bpb`, `table9_macro_bpb`: lower-is-better targets.
- `aggregate_kl_coefficient`, `phase_information_budget`, `anchor_id`, `direction_id`, `radius_fraction`, `seed_block`: proposal provenance when available.

## Exact Pair Geometry

For phase fractions \(\alpha_0\) and \(\alpha_1\),

\[
a=\alpha_0w^{(0)}+\alpha_1w^{(1)},\qquad d=w^{(0)}-w^{(1)}.
\]

The paired one-phase row uses \(w^{(0)}=w^{(1)}=a\). Its score subtracted from the corresponding two-phase score directly measures the observed phase-order effect at fixed aggregate exposure.

## Heldout Caveat

The heldout archive combines historical validation, model-proposed optima, adversarial stress panels, one-phase policies, exact-fiber populations, and repeat controls. It is deliberately heterogeneous. Pooled RMSE is insufficient; stratify by target, policy class, proposal series, candidate kind, and support.
