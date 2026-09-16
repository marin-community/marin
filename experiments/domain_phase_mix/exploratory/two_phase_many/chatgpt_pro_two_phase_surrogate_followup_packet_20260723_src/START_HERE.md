# Start Here

## Objective

Find a simple, principled parametric surrogate that predicts smooth BPB from a one-phase or two-phase 39-bucket Delphi 3e18 policy and supports reliable global two-phase policy selection. The practical failure remains unresolved: models often fit and rank ordinary policies while becoming optimistic outside support, compressing the response range, or returning a tied or inferior raw two-phase optimum.

## What Changed Since the Original Packet

- The append-only Delphi 3e18 development archive now contains every completed observation available at packet build time, with both Uncheatable and Table-9 BPB.
- Five independent ChatGPT Pro investigations have been consolidated.
- Exact aggregate-matched one-phase/two-phase fit pairs make the phase effect directly observable on 280 fibers.
- A two-parameter finite-potential transport law identifies a stable recency share near \(0.54\), but does not improve exact-fiber selection or produce a useful asymmetric raw optimum.
- A structural audit shows why a phase law constructed only from the aggregate potential \(F\) becomes tied near the constrained aggregate optimum.
- The next valid route must identify a phase-specific marginal-value state \(G(a)\) that is not merely \(\nabla F(a)\), or establish that the policy is only partially identified under the present design.

## Packet Layout

- `data/`: canonical fit and heldout tables, pair index, bucket catalog, and phase exposure multipliers.
- `evidence/cross_session_*`: the new synthesis, metrics, coefficients, exact-fiber tests, and raw optimum audit.
- `evidence/prior_sessions/`: selected terminal reports and registries from all five sessions.
- `evidence/prior_local_search/`: the earlier 41-route local search and frozen gate.
- `standalone_code/`: reference models and portable reproduction scripts.
- `protocol/`: frozen batch protocols used in the synthesis.
- `prompts/`: common task, independent assignments, and complete ready-to-send prompts.

## Data Boundary

All outcomes in this packet are exposed development evidence. Do not fit directly to heldout targets, candidate-series identity, or residual labels. Freeze a candidate's equation, fitting procedure, hyperparameters, and ablations before evaluating it on heldouts. Any surviving proposal needs a new untouched confirmation panel.
