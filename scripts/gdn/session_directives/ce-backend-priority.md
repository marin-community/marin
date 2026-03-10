# Session Directive: CE Is Now A Bounded Side-Arm

Current diagnosis:
- forcing TPU fused CE to `pallas_tpu` was the last giant false wall and produced the largest recent win,
- the current validated regime already uses `pallas_tpu` CE with `pallas` backward,
- the remaining CE-attributed `while` is still worth measuring, but it is no longer the mainline explanation
  for the hybrid-vs-attention gap.

Policy:
- For non-CE experiments, hold CE fixed at:
  - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
  - CE backward mode `pallas`
- Do not let CE settings drift while comparing GDN or model-boundary experiments.
- Treat CE work as a bounded side-arm:
  - useful for cleanup or diagnostic clarity,
  - not the mainline optimization target.

When CE work is justified:
- residual `while` attribution becomes ambiguous again,
- you have a specific CE backward-shell hypothesis,
- or you are evaluating a narrow CE diff with a strict end-to-end promotion bar.

Required recording:
- `CE backend selected: ...`
- `CE bwd mode: pallas | xla_streaming`
- `CE-attributed while: ... ms -> ... ms` when trace attribution is available

Promotion bar for CE-only work:
- accept only if `step_duration_ms` improves materially,
- and either `ce_attributed_while_ms` drops clearly or `remainder_budget_ms` does not regress.
