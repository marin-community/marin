# Session Directive: Model-Boundary Sweep Before More Kernel Hillclimbing

Goal:
- measure the throughput cost of GDN layer fraction directly,
- decide whether the mainline problem is the current GDN implementation boundary or the cost of using
  many GDN layers at all,
- avoid spending more budget on kernel tweaks before quantifying the model-level tradeoff.

Required sweep:
- keep `gdn_block_size = 4`,
- run with `gdn_layers_per_block` in `{0, 1, 2, 3}`,
- keep CE fixed at:
  - `LEVANTER_FUSED_CE_IMPLEMENTATION=pallas_tpu`
  - CE backward mode `pallas`
- keep the same benchmark family, TPU family, steps, and batch shape.

Required outputs:
- `gdn_layer_fraction`
- `throughput/mfu`
- `throughput/tokens_per_second`
- `step_duration_ms`
- `train_path_budget_ms`
- `remainder_budget_ms`
- `upper_bound_gap_ms`
- per-fraction delta vs the `0/4` attention-only control

Interpretation rule:
- if throughput degrades roughly monotonically with GDN fraction, treat that as evidence that the
  model/product boundary is a stronger lever than another same-boundary GDN kernel tweak.
- if one fraction shows a favorable tradeoff, record it as a product-side option, not a kernel win.

Guardrails:
- this is a measurement iteration first; do not combine it with new GDN kernel changes.
- if you must change code, restrict changes to benchmark/config plumbing or attribution support.
