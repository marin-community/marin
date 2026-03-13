# Session Directive: Remainder Attribution Must Now Use A Hybrid-Specific Shell Delta

Current diagnosis:
- the broad decoder-layer shell family is informative but too coarse,
- the attention-only control still carries normal layer-body compute inside that family,
- the next actionable attribution target is a matched hybrid-vs-attention generic shell delta.

Mainline metrics:
- `upper_bound_gap_ms`
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `layout_shell_delta_ms`
- `residual_add_shell_delta_ms`
- `interaction_remainder_ms`
- `xprof_idle_attributed_ms`
- `train_path_budget_ms` (secondary diagnostic)

Policy:
1. Treat `dispatch_shard_shell_delta_ms` as the immediate mainline budget.
2. Treat `ad_wrapper_shell_delta_ms` as the second mainline budget.
3. Treat `hybrid_generic_shell_delta_budget_ms` and `interaction_remainder_ms` as first-class metrics.
3. Keep `decoder_layer_shell_budget_ms` only as a coarse upper bound.
4. Record `hybrid_generic_shell_delta_topk`, not just generic `remainder_topk`.
5. Record `xprof_idle_attributed_ms` whenever an XPlane pair is available.
6. Reject namespace-only bucket movement that leaves the generic shell delta or interaction remainder unchanged.
