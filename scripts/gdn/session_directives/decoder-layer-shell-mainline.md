# Session Directive: Hybrid-Specific Generic Shell Delta Is The Mainline

Goal:
- stop treating the broad `HackableDecoderLayer/*` family as the actionable shell target,
- make the next iterations optimize against a namespace-invariant hybrid-specific shell delta budget.

Mainline budget:
- `hybrid_generic_shell_delta_budget_ms`

Required shell sub-budgets:
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `layout_shell_delta_ms`
- `residual_add_shell_delta_ms`
- `interaction_remainder_ms`

Grouping rule:
- compute the shell target from a matched hybrid vs attention-only pair,
- canonicalize buckets by generic family rather than full namespace,
- subtract the attention-only control contribution,
- keep only positive hybrid-only deltas.

What this replaces:
- `decoder_layer_shell_budget_ms` remains useful as a coarse upper bound,
- but it is no longer the mainline promotion target because it over-charges normal attention/MLP body compute.

Promotion rule:
- a candidate is not promotable unless `step_duration_ms` improves,
- and `hybrid_generic_shell_delta_budget_ms` improves,
- and `interaction_remainder_ms` does not grow.
- classify `train_path_budget_ms down, hybrid_generic_shell_delta_budget_ms flat/up` as `namespace-only / renamed-bucket progress`.
