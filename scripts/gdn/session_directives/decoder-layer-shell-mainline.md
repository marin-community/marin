# Session Directive: Hybrid-Specific Generic Shell Delta Is The Mainline

Mainline budgets:
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `interaction_remainder_ms`
- `xprof_idle_attributed_ms` when available

Grouping rule:
- compare matched hybrid vs attention-only runs
- canonicalize by generic shell family, not full namespace
- subtract the attention-only control
- keep only positive hybrid-only deltas

Use coarse `decoder_layer_shell_budget_ms` only as an upper bound, not as the main promotion target.

Immediate implication:
- do not optimize the broad `HackableDecoderLayer/*` or `HackableDecoderBlock/*` shell directly,
- do not optimize another broad GDN-branch wrapper directly,
- do not treat summary-side shell wins as sufficient,
- optimize the prepared-array branch-core subgraph only when it is paired with a true sharding envelope and matched xprof confirmation.
