# Session Directive: Hybrid-Specific Generic Shell Delta Is The Mainline

Goal:
- stop treating the broad `HackableDecoderLayer/*` family as the actionable shell target,
- make the next iterations optimize against a namespace-invariant hybrid-specific shell delta budget,
- and treat `dispatch_shard_shell_delta_ms` as the immediate mainline budget inside that shell delta.

Mainline budget:
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `hybrid_generic_shell_delta_budget_ms`
- `interaction_remainder_ms`
- Prefer xprof-backed accounting for this budget whenever you have a matched hybrid vs attention-only XPlane pair.
- Use `gdnctl xprof-compare-runs` or `gdnctl xprof-compare`, then feed the JSON into `gdnctl summary-attribution --xprof-compare-json ...`.

Required shell sub-budgets:
- `dispatch_shard_shell_delta_ms`
- `ad_wrapper_shell_delta_ms`
- `layout_shell_delta_ms`
- `residual_add_shell_delta_ms`
- `interaction_remainder_ms`
- xprof-specific counterparts when available:
  - `xprof_dispatch_shard_shell_delta_ms`
  - `xprof_ad_wrapper_shell_delta_ms`
  - `xprof_layout_shell_delta_ms`
  - `xprof_residual_add_shell_delta_ms`
  - `xprof_idle_attributed_ms`

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
- and `dispatch_shard_shell_delta_ms` improves,
- and `ad_wrapper_shell_delta_ms` does not regress,
- and `xprof_idle_attributed_ms` does not regress when an XPlane pair is available,
- and `hybrid_generic_shell_delta_budget_ms` improves,
- and `interaction_remainder_ms` does not grow.
- classify `train_path_budget_ms down, hybrid_generic_shell_delta_budget_ms flat/up` as `namespace-only / renamed-bucket progress`.
- classify `hybrid_generic_shell_delta_budget_ms down, xprof_idle_attributed_ms flat/up` as `waiting/serialization still dominant`.
