# Session Directive: Attribution Is Diagnostic Only Unless Tooling Changed

- `S3` attribution is already established on the current harness.
- Use attribution-only work only when the iteration changes xprof extraction, matched-pair logic, or shell grouping.
- Do not spend another mainline pass on attribution refresh just to reconfirm the same shell ranking.
- When attribution runs are required, record:
  - `hybrid_generic_shell_delta_budget_ms`
  - `dispatch_shard_shell_delta_ms`
  - `ad_wrapper_shell_delta_ms`
  - `interaction_remainder_ms`
  - `xprof_idle_attributed_ms`
  - `hybrid_generic_shell_delta_topk`
