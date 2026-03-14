# Session Directive: D2 Branch-Core Sharding Diagnostic

Goal:
- reduce `dispatch_shard_shell_delta_ms` first
- keep `ad_wrapper_shell_delta_ms` from regressing
- avoid another broad branch wrapper that merely renames shell tax

Hard requirements for `D2`:
- use a smaller cut than the rejected broad `G1` wrappers
- own the branch-core sharding contract
- own the branch-core layout contract
- reuse the current GDN leaf-kernel island or the smallest deterministic subgraph around it
- do not add a new custom VJP at this stage
- carry forward head-first layout discipline where it fits naturally

Interpretation rules:
- `D1` was a useful diagnostic lead, not a promotable win
- broad branch wrappers are already rejected
- the next cut must attack sharding ownership first, not backward ownership first

Reject `D2` if:
- `step_duration_ms` does not improve
- `dispatch_shard_shell_delta_ms` stays flat/up
- `interaction_remainder_ms` grows
- `xprof_idle_attributed_ms` stays flat/up when available
- `ad_wrapper_shell_delta_ms` improves but dispatch/shard does not

Only after a positive `D2`:
- attempt `A2`, holding the `D2` forward/sharding cut fixed while changing only AD ownership
