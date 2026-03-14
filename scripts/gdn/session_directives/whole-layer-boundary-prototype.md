# Session Directive: W1 Prepared-Array Sharding Envelope

Goal:
- keep the best `D2` subgraph: the prepared-array leaf-call cut
- turn that cut into one true sharding envelope instead of merely reasserting sharding on arrays
- reduce matched xprof `dispatch_shard_shell_delta_ms` first and `xprof_idle_attributed_ms` second

Hard requirements for `W1`:
- reuse the prepared-array leaf-call cut from Iteration 109
- own one explicit sharding envelope at that boundary
- own one explicit layout contract inside the island
- keep AD/backward unchanged on the first `W1` pass
- avoid nested inner `shard_map` / `closed_call/shard_map` wrappers inside the island
- hoist or consolidate collectives to the envelope boundary when possible

Interpretation rules:
- summary-side shell wins are only a prefilter
- matched xprof dispatch and idle are the real confirmation gate
- `D2` already proved that smaller cuts alone are not enough if waiting/serialization does not improve

Reject `W1` if:
- `step_duration_ms` does not improve materially
- summary-side `dispatch_shard_shell_delta_ms` does not improve materially
- matched xprof `dispatch_shard_shell_delta_ms` does not improve materially
- `xprof_idle_attributed_ms` stays flat/up
- `interaction_remainder_ms` grows materially
- `ad_wrapper_shell_delta_ms` improves only by paying back the win as idle/waiting

Only after a positive `W1`:
- attempt `A2`, holding the `W1` forward/sharding envelope fixed while changing only AD ownership
