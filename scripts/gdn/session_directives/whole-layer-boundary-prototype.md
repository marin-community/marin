# Session Directive: P3 Fixed-4-Layer Block Prototype

Goal:
- optimize the fixed `3 GDN + 1 attention` block as one unit
- reduce `dispatch_shard_shell_delta_ms` first and `ad_wrapper_shell_delta_ms` second

Hard requirements:
- own the forward boundary
- own the backward/custom-VJP contract
- own the sharding contract
- own the layout contract

Preferred first prototype:
- XLA-visible shell
- existing leaf kernels reused initially
- manual/custom VJP at the block boundary
- explicit sharding instead of generic nested wrappers

Reject the prototype if:
- `step_duration_ms` does not improve
- `dispatch_shard_shell_delta_ms` stays flat/up
- `ad_wrapper_shell_delta_ms` grows
- `interaction_remainder_ms` grows
- `xprof_idle_attributed_ms` stays flat/up when available
