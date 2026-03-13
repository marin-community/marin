# Session Directive: G1 GDN-Branch Primitive Prototype

Goal:
- optimize only the hybrid-specific GDN branch inside a GDN-bearing decoder layer
- reduce `dispatch_shard_shell_delta_ms` first and `ad_wrapper_shell_delta_ms` second

Hard requirements:
- own the forward boundary
- own the backward/custom-VJP contract
- own the sharding contract
- own the layout contract

Preferred first prototype:
- input is normalized hidden state plus mask
- output is the GDN branch contribution in model space
- reuse existing GDN leaf kernels initially
- manual/custom VJP at the branch boundary
- one explicit branch-local sharding contract instead of nested outer wrappers
- do not swallow generic MLP/residual shell in the first prototype

Reject the prototype if:
- `step_duration_ms` does not improve
- `dispatch_shard_shell_delta_ms` stays flat/up
- `ad_wrapper_shell_delta_ms` grows
- `interaction_remainder_ms` grows
- `xprof_idle_attributed_ms` stays flat/up when available
