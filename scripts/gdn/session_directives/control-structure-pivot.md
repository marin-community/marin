# Session Directive: Train-Path Control-Structure Pivot

Current diagnosis:
- Recent variants repeatedly cut forward/backward `shard_map/pallas_call` times by ~40-52%.
- End-to-end MFU still regressed because a device-side `while` bucket around `31.5-31.7 ms` became dominant.
- Therefore the bottleneck is now the lowered train-path control structure around the kernels, not just chunk-local math.
- After Iterations 64-66, the remaining `while` is likely mostly fused CE on XLA rather than GDN-train-shell work.

Implications for this session:
- Closed-call reduction is not a success criterion by itself.
- CE backend selection is now a first-class optimization lever, not background infra.
- The candidate must either:
  - change CE backend / CE-attributed control cost,
  - reduce/remove hot-path scan/control-flow overhead, or
  - move train-path orchestration to a different layer where that overhead does not dominate.

Required questions to answer in the writeup:
1. Where is the hot-path `while` / `conditional` coming from in this design?
2. Does this candidate add or preserve a hot-path `lax.scan`?
3. Does it add a hot-path `lax.cond` / runtime branch?
4. Why should that not lower to the same losing `WhileOp` / `Conditional` pattern?
5. Is the residual `while` still CE-attributed in this design?

Hard guardrail:
- If the result is another “closed-call down, `while` up” iteration, revert it and pivot again.
- If CE backend remains `xla` and residual `while` stays large, prioritize CE/backend, Macro O, or Macro M next.
