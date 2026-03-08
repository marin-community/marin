# Session Directive: Train-Path Control-Structure Pivot

Current diagnosis:
- Pre-CE, recent variants repeatedly cut forward/backward `shard_map/pallas_call` times by ~40-52% while the step still regressed.
- Iteration 67 removed the giant CE/XLA false wall.
- Iteration 68 then showed train-path budget can still drop while step duration gets worse.
- Therefore the bottleneck is no longer "just reduce GDN closed-call time"; it is the full step critical path, including CE backward/control and the post-train-path remainder.

Implications for this session:
- Closed-call reduction is not a success criterion by itself.
- CE backward mode and step remainder are now first-class optimization levers, not background infra.
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
6. What happens to `remainder_budget_ms` in this design?

Hard guardrail:
- If the result is another “train-path budget down, step not faster” iteration, classify it as `off-critical-path` / `overlap-loss`, revert it, and pivot again.
- If CE backend remains `xla` and residual `while` stays large, prioritize CE/backend next.
- Do not use Macro O or Macro M as mainline promotion candidates until CE backward mode has been explicitly benchmarked under forced `pallas_tpu` CE.
