# Session Directive: Whole-Layer Boundary Pivot

Current diagnosis:
- the unresolved hybrid-vs-attention gap is now dominated by GDN-bearing decoder-layer shell cost,
  not just the already-tracked GDN core or CE shell.
- the dominant unexplained buckets live under `HackableDecoderLayer/*`,
  `jvp(HackableTransformer)/HackableDecoderLayer/*`, and
  `transpose(jvp(HackableTransformer))/HackableDecoderLayer/*`.
- current-boundary GDN train-path hillclimbing is therefore the wrong mainline boundary.

Implications for this session:
- do not optimize toward smaller GDN closed-call buckets alone;
- optimize toward a shorter full-step critical path at the decoder-layer boundary;
- prefer:
  - widened decoder-layer-shell attribution,
  - whole-layer design/skeleton work,
  - or one serious whole-layer prototype.

Required questions to answer in the writeup:
1. What fraction of the hybrid-vs-attention gap is explained by the whole decoder-layer shell?
2. Which shell sub-budgets dominate: AD, sharding, layout, residual/add, or something else?
3. Does this candidate shorten the full step, or only move cost between train-path and shell buckets?
4. If this changes outer control structure, why should it beat the existing `HackableDecoderLayer/*` shell tax?
5. If this is still a same-boundary GDN move, why is it justified after the whole-layer-shell evidence?

Hard guardrail:
- If the result is another `train_path_budget_ms down, decoder_layer_shell_budget_ms flat/up, step_duration_ms flat/up`
  iteration, classify it as `wrong-boundary progress`, revert it, and pivot again.
- Treat same-boundary Macro O/M/N/R work as diagnostic unless it clearly improves `step_duration_ms`
  and reduces `decoder_layer_shell_budget_ms`.
- Keep CE fixed unless the point of the run is an explicit CE side-arm.
