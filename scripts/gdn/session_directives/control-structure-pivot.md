# Session Directive: Full-Step Critical-Path Pivot

Current diagnosis:
- the hybrid-vs-attention gap is now the dominant fact:
  - hybrid is around `6.09 MFU` and `~166 ms`,
  - attention-only is around `21.09 MFU` and `~57.9 ms`,
  - the current tracked train-path budget does not explain most of that gap.
- current-boundary GDN train-path hillclimbing is therefore no longer the mainline.

Implications for this session:
- do not optimize toward smaller GDN closed-call buckets alone;
- optimize toward a shorter full-step critical path;
- prefer:
  - better remainder attribution,
  - model-boundary understanding,
  - or genuinely different decomposition boundaries.

Required questions to answer in the writeup:
1. What fraction of the hybrid-vs-attention gap is explained by the tracked train-path budget?
2. What are the largest top-k remainder categories outside the tracked train path?
3. Does this candidate shorten the full step, or only move cost between buckets?
4. If this changes outer control structure, why should it beat the current off-critical-path failure mode?
5. If this is still a same-boundary GDN move, why is it justified after the attention-only upper bound evidence?

Hard guardrail:
- If the result is another `train_path_budget_ms down, step_duration_ms flat/up` iteration, classify it as
  `off-critical-path` / `overlap-loss`, revert it, and pivot again.
- Treat same-boundary Macro O/M/N/R work as diagnostic unless it clearly improves `step_duration_ms`.
- Keep CE fixed unless the point of the run is an explicit CE side-arm.
