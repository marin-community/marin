# Session Directive: Macro Coverage Before Next Pivot

Goal: produce strategy-level guidance by covering multiple macro directions, not repeated retries of one direction.

Coverage rule for this session:
- Before repeating a macro move, complete at least one **validated** attempt (TPU tests + profiled run) for each of:
  - `I` segmented forward prepare+recurrent fusion (with reusable heavy intermediates),
  - `J` chunk/segment sweep with a compact benchmark table,
  - `E` V-tiling/shared-K precompute direction,
  - `H` shared-RHS matmul batching,
  - `G` exp-diff reformulation (only if materially different from prior G attempts).

Selection order guidance:
1. Prefer `I`, then `J`, then `E`, then `H`.
2. Treat `G` as cooldown unless the implementation is materially different from prior G regressions.

Repeat-avoidance rules:
- Do not pick the same macro move as the immediately previous validated attempt.
- If a macro move is infra-blocked twice in a row, switch to a different macro move in the next iteration.
- If a macro move regresses twice in a row (<3% gain or negative MFU), defer it until all other macro moves above have one validated attempt.

Writeup requirement:
- At top of each iteration writeup, include:
  - `Coverage slot: <macro> (<n>/<5>)`
  - `Covered set so far: {...}`
- For `J`, include a table of attempted `(Ct, Seg)` combos with compile/run status and throughput metrics.

Guardrails:
- Keep focus on training chunk kernels.
- Keep test tolerances/semantics unchanged.
- Revert speculative code on failed/regressed attempts per governance policy.
