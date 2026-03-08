# Session Directive: Macro Move G (ExpDiff Outer Product)

## Objective
Eliminate Ct x Ct exponential-heavy work in train flash kernels by replacing
`exp(clip(g_i - g_j))` construction with a centered outer-product exp formulation.

## Required scope
Apply in all three train-path chunk kernels if feasible in one iteration:
- prepare
- recurrent forward
- backward

## Implementation constraints
1. Add helper `_exp_diff_and_mask_from_g(g, clip)`:
   - Fast path:
     - `center = 0.5 * (g_max + g_min)`
     - `er = exp(g - center)`, `ec = exp(center - g)` (vector exps only)
     - `exp_factor = er[:, None] * ec[None, :]`
     - `exp_diff = clip(exp_factor, exp(-clip), exp(clip))`
     - `mask = (exp_factor > exp(-clip)) & (exp_factor < exp(clip))`
   - Fallback path:
     - existing exact `diff/clip/exp` path for out-of-range cases.
2. Preserve model semantics and gradient correctness.
3. Keep output tape contracts stable unless explicitly justified in the log.

## Measurement requirements
- Compare before/after train-path hotspot times for:
  - forward closed-call shard_map/pallas_call
  - backward closed-call shard_map/pallas_call
- Report whether exponential op count dropped in custom-call IR or trace-derived op tables.

## Guardrails
- Do not ship approximation-only probe code as champion.
- If gain is <3% and hotspots are unchanged, revert and record why.
