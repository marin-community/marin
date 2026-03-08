# Session Directive: Control-Structure Coverage Before More Kernel-Math Retries

Goal: use iteration budget on the train-path control-shell bottleneck, not more kernel-local wins that preserve the same `while` / `conditional` structure.

Coverage rule for this session:
- Before repeating a kernel-math move (`E`, `H`, `G`, `I`, `J`) in isolation, complete at least one **validated** attempt for each of:
  - `L` associative chunk summaries / affine scan reformulation,
  - `M` XLA-first outer train path with Pallas only as leaf kernels,
  - `N` backward tape-contract redesign tied to a real control-structure change,
  - `O` reduced-Pallas / XLA control arm benchmark.

Selection order guidance:
1. Prefer `L`, then `M`, then `N`, then `O`.
2. Only choose `E/H/G/I/J` when they are embedded inside one of the moves above or when you can explain why the current control-flow diagnosis does not apply.

Repeat-avoidance rules:
- Do not pick the same macro move as the immediately previous validated attempt unless you are changing the outer train-path control structure materially.
- If a macro move regresses twice in a row because `while`/`conditional` grows or stays dominant, place it on cooldown.
- Do not keep retrying “closed-call down, `while` up” variants. That pattern is already established.

Writeup requirement:
- At the top of each iteration writeup, include:
  - `Coverage slot: <macro>`
  - `Why this attacks the train-path control bottleneck:`
  - `Hot-path scan/cond status:`
- In the perf section, always include:
  - `Forward closed-call`
  - `Backward closed-call`
  - `while`
  - `conditional`
  - `Kernel budget`
  - `Control budget`
  - `Train-path budget`

Guardrails:
- Keep focus on the training chunk path.
- Avoid introducing new hot-path `lax.cond` / runtime dispatch unless the end-to-end gain case is overwhelming.
- Revert speculative code on failed/regressed attempts per governance policy.
