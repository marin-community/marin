# Session Directive: CE-First / Control-Structure Coverage Before More Kernel-Math Retries

Goal: use iteration budget on the residual CE/control bottleneck, not more kernel-local wins that preserve the same `while` / `conditional` structure.

Coverage rule for this session:
- Before repeating a kernel-math move (`E`, `H`, `G`, `I`, `J`) in isolation, complete at least one **validated** attempt for each of:
  - `P` CE backend forcing / A-B benchmark on the real train run,
  - `O` reduced-Pallas / XLA control arm benchmark,
  - `M` XLA-first outer train path with Pallas only as leaf kernels.
- `N` and `L` should only consume mainline budget when nested inside `M`/`O` or when fresh CE evidence says GDN is dominant again.

Selection order guidance:
1. Prefer `P`, then `O`, then `M`.
2. Choose `N` only inside `M`/`O` or when CE is no longer the dominant unresolved while source.
3. Choose `L` only when paired with `M`/`O` or when you can explain why standalone associative-summary work is no longer low leverage.
4. Only choose `E/H/G/I/J` when they are embedded inside one of the moves above or when you can explain why the current CE/control diagnosis does not apply.

Repeat-avoidance rules:
- Do not pick the same macro move as the immediately previous validated attempt unless you are changing the outer train-path control structure materially.
- If a macro move regresses twice in a row because `while`/`conditional` grows or stays dominant, place it on cooldown.
- Do not keep retrying “closed-call down, `while` up” variants. That pattern is already established.
- Do not spend a fresh iteration on standalone `L` if CE is still selected as `xla` and residual `while` is still large.

Writeup requirement:
- At the top of each iteration writeup, include:
  - `Coverage slot: <macro>`
  - `Why this attacks the train-path control bottleneck:`
  - `Hot-path scan/cond status:`
  - `Change class: CE backend | outer control structure | inner kernel math`
- In the perf section, always include:
  - `CE backend selected`
  - `CE-attributed while` when available
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
