# Session Directive: CE-Backward / Remainder-Budget Coverage Before More GDN Retuning

Goal: use iteration budget on the residual CE backward/control bottleneck and on measuring the off-train-path remainder, not more kernel-local wins that preserve the same step-critical-path ambiguity.

Coverage rule for this session:
- Before repeating a kernel-math move (`E`, `H`, `G`, `I`, `J`) in isolation, complete at least one **validated** attempt for each of:
  - `R` ejkernel-style minimal-tape backward recompute control arm,
  - `P` CE backward-mode A/B on the real train run,
  - `O` reduced-Pallas / XLA control arm benchmark,
  - `M` XLA-first outer train path with Pallas only as leaf kernels.
- `N` and `L` should only consume mainline budget when nested inside `M`/`O` or when fresh evidence says CE backward and remainder budget are no longer the dominant unknowns.

Selection order guidance:
1. Prefer `R` first now that the CE false wall is removed and the next high-value external idea is the ejkernel-style backward/tape tradeoff.
2. Keep `P` available to refresh CE evidence when attribution becomes unclear.
3. Use `O` as a diagnostic control arm, not a mainline promotion target.
4. Use `M` only after `R` or when you can explain why the ejkernel-style backward/tape diagnosis no longer applies.
5. Choose `N` or `L` only inside `M`/`O`/`R`, or when you can explain why the current CE backward diagnosis no longer applies.
6. Only choose `E/H/G/I/J` when they are embedded inside one of the moves above or when you can explain why the current CE/remainder diagnosis does not apply.

Repeat-avoidance rules:
- Do not pick the same macro move as the immediately previous validated attempt unless you are changing the outer train-path control structure materially.
- If a macro move regresses twice in a row because `while`/`conditional` grows or stays dominant, place it on cooldown.
- Do not keep retrying “train-path budget down, step not faster” variants as if they were near-wins. Classify them as `off-critical-path` / `overlap-loss`.
- Do not spend a fresh iteration on standalone `L` if CE backward mode has not yet been benchmarked and remainder budget is still unexplained.
- Do not spend a fresh iteration on tape-heavy variants if the point of the run is to validate the recompute-heavy backward tradeoff.

Writeup requirement:
- At the top of each iteration writeup, include:
  - `Coverage slot: <macro>`
  - `Why this attacks the train-path control bottleneck:`
  - `Hot-path scan/cond status:`
  - `Change class: CE backend | outer control structure | inner kernel math`
- In the perf section, always include:
  - `CE backend selected`
  - `CE bwd mode`
  - `CE-attributed while` when available
  - `Forward closed-call`
  - `Backward closed-call`
  - `while`
  - `conditional`
  - `Kernel budget`
  - `Control budget`
  - `Train-path budget`
  - `Step duration`
  - `Remainder budget`

Guardrails:
- Keep focus on the training chunk path.
- Avoid introducing new hot-path `lax.cond` / runtime dispatch unless the end-to-end gain case is overwhelming.
- Hold CE fixed across GDN experiments unless the point of the run is the CE matrix itself.
- Revert speculative code on failed/regressed attempts per governance policy.
