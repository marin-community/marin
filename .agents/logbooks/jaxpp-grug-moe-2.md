---
topic: jaxpp-grug-moe
description: Continuation of the JaxPP pipeline parallelism research logbook
author: dlwh
---

# JaxPP Grug MoE: Task Logbook, Part 2

Continues [jaxpp-grug-moe.md](jaxpp-grug-moe.md).

### 2026-07-26 15:01 PDT - GPU isolation clears ring VJP and localizes packed dense backward
- Hypothesis: The full-stage order-one gradient failure originates either in composing two learned-router exact-ring calls in one reverse pass or in the packed MoE-side dense preparation before the two ring calls.
- Commit Hash: `36ce41274a` adds target-shape MoE-pair diagnostics, exact selected-route/margin reporting, and per-leaf actual/reference norms, norm ratios, and cosine similarity. CPU/reference controls pass the fixed `0.002` gate for joined learned-router MoE VJP, checkpoint modes, packed reference attention, and the unequal-weight final head. A separate CPU BF16 full-stage comparison reaches `0.00251` and remains non-promotable under the unchanged policy.
- Command: Parent `/dlwh/jaxpp-group2-moe-pair-diagnostic-r2-20260726-1457` ran `--diagnostic moe-pair` on one H100x8 from clean commit `36ce41274a`. Child: `/dlwh/jaxpp-group2-moe-pair-diagnostic-r2-20260726-1457/0`.
- Results:
  - Ordered and packed RMS/GatedNorm preparation matched exactly. Selected-expert mismatch counts were zero for both assignments and tokens; ordered and grouped minimum top-k boundary margins were both `2.3841858e-07`.
  - Two learned-router exact-ring calls in one `value_and_grad` matched two separately differentiated calls bitwise for values and all gradients. A representative input gradient norm was `626.3990` in both arms with relative-L2 `0`, norm ratio `1`, and cosine `1`.
  - Production grouped-boundary values passed. The largest value relative-L2 was `5.15977e-05` with cosine `1`.
  - Production grouped-boundary gradients failed first in the packed dense path. `mlp_gated_norm.w_up` was relative-L2 `0.003580288`, actual/reference norms `21.268112/21.267874`, norm ratio `1.00001121`, cosine `0.999993702`, absolute-L2 `0.0761451`, and maximum absolute error `0.0009765625`.
  - `mlp_gated_norm.w_down` was `0.00357906`; shared-expert weights were `0.0035712-0.0035760`; `rms_mlp.weight` was `0.00287548`. Routed expert gradients passed at `0.000218-0.000273`.
  - All phases completed in under `6s`. The child exited `1` after the expected `passed=false`; Iris is terminal and no live allocation remains.
- Interpretation:
  - Learned routing and paired exact-ring custom VJP composition are cleared. The first failing boundary is reduction order from running MoE-side RMSNorm, MLP GatedNorm, and shared-expert dense operations on the packed batch.
  - The correctness-preserving formulation should keep attention packed but split immediately afterward, before every MoE-side dense operation and routing call. Packed CuTe attention remains the next unresolved GPU boundary.
- Next action:
  - Move the split ahead of `moe_residual`, rerun local parity, then run a target H100x8 packed-CuTe-attention value/VJP and route-stability diagnostic. Only a passing attention arm justifies another full-stage parity gate.
