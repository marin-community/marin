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

### 2026-07-26 15:18 PDT - packed CuTe attention fails gradients and route stability
- Hypothesis: After splitting before all MoE-side preparation, one packed CuTe FA4 attention call will remain within relative-L2 `0.002` for outputs and every parameter/input gradient and will preserve the ordered selected experts.
- Commit Hash: `a5c58171ff` moves the split immediately after packed attention and before each complete `moe_residual`. Local full-stage relative-L2 is `9.2146e-08` for both `recompute_all` and `save_moe`; joined MoE VJP and post-attention preparation are exact. The same commit adds a target H100x8 packed-versus-ordered attention diagnostic.
- Command: Parent `/dlwh/jaxpp-group2-attention-parity-a5c58171-20260726-221215` ran `--diagnostic attention` from a clean `a5c58171ff` workspace. Child: `/dlwh/jaxpp-group2-attention-parity-a5c58171-20260726-221215/0`.
- Results:
  - Ordered attention lower/compile/execute took `20.556/4.670/1.887s`; packed attention took `9.696/3.352/0.053s`. Route preparation completed in approximately `1.2s` total.
  - Attention outputs narrowly passed. The maximum was `values['outputs'][0]`: relative-L2 `0.0019524225`, actual/reference norms `412.0878296/412.0885315`, norm ratio `0.9999982967`, cosine `0.9999980874`, absolute-L2 `0.8045709`, and maximum absolute error `0.001953125`.
  - Attention parameter gradients failed. `attn_gated_norm.w_down` was worst at relative-L2 `0.0036578454`, actual/reference norms `17.8711338/17.8710709`, norm ratio `1.0000035220`, cosine `0.9999932585`, absolute-L2 `0.0653696`, and maximum absolute error `0.0009765625`. Eight used attention parameter leaves exceeded `0.002`.
  - Both input cotangents passed; maximum relative-L2 was `0.0009632341`.
  - Post-attention MoE preparation failed at relative-L2 `0.0021891496` on `mlp_inputs[0]`. The packed and ordered outputs selected different experts for `10,429` assignments across `6,362` tokens. Minimum top-k margins were ordered `0.0` and packed `1.7881393e-07`.
  - The child exited `1` after `passed=false`. Iris is terminal with no live resource; no full-stage or L24 job launched.
- Interpretation:
  - Packing CuTe attention changes BF16 outputs enough to cross many near-tied top-k boundaries even though the output norm error narrowly passes. Its parameter VJP independently exceeds the fixed gate.
  - The exact-ring pair remains viable, but an acceptable formulation must execute attention independently per original microbatch and join only the exact-ring MoE work. Packed attention is closed under the current numerical policy.
- Next action:
  - Test two independent CuTe attention calls inside one reverse pass against two separately differentiated calls. If exact, design block-level paired execution that preserves independent attention and dense gradients while joining only the two exact-ring calls, avoiding the whole-stage tuple compile cliff.
