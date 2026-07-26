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

### 2026-07-26 15:32 PDT - joined independent CuTe calls pass the fixed gate
- Hypothesis: Two original-batch CuTe FA4 attention calls can coexist inside one reverse pass without the packed-batch numerical changes, clearing the kernel premise for block-level paired execution.
- Commit Hash: `ab825ed5d2` adds a target H100x8 `attention-pair` diagnostic. Both arms use two independent batch-32 attention calls with identical masks, PKO/rope settings, and cotangents; the joined arm differentiates both calls in one reverse pass while the reference differentiates each call separately.
- Command: Parent `/dlwh/jaxpp-group2-attention-pair-ab825ed5-20260726-222830` ran from clean commit `ab825ed5d2`. Child: `/dlwh/jaxpp-group2-attention-pair-ab825ed5-20260726-222830/0`.
- Results:
  - Ordered lower/compile/execute took `20.569/4.555/1.891s`; joined took `19.326/1.580/0.048s`.
  - Both attention outputs and the projection loss were bitwise exact.
  - All eight used attention parameter gradients and both input gradients passed. The maximum was `attn_gated_norm.w_down`: relative-L2 `0.000414435`, actual/reference norms `17.871082/17.871096`, norm ratio `0.99999925`, cosine `0.99999991`, absolute-L2 `0.007406`, and maximum absolute error `0.000488`.
  - Attention matrix gradients were relative-L2 `0-0.00004142`; RMS attention was `0.0001110`; input cotangents were `0.00001088` and `0.00000990`.
  - Iris succeeded in `97s`; the task exited `0` and no resource remains live.
- Interpretation:
  - Separate CuTe call boundaries preserve forward values and keep joined gradient accumulation comfortably within the fixed gate. Combined with the bitwise-exact joined learned-router ring result, block/component-level pairing is numerically viable.
  - The bounded architecture is three forward components per block: two independent non-MoE pre-tasks and one joined `MoEMLP` task. Backward replays these components, uses one joined MoE VJP, independently differentiates the two non-MoE paths, and assembles a stage-shaped gradient. Only completed stage tuples cross ranks.
  - The main remaining risk is operational: approximately `9L + 1` stage-local tasks per paired backward for `L` blocks may add substantial JaxPP task/launch overhead.
- Next action:
  - Implement pure one-block component forward/VJP helpers and prove full block value/QB/every-gradient parity locally. Then add a one-block JaxPP component compile gate before integrating the full standard-1F1B scheduler.

### 2026-07-26 16:07 PDT - exact paired MoE block components pass local parity
- Hypothesis: A block can preserve the original per-microbatch BF16 attention, normalization, shared-expert, and residual paths while joining only two learned-router exact-ring `MoEMLP` calls, with loss and every gradient leaf at relative-L2 `<=0.002`.
- Commit Hash: `0adaf6156d` adds transient attention and dense-side block views, paired MoE forward/VJP helpers, paired block forward/VJP composition, and master-parameter/QB adapters. The transient views do not alter model or checkpoint serialization.
- Command: `uv run pytest -q tests/test_grug_moe_explicit_stage_task_grouping.py`; `./infra/pre-commit.py --changed-files --fix`.
- Results:
  - All `22` focused tests passed in `17.76s`; changed-files precommit, including Pyrefly, passed.
  - Both `recompute_all` and `save_moe` matched two separately differentiated ordered block calls for outputs, projection losses, router metrics, QB values, routing counts, block-parameter gradients, and both input gradients.
  - The worst reported input-gradient relative-L2 was `4.4402259e-08` with maximum absolute error `2.9960808e-07`, far below the explicit `0.002` ceiling.
  - Structural Jaxpr inspection found exactly two router einsums in the joined MoE component and no attention operation.
- Interpretation:
  - The component decomposition clears the local numerical gate without packing attention or dense-side work. BF16 addition order remains routed-plus-shared followed by the residual.
  - This is not yet a JaxPP compiler or performance result. The remaining risk is whether the finer-grained component task graph compiles and executes under MPMD without prohibitive task overhead.
- Next action:
  - Run a one-block target-shape JaxPP H100 compile/parity gate on `cw-rno2a`. Require loss and every gradient leaf relative-L2 `<=0.002`; do not launch L8 or L24 until it passes.
