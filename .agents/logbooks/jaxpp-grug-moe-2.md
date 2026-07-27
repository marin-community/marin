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

### 2026-07-26 16:15 PDT - one-task JaxPP component gate compiles but fails target parity
- Hypothesis: The locally exact paired block composition will retain loss, routing, QB/router metrics, every block-parameter gradient, and both input gradients within relative-L2 `0.002` when compiled as one target-shape JaxPP MPMD task. This r3 harness compares JaxPP-paired directly with ordered VJPs; it does not include a target-H100 direct-paired executable and therefore cannot attribute a failure specifically to JaxPP.
- Commit Hash:
  - `54f3265262` adds the one-block d2560/e64/top-k4/sequence-4096/global-microbatch-32/local-4 H100x8 gate.
  - `695f389d9c` separates the pure paired-MoE structural contract from the expanded full-VJP task Jaxpr after r2 showed that reverse-mode and `save_moe` rematerialization expand the full task to four ring bodies and ten router einsums.
- Command: The authoritative r3 parent `/dlwh/jaxpp-group2-component-mpmd-r3-695f389d-20260726-231209` cloned and verified clean commit `695f389d9c52bcadf1ffa6adc1a50ec9db95db61`, installed patched JaxPP `7091a9b5`, and ran:
  ```bash
  TF_GPU_ALLOCATOR=cuda_malloc_async \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
  JAX_COMPILATION_CACHE_DIR=/mnt/local/iris-cache/jaxpp-group2-component \
  RAGGED_DOT_IMPL=triton \
  HALIAX_RAGGED_DOT_TRITON_BLOCK_K=32 \
  HALIAX_RAGGED_DOT_TRITON_NUM_WARPS=8 \
  python -u experiments/grug/moe/check_jaxpp_group2_component_mpmd_parity.py
  ```
  Child: `/dlwh/jaxpp-group2-component-mpmd-r3-695f389d-20260726-231209/0`. The initial parent `/dlwh/jaxpp-group2-component-mpmd-54f32652-20260726-230235` was setup-invalid because the Iris bundle at `/app` had no `.git`; r2 `/dlwh/jaxpp-group2-component-mpmd-r2-54f32652-20260726-230737` reached lowering and exposed only the over-strict full-VJP structural assertion.
- Results:
  - Provenance was `base_commit=695f389d9c52bcadf1ffa6adc1a50ec9db95db61`, `dirty=false`. The one-block parameter tree retained `19` leaves with signature `494b2a22ded3dad586e5155ce133c69f716d4695b78f3f4ae4fb4eb7971e63b1`.
  - The target-shape pure paired-MoE trace contained exactly two ring `shard_map` bodies, two router einsums, and zero attention equations under `_paired_moe_calls`. The complete VJP task retained the observed expanded counts: four ring bodies, ten router einsums, `338` joined-boundary equations, and zero attention equations inside the boundary.
  - Ordered lower/compile/execute took `21.092/7.267/0.719s`. JaxPP lower/precompile/execute took `0.799/23.932/0.089s`; precompile reported exactly one task and zero receive buffers.
  - All reported values were finite and the summed projection loss passed at relative-L2 `0.000837765`. The individual microbatch losses failed at `0.0131914` and `0.00347043`.
  - Both block outputs failed at relative-L2 `0.0205638` and `0.0202413`. Exact routing-count comparison found `59` and `61` mismatched entries. QB relative-L2 was `0.00343787` and `0.00284331`; router z-loss, entropy, load-balancing loss, and capacity-overflow metrics passed the norm gate.
  - `20/21` gradient leaves failed. The worst was `mlp.router` at relative-L2 `1.3889344`, actual/reference norms `54.19565/54.15769`, cosine `0.03611`; routed expert gradients were `1.38785-1.38837`. Input gradients were `0.66798/0.66936`; attention parameter leaves were `0.53762-0.64363`; shared-expert leaves were `0.00532-0.00580`. Only the overwritten `mlp.router_bias` gradient was exact.
  - The child exited `1` on the unchanged numerical assertion. Iris was terminal after `2m04.87s`; no retry or live allocation remained.
- Interpretation:
  - The component task graph is compile-compact and operational: target lowering, one-task precompile, and execution complete quickly. The full-task structural count expansion was harness-only and is now recorded instead of rejected.
  - The composed target path is a hard numerical negative, but r3 does not isolate its source. The divergence may already occur in a target-H100 direct `jax.jit(paired_block_value_and_grads)` arm, or it may be introduced by JaxPP task compilation/execution. CPU tiny parity and the separate target attention/ring controls do not prove the complete target direct-paired composition.
- Next action:
  - Add target-H100 direct paired lower/compile/execute and report direct-paired versus ordered, JaxPP-paired versus direct-paired, and JaxPP-paired versus ordered. Only attribute the failure to JaxPP if direct-paired passes. Keep exact routing equality and the per-leaf `0.002` policy unchanged; do not launch L8 or L24.

### 2026-07-26 16:22 PDT - direct control clears the one-task JaxPP boundary
- Hypothesis: The r3 failure is introduced by JaxPP rather than the complete target-H100 direct paired composition.
- Commit Hash: `1904bf52b5` adds a direct `jax.jit(paired_block_value_and_grads)` arm and separates direct-paired-versus-ordered from JaxPP-paired-versus-direct-paired reporting.
- Command: Parent `/dlwh/jaxpp-group2-component-mpmd-r4-1904bf52-20260726-231852` cloned clean commit `1904bf52b54ff72d68b71412bac7ffc69e847d75`; child `/dlwh/jaxpp-group2-component-mpmd-r4-1904bf52-20260726-231852/0`.
- Results:
  - Direct paired versus ordered failed before JaxPP lowering and reproduced r3: output relative-L2 `0.0205638/0.0202413`, routing mismatches `59/61`, and worst gradient `mlp.router` at relative-L2 `1.3889344`.
  - JaxPP paired versus direct paired passed. Both outputs, both per-microbatch losses, all router metrics, and routing counts were exact. All `21` gradient leaves passed; the maximum was `attn_gated_norm.w_down` at relative-L2 `0.000521081`, actual/reference norms `24.449770/24.449783`, cosine `0.999999876`, absolute-L2 `0.0127403`, and maximum absolute error `0.0009765625`.
  - Ordered lower/compile/execute took `20.913/7.114/0.724s`; direct paired took `20.444/3.670/0.086s`; JaxPP lower/precompile/execute took `0.005/4.010/0.087s`. The JaxPP execution retained one task and zero receive buffers.
  - The pure paired-MoE trace retained two ring bodies, two router calls, and no attention. The full VJP trace retained four ring bodies, ten router calls, and no attention inside the joined boundary.
  - The strict assertion made the child terminal after `2m09s`; no retry or live allocation remains.
- Interpretation:
  - JaxPP is cleared for this one-task boundary. It preserves the direct paired executable within the fixed `0.002` gate and preserves exact outputs and routes.
  - The hard failure belongs to the complete target-H100 direct hand-assembled VJP composition. Separate target attention and MoE controls are insufficient to validate their manual cotangent composition.
- Next action:
  - Compare ordered execution with one monolithic ordinary `value_and_grad` around the existing paired forward. If this passes, use the monolithic VAG as the block task and recheck it once under JaxPP; only add direct boundary probes if the monolithic control also fails.

### 2026-07-26 16:28 PDT - strict Sonic/QuACK target-shape gate is a hard negative
- Hypothesis: Sonic MoE through QuACK `0.5.0` will satisfy the unchanged relative-L2 `0.002` gate at the exact 16,384-token, 64-expert, top-k-4 target shape, justifying an H100x8 FSDP A/B.
- Commit Hash: `1cbfbf9bd1` adds the strict target-shape numerical gate.
- Command: Parent `/dlwh/sonic-quack-strict-h100-1cbfbf9b-20260726-162347` ran from clean commit `1cbfbf9bd1` on one H100 with 65,536 balanced assignments.
- Results:
  - Loss passed at relative-L2 `0.0002104686`.
  - The output diagnostic failed at relative-L2 `0.0060693517`.
  - Every gradient class failed: tokens `1.2829373`, routing weights `0.0029550040`, `w13` `1.4123952`, and `w2` `0.0069705437`.
  - Timings were correctly skipped after the numerical failure. Iris exited `1` as expected and is terminal with no live allocation.
- Interpretation:
  - This is a strict numerical hard negative under the accepted policy. The passing scalar loss does not offset output and gradient failures.
- Next action:
  - Do not launch the H100x8 FSDP A/B and do not change the tolerance.

### 2026-07-26 16:35 PDT - monolithic paired VAG reproduces the direct failure
- Hypothesis: Replacing the hand-assembled component VJP chain with one ordinary `value_and_grad` around the complete paired forward will preserve ordered target-H100 numerics.
- Commit Hash: `e3227ba4f5` adds the monolithic paired VAG and a direct-first gate that skips JaxPP unless the direct comparison passes.
- Command: Parent `/dlwh/jaxpp-group2-component-monolithic-r5-e3227ba4-20260726-1640` cloned clean commit `e3227ba4f5bdace82314ce5a441d90914a83d0eb`; child `/dlwh/jaxpp-group2-component-monolithic-r5-e3227ba4-20260726-1640/0`.
- Results:
  - The monolithic direct arm reproduced r4 exactly. Output relative-L2 was `0.0205638/0.0202413`, routing-count mismatches were `59/61`, and the worst gradient was `mlp.router` at `1.3889344`.
  - The summed projected loss still passed at `0.000837765`; individual losses failed at `0.0131914/0.00347043`. QB failed at `0.00343787/0.00284331`.
  - Ordered lower/compile/execute took `21.028/7.485/0.759s`; monolithic paired took `20.192/3.936/0.089s`.
  - JaxPP was correctly skipped. The strict assertion made the child terminal after `2m06s`; no retry or live allocation remains.
- Interpretation:
  - Hand-assembled cotangent propagation is not the cause. Both automatic monolithic reverse mode and the manual VJP chain produce the same wrong paired forward identity at target shape.
  - The `59/61` routing-count mismatches and near-orthogonal routed-expert gradients make cross-matched collective calls a ranked possibility. The current paired MoE path also differs from the previously passing joined control by enclosing both ring calls in one checkpoint.
- Next action:
  - On identical prepared MLP inputs, compare the current joint checkpoint with no encompassing checkpoint and one `save_moe` checkpoint per call. Report pre-ring routes, same-index and cross-index post-ring outputs/statistics, routing-count mismatches, and local VJPs before changing ring channels or tuple order.

### 2026-07-26 16:44 PDT - MoE call order and checkpoint scope are not the full-block failure
- Hypothesis: The target-shape paired block failure comes from cross-matched exact-ring calls or the joint `save_moe` checkpoint around both calls.
- Commit Hash: `7690addcb0` adds same-index and cross-index reporting for joint-checkpoint, no-checkpoint, and per-call-checkpoint MoE pairs on identical post-attention MLP inputs.
- Command: Parent `/dlwh/jaxpp-group2-moe-call-order-r6-7690addc-20260726-1644` ran `--diagnostic moe-call-order` from clean commit `7690addcb030059c88d416811742e48b34a017e8`; child `/dlwh/jaxpp-group2-moe-call-order-r6-7690addc-20260726-1644/0`.
- Results:
  - No arm passed the complete fixed relative-L2 `0.002` gate, and no arm cross-matched. Same-index selected routes and routing counts were exact in all three arms. Cross-index comparisons differed on all `515,433` assignments, all `131,072` tokens, and all `64` routing-count entries per microbatch.
  - The no-checkpoint arm was bitwise exact for same-index pre-ring values, MoE outputs, input gradients, losses, and router statistics. The joint and per-call checkpoint arms had only `1.63639e-05` output relative-L2 and `1.49426e-05` combine-weight relative-L2.
  - Routed expert gradients and input gradients passed in every arm. The sole meaningful same-index failure was the summed `mlp.router` gradient at relative-L2 `0.0057684313`, actual/reference norms `54.265850/54.267025`, norm ratio `0.99997835`, cosine `0.99998340`, absolute-L2 `0.31303561`, and maximum absolute error `0.0078125`.
  - Iris reached the expected strict assertion after `1m31s`; the parent and child are terminal with no retry or live allocation.
- Interpretation:
  - Call swapping and the encompassing checkpoint boundary are ruled out. On identical MLP inputs, forward routing, ring values, routed expert VJPs, statistics, and input VJPs preserve program order.
  - The router failure is consistent with summing two cotangents into one BF16 compute leaf before promotion. It is independent of the much larger r4/r5 full-block forward divergence, which must occur before or while constructing the paired MoE inputs.
- Next action:
  - Compare ordered and paired target-shape full blocks at post-attention residuals, MLP inputs, shared outputs, pre-MoE logits/routes, routed outputs, and final outputs. Include an ordered full-block-checkpoint versus no-checkpoint control and report each microbatch's router gradient separately before testing master-precision summation of two disjoint compute-MLP gradients.

### 2026-07-26 16:56 PDT - full-block remat context is the first direct divergence
- Hypothesis: The r4/r5 paired full-block failure begins at a component boundary unique to paired execution rather than the ordered block's complete `save_moe` checkpoint.
- Commit Hash: `02e99ee05b` adds target-shape forward-boundary reporting, an ordered complete-checkpoint versus no-checkpoint control, per-microbatch router-gradient reporting, and a distinct-compute-MLP gradient arm.
- Command:
  - Setup-invalid parent `/dlwh/jaxpp-group2-full-block-boundaries-r7-02e99ee0-20260726-1655` stopped after `9.17s` because its full-SHA assertion contained an incorrect suffix. It reached no diagnostic phase and left no live allocation.
  - Authoritative parent `/dlwh/jaxpp-group2-full-block-boundaries-r7b-02e99ee0-20260726-1656` ran `--diagnostic full-block-boundaries` from clean commit `02e99ee05beff6acd63e731cef9432ee7c94d842`; child `/dlwh/jaxpp-group2-full-block-boundaries-r7b-02e99ee0-20260726-1656/0`.
- Results:
  - Ordered complete-checkpoint versus ordered no-checkpoint and paired versus ordered produced the same boundary signature. Post-attention passed at `0.00103886` and MLP inputs passed at `0.00116565`. Shared-expert outputs were the first failures at `0.00400094/0.00399753`.
  - Pre-MoE boundary margins reached `0.00927861`; selected routes differed on `2,698/2,732` assignments across `1,647/1,663` tokens. Routing counts differed in `59/61` expert entries.
  - MoE outputs reached `0.04815`; final block outputs reproduced r4/r5 at `0.0205638/0.0202413`. Parameter gradients reached approximately `1.39`.
  - The checkpoint-versus-no-checkpoint individual projected losses were `0.0130294/0.00348135`; paired-versus-ordered losses were `0.0131915/0.00347034`.
  - The distinct-MLP arm had exact MoE outputs, router statistics, and routing counts, but its router-gradient comparison crossed the already divergent full-block remat contexts. Its approximately `1.39` gradient errors do not test master-precision summation in isolation and are not interpreted.
  - All lower/compile/execute phases completed; the longest lower was `19.60s` and the longest compile was `6.58s`. Iris reached the intended strict assertion after `2m40s`; the parent and child are terminal with no live allocation.
- Interpretation:
  - The complete ordered `save_moe` remat compiler context is the source of the r4/r5 forward identity. The current paired formulation behaves like ordered no-checkpoint execution at the reported boundaries.
  - The first threshold violation is shared-expert output, but it is downstream of small checkpoint-context changes in CuTe attention and MLP inputs. Those passing perturbations are sufficient to flip thousands of near-tied routes.
- Next action:
  - Compare the ordered checkpoint oracle with a complete paired checkpoint and with one pre-MoE checkpoint per microbatch followed by the joined MoE checkpoint. Gate gradients on exact routes and all forward boundaries passing `0.002`; do not launch L8 or L24.

### 2026-07-26 17:06 PDT - remat scopes inside one executable do not recover the oracle
- Hypothesis: Either one complete paired `save_moe` checkpoint or one pre-MoE checkpoint per microbatch followed by the joined-MoE checkpoint will reproduce the ordered full-checkpoint forward identity.
- Commit Hash: `b7d82f591f` adds both forward-first candidates and skips candidate gradients unless every forward boundary passes relative-L2 `0.002` with exact routes.
- Command: Parent `/dlwh/jaxpp-group2-remat-scope-r8-b7d82f59-20260726-1712` ran `--diagnostic full-block-remat-scope` from clean commit `b7d82f591ff6c0511fba2d8e083bd8e0eec1bfed`; child `/dlwh/jaxpp-group2-remat-scope-r8-b7d82f59-20260726-1712/0`.
- Results:
  - The complete-paired-checkpoint and per-microbatch-pre-MoE-checkpoint arms were numerically identical and both failed forward parity.
  - Post-attention passed at approximately `0.001933`. MLP inputs were the first failures at approximately `0.002160`; shared outputs reached approximately `0.005296`.
  - Selected routes differed on `5,137/5,305` assignments across `3,144/3,249` tokens. Final block outputs reached approximately `0.02879/0.02924`.
  - The forward gate correctly skipped both candidate VAGs. No gradient result is attributed to either arm.
  - Iris reached the intended strict assertion after `1m44s`; the parent and child are terminal with no live allocation.
- Interpretation:
  - Changing remat scopes while both microbatch calls remain in one executable does not reproduce the ordered full-checkpoint compiler identity. The two arms being identical indicates that the relevant distinction is executable/task compilation context, not merely the placement of remat primitives in one Jaxpr.
- Next action:
  - Compile one single-microbatch pre-MoE executable, reuse it for both inputs, and pass both prepared results to a separately compiled joined-MoE/finish executable. Compare forward boundaries and exact routes before adding VJPs. If the pre-task still misses, test an optimization barrier after attention and MLP-input formation.

### 2026-07-26 17:20 PDT - split forward executables are exact; VJP compilation changes their primals
- Hypothesis: Compiling one single-microbatch pre-MoE executable and reusing it twice, then running a separately compiled joined-MoE/finish executable, will reproduce the ordered complete-checkpoint oracle.
- Commit Hash: `d28f24d782` adds the split-executable forward and VJP gate, with a conditional optimization-barrier arm when the base forward fails.
- Command: Parent `/dlwh/jaxpp-group2-split-exec-r9-d28f24d7-20260726-1717` ran `--diagnostic split-executable-boundaries` from clean commit `d28f24d78201163e4a82fa71002405d31b26300c`; child `/dlwh/jaxpp-group2-split-exec-r9-d28f24d7-20260726-1717/0`.
- Results:
  - The base forward arm passed exactly. Post-attention residuals, MLP inputs, shared outputs, pre-MoE logits/weights/margins, routed outputs, router statistics, and final block outputs all had relative-L2 `0.0`. Selected-route assignment/token mismatches and routing-count mismatches were all `0`.
  - The single pre-MoE task lowered in `5.718s`, compiled in at most `1.113s`, and executed twice in at most `0.009s`. The separate joined-MoE/finish task lowered in `0.122s`, compiled in at most `1.348s`, and executed in at most `0.043s`. The barrier arm correctly did not run.
  - Because forward passed, the gate compiled matched VJP executables. Their returned primals no longer matched the ordered VJP oracle: post-attention still passed at `0.00193276`, but MLP inputs were the first failures at `0.00216019`, shared outputs reached `0.00529552`, and pre-MoE boundary margins reached `0.0173280`.
  - The VJP arm differed on `5,137/5,305` selected-route assignments across `3,144/3,249` tokens and on `57/62` routing-count entries. Routed outputs reached `0.0689444`, final block outputs `0.0292388`, and the two projected losses `0.00473397/0.0129419`.
  - VJP parameter gradients reached relative-L2 `1.3823348` at `mlp.expert_mlp.w_gate`; input gradients reached `0.6709662`. QB reached `0.00485684`. All reported values were finite.
  - Iris reached the intended strict assertion after `2m01s`; the parent and child are terminal with no retry or live allocation.
- Interpretation:
  - The architecture-realistic forward decomposition is validated exactly at target shape. Separate executable contexts, not an optimization barrier, recover the ordered full-checkpoint forward identity.
  - The gradient comparison is contaminated: its reference is the ordered VJP executable, whose auxiliary primal boundaries differ from the standalone ordered-forward executable, while the split gradient arm uses saved standalone-forward boundaries. The result isolates an AD compiler-context shift but does not evaluate split gradient assembly on matched primals.
- Next action:
  - Always run the barrier pre-task and compare standalone and VJP-exposed task primals against the ordered VJP auxiliary boundaries. Add an ordered `save_moe` VAG with `prevent_cse=False` to test whether remat's anti-CSE barrier causes the shift. Assemble gradients only when an arm has exact routes and every forward boundary passes relative-L2 `0.002`; do not launch L8 or L24.

### 2026-07-26 17:38 PDT - split VJP primals are exact; gradient assembly still fails
- Hypothesis: Exposing the independently compiled pre-task and joined-finish VJP primals will reproduce the ordered VAG compiler context, allowing a matched-primal gradient comparison. An optimization barrier or `prevent_cse=False` may explain the remat-context shift.
- Commit Hash: `6a564c644d` adds unconditional base/barrier task arms, VJP-primal reporting, matched-primal gradient gating, and ordered default-remat, `prevent_cse=False`, and no-checkpoint controls.
- Command: Parent `/dlwh/jaxpp-group2-vjp-context-r10-6a564c64-20260726-1734` ran `--diagnostic split-executable-boundaries` from clean commit `6a564c644d188dd57a7cff06c32d32180b94570f`; child `/dlwh/jaxpp-group2-vjp-context-r10-6a564c64-20260726-1734/0`.
- Results:
  - Both base and optimization-barrier standalone split forwards remained bitwise exact against the standalone ordered forward at every boundary, selected route, routing count, router statistic, and final output.
  - Both standalone arms failed against the ordered VAG auxiliary primals at the known signature: MLP inputs `0.00216019`, shared outputs `0.00529552`, pre-MoE margin `0.0173280`, `5,137/5,305` assignment mismatches across `3,144/3,249` tokens, `57/62` routing-count mismatches, and final outputs `0.0292388`.
  - Recompiling the pre-task as a VJP and rerunning the joined-finish VJP on those exposed primals reproduced the default ordered VAG auxiliary primals exactly for both arms. Every numeric boundary and final output had relative-L2 `0.0`; selected-route, token, and routing-count mismatches were all `0`.
  - The matched-primal gate therefore ran gradient assembly. Both arms had exact losses, forward boundaries, routes, and router statistics, but `15/19` parameter leaves failed. The worst was `mlp.router` at relative-L2 `1.3939911`, actual/reference norms `54.09361/54.15754`, cosine `0.02725`, absolute-L2 `75.49513`, and maximum absolute error `0.9970703`. Routed-expert leaves were `1.39021-1.39060`, attention leaves `0.53804-0.64412`, dense normalization/gating leaves `0.86539-0.87942`, and input gradients reached `0.6698257`. Shared-expert gradients and overwritten router bias were exact.
  - The optimization-barrier arm was numerically identical to the base arm for both primals and gradients.
  - The ordered `prevent_cse=False` VAG matched ordinary no-checkpoint execution: exact forward boundaries/routes/losses, maximum parameter-gradient relative-L2 `0.00043279`, and input-gradient relative-L2 `0.00001487`. It failed against the default ordered VAG beginning at shared outputs `0.00400094`, with `2,698/2,732` assignment mismatches and final outputs `0.0205638/0.0202413`.
  - Iris reached the intended strict assertion after `2m52.62s`; the parent and child are terminal with no retry or live allocation.
- Interpretation:
  - Executable splitting can reproduce both the standalone-forward and default ordered-VAG forward identities exactly when each task is compiled in the corresponding context. Forward task boundaries and route stability are no longer blockers.
  - The remaining hard failure is matched-primal gradient composition across the pre-MoE and joined-finish VJP tasks. It is not explained by route changes, the optimization barriers, or the earlier contaminated reference.
  - `prevent_cse=False` confirms that remat's default anti-CSE behavior selects the ordered-VAG identity. It is not a promotable workaround without HBM/rematerialization measurements and does not fix the matched-default gradient assembly.
- Next action:
  - Stop before L8/L24 or scheduler integration. Isolate joined-finish and pre-task parameter/input cotangents independently against matching cuts from the monolithic ordered VAG, preserving the fixed per-leaf relative-L2 `0.002` gate.

### 2026-07-26 17:50 PDT - single-finish VJPs do not fix full gradient assembly
- Hypothesis: Replacing the joined two-call MoE backward with two executions of one independently compiled single-microbatch finish VJP, then summing master gradients and applying the resulting boundary cotangents to two pre-task VJPs, will match the ordered full-block VAG.
- Commit Hash: `195b42326a` adds the single-finish VJP executable, direct joined-versus-combined reporting, and full split-gradient assembly.
- Command: Parent `/dlwh/jaxpp-group2-single-finish-vjp-r11-195b4232-20260726-1748` ran `--diagnostic split-single-finish-vjp` from clean commit `195b42326ad1972d1cf6586cf1d7ecc208d9489a`; child `/dlwh/jaxpp-group2-single-finish-vjp-r11-195b4232-20260726-1748/0`.
- Results:
  - The saved VJP-context primals matched the ordered VAG exactly at every forward boundary, selected route, routing count, router statistic, and final output.
  - Joined finish versus two single finishes had exact losses, MoE and block outputs, post-attention cotangents, MLP-input cotangents, shared-output cotangents, and routing counts. Routed-expert parameter gradients passed at relative-L2 `0.0018014-0.0018017`. The only direct finish failure was the summed `mlp.router` gradient at `0.00551110`, actual/reference norms `54.09361/54.09320`, cosine `0.99998492`, absolute-L2 `0.298113`, and maximum absolute error `0.00683594`.
  - Full single-finish assembly still reproduced the r10 hard failure despite exact saved primals and routes. Input gradients were `0.669319/0.669826`; attention leaves reached `0.644121`; dense normalization/gating leaves reached `0.879423`; routed-expert leaves were `1.39021-1.39060`; and the worst leaf was `mlp.router` at `1.393981`. Shared-expert gradients remained exact.
  - Ordered full-block lower/compile took `10.917/4.231s`; the joined finish VJP lower/compile took `0.367/1.062s`; the single finish VJP lower/compile took less than `1s` each. Iris reached the intended strict assertion after `2m04.21s`; no retry or live allocation remains.
- Interpretation:
  - The joined two-call finish backward is not the source of the large full-block error. Its expert gradients pass the fixed `0.002` gate, and all three boundary-cotangent streams are exact against two single finishes.
  - The remaining discontinuity is between the ordered full-block VAG reference and the split pre-task/finish assembly. A ranked cause is the reference itself: the whole-block `save_moe` checkpoint saves dispatch and expert tensors but not router logits, top-k indices, or combine weights, so its backward recompute may route differently from its exact auxiliary forward. Reference cut mismatch, microbatch/tree ordering, and cross-fed cotangents remain controls. The summed router leaf remains a separate BF16 accumulation issue.
- Next action:
  - Report ordered versus joined finish gradients on identical saved primals, joined versus ordered pre-task boundary cotangents, and unsummed same-index and cross-index per-microbatch finish/pre-task gradients before any master-tree reduction. Compare split assembly against the no-checkpoint full-block VJP and report default-checkpoint versus no-checkpoint gradients/routes. If split matches no-checkpoint, test saving router/top-k/combine intermediates or a narrower per-MoE checkpoint. Keep exact routes and the per-leaf `0.002` gate; do not launch L8 or L24.

### 2026-07-26 18:04 PDT - no existing remat context is a canonical gradient reference
- Hypothesis: The r11 discontinuity is either a microbatch/tree-order swap or evidence that the whole-block `save_moe` reference recomputes unsaved routing state during backward.
- Commit Hash: `22f2e80bfd` adds unsummed same-index/cross-index reports, default-checkpoint versus no-checkpoint controls, and split assembly versus no-checkpoint reporting.
- Command: Parent `/dlwh/jaxpp-group2-reference-assembly-r12-22f2e80b-20260726-1804` ran `--diagnostic reference-assembly-discontinuity` from clean commit `22f2e80bfd603a5c34481a724283381228162553`; child `/dlwh/jaxpp-group2-reference-assembly-r12-22f2e80b-20260726-1804/0`.
- Results:
  - Joined-finish versus independently computed ordered-cut boundary cotangents passed same-index exactly and failed crossed. The maximum crossed relative-L2 was `1.41724`. Microbatch/tree swapping is ruled out.
  - Single-finish MoE gradients matched neither ordered microbatch. Same-index maximum relative-L2 was `1.39611/1.39808`; crossed was `1.41943/1.41080`. The summed joined-finish versus ordered MoE gradient remained `1.39399`.
  - Pre-task parameter/input gradients also matched neither order. Same-index maximum relative-L2 was `0.877953/0.880515`; crossed was `1.42584/1.43008`.
  - Default whole-block checkpoint versus no checkpoint reproduced the known forward-context split: shared outputs were the first failure at `0.00400094`, routes differed on `2,698/2,732` assignments across `1,647/1,663` tokens, routing counts differed in `59/61` entries, parameter gradients reached `1.39361`, and input gradients reached `0.673786`.
  - Split assembly did not match no checkpoint either: parameter gradients reached `1.35558` and input gradients reached `0.654292`. Its saved default-context forward primals and routes remained exact.
  - Iris reached the intended strict assertion after `2m10.12s`; no retry or live allocation remains.
- Interpretation:
  - Program order and microbatch tree ordering are not the cause. The exact same-index cut cotangents establish that the split finish sends each cotangent to the intended pre-task input.
  - Neither default whole-block checkpoint nor no checkpoint is a canonical reference for the saved default-context forward. Their routes differ, while the split derivative is evaluated at the default-context saved primals. The ordered default checkpoint's MoE gradients matching neither exact saved-primal finish derivative is consistent with backward recomputation using unsaved routing state.
- Next action:
  - Add checkpoint names for selected experts and the `T×K` continuous routing intermediates required by the router VJP, avoiding retention of `T×E` logits. First require default whole-block checkpoint versus no checkpoint to pass every gradient leaf at relative-L2 `0.002` with exact routes; only then rerun split assembly against the stabilized reference. Do not launch L8 or L24.

### 2026-07-26 18:22 PDT - compact router saves improve gradients but do not align forward contexts
- Hypothesis: Saving selected experts and the compact `T×K` router VJP intermediates across `save_moe` remat will stabilize the whole-block default-remat gradient without retaining `T×E` logits.
- Commit Hash: `5d86a0722b` adds the production remat names and a target H100 gate that initially required default-remat versus no-checkpoint parity before split assembly.
- Command: Parent `/dlwh/jaxpp-group2-router-remat-r13-5d86a072-20260726-1820` ran `--diagnostic router-remat-reference` from clean commit `5d86a0722baf11f6d2a488d15208b8d6f02e5c6f`; child `/dlwh/jaxpp-group2-router-remat-r13-5d86a072-20260726-1820/0`.
- Results:
  - Default-remat versus no-checkpoint parameter-gradient relative-L2 fell from r12's `1.39361` to `0.0568697`; input-gradient relative-L2 fell from `0.673786` to `0.0247935`.
  - Forward compiler contexts remained distinct, as expected: shared outputs were the first failing boundary at `0.00400094`; routes differed on `2,698/2,732` assignments across `1,647/1,663` tokens, and routing counts differed in `59/61` entries.
  - The largest parameter-gradient error was `mlp.router` at `0.0568697`; expert gradients were `0.0476692-0.0477292`. Attention leaves were approximately `0.02065-0.02506`, dense normalization/gating leaves approximately `0.03007-0.03181`, and shared-expert leaves `0.00526-0.00574`.
  - The initial gate incorrectly skipped split assembly because it treated forward identity with the no-checkpoint compiler context as a prerequisite. Iris reached that intended assertion after `1m53.87s`; the job is terminal with no live allocation.
- Interpretation:
  - Compact router residuals are a strong partial improvement, but no-checkpoint remains a different forward program and is not the canonical gate.
  - The actual correctness test is split assembly against the ordered default-remat derivative using the same saved VJP primals and routes. No-checkpoint should remain diagnostic-only.
- Next action:
  - Always run split single-finish assembly and require its saved default-context forward plus every assembled gradient leaf to match ordered default remat at relative-L2 `0.002`. Use the resulting exact failing leaves to rank any additional compact saved boundaries; do not add large activation residuals or launch L8.

### 2026-07-26 18:29 PDT - split assembly is reduced to four parameter leaves
- Hypothesis: With compact router state saved, split single-finish assembly against ordered default remat will identify the remaining component-boundary errors without relying on the no-checkpoint program.
- Commit Hash: `94df400795` makes no-checkpoint diagnostic-only and always gates split assembly against ordered default remat using exact saved VJP primals and routes.
- Command: Parent `/dlwh/jaxpp-group2-router-remat-r13b-94df4007-20260726-1826` ran `--diagnostic router-remat-reference` from clean commit `94df400795e2de206f68bafc01383a22f554eca1`; child `/dlwh/jaxpp-group2-router-remat-r13b-94df4007-20260726-1826/0`.
- Results:
  - Saved split primals matched ordered default remat exactly: both projected losses, post-attention, MLP inputs, shared outputs, pre-MoE values, routed outputs, block outputs, and router statistics had relative-L2 `0.0`. Selected-route assignment/token mismatches and routing-count mismatches were all `0`.
  - Split input gradients passed at `0.00129074/0.00128271`. Fifteen of nineteen parameter leaves passed. Four failed:
    - `attn_gated_norm.w_down`: `0.00382351`, absolute-L2 `0.0933898`, maximum absolute error `0.0009765625`.
    - `attn_gated_norm.w_up`: `0.00297817`, absolute-L2 `0.0718864`, maximum absolute error `0.0009765625`.
    - `attn.w_k`: `0.00227959`, absolute-L2 `2.08499`, maximum absolute error `0.0166016`.
    - `mlp.router`: `0.00504063`, absolute-L2 `0.273595`, maximum absolute error `0.00463867`.
  - MoE expert and shared-expert gradients passed exactly. Joined finish versus two single finishes was exact except for the summed router leaf at `0.00549688`; same-index per-microbatch router gradients were `0.00504075/0.00505789`.
  - Same-index pre-task VJPs isolated the dense residuals: `attn_gated_norm.w_down` reached `0.00402792/0.00400447`, `attn_gated_norm.w_up` `0.00314646/0.00312173`, and `attn.w_k` `0.00228338/0.00227572`. Cross-index controls remained strongly negative.
  - The no-checkpoint diagnostic remained the r13 signature and did not affect the gate. Iris reached the strict split assertion after `2m14.3s`; the parent and child are terminal with no retry or live allocation.
- Interpretation:
  - Saving compact router state converts the earlier order-one split failure into a near-pass. The remaining router leaf is the known per-call BF16 accumulation issue. Three pre-task leaves independently exceed the threshold by small margins.
  - Further broad activation retention is not justified. The next remat experiment should rank compact normalization/attention residuals that can affect only these three dense leaves, with retained-byte accounting.
- Next action:
  - Inspect RMS/gated-normalization and CuTe attention backward residuals. Add only compact named intermediates with a plausible dependency path to the three dense failures, and keep the distinct-MLP/master-precision router-gradient arm separate. Do not launch L8 or retain full token-by-hidden activations without another explicit decision.

### 2026-07-26 18:56 PDT - explicit router state passes the full direct block gate
- Hypothesis: Computing router state in each independent pre-task, passing selected experts and combine weights to an expert-only joined finish, and returning combine-weight cotangents to the same pre/router VJP will remove r13b's router and dense compiler-context errors.
- Commit Hash: `5c411170fa` adds the transient explicit router state, expert-only paired production helper, direct one-block diagnostic, and focused JIT parity coverage.
- Command: Parent `/dlwh/jaxpp-group2-explicit-routing-r14-5c411170-20260726-1854` ran `--diagnostic explicit-routing` from clean commit `5c411170fab4e2ecec05e185c01fa8dc15b36612`; child `/dlwh/jaxpp-group2-explicit-routing-r14-5c411170-20260726-1854/0`. The target remained d2560/e64/top-k4/sequence 4096/global microbatch 32/local 4, H100x8, BF16 CuTe FA4, exact ring EP8, `save_moe`, Triton block-k 32/eight warps, JAX `0.11.1.dev20260725`, NCCL `2.30.7`, and patched JaxPP `7091a9b5`.
- Results:
  - The gate passed. Losses, post-attention residuals, MLP inputs, shared outputs, routed outputs, final block outputs, router statistics, and routing counts were exact. Selected-route assignment and token mismatches were `[0, 0]`; routing-count mismatches were `[0, 0]`.
  - Per-microbatch router gradients were exact at relative-L2 `0.0`; crossed controls remained negative. The assembled `mlp.router` gradient was exact.
  - All 19 parameter leaves passed relative-L2 `0.002`. The maximum was `mlp.expert_mlp.w_up=0.00180177` with absolute-L2 `1.19796`, reference norm `664.879`, cosine `0.999998335`, and maximum absolute error `0.0009765625`. The other expert leaves were `0.00180156` and `0.00180155`.
  - The previous dense failures passed: `attn_gated_norm.w_down=0.000781367`, `attn_gated_norm.w_up=0.000390953`, and `attn.w_k=0.0000439788`. Input-gradient relative-L2 was `0.0000315612/0.0000304466`.
  - The pre-task emitted explicit logits, selected experts, combine weights, boundary margins, and router metrics. Router logits and boundary margins were exact; combine-weight relative-L2 was at most `0.0000149426`.
  - Ordered VJP lower/compile took `10.91/4.57s`; pre-task forward `5.78/1.33s`; pre-task VJP `9.66/1.02s`; expert-only joined finish VJP `0.32/0.61s`. Iris succeeded with exit `0` after `1m58.75s`, with zero failures or preemptions and no live allocation.
- Interpretation:
  - Explicit router state is the correct direct component boundary. It removes r13b's router-only BF16 accumulation error and also restores the three dense leaves without changing the dense remat policy.
  - The joined finish now differentiates only expert execution and returns per-microbatch combine-weight cotangents. Each pre/router VJP owns its original attention, dense, routing, and auxiliary-loss derivative before master-precision gradient summation.
- Next action:
  - Stop at this direct milestone. The next separate gate is JaxPP task integration of the same pre/router and expert-only finish boundaries; do not launch L8 or L24 before that component task graph passes the same per-leaf `0.002` and exact-route checks.

### 2026-07-26 19:24 PDT - explicit router state passes the seven-task JaxPP graph
- Hypothesis: The r14 explicit-router boundary will retain exact routing and per-leaf relative-L2 at most `0.002` when represented as separate JaxPP MPMD pre, joined-expert, backward, and master-gradient reduction tasks.
- Commit Hash: `cd87b638fc` adds the one-block seven-task JaxPP gate, real stage shardings, distinct combine-weight cotangents, and structural assertions.
- Command: Parent `/dlwh/jaxpp-group2-explicit-routing-mpmd-r15-cd87b638-20260727-022028` ran `--diagnostic explicit-routing-mpmd` from clean commit `cd87b638fc8c51dff47dcfb858a4bb10eaa4b5df`; child `/dlwh/jaxpp-group2-explicit-routing-mpmd-r15-cd87b638-20260727-022028/0`. The target remained d2560/e64/top-k4/sequence 4096/global microbatch 32/local 4, one H100x8 node, BF16 CuTe FA4, exact ring EP8, `save_moe`, Triton block-k 32/eight warps, JAX `0.11.1.dev20260725`, NCCL `2.30.7`, and patched JaxPP `7091a9b5`.
- Results:
  - JaxPP versus ordered passed with exact selected routes and routing counts, no failed leaves, and no divergent forward boundary. Maximum parameter relative-L2 was `0.0018017726988870581`; maximum input relative-L2 was `0.000030204541776493355`.
  - JaxPP versus the matching direct graph passed. Maximum parameter relative-L2 was `0.00041523254248327415`; maximum input relative-L2 was `0.000011830357357974469`.
  - The pre-task JAXPR contained `338` attention operations, `4` router operations, and no ring body. The joined-expert JAXPR contained exactly two ring bodies and no attention or router operation.
  - Joined-expert backward returned two distinct combine-weight cotangent leaves, each with shape `[131072, 4]`. The graph used seven tasks and no receive buffers.
  - Lowering took approximately `0.01734s`, precompilation `2.514s`, and execution `0.1513s`.
  - Iris succeeded with exit `0` after `2m15.26s`, with zero failures or preemptions and no live allocation.
- Interpretation:
  - The explicit-router component boundary survives JaxPP task localization. JaxPP introduces no threshold failure relative to either the ordered reference or the direct r14 graph.
  - Separate pre/router tasks, one expert-only paired task, distinct combine-weight cotangents, separate pre backwards, and master-precision gradient reduction form the correctness architecture for production group-size-two execution.
  - This is a one-block correctness and compile gate. Production scheduling, multi-block caching, and L8 throughput remain untested.
- Next action:
  - Integrate the same graph into production explicit-MPMD `std_1f1b` group-size-two scheduling without duplicating the harness task functions. Preserve the existing tuple wire format and group-size-one path. Review and locally validate the production integration before launching an L8 smoke.

### 2026-07-26 20:05 PDT - production grouped L4 gate compiles and executes
- Hypothesis: The r15 explicit-router task graph can replace the production group-size-two `std_1f1b` stage wrappers while preserving group-size-one and non-ring paths.
- Commit Hash:
  - `df6ad012d3` moves the r15 numerical kernels into `train.py`, makes the parity harness import them, and integrates the grouped production task graph.
  - `964a0b5892` moves block QB extraction inside each MPMD task and validates the default `moe_implementation=None` as exact ring.
- Command: `TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ring-explicit-routing-prod-g2-l4-e64k4-b128-s4096-p4m4-r3-20260726-2001 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-stage-task-microbatch-group-size 2 --physical-stages 4 --logical-stages 4 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 4 --experts 64 --top-k 4 --vocab-size 8192 --batch 128 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 6 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Results:
  - Focused validation passed: `40 passed in 33.13s`; changed-files precommit, including Pyrefly, passed.
  - The first parent, `/dlwh/iris-run-job-20260727-024811`, failed setup when one worker could not connect to GitHub while cloning `jax-tvm-ffi`; it was stopped and is terminal.
  - The second parent, `/dlwh/iris-run-job-20260727-025222`, reached `explicit_mpmd_train_step.lower` but failed before task compilation because top-level `stage_qb_betas[block_index]` emitted unsupported MPMD primitive `squeeze`. The parent and child were stopped and are terminal. Passing the stage QB vector into each task and indexing with static `block_index` inside the task removed the unsupported top-level primitive.
  - Authoritative parent `/dlwh/iris-run-job-20260727-025904` and child `/dlwh/iris-run-job-20260727-025904/grug-train-jaxpp-rno2a-ring-explicit-routing-prod-g2-l4-e64k4-b128-s4096-p4m4-r3-20260726-2001` ran the fixed production graph. All four workers succeeded with exit `0`; parent and child are terminal with zero failures or preemptions and no live allocation.
  - The graph compiled the separate pre-forward, joined-expert forward/backward, pre-backward, gradient-reduction, embedding/head, transfer, accumulation, and update tasks. It completed all six requested steps.
  - All four measured W&B rows were finite:

    | step | loss | duration (s) | MFU | tokens/s |
    | ---: | ---: | ---: | ---: | ---: |
    | 2 | 8.8157654 | 0.5096568 | 7.9427940 | 1,028,708.0 |
    | 3 | 8.7312212 | 0.5121544 | 7.9040594 | 1,023,691.3 |
    | 4 | 8.6664658 | 0.5116922 | 7.9111997 | 1,024,616.0 |
    | 5 | 8.6214437 | 0.5109474 | 7.9227311 | 1,026,109.5 |

  - W&B summary: mean MFU `7.9207030`, p50 `7.9227311`, p90 `7.9508192`, standard deviation `0.0146932`, sample count `5`, final duration `0.5109474s`, and final throughput `1,026,109.5 tok/s`. Run: https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-explicit-routing-prod-g2-l4-e64k4-b128-s4096-p4m4-r3-20260726-2001
- Interpretation:
  - The production grouped task graph now passes the L4 compile/functionality gate. The prior `squeeze` was graph-construction leakage from traced scalar indexing, not a numerical or task-kernel failure.
  - This run is too short and too shallow to compare throughput with the established L8 group-size-one control. It establishes only production lowering, task compilation, finite execution, and clean teardown.
- Next action:
  - Stop at L4 as requested. The next separate decision is whether to run the matched L8 group-size-two throughput gate against the `16.116235` MFU control.

### 2026-07-26 20:26 PDT - production grouped L8 compiles but first execution exhausts HBM
- Hypothesis: The correctness-preserving explicit-router production graph will fit and execute at the matched L8/d2560/e64/top-k4/sequence-4096/b512/m16 shape.
- Command: Parent `/dlwh/iris-run-job-20260727-030719` launched four H100x8 stages with two layers per stage, exact ring EP8, CuTe FA4, Pallas-Triton block-k 32/eight warps, BF16 wire, `save_moe`, group size two, 16 original microbatches, XLA memory fraction `0.70`, and 20 steps. Child: `/dlwh/iris-run-job-20260727-030719/grug-train-jaxpp-rno2a-ring-explicit-routing-prod-g2-l8-e64k4-b512-s4096-p4m16-20260726-2008`.
- Results:
  - Every pre-forward, joined-expert forward/backward, pre-backward, reduction, update, and keep-step task compiled. Task compilation ran from approximately `03:10:37Z` through `03:23:20Z`.
  - Initial execution created the stage-3-to-stage-0 DIME streams and communicators, then exhausted the XLA BFC pool. Stage 1 and stage 2 requested another `5.62 GiB` per GPU; stage 3 requested `6.14 GiB` per GPU.
  - Rank 2 and rank 3 segfaulted after the allocator failure. Rank 0's thunk-initialization rendezvous timeout and rank 1's clique/coordination abort were secondary distributed failure propagation.
  - No training step, loss, duration, throughput, or MFU row was produced. Iris began an unchanged retry, so the parent and child were stopped. Both are terminal killed and no live resource remains.
- Interpretation:
  - This is a real first-execution HBM failure after successful lowering and compilation, not a communication deadlock. Lowering the XLA preallocation fraction would reduce the BFC ceiling and is not a correction for this failure.
  - The graph constructed fresh bound phase callables for every microbatch pair and block. Explicit `MpmdFun.lower()` does not apply JaxPP's automatic task-jaxpr deduplication, so the L8 run compiled and retained many equivalent executables before first execution.
- Next action:
  - Reuse stable phase callables per stage/block and prove that unique phase call-jaxpr identity count is independent of microbatch count. Retry L8 at the unchanged `0.70` fraction first; increase the fraction only if the deduplicated graph still has a measured BFC-limit failure.

### 2026-07-26 20:30 PDT - stable phase callables remove microbatch-scaled compile identities
- Hypothesis: Binding each explicit-router phase once per stage/block will let JAX and JaxPP reuse equivalent phase jaxprs without changing task boundaries, routes, values, or reduction order.
- Commit Hash: `3f06cc8505` prebuilds stable pre-forward, joined-expert forward, joined-expert backward, and pre-backward callable tables indexed by stage/block.
- Results:
  - The lowering regression emits 12 heavy phase calls for m4 and 24 for m8, while both retain exactly four unique phase call-jaxpr identities.
  - The production L24/m256 shape has 128 microbatch-pair slots. The previous fresh-partial construction implied approximately 12,288 distinct heavy phase jaxprs; stable stage/block/phase identities reduce the nominal heavy compile keys to 96.
  - The full grouped-stage suite passes `41/41`. Changed-files precommit, including Pyrefly, and `git diff --check` pass.
- Interpretation:
  - This is a compile-identity and likely executable-residency correction, not a steady-state dispatch reduction. The runtime graph still contains the same task calls and exact numerical ordering.
  - Full-stage gradient assembly/accumulation and tuple-wise DIME transfers remain separate runtime optimization candidates, but they should not be changed until the stable-callable L8 gate executes.
- Next action:
  - Launch one fresh matched L8 run from `3f06cc8505` at XLA fraction `0.70`. Require finite steps, exact W&B metrics, and unique compile counts that no longer scale with all eight microbatch pairs before any L24 promotion.

### 2026-07-26 20:54 PDT - stable callables reduce compilation but L8 still exhausts HBM at 0.70 and 0.80
- Hypothesis: Stable phase-callable identities will reduce executable residency enough for the matched L8 group-size-two graph to execute; if `0.70` remains narrowly short, increasing only the XLA memory fraction to `0.80` will provide sufficient bounded headroom.
- Commit Hash: `3f06cc8505` is the tested code; `82fa0a92cd` records the preceding L8 memory evidence.
- Commands:
  - r5 parent `/dlwh/iris-run-job-20260727-033624`, child `/dlwh/iris-run-job-20260727-033624/grug-train-jaxpp-rno2a-ring-explicit-routing-stablecalls-g2-l8-e64k4-b512-s4096-p4m16-r5-20260726-2032`, used XLA fraction `0.70`.
  - r6 parent `/dlwh/iris-run-job-20260727-034419`, child `/dlwh/iris-run-job-20260727-034419/grug-train-jaxpp-rno2a-ring-explicit-routing-stablecalls-g2-l8-e64k4-b512-s4096-p4m16-xla080-r6-20260726-2044`, changed only the XLA fraction to `0.80`.
- Results:
  - Compile reuse is real. The pre-stable r4 graph emitted `298` compile lines with `298` unique names, including `256` heavy pre-forward/joined-expert-forward/joined-expert-backward/pre-backward names across eight microbatch-pair prefixes. Both r5 and r6 emitted `74` compile lines with `74` unique names, including `32` heavy names under only the `mb0_1` prefix.
  - r5 nevertheless reproduced the first-execution BFC failure at `0.70`: stages 1 and 2 requested `5.62 GiB` per GPU and stage 3 requested `6.14 GiB` per GPU. No W&B history row was produced; the run is `crashed`. Iris began an unchanged retry, so the parent prefix was stopped. Parent and child are terminal killed with no live allocation.
  - r6 reproduced the same first-execution failure at `0.80` at `03:50:37Z`. Every GPU on stages 1 and 2 failed a `5.62 GiB` allocation (`6039812864` bytes); every GPU on stage 3 failed a `6.14 GiB` allocation (`6595557376` bytes). Rank 1, rank 2, and rank 3 then segfaulted in the DIME transfer path; the other coscheduled rank failures were secondary.
  - r6 produced no loss, duration, throughput, or MFU row. Iris started a second attempt; only `/dlwh/iris-run-job-20260727-034419` and its child were stopped. Both are terminal killed, the child records one failed attempt followed by the stopped retry, and no live allocation remains.
- Interpretation:
  - Stable phase callables remove microbatch-scaled compilation but do not materially lower first-execution peak buffer residency.
  - Raising the XLA fraction from `0.70` to `0.80` is a hard negative: it leaves the failed allocation sizes and stage pattern unchanged. A further fraction-only retry is not justified.
  - The next memory experiment must reduce live task/transfer buffers or activation residency. It should measure retained transfer and task output buffers before changing schedule capacity, rematerialization, or batch shape.
- Next action:
  - Stop before L24. Inspect explicit-MPMD task and DIME buffer lifetimes, prioritizing grouped forward/backward outputs and transfer reuse. Require a quantified memory reduction before another matched L8 run.

### 2026-07-26 21:24 PDT - component gradients do not remove the L8 transfer allocation
- Hypothesis: Keeping embedding, block, and head gradients componentized until the stage update will avoid materializing another full stage-gradient tree across a JaxPP task boundary and make the matched L8 group-size-two graph fit at XLA fraction `0.70`.
- Commit Hash: `4e0e01329cbece93aef21d51ad933e45a2477566`.
- Command: Parent `/dlwh/iris-run-job-20260727-041239` launched child `/dlwh/iris-run-job-20260727-041239/grug-train-jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-r7-20260726-2112` with:

  ```bash
  TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-r7-20260726-2112 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-stage-task-microbatch-group-size 2 --physical-stages 4 --logical-stages 4 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 8 --layers 8 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 20 --tracker wandb --jax-nightly-version 0.11.1.dev20260725 --xla-memory-fraction 0.70 --remat save_moe
  ```

- Results:
  - Setup completed on all four H100x8 ranks. The first attempt emitted `68` compile events, covering the first grouped pair's embedding, per-block pre/router and joined-expert forward/backward tasks, master-gradient reductions, component accumulations, stage averages, grouped-component updates, and `keep_step`.
  - First execution failed at `04:18:54-04:18:55Z`. Stage 1 requested `5.62 GiB` (`6039815424` bytes) per GPU. Stage 2 requested `5.62 GiB` (`6040339456` bytes) per GPU. Stage 3 requested `6.09 GiB` (`6543655424` bytes) per GPU.
  - Ranks 2 and 3 then exited `139`. Their fatal stacks were in `jax._src.dlpack._to_dlpack` through JaxPP `dime2.get_shard_ops_and_capsules`, `enqueue_nccl_transfer_group`, and `start_transfer`, called from `experimental._mpmd.eval_local`. The segfaults followed the BFC failures while JaxPP was starting transfer capsules.
  - Iris started attempt `:1` and repeated dependency setup at `04:19:17Z`. The retry was stopped before model compilation. The parent is terminal `killed` with one preemption. The child is terminal `killed` with one failed attempt and four stopped retry tasks; task 2 records exit `139`. No live allocation remains.
  - [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-r7-20260726-2112) has zero history rows and no loss, duration, throughput, or MFU. Its API state remained `running` after the abrupt process failure.
- Interpretation:
  - Componentizing embedding and head gradients reduced the stage-3 failed request from r5/r6's `6.14 GiB` to `6.09 GiB`, but stages 1 and 2 retained the same `5.62 GiB` request. The change does not make the matched L8 graph fit.
  - The failure remains at the first JaxPP transfer/execution boundary after all tasks compile. Another XLA-fraction retry or L24 promotion is not justified.
  - No valid throughput comparison exists against the group-size-one L8 control (`16.116235` mean MFU) or the old numerically invalid group-size-two result (`15.7366` mean MFU).
- Next action:
  - Inspect the exact transfer payload and live producer/consumer buffers for the `5.62 GiB` stage-1/2 and `6.09 GiB` stage-3 requests. Require a measured removal or reuse of that allocation before another matched L8 run.

### 2026-07-26 21:48 PDT - disabling JaxPP receive reuse leaves the L8 failure unchanged
- Hypothesis: The `5.62 GiB` stage-1/2 allocation is JaxPP's hoisted reusable receive-buffer pool, so `JAXPP_REUSE_RECV_BUFFERS=false` will remove the single large first-execution allocation.
- Command: Parent `/dlwh/iris-run-job-20260727-043813` launched child `/dlwh/iris-run-job-20260727-043813/grug-train-jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-norecvreuse-r8-20260726-2137` from unchanged commit `d6b8b2e190` with the r7 L8 shape, XLA fraction `0.70`, and `JAXPP_REUSE_RECV_BUFFERS=false`.
- Results:
  - All four worker containers reported `JAXPP_REUSE_RECV_BUFFERS=false`. The run compiled the same `68` tasks and reached first execution.
  - The failed allocations were byte-for-byte identical to r7: stage 1 requested `6,039,815,424` bytes, stage 2 requested `6,040,339,456` bytes, and stage 3 requested `6,543,655,424` bytes. Stage 0 reported no failed allocation.
  - Ranks 1-3 failed after the allocator errors. The unchanged retry entered setup and was stopped before compilation.
  - W&B initialized but produced zero history rows. The parent and child are terminal killed; all four child tasks are terminal and no matching pod or Kueue workload remains.
- Interpretation:
  - Disabling receive-buffer reuse is a hard negative for this grouped-ring graph. The numerical match between the request size and a count of 80 MiB activation shards is not sufficient to identify the allocation as the reusable receive prologue.
  - The next supported memory control is outer-state donation. It can alias updated parameter and optimizer-state buffers but does not change task boundaries, routes, or arithmetic.
- Next action:
  - Add `donate_argnums=0` to the four explicit-MPMD entrypoints, validate lowering and donation metadata, then rerun the unchanged L8 gate before considering any L24 promotion.

### 2026-07-26 22:05 PDT - outer-state donation deletes reused QB state
- Hypothesis: Donating the explicit-MPMD outer state will alias updated parameter and optimizer-state buffers without changing the task graph, numerical order, or model configuration, allowing the matched L8 group-size-two graph to execute within HBM.
- Commit Hash: `6eb2cf8a87631ae0fba3ecda14562ebc4dc13e74`.
- Command: Parent `/dlwh/iris-run-job-20260727-045310` launched child `/dlwh/iris-run-job-20260727-045310/grug-train-jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-donate-r9-20260726-2151` with the unchanged r7 L8 shape, default receive reuse, XLA fraction `0.70`, and `donate_argnums=0` on the four explicit-MPMD entrypoints.
- Results:
  - All `68` JaxPP tasks compiled. The first execution then failed on every rank in JaxPP `task_impl` while JAX was sharding a task argument.
  - Every rank raised `RuntimeError: Array has been deleted with shape=float32[2,64]`. The shape identifies a two-layer-by-64-expert stage QB state leaf reused by multiple tasks in the same MPMD step.
  - W&B initialized but produced no training or metric rows. The unchanged retry was stopped before compilation. Parent and child are terminal killed, all child tasks are terminal, and Iris reports no live allocation.
- Interpretation:
  - Whole-state outer donation is invalid for the current multi-use stage-state graph. JaxPP propagates donation to a task before later tasks have finished reading the same state leaves.
  - The allocator failure was not reached, so this run provides no evidence that donation would reduce the `5.62 GiB` stage-1/2 allocation after task-liveness is corrected.
- Next action:
  - Revert whole-state donation. Before another matched L8 run, identify the executable and buffer category responsible for the large first-execution request or prove a narrower donation scheme preserves every stage-state consumer.

### 2026-07-26 22:34 PDT - add named executable and receive-pool memory attribution
- Hypothesis: The stage-1/2 `5.62 GiB` and stage-3 `6.09 GiB` requests can be attributed before execution by comparing the lowered receive-buffer plan with `CompiledMemoryStats` for every named JaxPP task.
- Changes:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN` precompiles each local task without executing it and emits argument, output, alias, temporary, peak, and code bytes under the exact task name.
  - The same diagnostic reports the hoisted receive-pool size and each transfer's destination bytes, allocation mode, and adjacent producer/consumer tasks.
  - The launcher forwards the flag only when explicitly set. Normal execution is unchanged.
- Evidence:
  - The last rank-local executables compiled before r7's failures were `grug_1f1b_stage{1,2,3}_update_grouped_components`. Stage 3's failed request exceeds stage 2 by exactly `503,315,968` bytes, equal to `24 * (8192 * 2560) - 512`, so the grouped optimizer update is the leading attribution. Compile order alone is not proof.
  - NVIDIA/JaxPP `main` remains the pinned `7091a9b5` revision. Current JAX main includes the July 24 XLA device-initiated ragged all-to-all change already exercised by the device-ragged negative controls; no later ragged runtime change is available.
  - Focused grouped-stage and task-validation tests pass `48/48`; changed-files precommit including Pyrefly passes.
- Next action:
  - Commit the diagnostic, then launch one unchanged matched L8 group-size-two run on `cw-rno2a` with memory-plan logging enabled. Match the exact failed request against a task or receive record before changing buffer lifetimes.

### 2026-07-26 22:42 PDT - task precompile executes L8 and attributes the failed buffers
- Hypothesis: Precompiling every named local task before `eval_local` will identify the previous allocation and may prevent compilation/execution overlap from exhausting HBM.
- Commit Hash: `e96ac89d21`.
- Command: Parent `/dlwh/iris-run-job-20260727-052358` launched the unchanged exact L8/d2560/e64/top-k4/sequence-4096/b512/m16 group-size-two ring/CuTe FA4 graph at XLA fraction `0.70`, with one step and `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=true`.
- Results:
  - Parent, child, and all four tasks succeeded. All `68` JaxPP tasks compiled, one training step completed with loss `9.04`, and no retry or live resource remains.
  - The previous BFC requests exactly match 256-byte-aligned `joined_expert_backward` temporary buffers:
    - `6,039,815,424 = 6,039,815,272 + 152` bytes on stage 1/2 block 1.
    - `6,040,339,456 = 6,040,339,304 + 152` bytes on stage 0 block 1.
    - `6,543,655,424 = 6,543,655,288 + 136` bytes on stage 1/2 block 0 and stage 3 blocks 0/1.
  - Receive pools were only `167,772,164`, `838,860,800`, `671,088,640`, and `335,544,320` bytes on stages 0-3.
  - `update_grouped_components` temporary allocations were only `52,506,648`, `138,488,344`, `138,488,344`, and `52,507,168` bytes. The compile-order attribution to optimizer update was false.
- Interpretation:
  - The OOM is not a DIME receive allocation. It is the large joined-expert-backward executable temporary becoming unsatisfiable during lazy compile/dispatch.
  - The diagnostic precompiles and caches every task before `eval_local`; that separation is sufficient to execute the exact graph without changing routes, arithmetic, batch shape, or XLA fraction.
- Next action:
  - Babysit 20-step parent `/dlwh/iris-run-job-20260727-053238` with the same precompile/cache path. Require finite steady-state MFU before extracting precompile into a non-verbose production mode or running the fixed `0.002` numerical gate.

### 2026-07-26 22:49 PDT - precompile clears the OOM but grouped L8 regresses 33.7%
- Hypothesis: Precompiling and caching all `68` local tasks will preserve r10's successful execution while recovering the old group-size-two steady-state throughput.
- Commit Hash: `e96ac89d21`.
- Command: Parent `/dlwh/iris-run-job-20260727-053238` launched child `/dlwh/iris-run-job-20260727-053238/grug-train-jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-precompile-r11-20260727-0532` for 20 steps with the same exact L8/d2560/e64/top-k4/sequence-4096/b512/m16 group-size-two ring/CuTe FA4 graph and XLA fraction `0.70`.
- Results:
  - Parent, child, and all four rank tasks succeeded. All `68` tasks compiled, no allocation failure or retry occurred, and no pod or workload remains live.
  - [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-explicit-routing-componentgrads-g2-l8-e64k4-b512-s4096-p4m16-precompile-r11-20260727-0532) recorded finite rows for steps 2-19. Loss declined from `8.529188` to `6.293761`.
  - MFU mean/p50/p90 was `10.683126/11.088952/11.121371`. Step duration mean/p50/p90 was `2.971911/2.845301/3.467565s`.
  - Mean MFU is `5.433109` points (`33.71%`) below the valid group-size-one L8 control at `16.116235`. It is also `5.053474` points (`32.11%`) below the old numerically invalid group-size-two result at `15.7366`.
- Interpretation:
  - Precompile/cache is a functional fix for the first-execution allocation, but the explicit-routing grouped architecture is a hard performance negative. The `11.09` median rules out a few startup or tail outliers as the explanation.
  - Do not promote this graph to L24 and do not extract precompile as a production optimization. Retain the opt-in memory-plan path for executable attribution and debugging.
  - The fixed `0.002` loss and per-gradient-leaf relative-L2 acceptance remains available, but a numerical gate cannot rescue a path that is already `33.71%` slower than the valid L8 control.
- Next action:
  - Return to the valid group-size-one L24 baseline at `18.2583` mean MFU. Use its existing profile to rank changes capable of closing the remaining `1.7417` MFU-point (`9.54%`) gap before launching another expensive run.

### 2026-07-26 23:00 PDT - profile selects exact-ring NCCL protocol tuning
- Hypothesis: The exact bulk-ring all-gather and reduce-scatter payloads are large enough that NCCL's observed `RING_LL` selection leaves material bandwidth unused; selecting `Simple` or NVLS only for those collectives can accelerate the valid group-size-one graph without changing arithmetic.
- Profile evidence:
  - Communication is `39.61%` of the global exclusive timeline. A `28.2%` communication speedup projects the `18.2583` L24 result to `20.00` MFU.
  - Six MoE all-gather/reduce-scatter paths account for `59.8862%` of stage-0 XProf operation self-time. The trace records `20,480` all-gathers at `0.986ms` average and `11,776` reduce-scatters at `1.500ms` average, all identified as `RING_LL`.
  - The profile has no step markers, so operation self-time is supporting evidence rather than a direct step-time decomposition. Its b512/m16 graph retains the target's microbatch shape and six-layer stage depth.
  - NVIDIA NCCL 2.30 supports per-function `NCCL_PROTO` and `NCCL_ALGO` selection. Protocol forcing remains an experiment and requires measured confirmation.
- Command:
  - One H100x8 RNO2A job `/dlwh/ep-ring-nccl-protocol-ab-r12-20260727` runs the exact d2560/i1280/e64/top-k4/microbatch32/sequence-4096/capacity-1.25 ring value and value-and-grad benchmark for 10 warmups and 50 samples per arm.
  - Arms are NCCL defaults, `NCCL_PROTO=allgather:simple;reducescatter:simple`, and `NCCL_ALGO=allgather:nvls;reducescatter:nvls`. NCCL INFO logging records the actual algorithm and protocol.
- Promotion gate:
  - Require exact value/VJP behavior and either `1.282x` improvement in the dominant collective aggregate or `1.10x` direct value-and-grad improvement before an L8 group-size-one pipeline run.
- Results:
  - The job succeeded in `2m17.54s`; its single task exited `0` with no retry or live resource. Every arm passed strict value and VJP parity.
  - Default forward/value-and-grad medians were `10.512845/23.054087ms`.
  - Selective Simple medians were `10.566797/23.126030ms`; value-and-grad regressed `0.31%`.
  - Selective NVLS medians were `11.213155/23.780595ms`; value-and-grad regressed `3.15%`.
  - NCCL INFO corrected the profile-level inference: the 80 MiB all-gather/reduce-scatter operations already use `RING/SIMPLE` by default. The 256 KiB operations primarily use `RING/LL`. Selective Simple changes only the smaller operations, while selective NVLS moves both sizes to `NVLS/SIMPLE`.
- Interpretation:
  - Both overrides are hard negatives and neither clears the direct promotion gate. The profile's anonymous `RING_LL` kernels did not establish that the large MoE payload used LL.
  - Do not run a pipeline protocol override. Retain NCCL defaults for the valid exact-ring graph.
- Next action:
  - Validate the bounded NCCL_EP-forward plus bulk-ring token-gradient control. It retains the previously exact FP32 NCCL_EP output and parameter gradients while replacing the sole failing dispatch-backward token reduction with the accepted ring VJP; require every relative-L2 metric at most `0.002` and measure the duplicate-backward cost before any pipeline integration.

### 2026-07-26 23:17 PDT - ring token gradient is exact but parameter-only TE VJP is non-finite
- Hypothesis: Differentiating TE only with respect to routing and expert parameters will let XLA remove TE's rejected token cotangent, while a separate bulk-ring token VJP supplies the exact input gradient.
- Commit Hash: `df76429128`.
- Command: One H100x8 RNO2A job `/dlwh/ncclep-h100-ring-token-gradient-r13-20260727` used bounded NCCL_EP, FP32 TE/ring combine, BF16 dispatch, balanced 65,536-assignment destination loads, receive capacity 81,920, and strict parity.
- Results:
  - The job succeeded in `9m03.23s`; all eight ranks exited `0`, no retry occurred, and no resource remains live.
  - Loss, output, and token gradient were bitwise equal to the FP32 ring oracle. Routing-gradient relative-L2 was `8.1766e-5`.
  - TE-supplied W13 and W2 gradients were non-finite, while every ring reference remained finite. The gate stopped before timing.
  - The hybrid StableHLO contained 4 all-gathers, 4 all-reduces, 2 reduce-scatters, and 17 custom calls, versus ring's 4, 5, 3, and 6.
- Interpretation:
  - The bulk-ring token-gradient substitution fixes the sole prior numerical discrepancy exactly. The failure is introduced by differentiating the TE graph only with respect to its parameter arguments; the earlier complete TE VJP produced finite, exact FP32-reference parameter gradients.
  - Restore the known-good complete TE VJP, discard only its token-gradient result, and substitute the ring token gradient. This retains extra TE token-gradient work, so the next direct gate must measure the cost rather than assuming DCE.
- Next action:
  - Run the same strict direct gate with the complete TE VJP plus ring token VJP. Require all metrics at most `0.002` and at least `1.12x` value-and-grad speedup before reduced pipeline integration.

### 2026-07-26 23:29 PDT - complete TE VJP remains non-finite when composed with ring VJP
- Hypothesis: Retaining the original complete TE VJP will restore the previously finite parameter gradients, while replacing only its reported token gradient with the exact bulk-ring result.
- Commit Hash: `ad7be4336b`.
- Command: Matched one-H100x8 strict rerun `/dlwh/ncclep-h100-ring-token-gradient-r14-20260727`.
- Results:
  - Iris succeeded in `9m02.9s`; all eight ranks exited `0`, no retry occurred, and no resource remains live.
  - Loss, output, and token gradient remained bitwise equal to the FP32 ring oracle. Routing-gradient relative-L2 remained `8.1766e-5`.
  - W13 and W2 gradients remained non-finite. Ring references were finite and routing was balanced with zero drops. The strict gate stopped before timing.
  - StableHLO counts were unchanged from r13: hybrid 4 all-gathers, 4 all-reduces, 2 reduce-scatters, and 17 custom calls; ring 4, 5, 3, and 6.
- Interpretation:
  - The full-MLP bulk-ring token gradient is exact, but composing it with the NCCL_EP VJP in one executable destabilizes the TE expert-weight gradient path. Retaining the complete TE VJP does not fix the interaction.
  - Executing separate TE and ring backward executables would avoid this compilation interaction but necessarily adds a second full routed-MLP backward. Given ring's `23.4ms` and TE's `17.6ms` prior direct medians, the optimistic sum cannot beat the ring baseline. Stop this hybrid rather than spend another build allocation on a guaranteed performance negative.
  - Keep `ring_token_gradient` as a bounded reproducer. Do not integrate or pipeline it.
- Next action:
  - Return to the valid group-size-one exact-ring graph. Prototype saving compact block inputs and running block-local combined VJPs so backward can eliminate recomputed primal MoE collectives without the measured split-backward duplication.

### 2026-07-26 23:40 PDT - saved block inputs do not remove backward collectives
- Hypothesis: Saving each transformer block input and differentiating each block locally inside the existing combined backward task will let XLA eliminate recomputed primal MoE collectives without splitting input and parameter gradients.
- Results:
  - A forced two-device CPU optimized-HLO probe compared one exact-ring middle-stage block. Forward had 4 all-gathers and 1 reduce-scatter.
  - Current combined backward had 5 all-gathers, 2 reduce-scatters, and 2 all-reduces.
  - Saved block inputs plus block-local combined VJP had the same 5 all-gathers, 2 reduce-scatters, and 2 all-reduces. Gradient parity passed at `rtol=atol=2e-5`.
  - Focused input-gradient-first tests pass `4/4`; changed-file precommit passes. No code change or GPU job was produced.
- Interpretation:
  - Block inputs are not the residuals needed to avoid the MoE primal replay. Ring weight gradients still require gathered token activations, and `save_moe` residuals exist only inside one AD transform; they do not cross explicit JaxPP's separate forward and backward tasks.
  - A dedicated model/ring forward-residual and backward API would need to export dispatch inputs, expert hidden activations, dispatch outputs, and routing state. That is not a compact-input optimization and has a substantially larger memory/interface cost.
  - Do not launch this unchanged graph.
- Next action:
  - Test an FP8 wire format inside the bulk ring itself. Per-token scaled FP8 all-gather plus shared-scale FP8 reduce-scatter targets the dominant MoE payloads, unlike the previously tested FP8 inter-stage wire and FP8 expert GEMMs. Require direct output/every-gradient relative-L2 at most `0.002` and at least `1.10x` value-and-grad speedup.

### 2026-07-26 23:53 PDT - FP8 bulk-ring wire lowers forward and backward payloads
- Hypothesis: Per-token scaled E4M3 activation all-gather and shared-scale E4M3 output reduce-scatter will halve the dominant exact-ring payloads while retaining BF16 expert GEMMs and bounded final tensor error.
- Changes:
  - Benchmark-only `ring_fp8_wire_approx` preserves the existing routing, capacity, BF16 expert weights, and BF16 Pallas-Triton GEMMs.
  - Dispatch quantizes each local token with a stop-gradient FP32 scale, gathers E4M3 values and scales separately, then dequantizes before expert compute.
  - Combine computes a shared expert-axis per-token scale, leaves headroom for at most `min(top_k, expert_axis_size)` contributing ranks, reduce-scatters E4M3 values, and dequantizes the local shard.
  - The default `ring` path is unchanged and the new mode is explicitly marked approximate and non-promotable by the benchmark itself.
- Validation:
  - Forced EP8 forward and full value-and-grad StableHLO contain E4M3 all-gather and reduce-scatter payloads with no BF16 all-gather payload.
  - Focused tests pass `20/20`; changed-file precommit including Pyrefly passes; `git diff --check` is clean.
- Next action:
  - Commit and run the one-H100x8 exact d2560/i1280/e64/top-k4/microbatch32/sequence-4096 direct A/B. Require finite output and all gradients, relative-L2 at most `0.002` for each, zero drop mismatch, and at least `1.10x` median value-and-grad speedup.

### 2026-07-27 00:02 PDT - FP8 bulk-ring wire is faster but numerically invalid
- Commit Hash: `0b2c2ec387`.
- Command: One H100x8 RNO2A job `/dlwh/ep-ring-fp8-wire-approx-r15-20260727` compared ring and `ring_fp8_wire_approx` at d2560/i1280/e64/top-k4/microbatch32/sequence-4096/capacity-1.25, with 10 warmups and 50 alternating samples.
- Results:
  - Iris succeeded in `1m14.01s`; the task exited `0`, no retry occurred, and no resource remains live.
  - Routing remained balanced at 8,192 assignments per expert, 65,536 accepted assignments per rank, 81,920 local capacity, and zero drops in both arms.
  - Output relative-L2 was `0.0462873`, with `23.36%` mismatches and maximum absolute error `0.125`.
  - Candidate x, routing-weight, W13, and W2 gradient norms were all exactly zero, producing relative-L2 `1.0` against finite nonzero references. The FP8 cast/collective transpose does not provide the required training gradient.
  - Forward median improved from `10.3982ms` to `9.49133ms` (`1.09555x`). Value-and-grad median improved from `22.8937ms` to `21.0599ms` (`1.08708x`).
- Interpretation:
  - The FP8 payload collectives produce a real direct speedup, but output error is over 23 times the accepted `0.002` ceiling and gradients are invalid. The VAG gain also misses the independent `1.10x` promotion threshold.
  - A straight-through custom transpose could make gradients nonzero but cannot repair the already-invalid forward output. E4M3's quantization error is the blocker, not only autodiff plumbing.
  - Keep the benchmark as evidence and do not integrate or pipeline this mode.
- Next action:
  - Inspect the default NCCL channel count from the r12 logs. Run a bounded exact channel-count gate only if default large-payload ring collectives leave plausible unused channel parallelism; otherwise stop NCCL tuning.

### 2026-07-27 00:10 PDT - default NCCL already uses 24 channels
- Hypothesis: The exact ring collectives may leave unused channel parallelism that a forced channel count can recover.
- Evidence:
  - The default arm of `/dlwh/ep-ring-nccl-protocol-ab-r12-20260727` reports the 80 MiB all-gather and reduce-scatter as `RING/SIMPLE channel{Lo..Hi}={0..23}`.
  - The 256 KiB collectives also span channels `0..23`, either as one operation or two ranges split at channels 11 and 12.
- Interpretation:
  - NCCL already selects 24 channels for the target collectives. There is no evidence of unused channel parallelism, so a forced 24- or 32-channel sweep lacks a credible promotion path.
  - Stop NCCL environment tuning. The exact bulk-ring path remains the valid implementation, but protocol, algorithm, and channel overrides have not produced a useful speedup.
- Next action:
  - Rank structural changes that reduce the number of exact MoE collectives or overlap them with expert compute. Require a direct value-and-grad projection capable of closing the remaining `9.54%` relative L24 gap before another pipeline allocation.

### 2026-07-27 00:28 PDT - EP4/data2 step-amortized ring gate launched
- Upstream freshness:
  - SonicMoE `main` remains the tested `0349404acd7952592f73d180ff0c1510f6d112c2`.
  - QuACK `0.6.1` and current `main` do not identify an SM90 gated-GEMM numerical or backward fix for the strict target-shape failures.
  - JAX `0.11.1.dev20260726` advances ten XLA commits beyond the tested nightly, but none changes GPU, NCCL, or ragged all-to-all. The July 24 device-kernel change was already in the failed control.
  - Do not repeat Sonic/QuACK or ragged all-to-all until an upstream change names the observed failure mode.
- Hypothesis:
  - Mapping one H100x8 node as `data=2, expert=4` reduces each exact ring from seven remote chunks to three while preserving complete replicated BF16 compute weights in each data group.
  - Data-local expert gradients are reduce-scattered and sharded BF16 weights are materialized once per optimizer step. At m256, these boundaries should be negligible compared with the per-microbatch ring saving.
- Projection:
  - The L24 baseline needs an `8.71%` step-time reduction, from `81.037785s` to `73.9806s`.
  - Six critical-stage layers and 256 microbatches require at least `4.5945ms` saved per layer-microbatch to reach 20 MFU.
  - A traffic-based 50% reduction in the profiled MoE collectives projects approximately `20.41` MFU. The direct gate requires at least `20.3` projected MFU to retain margin.
- Commit Hash: `253e81a2e7a93cd32b6739f593243cfa5d677092`.
- Changes:
  - `benchmark_ep_ring_data_axis.py` compares EP8 with EP4/data2 at the exact d2560/i1280/e64/top-k4/microbatch32/sequence-4096 geometry.
  - Replicated compute weights return data-local W13/W2 gradients. Separate once-per-step functions use FP32-accumulating reduce-scatter for gradients and all-gather for BF16 compute-weight materialization.
  - The gate checks loss, output, x/combine/W13/W2 gradients at relative-L2 `<=0.002`, exact drops, optimized-HLO collective placement, compiled memory, alternating forward/VAG timings, and the measured L24 projection.
- Validation:
  - Focused tests pass `15/15`; changed-file precommit including Pyrefly passes.
  - An eight-device CPU lowering reports zero data-axis collectives in the treatment local VAG, two data-axis reduce-scatters in gradient sync, and two data-axis all-gathers in weight materialization.
  - A tiny eight-device execution is bitwise exact for loss, output, x, and combine-weight gradients. Its W13/W2 relative-L2 is `0.002876/0.002540`, so the exact target-shape H100 gate remains numerically decisive; the threshold is unchanged.
- Command:
  - One H100x8 RNO2A job `/dlwh/ep-ring-data2-ep4-r16-20260727` runs 10 warmups and 50 alternating samples at capacity factor `1.0`, block-k 32, eight warps, XLA fraction `0.70`, and no retries.
- Promotion gate:
  - Require exact drops, every relative-L2 metric at most `0.002`, zero data-axis collective in local VAG, explicit data-axis collectives only at the step boundary, and projected L24 MFU at least `20.3`.

### 2026-07-27 00:32 PDT - EP4/data2 is faster but remains below target
- Results:
  - `/dlwh/ep-ring-data2-ep4-r16-20260727` reached the intended assertion in `59.68s`; the task exited `1`, no retry occurred, and no resource remains live.
  - Loss, output, x gradient, combine-weight gradient, and drops were bitwise exact. W13 and W2 gradient relative-L2 was `0.00282516/0.00273174`, above the accepted `0.002` threshold.
  - Forward median improved from `8.69998ms` to `6.71259ms` (`1.2961x`). Local VAG improved from `18.59005ms` to `15.47777ms` (`1.2011x`).
  - Once-per-step gradient synchronization and weight materialization took `2.01913ms` and `1.20992ms`. Amortized over m256, the boundary adds `0.01261ms` per layer-microbatch.
  - The measured saving is `3.09967ms` per layer-microbatch. The six-layer critical-stage projection is `76.27669s`, `1.06242x`, and `19.39796` MFU, below both 20 and the `20.3` promotion margin.
  - Compiled peak memory was `3,423,748,224` bytes for EP8 local VAG and `3,065,266,368` bytes for EP4/data2. Sync and materialization peaks were `1,101,004,816` and `786,432,016` bytes.
- HLO instrumentation:
  - GPU optimized HLO reports NCCL operations as custom calls, so the first collector incorrectly returned zero operations for every arm. The prior CPU StableHLO structurally verified the intended placement.
  - Commit the collector fix that inspects pre-compile StableHLO. Do not reinterpret the GPU zero counts as missing communication; measured sync/materialization time proves the operations execute.
- Interpretation:
  - EP4/data2 is a direct transport and VAG improvement, but it is independently non-promotable on numerics and projected target performance.
  - The tiny CPU and exact H100 gates show nearly the same W13/W2 error, approximately `0.25-0.29%`. Splitting BF16 weight-gradient accumulation across data groups is the numerical source; FP32 synchronization after each partial gradient is already rounded cannot recover EP8's one-pass result.
  - Do not integrate or pipeline EP4/data2.
- Next action:
  - Run one final direct topology point at `data=4, expert=2`. Linear traffic scaling predicts approximately another `1.55ms` local-VAG saving, enough to approach 20 MFU and exceed it only when composed with the already validated `1.79%` inter-stage FP8 transfer.
  - Treat EP2/data4 as exploratory and non-promotable unless it independently passes the `0.002` gate and projects at least `20.3` MFU. Stop topology narrowing if either gate fails.

### 2026-07-27 00:40 PDT - EP2/data4 reaches 19.99 projected MFU but fails parity
- Commit Hash: `cc43694bbe538a07f786e408321019b8c0ec884c`.
- Command: `/dlwh/ep-ring-data4-ep2-r17-20260727` used the same exact gate with `data=4, expert=2`; 10 warmups, 50 alternating samples, and no retries.
- Results:
  - Iris reached the strict assertion in `34.56s`; the task exited `1`, and no resource remains live.
  - Loss and drops were exact. Output, x-gradient, and combine-weight-gradient relative-L2 was `2.7625e-6`, `2.6255e-5`, and `1.2884e-5`, all within threshold.
  - W13 and W2 gradient relative-L2 was `0.0025987/0.0023340`, above the accepted `0.002`.
  - Forward median improved from `8.731ms` to `6.005ms`; local VAG improved from `18.583ms` to `13.987ms`.
  - Gradient sync and weight materialization took `3.866ms` and `2.349ms`, or `0.02428ms` amortized per m256 layer-microbatch.
  - The measured projection is `74.016s`, `1.09487x`, and `19.9905` MFU. This misses 20 by `0.0095` points and the `20.3` promotion margin by `0.3095` points.
  - StableHLO confirms local VAG has no data-axis collective. Its `4` all-gathers, `1` all-reduce, and `3` reduce-scatters use the expert groups. Materialization has two data-axis all-gathers and sync has two data-axis reduce-scatters.
  - Peak compiled memory was `3.424GB` for control VAG, `3.357GB` for treatment VAG, `2.045GB` for sync, and `1.311GB` for materialization.
- Interpretation:
  - Stop topology narrowing. EP2/data4 independently fails both the numerical and performance promotion gates.
  - The speed is close enough that preserving FP32 data-local weight gradients before reduction is worth one code-level feasibility check. A direct H100 gate is justified only if the change avoids data collectives per microbatch and its added local memory traffic leaves enough headroom to exceed 20 after the separately measured `1.79%` inter-stage FP8 transfer.
  - Do not launch a JaxPP pipeline or L24 run from the current BF16-gradient path.
- Next action:
  - Determine whether a BF16-forward/FP32-weight-gradient custom VJP can return FP32 cotangents legally without making replicated FP32 weights the differentiable primal. Quantify the extra per-microbatch bytes before implementing.

### 2026-07-27 00:56 PDT - Launch FP32 weight-gradient accumulation gate
- Hypothesis: retaining only W13/W2 cotangents in FP32 will bring EP2/data4 below the accepted `0.002` relative-L2 error while donated FP32 accumulation stays within the `0.839ms` per layer-microbatch headroom needed to compose above 20 MFU.
- Commit Hash: `cc636b7f076f023942860256e2966b8f60af5c27`.
- Command:
  - `uv run iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256GB --disk 128GB --timeout 1800 --max-retries 0 --priority interactive --extra gpu --sync-package marin-levanter --job-name ep-ring-data4-ep2-fp32acc-r18-20260727 -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.70 -e TF_GPU_ALLOCATOR cuda_malloc_async -e HALIAX_RAGGED_DOT_TRITON_BLOCK_K 32 -e HALIAX_RAGGED_DOT_TRITON_NUM_WARPS 8 -- uv run --frozen --package marin-levanter --extra gpu python experiments/grug/moe/benchmark_ep_ring_data_axis.py --microbatch-size 32 --sequence-length 4096 --hidden-dim 2560 --intermediate-dim 1280 --num-experts 64 --top-k 4 --capacity-factor 1.0 --treatment-data-axis-size 4 --warmup 10 --iterations 50 --microbatches-per-step 256 --layers-per-stage 6 --baseline-mfu 18.2583 --baseline-step-seconds 81.037785 --interstage-speedup 1.0179 --promotion-mfu 20.0 --output both`
- Config:
  - The Triton grouped matmul keeps BF16 operands, forward output, and input cotangent. Only the W13/W2 cotangents use FP32 output buffers.
  - The timed VAG includes donated FP32 accumulator reads and writes in both arms. Gradient reduce-scatter remains once per step in FP32; BF16 compute-weight materialization remains once per step.
  - Job `/dlwh/ep-ring-data4-ep2-fp32acc-r18-20260727` runs on one H100x8 node in RNO2A with 10 warmups, 50 alternating samples, and no retries.
- Promotion gate:
  - Loss/output and every gradient leaf relative-L2 must be at most `0.002`; dropped routes/counts must match exactly.
  - Local VAG must contain zero data-axis collectives. FP32 reduce-scatter and BF16 all-gather may occur only at the step boundary.
  - The direct EP2/data4 projection composed with the separately measured `1.0179x` inter-stage FP8 gain must exceed `20.0` MFU.
- Next action:
  - If all gates pass, implement the same combination in a reduced JaxPP run before L24. Otherwise record the hard negative and stop this branch of optimization.

### 2026-07-27 01:01 PDT - Fix traced operator leaf and relaunch
- Commit Hash: `1940a7e866a6e664d237435c2b9cf28140d4e9bc`.
- Result:
  - `/dlwh/ep-ring-data4-ep2-fp32acc-r18-20260727` failed during JAX lowering after `22.39s` with `TypeError: Argument '<function _fp32_weight_gradient_ragged_dot ...>' of type '<class 'function'>' is not a valid JAX type`.
  - `MoeRaggedDotOps` passed the plain function through the ring custom-VJP identity as a dynamic pytree leaf. No GPU compilation, timing, or numerical result occurred. The task is terminal and no resource remains live.
- Fix:
  - Replace the plain function with a stateless `eqx.Module` callable. Its pytree has no dynamic leaves, so stateful operator handling remains intact without tracing a Python function.
  - Focused tests report `19 passed, 1 skipped`; changed-file pre-commit passes.
- Relaunch:
  - `/dlwh/ep-ring-data4-ep2-fp32acc-r19-20260727` uses the same resources, geometry, environment, warmups, samples, and promotion gates as r18, with no retries.
- Next action:
  - Capture the first compiled numerical and timing result. Do not promote on setup recovery alone.

### 2026-07-27 01:04 PDT - FP32 gradients pass parity but compose to 19.885 MFU
- Commit Hash: `1940a7e866a6e664d237435c2b9cf28140d4e9bc`.
- Command: `/dlwh/ep-ring-data4-ep2-fp32acc-r19-20260727` used the exact command and gates recorded above.
- Results:
  - The task succeeded in `44.16s` with no retries or preemptions; no resource remains live.
  - Loss was exactly `0.07439123839139938` in both arms and both dropped zero assignments.
  - Relative-L2 was `3.27972e-6` for output, `3.03248e-5` for x-gradient, `0` for combine-weight-gradient, `0.00165884` for W13, and `0.00165597` for W2. Every metric passes the accepted `0.002` threshold.
  - Forward median improved from `8.719ms` to `5.986ms`.
  - VAG plus donated FP32 accumulation improved from `18.823ms` to `15.348ms`.
  - FP32 gradient sync and BF16 weight materialization took `4.073ms` and `2.340ms`, or `0.025048ms` amortized per microbatch.
  - StableHLO reports zero data-axis collectives in local VAG, two data-axis reduce-scatters in gradient sync, and two data-axis all-gathers in weight materialization.
  - Compiler peak estimates were `3.482GiB` for EP8 local VAG, `4.299GiB` for EP2/data4 local VAG, `2.637GiB` for sync, and `1.221GiB` for materialization.
  - The direct projection is `75.740070s`, `1.069946x`, and `19.535395` MFU.
  - Composing the separately measured `1.0179x` inter-stage FP8 gain gives `74.408164s`, `1.089098x`, and `19.885079` MFU.
- Interpretation:
  - FP32 expert-weight cotangents solve the EP2/data4 numerical discrepancy under the accepted `0.2%` policy.
  - The measured accumulator cost removes the available headroom. The composed projection misses 20 MFU by `0.114921` points, so `promotable=false`.
  - Do not launch reduced JaxPP or L24 from this result. Stop replicated-expert topology narrowing unless a separate measured kernel change recovers at least `0.58%` end-to-end.
- Next action:
  - Seal this path as a numerical success and performance negative. Re-rank only independent optimizations with direct evidence at the exact geometry.

### 2026-07-27 01:25 PDT - Launch fused FP32 accumulator gate
- Hypothesis:
  - The r19 treatment writes a 1.2GB FP32 W13/W2 gradient and then reads it again for a separate accumulator add on every layer-microbatch.
  - Adding the prior accumulator inside the Triton weight-gradient epilogue removes approximately 2.4GB of avoidable write/read traffic per GPU. Recovering at least `0.283ms` from the `15.348ms` r19 treatment median is sufficient to compose above 20 MFU.
- Commit Hash: `d5abcea1ab`.
- Command:
  - `uv run iris --config lib/iris/config/cw-rno2a.yaml job run --no-wait --enable-extra-resources --gpu H100x8 --cpu 32 --memory 256GB --disk 128GB --timeout 1800 --max-retries 0 --priority interactive --extra gpu --sync-package marin-levanter --job-name ep-ring-data4-ep2-fused-fp32acc-r20-20260727 -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.70 -e TF_GPU_ALLOCATOR cuda_malloc_async -e HALIAX_RAGGED_DOT_TRITON_BLOCK_K 32 -e HALIAX_RAGGED_DOT_TRITON_NUM_WARPS 8 -- uv run --frozen --package marin-levanter --extra gpu python experiments/grug/moe/benchmark_ep_ring_data_axis.py --microbatch-size 32 --sequence-length 4096 --hidden-dim 2560 --intermediate-dim 1280 --num-experts 64 --top-k 4 --capacity-factor 1.0 --treatment-data-axis-size 4 --warmup 10 --iterations 50 --microbatches-per-step 256 --layers-per-stage 6 --baseline-mfu 18.2583 --baseline-step-seconds 81.037785 --interstage-speedup 1.0179 --promotion-mfu 20.0 --fuse-fp32-weight-gradient-accumulation --output both`
- Implementation:
  - The custom VJP returns the ordinary BF16 forward output plus a numerically zero FP32 token. The loss adds the token with coefficient exactly one after current-microbatch normalization.
  - In backward, the token cotangent scales the prior FP32 accumulator, which the Triton W13/W2 gradient epilogues add to the new FP32 gradient. The enclosing JIT donates both accumulators, and Pallas requests input/output aliasing.
  - Focused tests report `23 passed, 1 skipped`; changed-file pre-commit including Pyrefly passes. Token scales `0`, `0.25`, `1`, and `2` preserve VJP linearity in the CPU reference test.
- Promotion gate:
  - Loss/output and every gradient leaf relative-L2 must be at most `0.002`; drops remain exact.
  - Local VAG must contain zero data-axis collectives. FP32 reduce-scatter and BF16 all-gather remain step-boundary operations.
  - The measured direct projection composed with the independently validated `1.0179x` inter-stage FP8 gain must exceed `20.0` MFU before a reduced JaxPP run.
- Next action:
  - Babysitter `019fa2ad-61ec-7083-8940-8def207885fa` owns the run through terminal state. If the direct gate passes, test a reduced JaxPP integration before any exact L24 allocation.

### 2026-07-27 01:27 PDT - Fused accumulator composes to 20.014 MFU
- Commit Hash: `d5abcea1ab`.
- Result:
  - `/dlwh/ep-ring-data4-ep2-fused-fp32acc-r20-20260727` succeeded in `64.3s` with exit `0`, no retries or preemptions, and no live resources.
  - Loss was exact. Output, x-gradient, combine-gradient, W13-gradient, and W2-gradient relative-L2 was `3.7290e-6`, `1.6161e-5`, `3.5092e-7`, `0.00165882`, and `0.00165596`. Every metric passes the accepted `0.002` threshold; drops were exactly zero in both arms.
  - Forward median improved from `8.74646ms` to `6.01689ms`.
  - Local VAG plus donated FP32 accumulation improved from `18.85226ms` to `15.05985ms`. This recovers `0.28841ms` versus the unfused r19 treatment and clears the modeled `0.283ms` requirement by approximately `0.005ms`.
  - FP32 gradient sync and BF16 weight materialization took `4.06902ms` and `2.34195ms`, or `0.025043ms` amortized per m256 layer-microbatch.
  - StableHLO reports zero data-axis collectives in local VAG, two data-axis reduce-scatters in sync, and two data-axis all-gathers in materialization.
  - Compiler alias size for treatment VAG is `1,258,291,200` bytes. Peak and temporary estimates are `4,615,717,188` and `2,475,840,792` bytes. No Pallas alias, `vmap`, compiler, or OOM failure occurred.
  - Direct projection is `75.25110s` and `19.66233` MFU. Composing the independently validated `1.0179x` inter-stage FP8 gain gives `73.92780s` and `20.01429` MFU.
- Interpretation:
  - The direct exact-geometry gate is promotable, but its composed margin is only `0.01429` MFU (`0.071%`). Treat the result as evidence to attempt reduced JaxPP integration, not as proof that exact L24 exceeds 20.
  - The fused epilogue saved the required memory traffic without changing the accepted numerical result. The next risk is lifecycle integration: JaxPP must keep replicated BF16 expert compute weights and donated FP32 data-local accumulators across microbatches, then reduce-scatter/materialize once per optimizer step.
- Next action:
  - Design and run the smallest reduced JaxPP gate. Require numerical sanity, correct per-stage sharding and accumulator lifecycle, and measured throughput improvement before any exact L24 run.

### 2026-07-27 02:06 PDT - Reduced JaxPP integration checkpoint
- Commit Hash: `5ef99566394b9e343f7d17af57d767218886d7b8`.
- Implementation:
  - Explicit-MPMD `std_1f1b` tasks materialize Ring expert compute weights once per stage and step with `P("expert", None, None)`.
  - Each backward task carries ordinary-shaped, FP32 W13/W2 accumulators with the same sharding. Their physical values may diverge across `data`; JaxPP passes the buffers directly between same-stage tasks.
  - The fused Triton VJP adds each new expert-weight gradient into the prior accumulator. Ordinary gradient leaves continue through the existing tree sum.
  - The optimizer-boundary task performs one explicit `psum_scatter` over `data`, using W13 scatter dimension 1 and W2 scatter dimension 2, averages over microbatches, restores the expert leaves, and applies the existing optimizer.
  - Configuration rejects non-Ring MoE, `replica_axis_size != 1`, missing expert/data parallelism, grouped stage tasks, alternate explicit schedules, and research FP8 expert GEMMs.
- Validation:
  - Targeted model, Ring, explicit-stage, pipeline-wire, config, and launcher tests report `79 passed, 21 skipped`.
  - Follow-up topology/config tests report `8 passed`.
  - `./infra/pre-commit.py --changed-files --fix` passes, including Pyrefly.
  - A context-isolated review found no pre-sync reshard/host read, microbatch-scaling error, dropped gradient leaf, optimizer-tree mismatch, or explicit-MPMD lifecycle error for the accepted topology.
- Reduced gate:
  - L4, one layer on each of four physical stages, d2560/i1280, e64/top-k4, global batch 128, four microbatches, sequence 4096, H100x8 per stage, expert 2/data 4, FP8 inter-stage wire, CuTe FA4, Ring MoE, and fused FP32 data-local expert gradients.
  - The gate is for compilation, lifecycle, sharding, and numerical sanity. It is not a throughput claim for L24.
- Next action:
  - Submit the L4 gate on `cw-rno2a` from `5ef9956639` with a dedicated babysitter. Run a matched L8 control/treatment only after L4 reaches training metrics without compiler, donation, VMA, or optimizer-tree failure.

### 2026-07-27 02:08 PDT - Launch L4 fused-accumulator lifecycle gate
- Commit Hash: `73c33a5f7d34ae3f0542236a21add581b932a9cc`.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=true TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r1-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 1,1,1,1 --microbatches 4 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 4 --experts 64 --top-k 4 --vocab-size 8192 --batch 128 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 6 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-090740` on `cw-rno2a`.
  - Babysitter `019fa2d4-b845-7561-a8f3-0fff42a00c62` owns the parent and generated child through terminal state.
- Gate:
  - Require all four ranks to complete setup, lower, compile, and six finite training steps.
  - Reject compiler, VMA, donation, accumulator synchronization, optimizer-tree, OOM, or cross-rank lifecycle failure.
  - Do not infer target throughput from L4. A pass promotes only to a matched L8 EP2/data4 ordinary-versus-fused A/B.

### 2026-07-27 02:17 PDT - L4 r1 rejects expert-axis scalar slice
- Result:
  - Parent `/dlwh/iris-run-job-20260727-090740` and its child are terminal killed with no live resource.
  - All four ranks completed setup and entered `explicit_mpmd_train_step.lower`; none reached compile or execute.
  - Every rank failed in `expert_materialization_completion_token` with `ShardingTypeError: slicing on sharded dims where out dim (1) is not divisible by mesh axes (2) with spec (expert)`.
  - The token indexed expert `0` from W13/W2 before reduction. This collapsed the size-2 expert-sharded dimension to size 1.
  - Iris started an automatic retry. The babysitter stopped only this parent and child; later connection errors are shutdown noise.
  - W&B [jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r1-20260727](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r1-20260727) is finished with config metadata and no training metrics.
- Fix:
  - Commit `b881d9035273c4bc456a318be72bd0115e70883e` changes each dependency-token slice from expert index `0` to `[:, 0, 0]`, preserving the expert dimension until `jnp.sum`.
  - Focused explicit-stage/config/wire tests report `57 passed`; changed-file precommit including Pyrefly passes.
- Interpretation:
  - This is a deterministic sharding-contract bug in the new materialization dependency token. There is no evidence yet about compilation, VMA, donation, synchronization, optimizer state, memory, or throughput.
- Next action:
  - Relaunch the unchanged L4 gate as r2 from `b881d90352`. Require terminal six-step success before L8.

### 2026-07-27 02:18 PDT - Launch L4 lifecycle gate r2
- Command: unchanged from r1 except run ID `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r2-20260727`.
- Code:
  - Functional fix `b881d9035273c4bc456a318be72bd0115e70883e`.
  - Launch snapshot `72724c139e3cf560e2cff3b8c50038db590c7052`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-091630`.
  - Babysitter `019fa2dc-c146-7c10-b182-7c1de68dc32b`.
- Gate: unchanged six-step, four-rank lower/compile/execute requirement. L8 remains blocked until terminal success.

### 2026-07-27 02:29 PDT - L4 r2 exposes data-sharded hidden slice
- Result:
  - Parent `/dlwh/iris-run-job-20260727-091630` and child are terminal killed with no live retries.
  - All four ranks entered `explicit_mpmd_train_step.lower`; none reached compile or execute, and no training metric was emitted.
  - Every rank failed with `ShardingTypeError: slicing on sharded dims where out dim (1) is not divisible by mesh axes (4) with spec (data)`.
  - r1's fix preserved `expert` but `[:, 0, 0]` still collapsed the data-sharded hidden dimension.
  - W&B [r2](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r2-20260727) is finished with no metrics.
- Fix:
  - Commit `2349bae4452f9ae8e2ff3d40778ffaf5aad61518` uses `w_gate[:, :, 0]`, `w_up[:, :, 0]`, and `w_down[:, 0, :]`. The dependency probe now preserves both potentially sharded dimensions until reduction while reading only an `E x D` slice per weight.
  - Focused tests report `57 passed`; changed-file precommit including Pyrefly passes.
- Next action:
  - Run the unchanged L4 gate as r3. L8 remains blocked.

### 2026-07-27 02:40 PDT - L4 r3 reaches accumulator sharding contract
- Launch:
  - Parent `/dlwh/iris-run-job-20260727-092646`, unchanged L4 configuration with run ID suffix `r3`.
- Result:
  - All four ranks entered explicit-MPMD lowering, then failed before compile with `in_specs P('expert', None, None) does not match input P(None, None, None)` for the FP32 W13 accumulator.
  - The MPMD initialization task advertised expert sharding, but its `jnp.zeros` result remained physically replicated when consumed by the accumulating Ring `shard_map`.
  - Parent, child, and retry are terminal killed; no resource remains live. No loss, MFU, or step-time metric was emitted.
  - W&B [r3](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r3-20260727) contains config metadata only and remained stale `running` after forced shutdown.
- Fix:
  - Commit `10a5f44f9672e1475eb3542208e4afcd07733c35` explicitly reshards only the newly constructed all-zero accumulators to `P("expert", None, None)`.
  - An eight-device explicit-mesh CPU probe confirms the result has expert sharding and local expert dimension 32. `with_sharding_constraint` is invalid for this explicit-axis conversion, while `jax.sharding.reshard` succeeds.
  - No generic reshard was added after accumulators diverge across `data`.
  - Focused tests report `57 passed`; changed-file precommit including Pyrefly passes.
- Next action:
  - Run L4 r4. L8 remains blocked until six finite steps.

### 2026-07-27 02:51 PDT - L4 r4 identifies missing in-task weight materialization
- Launch:
  - Parent `/dlwh/iris-run-job-20260727-093433`, unchanged L4 configuration with run ID suffix `r4`.
- Result:
  - All four ranks entered explicit-MPMD lowering, then failed before compile with accumulator input sharding `P("expert", "data", "model")` instead of required `P("expert", None, None)`.
  - The initial explicit zero reshard changed the first observed failure, but the accumulator returned through the expert-weight cotangent inherited canonical parameter sharding.
  - Parent, child, and retry are terminal killed; no resource remains live. No loss, MFU, or step metric was emitted.
  - W&B [r4](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r4-20260727) is finished with config metadata only.
- Root cause and fix:
  - `materialize_compute_params` was an identity function that relied on the MPMD task's declared output sharding. Explicit-axis lowering did not perform the intended expert-weight reshard inside the task, as also shown by r1/r2's token seeing canonical data sharding.
  - Commit `f506b057f1d7e7c45e4513e02985da313c9b397c` explicitly reshards Ring expert weights to `P("expert", None, None)` inside the once-per-step materialization task. Sonic retains its existing fully replicated `P()` target.
  - This is the intended once-per-step compute-weight all-gather. It should keep the custom VJP's expert-weight cotangent data-local instead of mapping it back to canonical parameter sharding.
  - Focused tests report `57 passed`; changed-file precommit including Pyrefly passes.
- Next action:
  - Run L4 r5. L8 remains blocked.

### 2026-07-27 03:03 PDT - L4 r5 clears sharding and reaches gradient reconstruction
- Launch:
  - Parent `/dlwh/iris-run-job-20260727-094731`, unchanged L4 configuration with run ID suffix `r5`.
- Result:
  - All four ranks entered explicit-MPMD lowering. Rank 0 then failed while restoring synchronized expert gradients into the ordinary gradient tree; ranks 1-3 were terminated as coscheduled siblings.
  - The failure was `eqx.tree_at` attempting to replace `None` W13/W2 leaves without treating `None` as a leaf.
  - No rank reached compile or execute. Parent and child are terminal with no retry or failed resource live.
  - W&B [r5](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r5-20260727) is finished with config metadata only.
- Fix:
  - Commit `28b3f757168bcd8ebf68a24c4a2d519205ca1f9a` passes `is_leaf=lambda value: value is None` when restoring expert leaves.
  - A direct regression now removes, extracts, and restores every stage expert-gradient leaf, checking exact W13 split and W2 values.
  - The expanded focused suite reports `58 passed`; the direct regression passes independently; changed-file precommit including Pyrefly passes.
- Interpretation:
  - r5 passed the prior materialization and accumulator-sharding boundaries. The new failure is local gradient-tree plumbing, not a repeated sharding mismatch.
- Next action:
  - Run L4 r6. L8 remains blocked.

### 2026-07-27 03:21 PDT - L4 r6 deadlocks on the materialization transfer chain
- Launch:
  - Parent `/dlwh/iris-run-job-20260727-100240` used the unchanged L4 lifecycle configuration with run ID `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r6-20260727`.
  - Babysitter `019fa307-01d2-7090-b1fc-3d17ac061dd4` owned the run through terminal cleanup.
- Result:
  - All four ranks completed `explicit_mpmd_train_step.lower` and local task compilation. No rank completed the first `eval_local`; the run emitted zero of six training steps and no loss, step-time, or MFU metric.
  - Two unchanged watchdog samples showed rank 0 in `enqueue_nccl_transfer_group`, rank 1 in `recv_done_impl`, and ranks 2-3 after compiling `apply_task`. GPU utilization was 100% on ranks 0, 2, and 3 and 0% on rank 1, with approximately 59GB allocated per GPU.
  - No log advanced for more than ten minutes after DIME stream creation. The parent and child were killed, retries were suppressed, and no live resource remains.
  - W&B [r6](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r6-20260727) remained stale `running` with config metadata and zero history after forced shutdown.
- Interpretation:
  - r6 clears lowering, task compilation, weight materialization sharding, accumulator sharding, and gradient-tree reconstruction. The first execution deadlock is localized to the scalar DIME transfer chain that serializes stage materialization.
  - That chain was added for Sonic's staged FSDP materialization. Fused Ring already has a local dependency from each stage's materialized parameters into its forward and backward tasks; it does not need a cross-stage scalar token.
- Fix:
  - Commit `83cb355895` gives every fused-accumulator stage the local step token and omits only fused Ring's cross-stage completion-token transfers. Sonic retains the serialized chain unchanged.
  - The focused accumulator, explicit-stage, and pipeline-wire suite reports `58 passed`; changed-file pre-commit including Pyrefly passes.
- Next action:
  - Relaunch the unchanged L4 gate as r7. Require six finite steps before the matched L8 ordinary-versus-fused A/B.

### 2026-07-27 03:23 PDT - Launch L4 lifecycle gate r7 without the scalar chain
- Code:
  - Functional fix `83cb355895` removes fused Ring's cross-stage materialization token transfers.
  - Launch snapshot `bbdddbea64` records the r6 failure and preserves Sonic's serialized materialization behavior.
- Command:
  - The L4 command is unchanged from r1-r6 except run ID `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r7-20260727`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-102233` on `cw-rno2a`.
  - Babysitter `019fa319-4896-7593-9a17-db9b1e214fdd` owns the run through terminal state.
- Gate:
  - Require all four ranks to complete lower, compile, first execution, and six finite training steps.
  - A pass promotes only to a matched L8 EP2/data4 ordinary-versus-fused throughput A/B. Exact L24 remains blocked.

### 2026-07-27 03:33 PDT - L4 r7 passes the fused accumulator lifecycle gate
- Result:
  - Parent `/dlwh/iris-run-job-20260727-102233`, its child, and all four rank tasks succeeded with exit `0`. Each rank completed in `2m40.94s`; no retry, preemption, failed task, or live resource remains.
  - Every rank completed lowering, local task compilation, DIME initialization, FP8 inter-stage transfers, fused expert-gradient accumulation, fused expert update, and execution.
  - Training completed `6/6` batches and `3,145,728` tokens. W&B [r7](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l4-e64k4-b128-s4096-p4m4-r7-20260727) is `finished` at `_step=5` and `global_step=5`.
  - W&B retained finite loss and performance rows for steps 2-5: loss declined `8.816470 -> 8.622681`; step time was `0.297718/0.305104/0.297947/0.296806s`; MFU was `13.5971/13.2679/13.5867/13.6389`.
  - Mean and p50 MFU were `13.5459` and `13.5971`; final throughput was `1,766,435 tokens/s`. Steps 0-1 completed but W&B retained hook/loading rows instead of loss rows.
  - There was no OOM, compiler, NCCL, DIME, traceback, or watchdog failure. CUDA VMM fabric-handle fallback and PJRT shutdown messages occurred after successful execution.
- Interpretation:
  - The fused FP32 data-local accumulator now has a complete reduced JaxPP lifecycle: once-per-step expert materialization, microbatch-local accumulation, step-boundary synchronization, optimizer update, and repeated state execution all complete.
  - L4 MFU is a functional-gate metric and is not comparable to the exact L24 target. The decision-relevant next measurement is a matched L8 EP2/data4 A/B.
- Next action:
  - Run 20-step ordinary and fused arms at L8, stage split `2,2,2,2`, batch 512, 16 microbatches, and sequence 4096. Promote only if fused materially improves steady-state MFU without a lifecycle regression.

### 2026-07-27 03:34 PDT - Launch matched L8 ordinary-versus-fused A/B
- Snapshot: `10cd21f5ae` records the passing L4 lifecycle gate. Both arms launch from the same clean workspace bundle.
- Shared configuration:
  - L8 d2560/i1280, e64/top-k4, physical/logical stages 4/4 with split `2,2,2,2`, batch 512, 16 microbatches, sequence 4096, 20 steps.
  - Four H100x8 stage nodes per arm, expert axis 2 and data axis 4, Ring MoE, CuTe FA4 attention, Triton ragged dot with block K 32 and eight warps, FP8 inter-stage wire, `save_moe`, XLA memory fraction `0.70`.
- Arms:
  - Ordinary BF16 expert-gradient control: parent `/dlwh/iris-run-job-20260727-103109`, run `jaxpp-rno2a-ring-ep2d4-ordinary-fp8-l8-e64k4-b512-s4096-p4m16-ab-20260727`, babysitter `019fa321-4188-7530-a4b2-dffb51b3c7c0`.
  - Fused FP32 data-local treatment: parent `/dlwh/iris-run-job-20260727-103121`, run `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l8-e64k4-b512-s4096-p4m16-ab-20260727`, babysitter `019fa321-61b0-7081-b89a-f2ab8750365d`.
- Interpretation:
  - The ordinary EP2/data4 arm is a topology-matched throughput control only; direct evidence already rejects its BF16 expert-weight gradients under the fixed `0.002` relative-L2 policy.
  - Compare steady-state steps 2-19 when telemetry permits. Exact L24 promotion requires the fused arm to complete cleanly and show enough measured gain to support a credible target above 20 MFU.

### 2026-07-27 03:45 PDT - Fused L8 succeeds; ordinary control deadlocks
- Fused treatment:
  - Parent `/dlwh/iris-run-job-20260727-103121`, child, and all four ranks succeeded with exit `0`. The ranks completed in `3m19.4s`; no retry, preemption, failure, or live resource remains.
  - All ranks compiled and completed `20/20` iterations. Maximum planned task memory was `10.566/10.118/10.118/10.657 GiB` across ranks 0-3. There was no OOM, compiler, NCCL, DIME, traceback, or watchdog failure.
  - W&B [fused L8](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l8-e64k4-b512-s4096-p4m16-ab-20260727) is `finished` at step 19 with `41,943,040` tokens.
  - Steps 2-19 had mean/p50/p90 MFU `16.1404/16.1365/16.1814`, mean/p50/p90 step time `1.95482/1.95528/1.95790s`, and mean throughput `1.07281M tokens/s`. Loss declined from `8.530657` to `6.300170`.
- Ordinary control:
  - Parent `/dlwh/iris-run-job-20260727-103109` completed lowering and compilation on all ranks but stalled in the first `eval_local` before a training metric.
  - Two stable stack samples showed ranks 0-2 in `eval_local -> apply_task` at 100% GPU utilization and rank 3 in `eval_local -> recv_done_impl -> dime2.enqueue_wait` at 0%. Memory was approximately `62.1/81.6 GB` per GPU.
  - The parent and child were stopped with no retry or live resource. W&B [ordinary L8](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-ordinary-fp8-l8-e64k4-b512-s4096-p4m16-ab-20260727) has zero history and remained stale `running`.
- Interpretation:
  - The matched A/B has no ordinary throughput denominator. The control-specific execution deadlock is a negative result for ordinary EP2/data4, not evidence of fused speedup.
  - The fused result is `0.0242` MFU points (`0.15%`) above the existing valid L8 EP8 group-size-one reference at `16.116235`; this cross-topology comparison is not a material L8 gain.
  - The independent exact six-layer-stage projection remains `20.01429` MFU, only `0.01429` points above the target. The L8 result neither validates nor decisively falsifies that projection because fixed task overhead and two-layer stages differ from L24.
- Next action:
  - Run one exact L24 d2560/e64/top-k4/b512/m16/s4096 fused measurement as the decisive gate. Require sustained mean MFU above 20; otherwise record the path as a performance negative.

### 2026-07-27 03:45 PDT - Launch exact L24 fused performance gate
- Snapshot: `f923322aa8` records the L8 result and the missing matched denominator.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=true TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --run-id jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-exact-r1-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 6,6,6,6 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 24 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 20 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-104406` on `cw-rno2a`.
  - Babysitter `019fa32c-eb14-7800-b742-f529989b0e23` owns the run through terminal state.
- Gate:
  - Require all four ranks and 20 finite steps, then compute steps 2-19 mean/p50/p90 MFU and step time.
  - Success is sustained mean MFU strictly above `20.0`. Compare with the exact baseline `18.2583` MFU and `81.037785s`; do not promote a modeled projection.

### 2026-07-27 04:02 PDT - Exact L24 fused path is a hard performance negative
- Result:
  - Parent `/dlwh/iris-run-job-20260727-104406`, child, and all four ranks succeeded with exit `0`. The run completed `20/20` iterations and `41,943,040` tokens with no retry, failure, preemption, or live resource.
  - W&B [exact L24 fused](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-exact-r1-20260727) is `finished` at step 19. Loss declined from `8.410286` to `5.776348`; every retained loss was finite.
  - Steps 2-19 had mean/p50/p90 MFU `15.9514/16.2516/16.3131`, mean/p50/p90 step time `5.848868/5.721756/6.078428s`, and mean/p50/p90 throughput `359,753/366,522/367,911 tokens/s`. No retained step reached 20 MFU.
  - Maximum planned task memory was `27.5969/26.5450/26.5451/27.3905 GiB` across ranks 0-3. There was no OOM, compiler, NCCL, DIME, traceback, or watchdog failure.
- Comparison:
  - Mean MFU is `2.3069` points (`12.63%`) below the exact `18.2583` baseline and `4.0486` points (`20.24%`) below the target.
  - Mean throughput is `54,306 tokens/s` (`13.12%`) below the `414,059.1 tokens/s` baseline.
  - Normalized from batch 512 to the baseline batch 8192, mean step time is `93.581886s`, `12.544101s` (`15.48%`) slower than `81.037785s`.
- Interpretation:
  - The exact direct projection of `20.01429` MFU is falsified. It omitted pipeline-level materialization, synchronization, scheduling, or optimizer costs totaling approximately `12.5s` per batch-8192-equivalent step.
  - Fused FP32 data-local accumulation is a numerical and lifecycle success but a hard end-to-end performance negative at the exact L24 target.
- Next action:
  - Profile the exact fused graph and attribute the regression to named stage tasks, collectives, and idle gaps before changing another kernel or topology. Do not tune the fused path without measured evidence.

### 2026-07-27 04:03 PDT - Launch exact L24 fused profile
- Snapshot: `598fbcabdb` records the exact performance negative.
- Command:
  - The exact L24 fused command is unchanged except run ID `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r1-20260727`, 12 steps, `--profiler-steps 2`, and memory-plan logging disabled. The profiler starts at step 8.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-105805` on `cw-rno2a`.
  - Babysitter `019fa339-c357-72a0-89b0-916b15597a36` owns terminal monitoring, profile download, and structured summary generation.
- Gate:
  - Require successful profiler start, stop, upload, and terminal training state.
  - Attribute the exact regression to named stage tasks, compute/communication/stall shares, collectives, and idle gaps. Compare with `scratch/jaxpp_profile_b96_gpipe_artifact_summary.json` or the best available exact baseline profile before selecting another optimization.

### 2026-07-27 04:20 PDT - Profile r1 capture succeeds but upload targets the wrong S3 backend
- Result:
  - Parent `/dlwh/iris-run-job-20260727-105805` reached profiler start at step 8 and stopped after step 9.
  - Upload to `s3://marin-na/tmp/ttl=30d/xprof/...` failed with `InvalidRegion: Region does not match`. Rank 0 raised `RuntimeError: Failed to upload XProf profile`, leaving the distributed job stranded after the profile window.
  - The parent and all four ranks were stopped with no retry or live resource. W&B [profile r1](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r1-20260727) is `finished` at step 9 with no profiler artifact.
  - Steps 2-7 before profiling averaged `16.2476` MFU and `5.7232s`; profiled steps 8-9 averaged `15.8469` MFU and `5.8679s`.
- Root cause:
  - The launcher configures `AWS_ENDPOINT_URL` for the selected CoreWeave cluster, here `cwlota.com`, but its default prefix remains `s3://marin-na/marin`, which is Cloudflare R2. The CoreWeave endpoint cannot upload to the R2 bucket.
  - `cw-rno2a` configures `MARIN_PREFIX: s3://marin-us-east-02a/marin`, matching the cluster's object-store endpoint and credentials. Passing that prefix explicitly keeps the XProf TTL artifact on the same backend and region.
- Next action:
  - Relaunch the unchanged profile as r2 with `--prefix s3://marin-us-east-02a/marin`. Require artifact upload before profile analysis.

### 2026-07-27 04:21 PDT - Relaunch exact fused profile on the CoreWeave bucket
- Snapshot: `2bb024260f` records the r1 upload failure and backend mismatch.
- Command:
  - The profile r1 command is unchanged except run ID `jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r2-20260727` and `--prefix s3://marin-us-east-02a/marin`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-111534`.
  - Babysitter `019fa349-bb81-7ff1-86d2-d4561bc92926` owns terminal monitoring, artifact download, and structured comparison.
- Gate:
  - Require profiler start/stop and a readable XPlane artifact under the CoreWeave TTL path.
  - Compare the fused trace with the stored same-shape FA4 and GPipe summaries before selecting another optimization.

### 2026-07-27 04:47 PDT - Exact fused profile attributes the regression to collective fanout
- Result:
  - Parent `/dlwh/iris-run-job-20260727-111534`, child, and all four ranks succeeded with exit `0`, no retry, failure, preemption, or live resource. W&B [profile r2](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r2-20260727) is `finished` at step 11.
  - Profiling covered steps 8-10 and uploaded to `s3://marin-us-east-02a/tmp/ttl=30d/xprof/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r2-20260727/plugins/profile/steps-8-to-10`.
  - W&B [profiler artifact v0](https://wandb.ai/marin-community/marin_moe/artifacts/profiler/jaxpp-rno2a-ring-ep2d4-fusedacc-fp8-l24-e64k4-b512-s4096-p4m16-profile-r2-20260727-profiler/v0) is 179,036,534 bytes with digest `ba01a6e851cb8621724fe6ca6f0fbb0b`.
  - Structured outputs are `scratch/jaxpp_rno2a_fusedacc_l24_profile_r2_summary.json`, `scratch/jaxpp_rno2a_fusedacc_l24_profile_r2_timeline_summary.json`, and `scratch/jaxpp_rno2a_fusedacc_l24_profile_r2_report.md`.
- Attribution:
  - Global timeline shares are `45.73%` compute, `46.66%` communication, and `7.61%` stall. The same-shape FA4 reference is `52.79/39.61/7.59%`; stall is unchanged while communication replaces useful compute.
  - Each six-layer `backward_accumulating` contains twelve FP32 all-reduces. `AllReduce_Sum_f32_RING_LL` averages `3.114ms`, and aggregate all-reduce time per profiled step is `4.65x` the reference.
  - `SendRecv` latency improved to `14.43ms` from `25.83ms`, but transfer count grew from 280 to 856 per step (`3.06x`). All-gathers grew from 5,120 to 12,448 (`2.43x`).
  - Ragged-dot improved slightly to `1.825ms` from `1.852ms`; CuTe FA4 is unchanged at `1.525ms` versus `1.530ms`. SendRecv and all-gather pre-op gaps also improved per event.
  - Excess communication occupancy is approximately `0.433s` per batch-512 step, or `6.92s` batch-8192-normalized, explaining about 55% of the measured `12.544s` regression. Remaining time overlaps pipeline/DIME dependency amplification.
- Interpretation:
  - The direct benchmark's local VAG runs inside a manual outer `shard_map` and emits explicit `P("data", "expert", ...)` accumulator buffers. The pipeline differentiates through the inner Ring `shard_map`; its transpose reconciles data-replicated expert weights with per-microbatch all-reduces.
  - The highest-confidence fix is to carry genuinely data-sharded local cotangents across the JaxPP task boundary and synchronize only at optimizer update. Bundling FP8 DIME payloads is second; FA4, ragged-dot, and GPipe are closed by this trace.
- Next action:
  - Implement the smallest manual-data-axis gradient boundary and prove zero data-axis collectives in local backward plus one step-boundary sync per weight before another GPU run.

### 2026-07-27 05:00 PDT - Move fused Ring VJP inside the manual shard map
- Commit Hash: `6593a34edc`.
- Implementation:
  - `moe_mlp_accumulating_weight_gradient` now has a global custom VJP whose backward executes the local Ring pullback inside `shard_map`.
  - Expert-weight cotangents remain intentionally non-VMA across `data` until the existing optimizer-boundary synchronization. This bypasses the automatic transpose of the inner Ring `shard_map`, which inserted one data-axis all-reduce per W13/W2 layer and microbatch.
  - The optimizer-boundary synchronization is unchanged and exposed as a testable helper.
- Validation:
  - Four-device CPU HLO for local backward contains zero all-reduces and zero reduce-scatters. Step-boundary HLO contains exactly two data-axis reduce-scatters and zero all-reduces.
  - Synchronized W13/W2 gradients match ordinary Ring at relative-L2 `0.00175065/0.00155534`; outputs and dropped-route counts are exact.
  - Expanded changed-path tests report `71 passed`; focused post-cleanup tests report `10 passed`. Changed-file pre-commit including Pyrefly passes.
  - `tests/test_grug_moe_jaxpp_input_placement.py` remains independently stale because it imports removed helper `_localize_automatic_jaxpp_input_shardings`; it was not selected.
- GPU gate:
  - Parent `/dlwh/iris-run-job-20260727-115941` runs the unchanged L4 lifecycle shape with run ID `jaxpp-rno2a-ring-ep2d4-manualvjp-fp8-l4-e64k4-b128-s4096-p4m4-r8-20260727`.
  - Babysitter `019fa372-2131-74a3-afd2-cde99d9c10f8` owns terminal state and cleanup.
- Next action:
  - Require six finite L4 steps and no custom-VJP/JaxPP compile or execution failure. A pass promotes to an L8 performance/profile gate.

### 2026-07-27 05:07 PDT - Manual-VJP L4 lifecycle gate passes
- Result:
  - Parent `/dlwh/iris-run-job-20260727-115941`, its child, and all four rank tasks succeeded with exit `0`. Training completed `6/6` iterations with no retry, preemption, failure, or live resource.
  - W&B [L4 r8](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-manualvjp-fp8-l4-e64k4-b128-s4096-p4m4-r8-20260727) is finished. Step 2 was a startup outlier at `3.9003` MFU and `1.037892s`; steady steps 3-5 were `14.0139/14.0273/14.0999` MFU.
  - Steady steps 3-5 averaged `14.0470` MFU and `0.288184s`, or approximately `1.819M tokens/s`. This is `0.5492` MFU points (`4.07%`) above the old L4 fused steps 3-5 average of `13.4978`.
  - All ranks completed lower, compile, first execution, and repeated state execution. There was no custom-VJP, compiler, OOM, NCCL, DIME, traceback, or watchdog failure.
- Numerical policy:
  - User explicitly accepts floating output, loss, and every gradient leaf at relative-L2 `<=0.002` under these circumstances. Routing, counts, and dropped-route values remain exact.
  - The local synchronized W13/W2 errors of `0.00175065/0.00155534` therefore pass the accepted gate.
- Interpretation:
  - The manual-VJP formulation is a lifecycle pass and an exploratory reduced-shape performance improvement. The L4 result does not prove the exact L24 target or prove runtime collective removal.
- Next action:
  - Profile a 20-step L8 run and compare non-profiled throughput with the old L8 fused `16.1404` MFU result. Require profile evidence that per-microbatch data-axis FP32 all-reduces disappeared before promoting to exact L24.

### 2026-07-27 05:08 PDT - Launch L8 manual-VJP performance and profile gate
- Snapshot: `6d22a1eaf4` records the implementation and L4 launch; functional implementation commit is `6593a34edc`.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-manualvjp-fp8-l8-e64k4-b512-s4096-p4m16-profile-r1-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 2,2,2,2 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 8 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 20 --tracker wandb --profiler-steps 2 --xla-memory-fraction 0.70 --remat save_moe`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-120800` on `cw-rno2a`.
  - Babysitter `019fa37a-b09a-7610-9b58-8b4d3a122eff` owns all-rank terminal monitoring, profile artifact download, structured summary generation, and cleanup.
- Gate:
  - Compare non-profiled steps, preferably steps 2-7 and 11-19, against old L8 fused mean `16.1404` MFU.
  - Verify from the trace that the twelve per-stage FP32 all-reduces observed in each old six-layer `backward_accumulating` are removed or materially reduced. Record communication share, all-reduce, SendRecv, and all-gather counts before deciding whether to launch exact L24.

### 2026-07-27 05:35 PDT - L8 manual-VJP r1 hits a two-layer compiler cliff
- Result:
  - Parent `/dlwh/iris-run-job-20260727-120800` reached distributed setup, lowering, DIME initialization, and local task compilation on all four ranks, but completed no training step and never entered the profiler window.
  - Rank 3 began compiling `grug_1f1b_mb0_stage3_loss_backward_accumulating` at `12:10:46Z`. Two thread dumps about three minutes apart remained in `backend_compile_and_load -> compile_or_get_cached -> pxla -> apply_task -> eval_local`.
  - Rank 3's eight GPUs remained at 100% utilization with about `61.4 GiB` allocated each. After more than 23 minutes without a phase change, the parent was stopped. Parent and child are terminal `killed`; all four child tasks exited `0`; no failed or live resource remains.
- Interpretation:
  - The one-layer L4 task compiles and executes, while the two-layer final-stage loss/backward does not finish compilation. This localizes the cliff to graph growth from the manual custom VJP's nested `jax.vjp` over the complete Ring forward in each block, not to a runtime collective or memory-capacity failure.
  - The run used pre-merge snapshot `674ed8af27` and JAX `0.10.1`; the branch now includes stable JAX `0.11.0`, but stable XLA still predates the July 24 device-initiated ragged-all-to-all kernel.
- External audit:
  - OpenXLA commit [`acb5aaffe4c0`](https://github.com/openxla/xla/commit/acb5aaffe4c0d844bacb57ad85234422f0ceaae0) adds opt-in device-initiated ragged all-to-all behind `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true`.
  - Stable JAX `0.11.0` does not contain that commit. The July 25 nightly does, but the authoritative JaxPP production graph was already tested with NCCL `2.30.7`: direct controls pass while the pipeline path deadlocks or produces catastrophic stage-0 gradients. There is no released runtime change that justifies another target sweep, so Ring remains the transport baseline.
- Next action:
  - Replace the nested whole-Ring VJP with an explicit local Ring pullback, then repeat the same L8 gate from the merged JAX `0.11.0` stack.

### 2026-07-27 05:46 PDT - Replace nested Ring AD with an explicit pullback
- Implementation:
  - Added an explicit accumulating ragged-dot backward entry point that reuses the existing Triton `dlhs` and FP32 accumulating `drhs` kernels.
  - The Ring custom VJP now recomputes routing, dispatch, expert activations, and combine state, then explicitly transposes the expert-axis gather/scatter path. It no longer calls `jax.vjp` over the complete local Ring graph inside `shard_map`.
  - Accumulator inputs are explicitly nondifferentiable. W13/W2 cotangents remain data-local under `check_vma=False` and still synchronize only at the optimizer boundary.
- Validation:
  - The four-device CPU regression still finds zero all-reduces and zero reduce-scatters in the one-block weight-only local backward, and exactly two reduce-scatters in the step-boundary synchronization.
  - Activation, router-combine, W13, and W2 cotangents all pass the accepted relative-L2 ceiling `<=0.002`; output values and dropped-route counts remain exact.
  - A two-distinct-block regression lowers, compiles, executes, and passes the same per-leaf numerical policy in `33.88s` test time. This exercises the graph shape that failed in the L8 task without incorrectly reusing one expert tensor across layers.
  - `uv run pytest -q tests/test_grug_moe_accumulating_api.py` reports `2 passed`; the combined accumulating API/config selection reports `9 passed`. Changed-file pre-commit, including Pyrefly, passes.
- Next action:
  - Snapshot this implementation and rerun the unchanged L8 EP2/data4 profile gate. Require compile completion, finite steps, and trace evidence before an exact L24 launch.

### 2026-07-27 05:53 PDT - Package the nested-VJP compile cliff as a CPU reproducer
- Snapshot: `a4a9f8d8ca`.
- Artifact:
  - `scratch/jaxpp_manual_vjp_compile_repro/repro.py` is independent of Marin, JaxPP, and GPU kernels. It keeps the 2x2 data/expert mesh, exact integer routing and drops, Ring-shaped gather/scatter collectives, accumulating grouped matmul contract, outer custom VJP, and nested full-local `jax.vjp`.
  - `README.md` records the command, scope, numerical policy, and interpretation. `results.json` contains complete lowering, compile, StableHLO, recursive-JAXPR, operation-count, and parity measurements.
- Result:
  - Forward-only StableHLO grows `23,306 -> 41,104` bytes from one to two blocks. Nested full-VJP StableHLO grows `59,319 -> 108,812` bytes and recursive JAXPR equations grow `351 -> 664`.
  - The second forward block adds `17,798` StableHLO bytes; the corresponding training graph adds `49,493` bytes, a `2.78x` amplification. Dot, reduce-scatter, and scatter deltas are each `4.00x` their forward-only deltas.
  - `jax.jit(inline=False)` outlining does not contain growth: two-block StableHLO is `114,655` bytes. A synthetic `lax.scan` formulation holds one- and two-block modules at `83,609` bytes and 363 recursive equations.
  - Every floating output, loss, and gradient leaf has zero relative-L2 difference in the synthetic fixture. Routing counts and drops are exact.
- Interpretation:
  - The compile cliff comes from replay and transpose construction across each unrolled full Ring block. It is not isolated to grouped-matmul lowering. The production explicit pullback in `eda3bac52a` removes that nested whole-block transformation without requiring a scanned transformer representation.

### 2026-07-27 05:54 PDT - L8 explicit-backward profile is blocked by an obsolete CUTLASS patch
- Snapshot: `eda3bac52a`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-124912`.
  - Child `/dlwh/iris-run-job-20260727-124912/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r2-20260727`.
- Result:
  - All four H100x8 ranks failed during setup in `patch_cutlass_dsl_mlir_type_guard.sh`. The patch expected `_isa` in `cutlass/_mlir/dialects/arith.py`; CUTLASS DSL `4.6.0` removed that helper, so every rank reported `Hunk #1 FAILED at 18`.
  - Iris retried the deterministic setup failure three times. The parent was stopped to end the retry loop. Parent and child are terminal `killed`; all four child tasks are complete and no resource remains live. The child summary reports `failures=3`, `preemptions=4`, task 0 exit `1`, and tasks 1-3 exit `0` after termination.
  - The run did not reach JAX distributed setup, lowering, `grug_1f1b_mb0_stage3_loss_backward_accumulating`, training, or profiling. W&B has no run with the requested ID, and no profiler artifact exists.
- Interpretation:
  - This attempt says nothing about explicit-backward compile time, MFU, FP32 all-reduces, SendRecv, all-gathers, or compute/communication/stall shares. The old L8 fused `16.1404` MFU result remains the comparison point.
  - Package inspection confirms that CUTLASS DSL `4.6.0` has no `_isa` implementation to patch. The setup helper must treat the missing helper as an obsolete-patch no-op before the unchanged L8 gate can run.
- Next action:
  - Make the CUTLASS patch helper version-aware, validate it against `4.6.0`, then relaunch the same L8 profile gate under a fresh run ID and dedicated babysitter.

### 2026-07-27 06:03 PDT - L8 explicit-backward r3 exposes a QuACK/CUTLASS import mismatch
- Snapshot: `b56235ebc9`.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r3-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 2,2,2,2 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 8 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 20 --tracker wandb --profiler-steps 2 --xla-memory-fraction 0.70 --remat save_moe`.
- Jobs:
  - Parent `/dlwh/iris-run-job-20260727-125851`.
  - Child `/dlwh/iris-run-job-20260727-125851/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r3-20260727`.
- Result:
  - All four H100x8 ranks installed CUTLASS DSL `4.6.0` and ran `patch_cutlass_dsl_mlir_type_guard.sh`. The helper returned successfully without applying the obsolete `_isa` patch, and setup advanced to step 4.
  - At `13:02:05Z` the ranks began deserializing the training callable. Rank 0 failed at `13:02:10Z` while importing QuACK `0.5.0`: `quack/layout_utils.py:281` evaluates the annotation `cute.core.ThrMma`, which CUTLASS DSL `4.6.0` no longer exports. The other three ranks emitted the same traceback before Iris completed coscheduled cancellation.
  - The parent is terminal `failed` with one failed task. The child is terminal `worker_failed`: rank 0 exited `1` after `28.18s`; ranks 1-3 are `cosched_failed`; failures=`4`, preemptions=`0`. No retry or live allocation remains.
  - JAX distributed setup, lowering, `grug_1f1b_mb0_stage3_loss_backward_accumulating`, training, W&B, and profiling never started. There is no W&B run or profiler artifact. No watchdog stack was taken because the failure was an immediate Python import exception rather than a stall.
- Interpretation:
  - `b56235ebc9` fixed the prior setup failure as intended. The run then exposed a separate package incompatibility between the branch pins `quack-kernels==0.5.0` and `nvidia-cutlass-dsl==4.6.0`.
  - This attempt provides no evidence about explicit-backward compile time, steady MFU, FP32 all-reduces, SendRecv, all-gathers, or compute/communication/stall shares. Old L8 fused `16.1404` MFU remains the comparison point.
- Next action:
  - Reconcile the QuACK and CUTLASS DSL pins or avoid importing QuACK on the Ring-only path, validate a clean callable import, then rerun the unchanged L8 profile gate under a fresh ID.

### 2026-07-27 06:35 PDT - QuACK 0.6.1 import gates expose an incomplete PyTorch CUDA lock split
- Package audit:
  - Commit `814d891706` updates the active GPU extras to `quack-kernels==0.6.1`, matching CUTLASS DSL `4.6.0`. QuACK 0.6.1 replaces the failing `cute.core.ThrMma` annotation with `cute.ThrMma`; the Sonic adapter's TVM-FFI, grouped-GEMM, scheduler, varlen, and epilogue APIs remain available.
  - The historical standalone NVIDIA/JaxPP reproducer remains pinned to QuACK `0.5.0` so its published environment is unchanged.
- Command:
  - Each one-H100 job ran `CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=.70 uv run python experiments/grug/moe/repro_quack_grouped_mlp_numerics.py --warmup 2 --iterations 4`.
  - The configured gate has `16,384` tokens, `65,536` assignments, 64 experts, top-k 4, and an exact balanced count of `1,024` assignments per expert.
- Failed setup gates:
  - `/dlwh/quack061-numerical-gate-20260727-131131` from `814d891706` exited `1` after `6.49s`: QuACK's eager PyTorch import failed to load `libcufile.so.0`.
  - `/dlwh/quack061-numerical-gate-r2-20260727-132008` from `348cd68c10` exited `1` after `6.32s`: Iris could expose a cuFile library directory when installed, but the pure Levanter GPU lock split had not installed `nvidia-cufile-cu12`.
  - `/dlwh/quack061-numerical-gate-r3-20260727-132457` from `2faaf1348d` installed cuFile and advanced to the next eager PyTorch dependency, then exited `1` after `8.70s` on missing `libcusparseLt.so.0`. It reported JAX `0.11.0`, QuACK `0.6.1`, CUTLASS DSL `4.6.0`, cuFile `1.13.1.3`, and PyTorch `2.11.0`.
  - All three jobs are terminal with no retry or live resource. None reached QuACK lowering, numerical comparison, timing, or drop reporting.
- Root cause and fix:
  - Adding individual NVIDIA runtime wheels would repeat the failure for each omitted PyTorch CUDA dependency. The pure `marin-levanter[gpu]` workspace split selected PyTorch's CUDA wheel through QuACK without a Levanter-local source mapping, so uv did not attach the wheel's complete CUDA runtime closure to that extra.
  - Commit `ce84c8868d` directly pins PyTorch `2.11.0` in the Levanter GPU extra and maps it to the cu128 index. `uv export --package marin-levanter --extra gpu --locked` now includes PyTorch `2.11.0+cu128`, cuFile `1.13.1.3`, cuSPARSELt `0.7.1`, cuDNN `9.19.0.56`, NCCL `2.28.9`, and NVSHMEM `3.4.5`.
  - `uv lock --check`, the ten Iris CUDA-staging tests, and changed-file pre-commit pass.
- Numerical policy:
  - Commit `d9bb05952e` makes the user-approved `0.002` relative-L2 ceiling mandatory for floating output, loss, and every gradient leaf. Routing and per-expert counts remain exact, and the gate now explicitly requires and reports exactly zero dropped assignments.
- Next action:
  - Run the one-H100 gate from `d9bb05952e`. Only a full numerical and exact-routing/drop pass promotes to the unchanged L8 explicit-backward profile gate.

### 2026-07-27 06:39 PDT - QuACK r4 confirms the full runtime set but exposes a narrow loader path
- Job:
  - `/dlwh/quack061-numerical-gate-r4-20260727-133307` ran from `ce84c8868d` on one `cw-rno2a` H100 with zero retries.
- Result:
  - Setup installed JAX `0.11.0`, PyTorch `2.11.0+cu128`, QuACK `0.6.1`, CUTLASS DSL `4.6.0`, cuFile `1.13.1.3`, and cuSPARSELt `0.7.1`.
  - The cuFile loader probe passed. A subsequent bare `ctypes.CDLL("libcusparseLt.so.0")` probe failed because `site-packages/nvidia/cusparselt/lib` was not on `LD_LIBRARY_PATH`.
  - The task exited `1` after `5.04s`; it never launched the numerical script, so there are no parity, timing, route-count, or drop results. The job is terminal and cleanup reports zero pods, workloads, and jobs.
- Interpretation:
  - The uv lock now contains the required libraries. The remaining setup fault is Iris's cuFile-specific loader-path activation, which exposes one NVIDIA wheel directory instead of the complete installed NVIDIA runtime set.
- Fix:
  - The current workspace generalizes GPU activation to append every installed `site-packages/nvidia/*/lib` directory to `LD_LIBRARY_PATH`. The behavior test covers both cuFile and cuSPARSELt while preserving an existing loader path.
  - The focused Iris suite reports `10 passed`; changed-file pre-commit including Pyrefly passes.
- Next action:
  - Commit and push the generalized loader setup, then rerun the full `d9bb05952e` numerical policy under a fresh one-H100 job.

### 2026-07-27 06:46 PDT - QuACK 0.6.1 executes but fails the full numerical gate
- Job:
  - `/dlwh/quack061-numerical-gate-r5-20260727-133924` ran from `8cc00c6f6252e663d54b09ba05acf0d8329b20dd` on one `cw-rno2a` H100.
  - Command: `CUDA_VISIBLE_DEVICES=0 TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_MEM_FRACTION=.70 uv run python experiments/grug/moe/repro_quack_grouped_mlp_numerics.py --warmup 2 --iterations 4`.
- Runtime:
  - Imports passed with JAX `0.11.0`, PyTorch `2.11.0+cu128`, QuACK `0.6.1`, CUTLASS DSL `4.6.0`, cuFile `1.13.1.3`, and cuSPARSELt `0.7.1` on an H100 80 GB.
  - This validates the complete PyTorch CUDA dependency closure and generalized NVIDIA-wheel loader path.
- Numerical result:

  | Tensor | Relative L2 | Cosine similarity | Accepted at `0.002` |
  | --- | ---: | ---: | --- |
  | loss | `0.000210469` | `0.999999980` | yes |
  | output | `0.00606944` | `0.999981726` | no |
  | gradient.tokens | `1.28293727` | `0.02049198` | no |
  | gradient.routing_weights | `0.00295452` | `0.999995699` | no |
  | gradient.w13 | `1.41239541` | `0.00258660` | no |
  | gradient.w2 | `0.00697065` | `0.999975816` | no |

  - Every tensor was finite. Near-zero cosine for token and W13 gradients makes this a backward-path correctness failure, not acceptable BF16 accumulation drift.
  - The fixed route table was exact with `65,536` assignments, 64 experts, and min/max `1,024` assignments per expert. Dropped assignments were exactly zero.
  - Timing is intentionally absent because the harness benchmarks only after numerical admission.
- Terminal state:
  - The gate returned exit `1`, so Iris records one failed task after `29.41s`; retries were disabled. Cleanup verified zero pods, workloads, and jobs.
- Interpretation:
  - Proper Sonic/QuACK remains closed as a performance candidate under the user-approved `0.002` policy. The import/runtime upgrade is successful, but the current JAX custom-VJP adapter is not numerically admissible with QuACK `0.6.1`.
  - Do not launch the L8 JaxPP profile from this result and do not weaken tolerances.
- Next action:
  - Audit the QuACK 0.6.1 grouped-GEMM backward contract against `sonic_quack.py`, focusing on layout/transposition semantics that could independently corrupt token and W13 gradients. Run only a smallest discriminating gate after identifying a concrete adapter hypothesis.

### 2026-07-27 06:57 PDT - Correct QuACK's gated backward permutation; residual numerics still fail
- Root cause:
  - QuACK's `concat_layout` interleaves a tensor's non-contiguous dimension. The forward W13 view presents `(2I,H,E)`, so its gated dimension is correctly interleaved.
  - The token-gradient and W13-gradient paths instead presented `(H,2I,E)`. Their concat kernels therefore interleaved hidden size rather than the gated `2I` dimension.
  - The r5 W13 cosine `0.0025866` with equal actual/reference norms predicts relative-L2 `sqrt(2 - 2c) = 1.412383`, matching the observed `1.412395` and confirming a permutation signature.
- Fix:
  - Commit `7511e190c6` removes the incorrect concat weight-gradient kernel. Backward now converts QuACK's interleaved dpreactivation to concatenated `[dgate; dup]` explicitly and uses the ordinary grouped GEMM kernels for token and W13 gradients.
  - Gate-policy unit tests report `5 passed`; the QuACK-focused Levanter selection reports `4 passed, 1 skipped` on CPU. Changed-file pre-commit including Pyrefly passes.
- Job:
  - `/dlwh/quack061-numerical-gate-r6-20260727-135404` ran the same one-H100 command from `7511e190c60e93a3b19cb418040975fa4c303fbb`.
- Result:

  | Tensor | r5 relative L2 | r6 relative L2 | r6 cosine | Accepted at `0.002` |
  | --- | ---: | ---: | ---: | --- |
  | loss | `0.000210469` | `0.000210469` | `0.999999980` | yes |
  | output | `0.00606944` | `0.00606941` | `0.999981663` | no |
  | gradient.tokens | `1.28293727` | `0.00690849` | `0.999976272` | no |
  | gradient.routing_weights | `0.00295452` | `0.00295354` | `0.999995699` | no |
  | gradient.w13 | `1.41239541` | `0.00707695` | `0.999974898` | no |
  | gradient.w2 | `0.00697065` | `0.00697064` | `0.999975932` | no |

  - The fix removes the catastrophic permutation: token and W13 relative-L2 improve by about `186x` and `200x`, and both cosines exceed `0.99997`.
  - The remaining output and gradient errors cluster around `0.30-0.71%`, above the unchanged `0.2%` ceiling. Every tensor is finite.
  - Routing remains exact at `65,536` assignments and `1,024` per expert; dropped assignments remain exactly zero. Timing is absent because admission failed.
  - Iris records task exit `1` after `28.31s` with zero retries. Cleanup reports zero pods, workloads, and jobs.
- Interpretation:
  - The adapter permutation was a real correctness bug and is fixed. The corrected QuACK BF16 path still fails the accepted full numerical policy, so proper Sonic remains closed as a performance candidate.
  - Do not relax the ceiling or launch a Sonic L8 run. The independent Ring explicit-backward profile remains the active target path.

### 2026-07-27 07:03 PDT - L8 explicit Ring pullback passes and removes the all-reduce bottleneck
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-134811` and child `/dlwh/iris-run-job-20260727-134811/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r4-20260727` launched from clean commit `b7352a8c335f`.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r4-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 2,2,2,2 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 8 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 20 --tracker wandb --profiler-steps 2 --xla-memory-fraction 0.70 --remat save_moe`.
- Lifecycle:
  - All four ranks imported the corrected QuACK/CUTLASS runtime, compiled their two-layer forward/backward tasks, and completed `20/20` steps. Parent and child succeeded with exit `0`, zero failures, and zero preemptions.
  - W&B [r4](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r4-20260727) retained 18 finite rows for steps 2-19; loss declined `8.53050 -> 6.29976`.
- Performance:
  - Non-profiled steps 11-19 had mean/p50 MFU `16.54746/16.54596`, range `16.44345-16.61353`, and mean step time `1.90674s`.
  - Mean MFU improves by `0.40706` points (`2.522%`) over the old L8 fused `16.1404` result.
- Profile:
  - Profiler artifact `marin-community/marin_moe/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r4-20260727-profiler:v0` is readable at 102,002,917 bytes with digest `54373646bf68cc9e1a7a9b52c66f173f`.
  - [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fjaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l8-e64k4-b512-s4096-p4m16-profile-r4-20260727) contains 1,874,454 events.
  - Aggregate kernel duration is `65.28%` compute and `34.72%` communication. Timeline attribution is `44.07%` compute, `35.85%` communication, and `20.08%` stall.
  - FP32 all-reduce has 928 calls and `117.80ms` aggregate. Layer-normalized against the old fused L24 profile, count falls `9.38%` and time falls `96.31%`; it is no longer the dominant regression.
  - SendRecv has 944 calls and `8,133.69ms`, accounting for `68.2%` of collective time. All-gather has 9,152 calls and `2,452.25ms`; reduce-scatter has 2,368 calls and `1,017.30ms`.
- Interpretation:
  - The explicit pullback achieves its intended system effect: per-microbatch FP32 all-reduce occupancy is effectively removed, and L8 throughput improves.
  - SendRecv/waiting and pipeline stall now dominate. The `2.522%` L8 gain alone projects below 20 MFU from the `18.2583` best exact baseline, but the all-reduce fix scales with microbatch count. The decisive remaining measurement is the exact L24 batch8192/m256 combination before closing this path.
- Cleanup:
  - No matching pod, workload, Kubernetes job, failed allocation, or stale profiler process remains live.

### 2026-07-27 07:18 PDT - Exact L24/m256 explicit pullback hits a six-layer compiler crash
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-140249` and child `/dlwh/iris-run-job-20260727-140249/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-exact-r1-20260727` launched from `3e852277ad`.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-exact-r1-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 6,6,6,6 --microbatches 256 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 24 --experts 64 --top-k 4 --vocab-size 8192 --batch 8192 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 12 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Result:
  - Both attempts imported and initialized all four six-layer ranks. Rank 3 failed twice during local backward compilation before step 0.
  - Attempt 0 exited `139` while compiling `stage3_loss_backward_accumulating`. Attempt 1 passed that task, then emitted `Fatal Python error: Aborted` while downstream backward stages compiled and was recorded as exit `139`.
  - Ranks 0-2 were atomically bounced on each rank-3 failure. There is no OOM, preemption, setup, import, NCCL, DIME-execution, or training evidence.
  - Iris began a third deterministic retry because this launch path did not expose a zero-retry setting. The parent was stopped immediately. Parent and child are terminal `killed`; the child summary records failures=`2`, preemptions=`4`, and rank 3 exit `139`. Direct Kubernetes inspection reports zero matching pods, jobs, or workloads.
  - W&B [exact r1](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-exact-r1-20260727) is stale `running` with zero metric rows.
- Interpretation:
  - The run provides no exact MFU verdict. The best exact baseline remains `18.2583` mean MFU and the target remains strictly above 20.
  - L8 proves two-layer explicit backward compilation and execution. The new boundary is the six-layer local loss/backward graph, not runtime collectives or HBM.
- Next action:
  - Reduce or outline the six-layer explicit pullback compile graph while preserving the accepted `0.002` floating-tensor policy and exact routing/drop behavior. Require a focused CPU graph/numerical regression before another H100x32 launch; do not rerun the identical command.

### 2026-07-27 07:47 PDT - Late-outlined Ring weight cotangents shrink the six-block graph
- Hypothesis:
  - The L24 crash comes from inlining all six blocks' two FP32 expert-weight cotangent custom calls into each stage backward. Keeping only those weight cotangents as reusable late-inline compiler units should reduce the monolithic graph without changing the activation cotangent or broad model numerics.
- Commit Hash:
  - `bd153fd5ee` adds `_outlined_accumulating_weight_cotangent` with `jax.jit(..., inline=jax.Inline.XLA_LATE)`. The explicit Ring pullback still evaluates the ordinary backward for its activation cotangent, while each W13/W2 cotangent is emitted through the outlined helper. Dead-code elimination can discard the unused half of each backward call.
- Local evidence:
  - A six-block CPU StableHLO probe finds `12` late-inline call sites backed by two reusable private functions.
  - Main-function StableHLO falls from `12,775` to `12,295` lines (`-3.8%`); whole-module text falls from `1,248,743` to `1,213,634` characters (`-2.8%`).
  - `88/106` pullback leaves remain bitwise identical. The only changed leaves are the `18` expert-weight gradients across six blocks, and the worst relative-L2 is `0.00191157`, within the user-approved `0.002` ceiling.
  - A broader whole-pullback outline was rejected locally because router or attention gradients exceeded `0.002`.
  - `uv run pytest -q tests/test_grug_moe_accumulating_api.py` and `./infra/pre-commit.py --changed-files --fix` pass.
- Setup-only failed launch:
  - Parent `/dlwh/iris-run-job-20260727-144335` did not submit a GPU child. The launcher selected its default JaxPP environment with JAX `0.10.1`; import failed because that release has no `jax.Inline.XLA_LATE`.
  - This is not compiler or numerical evidence about the outlined graph. No GPU allocation, W&B run, or live resource was created.
- Interpretation:
  - The narrow outline is locally numerically admissible and measurably reduces the six-block compiler unit. The first remote launch tested the wrong JAX runtime.
- Next action:
  - Rerun the three-step L24/b512/m16 compile gate from `bd153fd5ee` with explicit `--jax-nightly-version 0.11.1.dev20260725`. Only a successful four-rank compile/execute gate justifies the exact L24/b8192/m256 run with the same runtime.

### 2026-07-27 07:56 PDT - Defer outline construction across the host/worker JAX boundary
- Failed launch:
  - Parent `/dlwh/iris-run-job-20260727-144755` received `--jax-nightly-version 0.11.1.dev20260725` but failed before submitting a GPU child. The nightly setup applies only inside the eventual worker; the Iris host imports the launcher and model under locked JAX `0.10.1`.
  - The host therefore reproduced `AttributeError: module 'jax' has no attribute 'Inline'` at the module-level decorator. It produced no ranks, compiler phase, W&B run, step, or metric. Iris reports exit `1` after `30.93s`, and cleanup finds no child, pod, Kubernetes job, or workload.
- Fix:
  - Commit `87917d0469` keeps the raw weight-cotangent function at module scope and caches the `jax.jit(..., inline=jax.Inline.XLA_LATE)` wrapper only when the worker first traces it.
  - An isolated import using cached JAX/JAXlib `0.10.1` succeeds and reports zero cache hits or misses, proving that the host does not resolve `jax.Inline`.
  - A JAX `0.11.0` lowering probe with 12 calls reports 12 `call` sites to one private function, each carrying `inlineable = "xla_late"`. Deferred construction therefore preserves the compiler boundary.
  - `uv run pytest -q tests/test_grug_moe_accumulating_api.py` reports `2 passed`; changed-file pre-commit including Pyrefly passes.
- Interpretation:
  - Both failed parents were launcher-version mismatches before allocation, not evidence about the six-layer GPU compiler. The host and worker have intentionally different JAX versions, so worker-only compiler APIs must not be resolved at launcher import time.
- Next action:
  - Run a fresh three-step L24/b512/m16 gate from `87917d0469` with worker JAX `0.11.1.dev20260725`. Require all four ranks to reach compile and execution before the exact b8192/m256 launch.

### 2026-07-27 08:12 PDT - Nightly outline gate reaches six-layer compilation but exits in thunk initialization
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-145536` and its four-rank child ran from clean commit `87917d0469` with worker JAX/JAXlib `0.11.1.dev20260725`, JaxPP `7091a9b5ce02`, and NCCL `2.30.7`.
  - Config was L24/d2560/e64/top-k4/sequence4096, batch512/m16, split `6,6,6,6`, expert2/data4, explicit `std_1f1b`, FP8 wire, fused FP32 data-local expert gradients, CuTe FA4, and Triton grouped GEMM.
- Result:
  - The deferred wrapper cleared host import. All four ranks installed the intended nightly and initialized the distributed job.
  - Attempt 0 reached real six-layer compilation. Ranks 0/1 compiled stage 0/1 forward tasks. Rank 2 reached `stage2_backward_accumulating`. Rank 3 compiled `stage3_loss_backward_accumulating` in about `54s`, compiled its backward-wire pack, and began later reusable tasks.
  - Rank 3 then reported a device rendezvous stuck at `thunk initialization completion for device ordinal 0` and exited `139`. There was no Python exception, OOM, or training step.
  - Attempt 1 was not a valid retry: rank 1 reused the attempt-0 coordinator `10.168.192.17:8476`, while ranks 0/2/3 used `10.168.196.113:8476`. All four stacks remained in `jax.distributed.initialize()`.
  - The parent was stopped. Parent and child are terminal `killed`; rank 3 records exit `139`, failures=`1`, and preemptions=`4`. No live descendant remains.
  - [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-compile-r3-20260727) has zero rows and remains stale `running`.
- Interpretation:
  - The outline reaches farther than the old rank-3 crash because the stage-3 loss backward completes. This run does not prove that worker code is executable: it fails around DIME/device thunk initialization under the JAX nightly and NCCL `2.30.7`.
  - The nightly was needed only to work around the original module-level `jax.Inline` reference. Commit `87917d0469` removes that host constraint, and locked worker JAX `0.11.0` already supports `XLA_LATE`.
- Next action:
  - Run one stable-worker control from `87917d0469`, omitting the nightly flag so workers retain JAX/JAXlib `0.11.0` and NCCL `2.28.9`. Only stable four-rank execution justifies the exact b8192/m256 run.

### 2026-07-27 08:21 PDT - Stable workers reproduce the outlined L24 thunk failure
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-151235` and child `/dlwh/iris-run-job-20260727-151235/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-stable-compile-r4-20260727` ran clean commit `87917d0469`.
  - Workers used locked JAX/JAXlib `0.11.0`, JaxPP `7091a9b5ce02`, and NCCL `2.28.9`, removing the nightly JAX and NCCL `2.30.7` used by r3.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-stable-compile-r4-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 6,6,6,6 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 24 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 3 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Result:
  - All four ranks rendezvoused on one coordinator and entered six-layer compilation. Rank 0 compiled stage-0 forward tasks; rank 1 compiled stage-1 forward tasks and reached `stage1_backward_accumulating`; rank 2 compiled stage-2 forward and `stage2_backward_accumulating`; rank 3 compiled `stage3_loss_backward_accumulating` in about `54s` and its backward-wire pack.
  - Rank 3 then waited at `thunk initialization completion for device ordinal 0; run_id=622354233`. XLA warned after `10s` and aborted after `30s`: only `1/8` expected local-device threads reached the rendezvous.
  - The native stack is `xla::gpu::GpuExecutable::ExecuteThunksImpl -> xla::gpu::RendezvousAfterInitialization -> xla::Rendezvous`. Rank 3 exited `139`. There was no OOM, Python exception, loss, step, or MFU.
  - Iris began attempt 1 setup. The parent was stopped before another model compile. Parent and child are terminal `killed`; the child summary records one failure, four preemptions from coordinated cancellation, rank-3 exit `139`, and all `4/4` tasks complete. No live descendant remains.
  - [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-stable-compile-r4-20260727) remains stale `running` with `lastHistoryStep=-1`; it has topology/configuration summary data but no history rows.
- Interpretation:
  - The same thunk-initialization rendezvous fails under stable JAX/JAXlib `0.11.0` and NCCL `2.28.9` as under nightly JAX/JAXlib `0.11.1.dev20260725` and NCCL `2.30.7`. The failure is therefore independent of that runtime/transport change.
  - The late outline clears the former monolithic stage-3 compilation boundary, but the resulting six-layer executable is not runnable. This is an XLA GPU thunk-initialization failure, not DIME communicator setup, HBM exhaustion, or the retry coordinator bug.
  - No numerical parity result was produced. The accepted policy remains relative-L2 `<=0.002` for floating outputs, loss, and every gradient leaf, with routing/counts/drops exact.
- Decision:
  - Do not launch the exact L24 batch8192/m256 run from this outline. Minimize the six-layer thunk-initialization failure before another performance attempt.

### 2026-07-27 08:38 PDT - Unequal split clears outlined L24 compile and execution
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-152829` and child `/dlwh/iris-run-job-20260727-152829/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-split7655-compile-r5-20260727` ran clean commit `ab0e941ef2`.
  - Workers used JAX/JAXlib `0.11.0`, JaxPP `7091a9b5ce02`, NCCL `2.28.9`, and CUDA toolkit `12.8.1`. The L24/d2560/e64/top-k4/sequence4096 gate used batch512/m16, split `7,6,6,5`, expert2/data4, explicit `std_1f1b`, FP8 wire, fused FP32 data-local expert gradients, CuTe FA4, and Triton grouped GEMM.
- Command:
  - `GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false TF_GPU_ALLOCATOR=cuda_malloc_async experiments/grug/moe/run_cw_jaxpp_may_d2560.sh --submit --cluster cw-rno2a --prefix s3://marin-us-east-02a/marin --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-split7655-compile-r5-20260727 --schedule std_1f1b --implementation explicit_mpmd --explicit-mpmd-schedule-mode default --explicit-mpmd-pipeline-wire-format fp8 --explicit-mpmd-stage-task-microbatch-group-size 1 --expert-gradient-accumulation fused_fp32_data_local --physical-stages 4 --logical-stages 4 --stage-layer-counts 7,6,6,5 --microbatches 16 --nodes 4 --gpus-per-replica 8 --expert-axis 2 --layers 24 --experts 64 --top-k 4 --vocab-size 8192 --batch 512 --seq-len 4096 --moe-implementation ring --attention-implementation gpu_fa4_cute --ragged-dot-implementation triton --ragged-dot-block-k 32 --ragged-dot-num-warps 8 --loss-implementation xla --steps 3 --tracker wandb --xla-memory-fraction 0.70 --remat save_moe`.
- Compile and execution:
  - Rank 0's seven-layer forward compiled in about `30s`; its seven-layer backward took at most `58s`. Rank 1's six-layer forward/backward took about `25/48s`, rank 2's took about `25/49s`, and rank 3's five-layer loss/backward took about `51s`.
  - Rank 3 emitted its backward wire and rank 0 subsequently completed the seven-layer backward. All 16 microbatches traversed all four stages, and every rank reached its fused expert update.
  - The prior thunk-initialization failure neither recurred on rank 3 nor followed the seven-layer graph to rank 0.
- Metrics:
  - [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b512-s4096-p4m16-outline-split7655-compile-r5-20260727) is `finished` with three completed steps and two post-compile throughput samples.
  - Mean/p50 MFU was `14.9736655`; both samples had the same reported MFU. Step duration was `6.210066s`, throughput was `337,702.04` tokens/s and `82.44679` examples/s, and final loss was `8.4956408`.
  - The three-step run processed `6,291,456` tokens. Compile-dominated wall time was `5m58s`, so this small batch512/m16 gate is compile admission rather than the exact batch8192/m256 performance verdict.
- Terminal state:
  - Parent and child succeeded with exit `0`. All `4/4` rank tasks succeeded after `7m24.68s`; failures and preemptions were both zero.
  - Coordination-service connection warnings appeared only after W&B finished and all steps completed. Iris still recorded all tasks as successful. No retry or live descendant remains.
- Upstream freshness:
  - JaxPP main remains `7091a9b5`. JAX nightly `0.11.1.dev20260727` with XLA `841f8d2` has no ragged implementation change after device-kernel commit `acb5aaf`; do not rerun the JAX ragged path.
  - NCCL `2.30.7-1` adds experimental UB-X at `db0c814185a0415cc2e23dca387fecb9282de551` under `contrib/nccl_ubx`. It needs a one-node EP8 direct gate before any JAX wrapper experiment.
- Interpretation:
  - The outlined Ring program is executable at L24 when the last stage contains five layers. The deterministic `6,6,6,6` failure is sensitive to stage placement or the six-layer terminal loss/backward graph, not simply the largest local layer count.
  - The exact batch8192/m256 run was not launched. Its expected split and the `14.9736655` compile-gate MFU need interpretation before spending another H100x32 compile.

### 2026-07-27 08:54 PDT - NCCL UB-X direct gate stops in setup three times
- Snapshot:
  - Jobs `/dlwh/nccl-ubx-direct-ep8-r1-20260727`, `/dlwh/nccl-ubx-direct-ep8-r2-20260727`, and `/dlwh/nccl-ubx-direct-ep8-r3-20260727` used clean commit `5c09dc0e98` on one `H100x8` node in `cw-rno2a`.
  - The intended gate is EP8, d2560, 64 experts, top-k 4, 16,384 tokens and 65,536 assignments per source rank, with balanced and deterministic learned-skew routes. Admission requires exact route/count/drop maps, bitwise dispatch, floating output relative-L2 `<=0.002`, and UB-X at least `1.10x` faster than Ring by slowest-rank p50.
- R1 command:

```bash
uv run iris --cluster cw-rno2a job run --no-wait \
  --enable-extra-resources --gpu H100x8 --cpu 64 --memory 256GB --disk 256GB \
  --timeout 3600 --max-retries 0 --priority interactive \
  --extra gpu --sync-package marin-levanter \
  --job-name nccl-ubx-direct-ep8-r1-20260727 -- \
  bash -lc 'set -euxo pipefail
cd "$IRIS_WORKDIR"
SOURCE=/tmp/nccl-ubx-db0c814
rm -rf "$SOURCE"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE"
git -C "$SOURCE" fetch --depth 1 origin db0c814185a0415cc2e23dca387fecb9282de551
git -C "$SOURCE" checkout --detach FETCH_HEAD
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
test -x "$CUDA_HOME/bin/nvcc"
"$CUDA_HOME/bin/nvcc" --version
make -C "$SOURCE" -j32 src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export NCCL_HOME="$SOURCE/build"
export NCCL_INCLUDE_DIR="$SOURCE/build/include"
export NCCL_LIBRARY_DIR="$SOURCE/build/lib"
export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:${LD_LIBRARY_PATH:-}"
uv pip install --python "$IRIS_VENV/bin/python" --no-deps -e "$SOURCE/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a uv pip install --python "$IRIS_VENV/bin/python" --no-build-isolation --no-deps -e "$SOURCE/contrib/nccl_ubx"
export NCCL_UBX_SOURCE="$SOURCE"
export NCCL_UBX_OUTPUT_DIR=/tmp/nccl-ubx-direct-results
export UBX_GRAPH_POOL_SHARE=0.1
python -c "import torch, ubx; print(\"UBX runtime\", torch.__version__, torch.cuda.nccl.version(), ubx.get_version())"
ldd "$(python -c "import ubx._C; print(ubx._C.__file__)")" | grep libnccl
experiments/grug/moe/run_nccl_ubx_direct_gate.sh'
```

- R1 result:
  - Iris setup completed and the pinned NVIDIA/nccl commit `db0c814185a0415cc2e23dca387fecb9282de551` checked out. The command then evaluated `test -x /usr/local/cuda/bin/nvcc`, which returned exit `1`.
  - The single task failed after `11.39s`. NCCL and UB-X were not built, no GPU process started, and no JSON result exists.
- R2 command:

```bash
uv run iris --cluster cw-rno2a job run --no-wait \
  --enable-extra-resources --gpu H100x8 --cpu 64 --memory 256GB --disk 256GB \
  --timeout 3600 --max-retries 0 --priority interactive \
  --extra gpu --sync-package marin-levanter \
  --job-name nccl-ubx-direct-ep8-r2-20260727 -- \
  bash -lc 'set -euxo pipefail
cd "$IRIS_WORKDIR"
uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"
cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin -name nvcc -print -quit)"
test -n "$cuda_bin"
CUDA_HOME="$(dirname "$(dirname "$cuda_bin")")"
export CUDA_HOME
"$CUDA_HOME/bin/nvcc" --version
python -c "import torch; print(\"Torch CUDA\", torch.__version__, torch.version.cuda)"
SOURCE=/tmp/nccl-ubx-db0c814
rm -rf "$SOURCE"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE"
git -C "$SOURCE" fetch --depth 1 origin db0c814185a0415cc2e23dca387fecb9282de551
git -C "$SOURCE" checkout --detach FETCH_HEAD
make -C "$SOURCE" -j32 src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export NCCL_HOME="$SOURCE/build"
export NCCL_INCLUDE_DIR="$SOURCE/build/include"
export NCCL_LIBRARY_DIR="$SOURCE/build/lib"
export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
uv pip install --python "$IRIS_VENV/bin/python" --no-deps -e "$SOURCE/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a uv pip install --python "$IRIS_VENV/bin/python" --no-build-isolation --no-deps -e "$SOURCE/contrib/nccl_ubx"
export NCCL_UBX_SOURCE="$SOURCE"
export NCCL_UBX_OUTPUT_DIR=/tmp/nccl-ubx-direct-results
export UBX_GRAPH_POOL_SHARE=0.1
python -c "import torch, ubx; print(\"UBX runtime\", torch.__version__, torch.cuda.nccl.version(), ubx.get_version())"
ldd "$(python -c "import ubx._C; print(ubx._C.__file__)")" | grep libnccl
experiments/grug/moe/run_nccl_ubx_direct_gate.sh'
```

- R2 result:
  - Torch changed from `2.11.0+cu128` to `2.11.0+cu130` inside `/app/.venv`. The command found `/app/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc`, and `nvcc --version` reported CUDA `13.2`, build `V13.2.78`.
  - The next unqualified `python -c "import torch"` resolved outside `/app/.venv` and failed with `ModuleNotFoundError: No module named 'torch'`. The single task failed after `6.09s`; NCCL and UB-X were not built, no GPU process started, and no JSON result exists.
- R3 command:

```bash
uv run iris --cluster cw-rno2a job run --no-wait \
  --enable-extra-resources --gpu H100x8 --cpu 64 --memory 256GB --disk 256GB \
  --timeout 3600 --max-retries 0 --priority interactive \
  --extra gpu --sync-package marin-levanter \
  --job-name nccl-ubx-direct-ep8-r3-20260727 -- \
  bash -lc 'set -euxo pipefail
source "$IRIS_VENV/bin/activate"
cd "$IRIS_WORKDIR"
uv pip install --no-deps --reinstall --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"
cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin -name nvcc -print -quit)"
test -n "$cuda_bin"
CUDA_HOME="$(dirname "$(dirname "$cuda_bin")")"
export CUDA_HOME
"$CUDA_HOME/bin/nvcc" --version
python -c "import torch; print(\"Torch CUDA\", torch.__version__, torch.version.cuda)"
SOURCE=/tmp/nccl-ubx-db0c814
rm -rf "$SOURCE"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE"
git -C "$SOURCE" fetch --depth 1 origin db0c814185a0415cc2e23dca387fecb9282de551
git -C "$SOURCE" checkout --detach FETCH_HEAD
make -C "$SOURCE" -j32 src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export NCCL_HOME="$SOURCE/build"
export NCCL_INCLUDE_DIR="$SOURCE/build/include"
export NCCL_LIBRARY_DIR="$SOURCE/build/lib"
export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
uv pip install --no-deps -e "$SOURCE/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a uv pip install --no-build-isolation --no-deps -e "$SOURCE/contrib/nccl_ubx"
export NCCL_UBX_SOURCE="$SOURCE"
export NCCL_UBX_OUTPUT_DIR=/tmp/nccl-ubx-direct-results
export UBX_GRAPH_POOL_SHARE=0.1
python -c "import torch, ubx; print(\"UBX runtime\", torch.__version__, torch.cuda.nccl.version(), ubx.get_version())"
ldd "$(python -c "import ubx._C; print(ubx._C.__file__)")" | grep libnccl
experiments/grug/moe/run_nccl_ubx_direct_gate.sh'
```

- R3 result:
  - Venv activation fixed R2's interpreter error. Torch imported as `2.11.0+cu130`, CUDA runtime `13.0`, and NCCL source compilation began with CUDA compiler `13.2.78`.
  - Every affected device translation unit then failed in `cuda_fp16.h:4492` with `fatal error: nv/target: No such file or directory`. `make src.build` exited `2` because the synthesized CUDA root lacked the matching `nvidia-cuda-cccl` headers.
  - The single task failed after `17.13s`. The NCCL build did not complete, the UB-X extension and benchmark did not run, and no JSON result exists.
- Interpretation:
  - All three failures are launcher or CUDA-toolchain setup errors. They provide no numerical or speed evidence for UB-X.
  - A corrected retry must retain venv activation and install the CUDA 13 `nvidia-cuda-cccl` package before the NCCL build so `${CUDA_HOME}/include/nv/target` exists. The benchmark and admission thresholds do not need to change.
  - Even a passing retry would be an optimistic transport-only result with precomputed maps and identity expert compute. It would not prove JAX integration, autograd correctness, routing cost, expert GEMM performance, or end-to-end pipeline MFU.
- Terminal state:
  - R1, R2, and R3 are terminal `failed` with one failed task each, zero retries, and no live resource.

### 2026-07-27 09:00 PDT - NCCL UB-X R4 exposes mixed CUDA component versions
- Snapshot:
  - Job `/dlwh/nccl-ubx-direct-ep8-r4-20260727` used clean commit `9f4eb7a3b4` on one `H100x8` node in `cw-rno2a`.
  - It retained venv activation and added `nvidia-cuda-cccl==13.3.3.4.1` to supply the `nv/target` header missing in R3. The EP8 d2560/e64/top-k4 gate and all admission thresholds were unchanged.
- Command:

```bash
uv run iris --cluster cw-rno2a job run --no-wait \
  --enable-extra-resources --gpu H100x8 --cpu 64 --memory 256GB --disk 256GB \
  --timeout 3600 --max-retries 0 --priority interactive \
  --extra gpu --sync-package marin-levanter \
  --job-name nccl-ubx-direct-ep8-r4-20260727 -- \
  bash -lc 'set -euxo pipefail
source "$IRIS_VENV/bin/activate"
cd "$IRIS_WORKDIR"
uv pip install --no-deps --reinstall --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"
uv pip install --link-mode symlink "nvidia-cuda-cccl==13.3.3.4.1"
cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu*/bin -name nvcc -print -quit)"
test -n "$cuda_bin"
CUDA_HOME="$(dirname "$(dirname "$cuda_bin")")"
export CUDA_HOME
cccl_target="$(find "$CUDA_HOME/include/cccl" -path "*/nv/target" -print -quit)"
test -n "$cccl_target"
"$CUDA_HOME/bin/nvcc" --version
python -c "import torch; print(\"Torch CUDA\", torch.__version__, torch.version.cuda)"
SOURCE=/tmp/nccl-ubx-db0c814
rm -rf "$SOURCE"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE"
git -C "$SOURCE" fetch --depth 1 origin db0c814185a0415cc2e23dca387fecb9282de551
git -C "$SOURCE" checkout --detach FETCH_HEAD
make -C "$SOURCE" -j32 src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export NCCL_HOME="$SOURCE/build"
export NCCL_INCLUDE_DIR="$SOURCE/build/include"
export NCCL_LIBRARY_DIR="$SOURCE/build/lib"
export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
uv pip install --no-deps -e "$SOURCE/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a uv pip install --no-build-isolation --no-deps -e "$SOURCE/contrib/nccl_ubx"
export NCCL_UBX_SOURCE="$SOURCE"
export NCCL_UBX_OUTPUT_DIR=/tmp/nccl-ubx-direct-results
export UBX_GRAPH_POOL_SHARE=0.1
python -c "import torch, ubx; print(\"UBX runtime\", torch.__version__, torch.cuda.nccl.version(), ubx.get_version())"
ldd "$(python -c "import ubx._C; print(ubx._C.__file__)")" | grep libnccl
experiments/grug/moe/run_nccl_ubx_direct_gate.sh'
```

- Result:
  - The `nv/target` preflight passed and pinned NCCL compilation began. Device compilation then failed in CCCL `cuda_toolkit.h:41` with `CUDA compiler and CUDA toolkit headers are incompatible`.
  - The environment combined CUDA compiler `13.2.78`, CUDA runtime headers `13.0.96`, and CCCL from the CUDA `13.3` package line. The CCCL compatibility check compares compiler and runtime-header major/minor versions, so supplying only the missing header cannot make this mixed toolchain valid.
  - The single task exited `2` after `16.23s`. NCCL did not finish building, the UB-X extension and benchmark did not run, and no JSON, numerical, or timing result exists.
- Next action:
  - Install one coherent CUDA component set before deriving `CUDA_HOME`. PyPI publishes version `13.2.86` for `nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-crt`, `nvidia-nvvm`, and `nvidia-cuda-cccl`.
  - Verify `nvcc --version`, `CUDART_VERSION`, `libdevice.10.bc`, and `${CUDA_HOME}/include/cccl/nv/target` before compiling NCCL. Keep the direct-gate benchmark and its exact map, bitwise dispatch, relative-L2 `<=0.002`, and `>=1.10x` speedup requirements unchanged.
- Terminal state:
  - R4 is terminal `failed` with one failed task, zero retries, and no live resource.
  - This remains an unexecuted optimistic precomputed-map identity transport gate, not evidence for JAX integration, autograd, expert GEMMs, or end-to-end MFU.

### 2026-07-27 09:08 PDT - Exact unequal-split Ring run regresses and exhausts rank-0 memory
- Snapshot:
  - Parent `/dlwh/iris-run-job-20260727-154333` and child `/dlwh/iris-run-job-20260727-154333/grug-train-jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-outline-split7655-exact-r2-20260727` used clean commit `55be7f18ce`.
  - The exact target shape was L24/d2560/e64/top-k4/sequence4096, batch8192/m256, four physical and logical stages split `7,6,6,5`, explicit `std_1f1b`, FP8 pipeline wires, fused FP32 data-local expert-gradient accumulation, Ring MoE, CuTe FA4, and JAX/JAXlib `0.11.0`.
- Command:

```bash
GRUG_JAXPP_LOG_LOCAL_MEMORY_PLAN=false \
TF_GPU_ALLOCATOR=cuda_malloc_async \
experiments/grug/moe/run_cw_jaxpp_may_d2560.sh \
  --submit \
  --cluster cw-rno2a \
  --prefix s3://marin-us-east-02a/marin \
  --run-id jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-outline-split7655-exact-r2-20260727 \
  --schedule std_1f1b \
  --implementation explicit_mpmd \
  --explicit-mpmd-schedule-mode default \
  --explicit-mpmd-pipeline-wire-format fp8 \
  --explicit-mpmd-stage-task-microbatch-group-size 1 \
  --expert-gradient-accumulation fused_fp32_data_local \
  --physical-stages 4 \
  --logical-stages 4 \
  --stage-layer-counts 7,6,6,5 \
  --microbatches 256 \
  --nodes 4 \
  --gpus-per-replica 8 \
  --expert-axis 2 \
  --layers 24 \
  --experts 64 \
  --top-k 4 \
  --vocab-size 8192 \
  --batch 8192 \
  --seq-len 4096 \
  --moe-implementation ring \
  --attention-implementation gpu_fa4_cute \
  --ragged-dot-implementation triton \
  --ragged-dot-block-k 32 \
  --ragged-dot-num-warps 8 \
  --loss-implementation xla \
  --steps 12 \
  --tracker wandb \
  --xla-memory-fraction 0.70 \
  --remat save_moe
```

- Result:
  - The first attempt compiled and reached step `5/12`. [W&B](https://wandb.ai/marin-community/marin_moe/runs/jaxpp-rno2a-ring-ep2d4-explicitbwd-fp8-l24-e64k4-b8192-s4096-p4m256-outline-split7655-exact-r2-20260727) retained two throughput samples with mean/p50 MFU `16.515528197633373`, latest MFU `16.353105260469082`, latest duration `90.97961631510407s`, `368812.63473112084` tokens/s, `90.04214715115255` examples/s, and loss `6.690955638885498` at global step `2`.
  - Mean MFU is `1.7427718023666259` points (`9.5451%`) below the exact ordinary-Ring baseline `18.2583` and `3.4844718023666275` points below the strict `>20` target. The `7,6,6,5` split is a measured performance regression at the exact shape.
  - At `15:59:58Z`, all eight rank-0 GPUs warned that a `15.34 GiB` allocation had failed. Step 5 completed, then rank 0 segfaulted at `16:01:15Z`. Iris retains rank 0 attempt 0 as `failed`, exit `139 (SIGSEGV)`; ranks 1-3 are `cosched_failed` after rank 0 bounced.
  - Iris automatically started a second four-rank setup attempt. The parent was stopped at `16:02:49Z` because the first attempt had already produced the performance verdict. W&B remains stale `running` at global step 2 because rank 0 did not finalize it.
- Terminal state:
  - Parent and child are terminal `killed`. All four child tasks are complete, the retry is stopped, and `iris job list --prefix /dlwh/iris-run-job-20260727-154333` reports no live descendant.
  - Preserve the relative-L2 `<=0.002` acceptance ceiling for floating output, loss, and every gradient leaf, with exact routing, counts, and drops. This throughput run did not change the numerical policy.
- UB-X sidecar:
  - Commit `5c09dc0e98` contains the reviewed optimistic EP8 direct transport gate. Local tests and pre-commit passed; no UB-X GPU benchmark result exists yet.
- Decision:
  - Do not promote or retry the exact `7,6,6,5` Ring configuration. It is slower than the `18.2583` baseline and exceeds rank-0 memory before completing 12 steps.

### 2026-07-27 09:45 PDT - NCCL UB-X direct transport passes balanced and learned-skew gates
- Snapshot:
  - R12 `/dlwh/nccl-ubx-direct-ep8-r12-20260727` used clean commit `cf01920bda` on one `H100x8` node in `cw-rno2a`.
  - R13 `/dlwh/nccl-ubx-direct-ep8-r13-skew-20260727` used clean commit `b50ce7d1a1` on the same cluster and ran only `--routing learned_skew`.
  - Both jobs built NVIDIA/nccl commit `db0c814185a0415cc2e23dca387fecb9282de551` as NCCL `2.30.7-1`, nccl4py, and UB-X `0.01`. Runtime validation returned NCCL version `23007` with exactly one mapped library, `/tmp/nccl-ubx-db0c814/build/lib/libnccl.so.2.30.7`.
  - The accepted policy at `b50ce7d1a1` requires exact route/map/count/drop data, bitwise-exact dispatch, UB-X versus the FP32 identity reference relative L2 `<=0.002`, and slowest-rank p50 speedup `>=1.10x`. Ring-versus-FP32 and UB-X-versus-Ring remain diagnostics.
- Setup findings:
  - R5-R11 were setup or adapter failures and produced no transport result. The sequence identified the missing shared-cudart link override, Torch's older NCCL soname collision, the compile-time-versus-runtime NCCL version distinction, omitted nccl4py CUDA Python dependencies, and a pinned UB-X namespace bug.
  - Commit `cf01920bda` audits the unchanged upstream namespace behavior and redirects `ubx._nccl_backend._nccl_bindings` from the empty `nccl.bindings` package to the compiled `nccl.bindings.nccl` module. No NVIDIA source file is patched.
- R13 exact command:

```bash
uv run iris --cluster cw-rno2a job run --no-wait \
  --enable-extra-resources --gpu H100x8 --cpu 64 --memory 256GB --disk 256GB \
  --timeout 3600 --max-retries 0 --priority interactive \
  --extra gpu --sync-package marin-levanter \
  --job-name nccl-ubx-direct-ep8-r13-skew-20260727 -- \
  bash -lc 'set -euxo pipefail
source "$IRIS_VENV/bin/activate"
cd "$IRIS_WORKDIR"
uv pip install --no-deps --reinstall --index-url https://download.pytorch.org/whl/cu130 "torch==2.11.0+cu130"
uv pip install --link-mode copy --reinstall "nvidia-cuda-nvcc==13.2.78" "nvidia-nvvm==13.2.78" "nvidia-cuda-cccl==13.3.3.4.1" "nvidia-cuda-runtime==13.2.75"
uv pip install --link-mode copy "cuda-core==1.0.1" "cuda-pathfinder==1.5.4" "cuda-bindings==13.2.0"
python -c "import cuda.core; print(\"cuda.core import ok\", cuda.core.__file__)"
cuda_bin="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/cu13/bin -name nvcc -print -quit)"
test -n "$cuda_bin"
CUDA_HOME="$(dirname "$(dirname "$cuda_bin")")"
export CUDA_HOME
ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
grep "#define CUDART_VERSION" "$CUDA_HOME/include/cuda_runtime_api.h" | head -1
cccl_target="$(find "$CUDA_HOME/include/cccl" -path "*/nv/target" -print -quit)"
test -n "$cccl_target"
"$CUDA_HOME/bin/nvcc" --version
python -c "import torch; print(\"Torch CUDA\", torch.__version__, torch.version.cuda)"
SOURCE=/tmp/nccl-ubx-db0c814
rm -rf "$SOURCE"
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/nccl.git "$SOURCE"
git -C "$SOURCE" fetch --depth 1 origin db0c814185a0415cc2e23dca387fecb9282de551
git -C "$SOURCE" checkout --detach FETCH_HEAD
make -C "$SOURCE" -j32 src.build CUDA_HOME="$CUDA_HOME" CUDA_LIB="$CUDA_HOME/lib" CUDARTLIB=cudart NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export NCCL_HOME="$SOURCE/build"
export NCCL_INCLUDE_DIR="$SOURCE/build/include"
export NCCL_LIBRARY_DIR="$SOURCE/build/lib"
export LD_LIBRARY_PATH="$NCCL_LIBRARY_DIR:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
wheel_nccl="$(find "$IRIS_VENV"/lib/python*/site-packages/nvidia/nccl/lib -name libnccl.so.2 -print -quit)"
test -n "$wheel_nccl"
rm -f "$wheel_nccl"
ln -s "$NCCL_LIBRARY_DIR/libnccl.so.2" "$wheel_nccl"
readlink -f "$wheel_nccl"
uv pip install --no-deps -e "$SOURCE/bindings/nccl4py"
TORCH_CUDA_ARCH_LIST=9.0a uv pip install --no-build-isolation --no-deps -e "$SOURCE/contrib/nccl_ubx"
export LD_PRELOAD="$NCCL_LIBRARY_DIR/libnccl.so.2"
export NCCL_UBX_SOURCE="$SOURCE"
export NCCL_UBX_OUTPUT_DIR=/tmp/nccl-ubx-direct-results
export UBX_GRAPH_POOL_SHARE=0.1
python -c "from experiments.grug.moe.nccl_runtime import nccl_runtime; print(\"loaded NCCL\", nccl_runtime())"
python -c "import torch, ubx; print(\"UBX runtime\", torch.__version__, torch.cuda.nccl.version(), ubx.get_version())"
ldd "$(python -c "import ubx._C; print(ubx._C.__file__)")" | grep libnccl
mkdir -p "$NCCL_UBX_OUTPUT_DIR"
torchrun --standalone --nproc-per-node=8 experiments/grug/moe/benchmark_nccl_ubx.py \
  --ubx-source "$SOURCE" --routing learned_skew \
  | tee "$NCCL_UBX_OUTPUT_DIR/learned_skew.jsonl"'
```

- Balanced R12 result:
  - Route digest `04e82c2839611c078eb5063e3dcb5adb8bc908ced183744cf8b9c3bf7c36f910`; `524288/524288` assignments accepted, zero dropped, `65536` accepted by every expert rank, and `8192` assignments per expert.
  - Every map check passed on rank 0, route/count/drop data were exact on all ranks, gathered experts were exact, and dispatch was bitwise exact.
  - All outputs were finite. All-rank maximum relative L2 was `0.0016620228719725061` for UB-X versus FP32 identity, `0.002751047558687074` for Ring versus FP32 identity, and `0.003171801610466874` for UB-X versus Ring. Dispatch versus reference was exactly zero.
  - Slowest-rank p50 was `6.503631830215454ms` for Ring and `2.0476959943771362ms` for UB-X, a `3.176072936643955x` speedup over 30 measured iterations.
  - The raw R12 JSON used the earlier all-metrics policy and reported admission false. The unchanged data pass the user-approved `b50ce7d1a1` policy because UB-X versus FP32 is below `0.002`, routing is exact, and speedup exceeds `1.10x`.
- Learned-skew R13 result:
  - Route digest `38a85732daa3831b7344b545cdee636c7fbd3e282cc1858df27345a40677c1bb`; `417371/524288` assignments accepted and `106917` dropped. Accepted assignments by expert rank were `[41649, 57491, 65536, 62002, 65536, 52039, 41163, 31955]`; drops were `[0, 0, 84582, 0, 22335, 0, 0, 0]`.
  - Original per-expert counts ranged from `1309` to `102289`; accepted counts ranged from `0` to `65536`. Every map check, route/count/drop check, gathered-expert check, and bitwise dispatch check passed.
  - All outputs were finite. All-rank maximum relative L2 was `0.0016605777289329705` for UB-X versus FP32 identity, `0.0027192861254322436` for Ring versus FP32 identity, and `0.003141772833169258` for UB-X versus Ring. Dispatch versus reference was exactly zero.
  - Slowest-rank p50 was `6.5939040184021ms` for Ring and `2.295423984527588ms` for UB-X, a `2.8726300948533328x` speedup over 30 measured iterations.
  - Admission was true. Iris completed R13 successfully with one succeeded task, zero failures, and zero preemptions.
- Interpretation:
  - UB-X passes the direct EP8 identity-transport gate for balanced and capacity-limited learned-skew routing. This earns a JAX FFI integration experiment.
  - This is an optimistic precomputed-map identity transport result. It excludes routing-map construction, expert compute, gradients, JAX/JaxPP integration, pipeline scheduling, and end-to-end MFU. It does not establish or imply the `>20` MFU target.
- Terminal state:
  - R12 is terminal `failed` because its older policy rejected diagnostic Ring errors after emitting the complete balanced JSON. R13 is terminal `succeeded`. Neither job has a live task.

### 2026-07-27 10:00 PDT - Compact UB-X slot map removes sparse receive expansion
- Snapshot: `2ff09a7907` (`[grug] Compact UB-X expert slots`).
- Finding:
  - UB-X `a2av_token_bf16_bf16_topk` treats `topk_slot` as an arbitrary destination row. It derives only the destination rank from the expert id.
  - PUSH3 phase 1 similarly treats each valid `inverse_map` row as an arbitrary expert-output row and scans exactly `max_tokens_per_rank` rows.
  - The upstream Python map helper assigns each local expert a fixed `max_tokens_per_expert` segment. On R13 skew this expands the dispatch buffer to `8 * 65536 = 524288` rows even though Ring accepts at most `65536` assignments per expert rank.
- Change:
  - `build_ubx_routing_maps` now uses exclusive per-expert accepted-count prefixes to assign contiguous expert-major segments inside the existing Ring rank capacity.
  - Dispatch output, expert input/output, and `inverse_map` are therefore fixed at `65536` rows for the target shape. `group_sizes` already follows the same expert-major order, and `dispatch_valid` masks the unused tail.
  - The small JIT test covers a rank whose second local expert receives an accepted segment after the first expert and verifies dispatch slots, inverse maps, exact counts, one dropped route, and padded group sizes.
- Validation:
  - `uv run pytest lib/levanter/tests/grug/test_ep_ubx_maps.py`: `1 passed`.
  - `./infra/pre-commit.py --changed-files --fix`: passed, including Pyrefly.
- Interpretation:
  - The compact layout removes the planned sparse gather/scatter and reduces the learned-skew dispatch buffer by `8x` without changing accepted routes or drops.
  - GPU transport correctness and timing for this compact slot layout remain unmeasured. The next gate must validate it through the JAX FFI wrapper before any MoE or JaxPP run.

### 2026-07-27 10:15 PDT - Compact UB-X slot layout passes the direct EP8 gate
- Snapshot:
  - Commit `9bb89438dc` adds `--slot-layout compact` to the direct gate.
  - R14 `/dlwh/nccl-ubx-direct-ep8-r14-compact-20260727` ran both routing cases on one `H100x8` node in `cw-rno2a`.
  - The first task incarnation exited `126` before setup because the synced launcher was not executable. The babysitter resubmitted the same job with `bash experiments/grug/moe/run_nccl_ubx_direct_gate.sh --slot-layout compact`; the replacement task succeeded with no failures or preemptions, and no task remains live.
  - The remaining setup and pinned NVIDIA/nccl build command were identical to R13.
- Results:

| Routing | Accepted / dropped | UB-X vs FP32 rel. L2 | Ring p50 | UB-X p50 | Speedup | Dispatch bytes/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | `524288 / 0` | `0.001662022872` | `6.500688076ms` | `2.048016071ms` | `3.174139191x` | `335544320` |
| learned skew | `417371 / 106917` | `0.001660577729` | `6.604256153ms` | `1.912575960ms` | `3.453068684x` | `335544320` |

- Correctness:
  - Both cases passed exact route, map, count, and drop checks across all ranks. Dispatch was bitwise exact.
  - Learned-skew drops remained `[0, 0, 84582, 0, 22335, 0, 0, 0]`, matching R13.
  - Both candidate errors passed the user-approved `0.002` relative-L2 ceiling.
  - Runtime validation found NCCL version `23007`, exactly one mapped `/tmp/nccl-ubx-db0c814/build/lib/libnccl.so.2.30.7`, and UB-X `0.01`.
- Interpretation:
  - Balanced routing is unchanged because its upstream expert strides already total `65536` rows.
  - Learned-skew receive storage fell from `524288` rows (`2.5GiB`) to `65536` rows (`320MiB`) per rank. Its UB-X p50 improved from R13's `2.295423985ms` to `1.912575960ms`, and speedup over Ring increased from `2.872630095x` to `3.453068684x`.
  - The compact map is now admitted for the raw JAX FFI gate. Expert compute, gradients, JaxPP scheduling, and end-to-end MFU remain unmeasured.
