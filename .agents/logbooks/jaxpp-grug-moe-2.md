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
  - Transforming the split functions into fresh VJP executables changes their returned primals before gradient comparison. The remaining failure is therefore in reverse-mode compilation/recomputation context, not the split forward dataflow or joined-MoE forward.
- Next action:
  - Preserve outputs and residuals from the validated forward executables and compile transpose-only backward tasks that consume those saved residuals. Do not use a fresh `value_and_grad` wrapper that recomputes each task's primals, and do not launch L8 or L24 until every backward leaf passes relative-L2 `0.002`.
