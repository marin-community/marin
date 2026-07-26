# JaxPP group-2 parity debug log

## Problem

The target H100x8 grouped final-stage gate at
`/dlwh/jaxpp-group2-moe-boundary-parity-20260726-212357` produced a matching
loss but failed the per-leaf relative-L2 gradient gate. The investigation must
preserve the `0.002` threshold.

## Evidence

- Loss relative-L2: `3.5869572e-06`.
- QB beta relative-L2: `0.0034909921`.
- 24 of 25 gradient leaves failed. Expert and router leaves reached about
  `1.38` relative-L2; input and attention leaves reached about `0.68`.
- The prior EP8 direct ring benchmark compared two ring calls in one
  `value_and_grad` with two separately differentiated calls and reported
  bitwise-identical values and gradients. That benchmark supplied fixed expert
  selections and combine weights, so it did not exercise learned routing.
- A tiny CPU/reference full-stage comparison passes in FP32. In BF16 its
  gradients remain aligned and similarly scaled, with maximum relative-L2
  about `0.00251`. That narrowly fails the fixed gate and is not promotable,
  although its aligned gradients do not explain the target's order-one
  gradient-direction failure.

## Ranked hypotheses

1. Packed attention or packed dense preparation perturbs post-attention values
   enough to change learned top-k routes. Discrete route changes can preserve
   scalar loss while rotating router, expert, input, and attention gradients.
2. Two learned-router exact-ring calls inside one reverse pass interact
   incorrectly. The fixed-routing direct benchmark does not exclude this.
3. The `save_moe` checkpoint policy aliases same-name saved values from the two
   calls. Comparing no checkpoint, `recompute_all`, and `save_moe` isolates it.
4. The final norm/head weighted-loss oracle is wrong. The near-exact summed
   loss and separate denominators make this less likely.

## Next checks

- Add actual/reference norm ratio and cosine to every reported value and
  gradient leaf.
- Compare learned-router MoE pair value/VJP with joined and separately
  differentiated calls on identical post-attention inputs.
- Compare grouped block gradients across no checkpoint, `recompute_all`, and
  `save_moe`.
- Compare packed attention value/VJP with ordered attention calls.
- Compare the final norm/head pair loss and gradients independently with
  unequal loss-weight denominators.

## CPU/reference isolation

All checks used JIT, one-device exact ring, reference attention, unequal
weighted-loss denominators, and the unchanged `0.002` gate. The learned-router
fixture uses top-k2 so its router gradient is nonzero.

- Two MoE calls in one `value_and_grad` versus two separate calls: maximum
  relative-L2 `0.0` across 36 leaves.
- Production `grouped_moe_residual` packed preparation versus two complete
  `moe_residual` calls: maximum relative-L2 `4.4373545e-07`, on the projected
  scalar; norm ratio `1.00000044`, cosine `1.0`.
- Grouped `recompute_all` versus no checkpoint: maximum relative-L2 `0.0`
  across 34 leaves.
- Grouped `save_moe` versus no checkpoint: maximum relative-L2
  `1.3314632e-07`, on `mlp.expert_mlp.w_gate`; norm ratio
  `0.99999991`, cosine `0.99999984`.
- Packed reference attention versus two ordered calls: maximum relative-L2
  `6.9115365e-08`, on `attn.w_k`; norm ratio `1.0`, cosine
  `0.99999986`.
- Joined final norm/head weighted losses versus separate differentiation:
  maximum relative-L2 `0.0` across 26 leaves.

A top-k1 trial initially showed a large relative error only on a roughly
`2e-11` router-gradient norm. Top-k1's normalized combine weight is constant,
so this was roundoff around a mathematically zero gradient. The same effect
occurred for a single checkpointed block, excluding pair-specific saved-value
aliasing.

## Current classification

- Harness oracle bug: not supported. The final-head oracle and full FP32
  reference path agree.
- Generic remat/custom-VJP aliasing: not supported by the CPU checkpoint
  comparison. A target-GPU comparison remains necessary because the production
  ring and CuTe paths have different custom calls.
- Exact-ring pair VJP bug: fixed-routing EP8 evidence and learned-routing CPU
  evidence both argue against it, but target EP8 learned routing is still
  untested.
- Packed attention behavior: leading hypothesis. BF16 packed attention can
  alter post-attention values, which may change discrete top-k assignments and
  explain why the scalar loss matches while expert/router gradient directions
  do not.

The smallest next GPU diagnostic is the existing H100x8 harness with
`--diagnostic moe-pair`. It reports two comparisons on identical BF16 inputs
with EP8 exact ring and the target Pallas-Triton expert kernels:

1. two complete learned-router MoE calls in one reverse pass versus separately
   differentiated calls, isolating ring pair-VJP behavior;
2. production packed RMS/GatedNorm preparation versus the two complete calls,
   including exact selected-expert mismatch counts and top-k boundary margins,
   isolating pre-router numerical changes and selected-route effects.

If both pass, the next diagnostic should compare packed CuTe attention
value/VJP before running the full stage.
