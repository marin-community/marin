# SonicMoE Token Gather/Sum Port-Aware Diff

This is a semantic diff, not a textual diff. It maps the isolated upstream
SonicMoE Triton token gather/sum kernel to the current JAX/Pallas port blocks.
GitHub anchors target the
[`codex/sonic-equivalent-pallas`](https://github.com/marin-community/marin/tree/codex/sonic-equivalent-pallas)
branch.

Scope: fixed-top-k token gather/sum

```text
out[t, h] = sum_k x[reverse_scatter[t, k], h] * weight[t, k]
```

For Grug forward combine, `weight` is the router combine weight. For the
backward token-broadcast analogue, `weight` can be all ones.

## Files

| Role | File |
|---|---|
| Real Sonic source | [`.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py`](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py) |
| Port adapter | [`.agents/scripts/sonicmoe_compare/pallas_token_gather_sum_port.py`](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/pallas_token_gather_sum_port.py) |
| Pallas implementation | [`lib/levanter/src/levanter/grug/sonic_moe.py`](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py) |
| Runner | [`.agents/scripts/sonicmoe_compare/compare_token_gather_sum.py`](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/compare_token_gather_sum.py) |

## Contract Map

| Concept | Real Sonic | Pallas port | Notes |
|---|---|---|---|
| Source rows | `x_ptr`, `stride_xM`, `stride_xH`, [lines 112-124](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L112-L124) and [156-158](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L156-L158) | `dispatch_output`, [lines 524-529](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L524-L529) / [560-565](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L560-L565); loads at [546-548](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L546-L548) / [581-583](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L581-L583) | Same logical buffer: expert-sorted or reverse-scatter-addressed rows by hidden column. |
| Reverse scatter positions | `m_perm_ptr`, [line 114](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L114); loaded at [154](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L154) | `dispatch_positions_ref`, loaded at [544](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L544) / [579](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L579) | Same fixed-top-k reverse map, but Pallas stores it as `[T, K]`; Sonic uses flattened `[T * K]`. |
| Optional weights | `w_ptr`, [line 113](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L113); `w_is_None`, [line 127](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L127); loaded at [162](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L162) | `combine_weights_ref`, loaded at [545](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L545) / [580](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L580) | Pallas side currently always has a weights array; unweighted mode passes ones. |
| Repeat offsets | `repeat_offsets_ptr`, [line 116](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L116); loaded at [148](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L148) | `repeat_offsets_ref`, loaded at [543](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L543) / [578](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L578) | Same benchmark trick for amortization / replicated-input experiments. |
| Output | `out_ptr`, [line 117](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L117); store at [165-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L165-L166) | `output_ref`, store at [557](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L557) / [590](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L590) | Same `[T, H]` output contract. |
| Top-k extent | `MAX_K`, [line 120](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L120); fixed path at [139-140](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L139-L140) | `topk`, [lines 531](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L531) / [567](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L567); `topk_offsets`, [lines 537](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L537) / [574](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L574) | Current Pallas path only mirrors fixed top-k, not Sonic varlen-K. |
| Hidden tile | `BLOCK_H`, [line 125](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L125); `h_idx` at [142-145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L145) | `hidden_block_size`, [lines 532](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L532) / [568](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L568); `hidden_offsets` at [540](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L540) / [573](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L573) | Same conceptual tile. The two Pallas variants assign this tile differently in the launch grid. |

## Kernel Body Map

| Sonic block | Real Sonic lines | Closest Pallas block | Pallas lines | Status |
|---|---:|---|---:|---|
| Autotune `BLOCK_H`, `BLOCK_K`, warps | [79-108](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L79-L108) | Manual block size config in adapter and wrapper | adapter [78-83](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/pallas_token_gather_sum_port.py#L78-L83), implementation [913](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L913) / [975](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L975) | Not a faithful port. Pallas currently takes explicit tile sizes; Sonic autotunes. |
| One program per token | [131-132](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L131-L132) | Token-loop variant: one program per token | [536](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L536) | Faithful. |
| One program per token and hidden block | [131-132](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L131-L132) + hidden tile loop [142](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142) | Token-kblock variant: grid `(token, hidden_block)` | [571-573](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L571-L573) and [971-974](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L971-L974) | Intentional divergence. This exposes more parallelism and avoids an inner hidden-tile loop. |
| Fixed top-k start/end | [138-140](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L138-L140) | `topk_offsets = arange(topk)` | [537](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L537) / [574](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L574) | Faithful for fixed top-k only. |
| Hidden tile offsets and mask | [142-145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L145) | Hidden offsets, no explicit H mask after padding | [539-541](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L539-L541) / [572-573](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L572-L573) | Semantically equivalent after `_pad_for_pallas`; padding removes per-load/store hidden masks. |
| Accumulator initialized in fp32 | [145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L145) | Accumulator initialized in output dtype | [541](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L541) / [576](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L576) | Divergence. Sonic accumulates `tl.float32`; current Pallas accumulates output dtype except for final repeat averaging. This is worth testing/tightening. |
| Repeat loop | [147-148](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L147-L148) | Repeat loop | [542-543](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L542-L543) / [577-578](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L577-L578) | Faithful. |
| K-tile loop over `BLOCK_K` | [149-154](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L149-L154) | Vector load over all top-k positions | [537](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L537), [544](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L544) / [574](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L574), [579](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L579) | Divergence. Sonic supports `BLOCK_K`; current Pallas assumes small fixed top-k and loads all K at once. |
| Gather source rows | [154-158](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L154-L158) | Gather source rows | [544-548](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L544-L548) / [579-583](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L579-L583) | Faithful at the logical level. |
| Weighted/unweighted accumulation | [159-163](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L159-L163) | Always weighted accumulation, with unit weights for unweighted tests | [545-550](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L545-L550) / [580-584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L580-L584) | Semantically equivalent for harness cases; not source-faithful to `w_is_None`. |
| Store output tile | [165-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L165-L166) | Store output tile | [557](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L557) / [590](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L590) | Faithful after padding. |

## Which Pallas Variant Lines Up Best?

`_gather_sum_pallas_triton_token_loop_kernel` is the closest source-shape port:

- Real Sonic uses one program per token, then loops hidden tiles inside the
  kernel.
- Token-loop Pallas also uses one program per token and loops hidden tiles
  inside the kernel.
- This is the right file block to compare when asking whether our source
  structure resembles Sonic.

`_gather_sum_pallas_triton_token_kblock_kernel` is a performance-oriented
variant:

- It splits hidden tiles into the launch grid: `(token, hidden_block)`.
- It is less source-faithful, but easier for XLA/Pallas to schedule and compare
  against the tiled Pallas path.

The older `_gather_sum_pallas_triton_kernel` is a block-token/block-hidden
variant and is not the faithful Sonic source-shape port.

## Main Divergences To Track

1. **Autotuning**: Sonic autotunes `BLOCK_H`, `BLOCK_K`, and warps. The port uses
   explicit `SonicGatherSumBlockSizes`.
2. **K blocking**: Sonic has a `BLOCK_K` loop. The current Pallas fixed-top-k
   path loads all top-k entries in one vector.
3. **Accumulator dtype**: Sonic accumulates in fp32. Current Pallas accumulator
   is output dtype for the hot loop. This may affect both numerics and generated
   code.
4. **Masks vs padding**: Sonic masks hidden and K lanes. Pallas pads tokens and
   hidden dimension before launch, then stores unmasked within the padded extent.
5. **Optional no-weight path**: Sonic has `w_is_None`. The Pallas harness uses
   unit weights to represent the unweighted case.
6. **Varlen-K**: Sonic supports `is_varlen_K`. The current Pallas comparison
   covers fixed top-k only.

## Reading The Next IR/PTX Report

Use this map as the source alignment before looking at IR:

1. Compare real Sonic [lines 142-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L166)
   against Pallas token-loop
   [lines 539-557](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L539-L557).
2. Compare real Sonic [lines 147-163](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L147-L163)
   against Pallas token-kblock
   [lines 577-584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L577-L584)
   if the benchmark uses `pallas_triton_token_kblock`.
3. Treat differences caused by padding and launch-grid decomposition separately
   from actual math differences.
4. If Pallas PTX still has many more global load instructions, the first source
   suspects are K blocking, accumulator dtype, and launch-grid structure.
