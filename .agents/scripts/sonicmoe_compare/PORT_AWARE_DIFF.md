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
| Source rows | `x_ptr`, `stride_xM`, `stride_xH`, [lines 112-124](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L112-L124) and [156-158](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L156-L158) | `dispatch_output`, [lines 563-568](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L563-L568); loads at [600-604](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L600-L604) | Same logical buffer: expert-sorted or reverse-scatter-addressed rows by hidden column. |
| Reverse scatter positions | `m_perm_ptr`, [line 114](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L114); loaded at [154](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L154) | `dispatch_positions_ref`, loaded at [592-598](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L592-L598) | Same fixed-top-k reverse map, but Pallas stores it as `[T, K]`; Sonic uses flattened `[T * K]`. |
| Optional weights | `w_ptr`, [line 113](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L113); `w_is_None`, [line 127](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L127); loaded at [162](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L162) | `combine_weights_ref`, loaded at [605-610](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L605-L610); branch controlled at [577](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L577) | Faithful path preserves a no-weight branch for the comparison harness. The public Grug API still passes weights. |
| Repeat offsets | `repeat_offsets_ptr`, [line 116](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L116); loaded at [148](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L148) | `repeat_offsets_ref`, loaded at [586-587](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L586-L587) | Same benchmark trick for amortization / replicated-input experiments. |
| Output | `out_ptr`, [line 117](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L117); store at [165-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L165-L166) | `output_ref`, store at [615-616](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L615-L616) | Same `[T, H]` output contract. |
| Top-k extent | `MAX_K`, [line 120](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L120); fixed path at [139-140](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L139-L140) | `topk`, [line 570](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L570); `k_block_size` / `k_tiles`, [lines 574-575](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L574-L575); `k_offsets`, [lines 588-591](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L588-L591) | Faithful for fixed top-k only, including explicit `BLOCK_K`-style tiling. |
| Hidden tile | `BLOCK_H`, [line 125](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L125); `h_idx` at [142-145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L145) | `hidden_block_size`, [line 572](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L572); `hidden_offsets` and mask at [581-584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L581-L584) | Same conceptual tile and mask structure. |

## Kernel Body Map

| Sonic block | Real Sonic lines | Closest Pallas block | Pallas lines | Status |
|---|---:|---|---:|---|
| Autotune `BLOCK_H`, `BLOCK_K`, warps | [79-108](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L79-L108) | Manual block size config in adapter and wrapper | adapter [86-91](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/pallas_token_gather_sum_port.py#L86-L91), dataclass [47-51](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L47-L51) | Remaining divergence. Pallas takes explicit tile sizes; manual sweep stands in for Sonic autotune. |
| One program per token | [131-132](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L131-L132) | Faithful Pallas token program | [579](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L579) | Faithful. |
| One program per token with hidden-tile loop | [131-132](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L131-L132) + hidden tile loop [142](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142) | Faithful Pallas hidden loop | [581-584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L581-L584) | Faithful. |
| Fixed top-k start/end | [138-140](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L138-L140) | `k_offsets` and mask | [588-591](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L588-L591) | Faithful for fixed top-k only. |
| Hidden tile offsets and mask | [142-145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L145) | Hidden offsets and mask | [581-584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L581-L584) | Faithful. |
| Accumulator initialized in fp32 | [145](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L145) | fp32 accumulator | [584](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L584) | Faithful. |
| Repeat loop | [147-148](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L147-L148) | Repeat loop | [586-587](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L586-L587) | Faithful. |
| K-tile loop over `BLOCK_K` | [149-154](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L149-L154) | K-tile loop over `k_block_size` | [588-604](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L588-L604) | Faithful for fixed top-k. |
| Gather source rows | [154-158](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L154-L158) | Gather source rows | [592-604](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L592-L604) | Faithful at the logical level. |
| Weighted/unweighted accumulation | [159-163](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L159-L163) | Weighted/unweighted branch | [605-613](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L605-L613) | Faithful for the comparison harness. |
| Store output tile | [165-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L165-L166) | Store output tile | [615-616](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L615-L616) | Faithful. |

## Which Pallas Variant Lines Up Best?

`_gather_sum_pallas_triton_faithful_kernel` is the source-shape control:

- Real Sonic uses one program per token, then loops hidden tiles inside the
  kernel.
- Faithful Pallas also uses one program per token and loops hidden tiles inside
  the kernel.
- It keeps a `BLOCK_K`-style loop, fp32 accumulator, masks, and weighted /
  unweighted branch. Use this path when asking whether Pallas can lower the
  same recipe as Sonic.

`_gather_sum_pallas_triton_token_kblock_kernel` is a performance-oriented
variant:

- It splits hidden tiles into the launch grid: `(token, hidden_block)`.
- It is less source-faithful, but easier for XLA/Pallas to schedule and compare
  against the tiled Pallas path.

`_gather_sum_pallas_triton_token_loop_kernel` is the older closest-source
attempt. It still differs from Sonic by loading all fixed top-k routes at once,
accumulating in output dtype, and relying on padding.

The older `_gather_sum_pallas_triton_kernel` is a block-token/block-hidden
variant and is not the faithful Sonic source-shape port.

## Main Divergences To Track

1. **Autotuning**: Sonic autotunes `BLOCK_H`, `BLOCK_K`, and warps. Faithful
   Pallas uses explicit `SonicGatherSumBlockSizes`, so manual sweeps are needed.
2. **Flattened vs 2D reverse map**: Sonic uses flattened `m_perm_ptr`; Pallas
   keeps `dispatch_positions_ref` as `[T, K]`. The address math is equivalent
   for fixed top-k, but the source is not byte-for-byte identical.
3. **Varlen-K**: Sonic supports `is_varlen_K`. The current Pallas comparison
   covers fixed top-k only.
4. **Compiler stack**: Sonic lowers through Triton directly. Pallas lowers
   through JAX/Pallas/Triton custom calls. Host-side JAX dispatch is material
   for standalone one-kernel microbenchmarks, but a compiled JAX executable can
   amortize many Pallas kernel launches; see the launch-overhead check below.

## PTX Check, 2026-05-11

Jobs:

- `/dlwh/sonicmoe-faithful-ptx2-20260511-121635`: real Sonic plus fastest
  faithful Pallas PTX dump.
- `/dlwh/sonicmoe-autotune-id-20260511-121825`: Sonic autotune identification.
- `/dlwh/sonicmoe-selected-ptx-20260511-121937`: selected Sonic PTX summary.
- `/dlwh/sonicmoe-pallas-ptx-match-20260511-122056`: Pallas PTX with the same
  tile/warp shape as Sonic's selected autotune config.

Shape: `tokens=8192`, `hidden=2048`, `experts=8`, `topk=4`, weighted BF16 on
GH200.

Sonic's Triton autotuner selected `BLOCK_H=2048`, `BLOCK_K=2`, `num_warps=8`.
The fastest Pallas timing from the manual sweep was still
`BLOCK_H=4096`, `BLOCK_K=4`, `num_warps=8`, but the matched-tile PTX is the
cleaner source-comparison artifact.

| Path | Tile | PTX bytes | Lines | `ld.global` | `st.global` | `add.*` | `mul.*` | `cvt.*` | `fma.*` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonic selected Triton | `H=2048`, `K=2`, `warps=8` | `19.6KB` | `522` | `9` | `1` | `28` | `17` | `45` | `16` |
| Pallas faithful, matched tile | `H=2048`, `K=2`, `warps=8` | `8.2KB` | `271` | `13` | `1` | `49` | `39` | `40` | `0` |
| Pallas faithful, fastest manual tile | `H=4096`, `K=4`, `warps=8` | `13.5KB` | `435` | `17` | `2` | `91` | `76` | `76` | `0` |

The matched-tile PTX changes the earlier diagnosis. The old token-loop Pallas
artifact was much larger than Sonic's PTX; the source-faithful matched-tile PTX
is smaller in bytes/lines than Sonic, but it still emits more global loads and
substantially more integer address arithmetic. The load pattern is:

- Sonic selected: one repeat-offset load, one vectorized reverse-position load
  for the `K=2` tile, several vectorized source/weight loads, and one vectorized
  output store.
- Pallas matched tile: one repeat-offset load, two scalar reverse-position
  loads, two vectorized source loads and two scalar weight loads per `K=2` tile,
  repeated for the two K tiles, and one vectorized output store.

### Load-Origin Check, 2026-05-11

Follow-up jobs isolated the `13` versus `9` load count:

- `/dlwh/sonicmoe-load-origin-20260511-ptx13b-132328`: selected the Pallas
  matched-tile artifact
  `/tmp/xla_pallas_faithful_ptx13/module_18531.jit_call.ptx` with `13`
  `ld.global` and `1` `st.global`.
- `/dlwh/sonicmoe-load-origin-20260511-sonic9-132509`: selected the real
  Sonic `K=2` artifact
  `/tmp/sonicmoe_load_origin/upstream/real_sonic/kernel.fn.device_caches.dict_0.tuple_0.dict_15.asm.ptx`
  with `9` `ld.global` and `1` `st.global`.

The extra four Pallas loads are not extra `dispatch_output` tensor loads. They
come from scalarization of the two-element K tile metadata:

| Logical load group | Sonic selected PTX | Pallas matched PTX | Difference |
|---|---:|---:|---:|
| `repeat_offsets[repeat_index]` | `1` scalar `b32` | `1` scalar `b32` | `0` |
| Reverse positions, tile `K=2` | `2` vector `v2.b32` loads, one per K tile | `4` scalar `b32` loads, one per route | `+2` |
| Source rows, two rows per K tile | `4` vector `v4.b32` loads | `4` vector `v4.b32` loads | `0` |
| Combine weights, tile `K=2` | `2` packed `b32` loads, one per K tile | `4` scalar `b16` loads, one per route | `+2` |
| Total loads | `9` | `13` | `+4` |

Real Sonic TTIR for the selected artifact has vector K-tile loads:

```text
%perm_idx_18 = tt.load ... : tensor<2x!tt.ptr<i32>>
%x_vals = tt.load ... : tensor<2x2048x!tt.ptr<bf16>>
%w_vals_35 = tt.load ... : tensor<2x!tt.ptr<bf16>>
```

Those lower in PTX to one `ld.global.v2.b32` for two reverse positions and one
packed `ld.global.b32` for two BF16 weights per K tile. The current faithful
Pallas source also expresses `K=2`, but XLA/Pallas emits scalar loads for
`dispatch_positions_ref.at[token_index, safe_k_offsets]` and
`combine_weights_ref.at[token_index, safe_k_offsets]`.

Concrete Pallas load map from
`/dlwh/sonicmoe-load-origin-20260511-ptx13b-132328`:

```text
45   repeat offset
54   dispatch_positions[t, 0]
58   dispatch_positions[t, 1]
76   dispatch_output[row0, h:h+8]
87   dispatch_output[row1, h:h+8]
115  combine_weights[t, 0]
119  combine_weights[t, 1]
159  dispatch_positions[t, 2]
163  dispatch_positions[t, 3]
180  dispatch_output[row2, h:h+8]
191  dispatch_output[row3, h:h+8]
217  combine_weights[t, 2]
221  combine_weights[t, 3]
266  output store
```

Concrete Sonic selected load map from
`/dlwh/sonicmoe-load-origin-20260511-sonic9-132509`:

```text
48   repeat offset
62   packed reverse positions for K tile 0
83   dispatch_output[row0, h:h+8]
90   dispatch_output[row1, h:h+8]
97   packed BF16 weights for K tile 0
105  packed reverse positions for K tile 1
125  dispatch_output[row2, h:h+8]
132  dispatch_output[row3, h:h+8]
139  packed BF16 weights for K tile 1
302  output store
```

Next source-level experiment: special-case exact fixed K tiles where
`topk % k_block_size == 0` in the faithful Pallas path and remove the `k_mask`
/ `safe_k_offsets` shape guards for the `K=2` metadata loads. If Pallas still
emits scalar `dispatch_positions` and `combine_weights` loads, reduce this to a
minimal Pallas repro that loads `i32[2]` and `bf16[2]` vectors from contiguous
memory and compare against Triton.

### Fixed-K Metadata Result, 2026-05-11

Job `/dlwh/sonicmoe-pallas-fixedk-jitcall-20260511-220348` confirmed that the
exact fixed-K branch fixes the textual global-load count:

| Path | Tile | PTX bytes | Lines | `ld.global` | `st.global` | `add.*` | `mul.*` | `cvt.*` | `fma.*` | Steady time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonic selected Triton | `H=2048`, `K=2`, `warps=8` | `19.6KB` | `522` | `9` | `1` | `28` | `17` | `45` | `16` | `0.072ms` CUDA event / `0.086ms` wall |
| Pallas faithful before fixed-K | `H=2048`, `K=2`, `warps=8` | `8.2KB` | `271` | `13` | `1` | `49` | `39` | `40` | `0` | `0.118ms` wall |
| Pallas faithful after fixed-K | `H=2048`, `K=2`, `warps=8` | `7.7KB` | `254` | `9` | `1` | `45` | `39` | `40` | `0` | `0.121ms` wall |

The selected Pallas `jit_call` memory instructions now line up structurally with
Sonic:

```text
45   repeat offset
53   packed reverse positions for K tile 0
72   dispatch_output[row0, h:h+8]
83   dispatch_output[row1, h:h+8]
109  packed BF16 weights for K tile 0
150  packed reverse positions for K tile 1
167  dispatch_output[row2, h:h+8]
178  dispatch_output[row3, h:h+8]
203  packed BF16 weights for K tile 1
249  output store
```

This means the tiny `i32[2]` / `bf16[2]` repro is no longer the next blocker for
this exact shape. Pallas can emit the packed loads when the source removes
partial-K guards. The remaining gap is now arithmetic/codegen shape: Pallas
still emits many more integer adds/muls and no textual `fma.*`, while Sonic
emits fewer address operations and `16` fma instructions.

### FMA And Flat-Metadata Result, 2026-05-11

There is no named `pltriton.fma` intrinsic in the local JAX/Pallas build. The
usable escape hatch is `pltriton.elementwise_inline_asm`. A whole-tile inline
ASM wrapper around the multiply side of the weighted sum lowers and emits real
`fma.rn.f32` instructions; a per-route slicing version does not lower because
Pallas GPU lowering does not currently handle that `slice` primitive in this
context. A `jnp.dot(weights, values)` rewrite also failed during Pallas/Triton
lowering with an invalid tensor shape assertion, so it is not a viable source
rewrite here.

Jobs:

- `/dlwh/sonicmoe-pallas-inlinefma-20260511-221005`: per-route inline-ASM
  attempt; failed during lowering on an unsupported `slice`.
- `/dlwh/sonicmoe-pallas-inlinefma2-20260511-221246`: whole-tile inline-ASM
  attempt; succeeded and emitted textual `fma.rn.f32`.
- `/dlwh/sonicmoe-pallas-flatmeta-20260511-221532`: flattened metadata inputs
  for the fixed-K source path, with and without inline FMA.

| Path | Metadata layout | FMA source | PTX bytes | Lines | `ld.global` | `st.global` | `add.*` | `mul.*` | `cvt.*` | `fma.*` | Steady time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pallas fixed-K | 2D source, fixed-K offsets | normal multiply/sum | `7.7KB` | `254` | `9` | `1` | `45` | `39` | `40` | `0` | `0.121ms` |
| Pallas inline FMA | 2D source, fixed-K offsets | whole-tile inline ASM | `9.3KB` | `322` | `9` | `1` | `49` | `7` | `40` | `32` | `0.115ms` |
| Pallas flat metadata | flattened metadata refs | normal multiply/sum | `7.7KB` | `254` | `9` | `1` | `45` | `39` | `40` | `0` | `0.117ms` |
| Pallas flat metadata + inline FMA | flattened metadata refs | whole-tile inline ASM | `9.3KB` | `322` | `9` | `1` | `49` | `7` | `40` | `32` | `0.116ms` |

Flattening `dispatch_positions` / `combine_weights` to one-dimensional metadata
refs did not materially change generated PTX or runtime once the fixed-K branch
was in place. Inline FMA does what we asked codegen to do, but the speedup is
small and noisy: it removes most textual `mul.*` instructions and introduces
`fma.rn.f32`, yet the best observed Pallas timing remains around `0.115ms`,
still slower than real Sonic's same-shape `0.086ms` wall / `0.072ms` CUDA event
measurement. The remaining gap is likely in address arithmetic, instruction
scheduling, or other lowering differences rather than just missing vector loads
or missing FMA contraction.

So the remaining isolated gather/sum gap is no longer explained by gross PTX
size or by K-metadata scalarization for the fixed `K=2` shape. The more
concrete PTX-level gap is duplicated address arithmetic, scheduling, and the
shape of the lowered arithmetic/reduction in the Pallas path. Host-dispatch
overhead can dominate
standalone JAX microbenchmarks, but the compiled-executable launch check below
suggests it should not be the fundamental limit inside a real JIT-compiled
training step.

## Launch Overhead Check, 2026-05-11

Benchmark script:
[`sonicmoe_launch_overhead_bench.py`](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_launch_overhead_bench.py).

Jobs:

- `/dlwh/sonicmoe-launch-overhead-20260511-123049`: Triton, Torch, JAX/XLA,
  and JAX/Pallas host-loop measurements.
- `/dlwh/sonicmoe-launch-overhead-compiled-20260511-123736`: Pallas calls
  chained inside one compiled JAX executable.

Shape: one float32 input/output element on one GH200. Each record reports the
median steady per-launch wall time at `launches=1000`; Triton/Torch also report
CUDA event timing for the batched enqueue case. JAX records are wall-time only.

| Path | Mode | Median per launch |
|---|---|---:|
| Triton tiny kernel | host loop, synchronize once | `6.06 us` wall / `6.04 us` CUDA event |
| Torch tiny add | host loop, synchronize once | `5.76 us` wall / `5.75 us` CUDA event |
| Triton tiny kernel | synchronize after each launch | `11.62 us` wall |
| Torch tiny add | synchronize after each launch | `12.60 us` wall |
| JAX/XLA tiny add | host loop, synchronize once | `48.98 us` wall |
| JAX/Pallas tiny kernel | host loop, synchronize once | `44.92 us` wall |
| JAX/XLA tiny add | synchronize after each call | `56.33 us` wall |
| JAX/Pallas tiny kernel | synchronize after each call | `55.08 us` wall |
| JAX/Pallas tiny kernel | 1000 chained Pallas calls in one compiled executable | `1.17 us` wall |

Interpretation:

- Direct Triton/Torch launch cadence on this GH200 is roughly `6 us` when
  launches are batched and roughly `12 us` with a host synchronize after each
  launch.
- A Python loop around a jitted JAX/Pallas one-kernel function costs roughly
  `45-55 us` per call. That is a dispatch benchmark, not a pure GPU-launch
  benchmark.
- When 1000 Pallas calls are chained inside a single compiled JAX executable,
  the effective cost drops to roughly `1.2 us` per call. This is best read as
  XLA executable/command-buffer cadence rather than raw `cudaLaunchKernel`
  latency, but it is the relevant lower-bound for kernels embedded in a
  training-step JIT.
- Therefore, host-dispatch overhead can explain bad standalone JAX
  one-kernel microbenchmarks, but it is unlikely to be the fundamental limit for
  the Grug MoE path inside a compiled training step. Continue looking at the
  kernel body and codegen differences first: scalarized reverse-position loads,
  duplicated weight/source loads, integer address arithmetic, and the backward
  token path.

## Reading The Next IR/PTX Report

Use this map as the source alignment before looking at IR:

1. Compare real Sonic [lines 142-166](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L142-L166)
   against faithful Pallas
   [lines 596-684](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L596-L684).
2. Compare real Sonic [lines 147-163](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/.agents/scripts/sonicmoe_compare/real_sonic_token_gather_sum.py#L147-L163)
   against faithful Pallas
   [lines 621-679](https://github.com/marin-community/marin/blob/codex/sonic-equivalent-pallas/lib/levanter/src/levanter/grug/sonic_moe.py#L621-L679).
3. Compare matched-tile PTX first (`H=2048`, `K=2`, `warps=8`) before comparing
   fastest-tile PTX, otherwise autotuning differences obscure source-lowering
   differences.
4. The fixed-K source path now removes the reverse-position and weight
   scalarization for the matched `K=2` case. The current first suspect is the
   remaining address/integer arithmetic and scheduling difference rather than
   missing `BLOCK_K`, accumulator dtype, masks, or launch-grid structure.
