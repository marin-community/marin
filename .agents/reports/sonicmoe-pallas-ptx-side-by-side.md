# SonicMoE vs Pallas Gather/Sum PTX Side by Side

## Scope

This compares the selected upstream SonicMoE Triton token gather/sum kernel with the closest Grug/JAX Pallas port on the replicated-input repeat-16 weighted gather/sum shape:

- `T=8192`, `K=2`, `H=2048`, `E=8`, BF16
- `kernel_repeat=16`, `replicate_input=true`, `weighted=true`
- Hardware: `NVIDIA GH200 480GB`
- Rerun job: `/dlwh/sonicmoe-ptx-side-by-side-20260511-013637`
- Raw log: `.agents/reports/ptx-snippets/sonicmoe-ptx-side-by-side-20260511-013637-log.txt`
- Decoded metrics: `.agents/reports/ptx-snippets/sonicmoe-ptx-side-by-side-20260511-013637-metrics.json`

Command:

```bash
KUBECONFIG=$HOME/.kube/cw-rno2a.yaml uv run --package marin-iris --extra controller --group dev iris --cluster=coreweave-rno2a job run \
  --job-name sonicmoe-ptx-side-by-side-20260511-013637 --no-wait --max-retries 0 \
  --enable-extra-resources --gpu GH200x1 --cpu 16 --memory 128g --disk 256g --timeout 3600 \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION 0.85 -e XLA_PYTHON_CLIENT_PREALLOCATE false \
  -- bash -lc 'bash .agents/scripts/run_sonicmoe_ir_compare.sh && python .agents/scripts/sonicmoe_ptx_side_by_side.py --emit-log-records'
```

The helper extracted snippets from the raw `/tmp/sonicmoe_ir_compare`, `/tmp/xla_pallas_tiled`, and `/tmp/xla_pallas_token_loop` dumps while the GH200 job still had them.

## Metrics

| Path | Grid/config | Runtime | Text IR | PTX | Global mem ops | Snippet |
| --- | --- | ---: | --- | --- | --- | --- |
| Upstream SonicMoE Triton | autotuned `BLOCK_H=2048`, `BLOCK_K=1`, `num_warps=4` | `0.3253ms` CUDA event / `0.3385ms` wall | TTIR `11.9KB` / `176` lines / `4` `tt.load` / `2` `tt.reduce` / `1` `tt.store` | `14.3KB` / `406` lines | `9` `ld.global`, `2` `st.global` | `ptx-snippets/sonic_selected_ttir_ops.ttir`, `ptx-snippets/sonic_selected_ptx_mem.ptx` |
| Pallas token-loop, closest port | `grid_x=8192`, `grid_y=1`, `token_block=1`, `hidden_block=2048`, `num_warps=4` | `0.3856ms` wall | readable TTIR not exposed; StableHLO is one `@__gpu$xla.gpu.triton` custom call with embedded Triton bytecode (`34.5KB`) | `80.1KB` / `2481` lines | `82` `ld.global`, `2` `st.global` | `ptx-snippets/pallas_token_loop_stablehlo_custom_call.stablehlo.mlir`, `ptx-snippets/pallas_token_loop_ptx_mem.ptx` |
| Pallas tiled, context | `grid_x=1024`, `grid_y=8`, `token_block=8`, `hidden_block=256`, `num_warps=4` | `0.3899ms` wall | readable TTIR not exposed; StableHLO custom call with embedded Triton bytecode (`57.4KB`) | `75.4KB` / `2262` lines | `88` `ld.global`, `2` `st.global` | `ptx-snippets/pallas_tiled_ptx_mem.ptx` |

Pallas compile-inclusive timings from the rerun were `0.459s` for token-loop and `0.526s` for tiled. The runtime comparison above uses the steady-state timings printed by the existing harnesses.

## TTIR Contrast

Sonic's selected TTIR is a compact two-loop program: one repeat loop, one top-k loop, vector load over the full hidden width, reduction over K, one vector store.

```mlir
34:     %acc_6 = scf.for %repeat_index = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%acc_10 = %acc) -> (tensor<2048xf32>)  : i32 {
36:       %repeat_offset_11 = tt.load %repeat_offset : !tt.ptr<i32> loc(#loc65)
38:       %acc_13 = scf.for %k_tile = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc_14 = %acc_10) -> (tensor<2048xf32>)  : i32 {
45:         %perm_idx_17 = tt.load %perm_idx_16, %k_mask_15, %cst_3 : tensor<1x!tt.ptr<i32>> loc(#loc72)
59:         %x_vals = tt.load %x_ptrs_26, %x_mask_28, %cst_0 : tensor<1x2048x!tt.ptr<bf16>> loc(#loc83)
63:         %w_vals_31 = tt.load %w_vals_30, %k_mask_15, %cst : tensor<1x!tt.ptr<bf16>> loc(#loc86)
68:         %acc_36 = "tt.reduce"(%acc_35) <{axis = 0 : i32}> ({
84:     tt.store %out_ptrs_9, %1, %h_mask_5 : tensor<2048x!tt.ptr<bf16>> loc(#loc47)
```

The closest Pallas lowering does not expose comparable readable TTIR in the normal dumps. The StableHLO is a single custom call:

```mlir
%0 = stablehlo.custom_call @__gpu$xla.gpu.triton(%arg0, %arg1, %arg2, %arg3) {
  mhlo.backend_config = {
    grid_x = 8192 : i32,
    grid_y = 1 : i32,
    grid_z = 1 : i32,
    ir = "<embedded Triton bytecode omitted>",
    name = "sonic_gather_sum_pallas_triton_token_loop",
    num_stages = 4 : i32,
    num_warps = 4 : i32
  }
}
```

That means the useful text-level comparison is Sonic TTIR/TTGIR/PTX versus Pallas StableHLO custom-call metadata plus XLA-emitted PTX.

## PTX Contrast

Sonic's selected PTX has a small global-memory footprint. The full extracted memory-instruction snippet is only 11 lines:

```ptx
76: 	ld.global.b32 { %r5 }, [ %rd8 + 0 ];
84: 	@%p2 ld.global.b32 { %r6 }, [ %rd9 + 0 ];
104: 	@%p2 ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd10 + 0 ];
111: 	@%p2 ld.global.v4.b32 { %r12, %r13, %r14, %r15 }, [ %rd11 + 0 ];
117: 	@%p2 ld.global.b16 { %rs1 }, [ %rd12 + 0 ];
124: 	@%p2 ld.global.b32 { %r16 }, [ %rd13 + 0 ];
142: 	@%p2 ld.global.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd14 + 0 ];
149: 	@%p2 ld.global.v4.b32 { %r21, %r22, %r23, %r24 }, [ %rd15 + 0 ];
154: 	@%p2 ld.global.b16 { %rs3 }, [ %rd16 + 0 ];
294: 	@%p2 st.global.v4.b32 [ %rd27 + 0 ], { %r75, %r76, %r77, %r78 };
297: 	@%p2 st.global.v4.b32 [ %rd28 + 0 ], { %r79, %r80, %r81, %r82 };
```

The token-loop Pallas PTX has the same high-level one-token/full-hidden shape, but the generated PTX repeats load groups through the body. The extracted file has all `82` global loads plus the `2` stores; the first few already show the pattern:

```ptx
40: 	ld.global.b32 { %r1 }, [ %rd1 + 0 ];
48: 	ld.global.v2.b32 { %r2, %r3 }, [ %rd2 + 0 ];
56: 	ld.global.b32 { %r4 }, [ %rd3 + 0 ];
83: 	ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd4 + 0 ];
94: 	ld.global.v4.b32 { %r9, %r10, %r11, %r12 }, [ %rd5 + 0 ];
105: 	ld.global.v4.b32 { %r13, %r14, %r15, %r16 }, [ %rd6 + 0 ];
116: 	ld.global.v4.b32 { %r17, %r18, %r19, %r20 }, [ %rd7 + 0 ];
206: 	ld.global.b32 { %r21 }, [ %rd8 + 0 ];
233: 	ld.global.v4.b32 { %r22, %r23, %r24, %r25 }, [ %rd9 + 0 ];
244: 	ld.global.v4.b32 { %r26, %r27, %r28, %r29 }, [ %rd10 + 0 ];
...
2469: 	st.global.v4.b32 [ %rd83 + 0 ], { %r276, %r277, %r278, %r279 };
2476: 	st.global.v4.b32 [ %rd84 + 0 ], { %r280, %r281, %r282, %r283 };
```

The key point is not just that Pallas emits more PTX text. It emits many more textual global-load instructions for the closest source shape: `82` versus Sonic's `9`. The tiled Pallas variant is no better on this axis (`88` loads). That matches the runtime plateau from earlier tile/warp sweeps: the remaining gap is dominated by generated code shape, not an obvious source-level tile choice.

## Interpretation

Sonic is cleaner because the selected Triton program keeps the source structure visible through lowering: one full-hidden token program, scalar metadata loads, vectorized hidden loads, a K reduction, and a single vectorized output store. The PTX is short enough that the memory traffic pattern is immediately inspectable.

The Pallas token-loop source is intentionally close to that grid shape, but XLA/Pallas lowers it through a custom call carrying embedded Triton bytecode. The final PTX is about `5.6x` larger by bytes and about `6.1x` larger by lines, with roughly `9.1x` as many textual `ld.global` instructions. That points at a Pallas/XLA codegen issue or at source constructs that Pallas lowers poorly, not at a missing obvious tile from the already-tested Pallas variants.

## Caveats

- This does not prove semantic equivalence for the full MoE block. It is the selected token gather/sum subkernel on the replicated-input repeat-16 weighted shape.
- Timing is cross-harness: upstream Sonic uses a Torch/Triton harness with CUDA-event and wall timings; Pallas uses a JAX harness with steady wall timings and separate compile-inclusive timings.
- The Pallas StableHLO carries embedded Triton bytecode, so this artifact cannot compare Pallas readable TTIR against Sonic TTIR without an additional bytecode decode/extraction path.
- The Pallas value checks are close but not bit-identical to the local references (`max_abs` in the rerun: token-loop `0.0625`, tiled `0.125`, Sonic `0.03125`). Treat this as a codegen shape comparison, not a final numerical-equivalence report.

## Next Actions

1. Find or build a way to decode the embedded Pallas Triton bytecode before XLA's PTX emission. That is the missing middle layer between Pallas source and the bloated final PTX.
2. Reduce the Pallas source to a minimal reproducer that still produces the `82`-load token-loop PTX. This would make an upstream JAX/Pallas issue actionable.
3. If chasing the last gather/sum gap is still important, port Sonic's actual `BLOCK_H=2048`, `BLOCK_K=1` Triton kernel shape through a direct Triton/custom-call path rather than continuing Pallas tile sweeps.
4. Keep production attention on the larger full-block/end-to-end work unless the decoded Pallas IR identifies a small local source rewrite that collapses the PTX.
