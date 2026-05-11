# Faithful Sonic Gather/Sum Pallas Benchmark

Date: 2026-05-11

## Scope

This note isolates the fixed-top-k token gather/sum kernel used in the SonicMoE
local-MoE comparison. The `pallas_triton_faithful` backend is intended as a
source-faithful Pallas control for Sonic's Triton kernel:

- one program per token,
- inner hidden-tile loop,
- explicit `BLOCK_K`-style loop,
- fp32 accumulator,
- hidden/K masks,
- repeat loop,
- weighted and no-weight branches.

The goal is to answer whether the remaining Pallas-vs-Sonic gap is caused by
source-level recipe differences or by compiler/runtime/codegen differences.

## Hardware And Shape

- Cluster: CoreWeave RNO2A
- Device request: `GH200x1`
- Shape: `tokens=8192`, `hidden=2048`, `experts=8`, `topk=4`
- Dtype: BF16
- Mode: weighted gather/sum
- Timing: `warmup=5`, `steps=20`

## Iris Jobs

- Smoke: `/dlwh/sonicmoe-faithful-smoke-k4-20260511-115451`
- Main sweep: `/dlwh/sonicmoe-faithful-sweep-k4-20260511-115620`
- Add-on `BLOCK_H=4096` sweep:
  `/dlwh/sonicmoe-faithful-sweep-k4-h4096-20260511-120218`
- PTX comparison:
  - `/dlwh/sonicmoe-faithful-ptx2-20260511-121635`
  - `/dlwh/sonicmoe-autotune-id-20260511-121825`
  - `/dlwh/sonicmoe-selected-ptx-20260511-121937`
  - `/dlwh/sonicmoe-pallas-ptx-match-20260511-122056`

## Result

| Path/config | Steady time | Median time | Notes |
| --- | ---: | ---: | --- |
| Real Sonic Triton | `0.0710ms` CUDA event / `0.0829ms` wall | `0.0708ms` CUDA event / `0.0822ms` wall | Sonic autotuned kernel |
| Best faithful Pallas, `BLOCK_H=4096`, `BLOCK_K=4`, `warps=8` | `0.1158ms` wall | `0.1152ms` wall | Best manual sweep result |
| Faithful Pallas, `BLOCK_H=512`, `BLOCK_K=1`, `warps=4` | `0.1175ms` wall | `0.1174ms` wall | Best result from first sweep |
| Current block-token Pallas, `hidden_block=128`, `warps=4` | `0.1161ms` wall | `0.1159ms` wall | Existing non-faithful path |
| Token-loop Pallas, `hidden_block=256`, `warps=4` | `0.1601ms` wall | `0.1598ms` wall | Older source-shape attempt |
| Token-kblock Pallas, `hidden_block=256`, `warps=4` | `0.1628ms` wall | `0.1628ms` wall | Grid-split variant |

## Interpretation

Making the Pallas source shape faithful does not close the isolated token
gather/sum gap. The best faithful Pallas configuration is effectively tied with
the existing block-token Pallas path and remains about 40% slower than real
Sonic wall time, or about 63% slower than Sonic CUDA-event kernel time.

That result removes the major source-recipe objection for this isolated kernel:
the comparison control now includes `BLOCK_K`, fp32 accumulation, masks, repeat
loop, and weighted/no-weight branching.

The follow-up PTX check found that Sonic autotuned to `BLOCK_H=2048`,
`BLOCK_K=2`, `warps=8`. With that same tile, faithful Pallas PTX is smaller
than Sonic's selected PTX (`8.2KB` / `271` lines vs `19.6KB` / `522` lines), but
has more global loads (`13` vs `9`) and more integer address arithmetic. The
remaining isolated gap now looks like scalarized / duplicated address and load
work in the Pallas lowering plus any residual custom-call/runtime launch tax,
not a missing source-level loop.
